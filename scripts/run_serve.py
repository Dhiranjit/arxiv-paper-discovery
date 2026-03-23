"""
scripts/serve.py

Online inference server: FastAPI + Ray Serve.

Ray Serve manages replicas and request batching so multiple concurrent HTTP
requests share the same model instance without race conditions.  FastAPI
provides the typed REST interface.

How Ray Serve is used here
--------------------------
@serve.deployment      → wraps TaggerDeployment into a Ray actor (separate
                          process/memory space).  The model loads once in
                          __init__ and stays resident between requests.

@serve.ingress(app)    → mounts the FastAPI router onto that actor so HTTP
                          requests are routed directly into it.

serve.run(...)         → deploys the actor and starts accepting traffic.
                          num_replicas controls horizontal scale-out; each
                          replica gets its own model copy.  Ray serialises
                          requests within a replica, so there are no
                          concurrent writes to model state.

Usage
-----
# Start the server (blocks until Ctrl-C):
python scripts/serve.py --checkpoint saved_models/my_run

# Open the dashboard in a browser:
open http://localhost:8000/

# Or hit the API directly:
curl -s -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"title": "Attention is all you need", "abstract": "..."}'

curl -s -X POST http://localhost:8000/predict_batch \
     -H "Content-Type: application/json" \
     -d '[{"title": "Paper A"}, {"title": "Paper B", "abstract": "..."}]'

curl http://localhost:8000/health

Endpoints
---------
GET  /                  → interactive dashboard (HTML)
POST /predict           → PredictRequest       → PredictResponse
POST /predict_batch     → list[PredictRequest] → list[PredictResponse]
GET  /health            → {"status": "ok", "threshold": float, "device": str}
"""

import argparse
import time
from importlib.resources import files
from pathlib import Path

import ray
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from ray import serve

from arxiv_paper_discovery.predictor import ArticleTagger


def _load_dashboard_html() -> str | None:
    """Load dashboard HTML bundled as package data."""
    try:
        return files("arxiv_paper_discovery.web").joinpath("dashboard.html").read_text(encoding="utf-8")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    title:    str
    abstract: str = ""


class PredictResponse(BaseModel):
    title:             str
    predicted_tags:    list[str]
    tag_probabilities: dict[str, float]


# ---------------------------------------------------------------------------
# Ray Serve deployment
# ---------------------------------------------------------------------------

app = FastAPI(title="arxiv paper tagger", version="1.0")


@serve.deployment(
    num_replicas=1,                      # increase for higher throughput
    ray_actor_options={"num_gpus": 0},   # set to 1 if GPU available
)
@serve.ingress(app)
class TaggerDeployment:
    """
    One Ray actor = one model loaded into memory.
    Ray serialises requests within a replica — no concurrent model-state
    mutations.  Scale out with num_replicas for parallel throughput.
    """

    def __init__(self, checkpoint_dir: str, threshold: float | None = None) -> None:
        self.tagger = ArticleTagger(
            checkpoint_dir=Path(checkpoint_dir),
            threshold=threshold,
        )
        self.dashboard_html = _load_dashboard_html()

    # ── Dashboard ──────────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    def dashboard(self) -> HTMLResponse:
        """Serve the interactive tagging dashboard."""
        if self.dashboard_html is not None:
            return HTMLResponse(self.dashboard_html)
        return HTMLResponse(
            "<h3>dashboard.html not found in arxiv_paper_discovery.web package data</h3>",
            status_code=404,
        )

    # ── API ────────────────────────────────────────────────────────────────

    @app.get("/health")
    def health(self) -> dict:
        return {
            "status":    "ok",
            "threshold": self.tagger.threshold,
            "device":    self.tagger.device,
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(self, request: PredictRequest) -> PredictResponse:
        result = self.tagger.predict(request.title, request.abstract)
        return PredictResponse(
            title             = request.title,
            predicted_tags    = result["tags"],
            tag_probabilities = result["probabilities"],
        )

    @app.post("/predict_batch", response_model=list[PredictResponse])
    def predict_batch(self, requests: list[PredictRequest]) -> list[PredictResponse]:
        if not requests:
            raise HTTPException(status_code=422, detail="Request list is empty.")

        results = self.tagger.predict(
            title    = [r.title    for r in requests],
            abstract = [r.abstract for r in requests],
        )
        return [
            PredictResponse(
                title             = req.title,
                predicted_tags    = res["tags"],
                tag_probabilities = res["probabilities"],
            )
            for req, res in zip(requests, results)
        ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve the arxiv paper tagger via Ray Serve + FastAPI."
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to a Hugging Face model directory saved via save_pretrained().",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1). Use 0.0.0.0 to expose externally.",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port to listen on (default: 8000).",
    )
    parser.add_argument(
        "--num-replicas", type=int, default=1,
        help="Number of Ray Serve replicas (default: 1). Increase for concurrent load.",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override sigmoid threshold from model config.",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint}")

    ray.init(ignore_reinit_error=True)
    serve.start(http_options={"host": args.host, "port": args.port})

    # .options() overrides deployment config at runtime without touching the decorator
    serve.run(
        TaggerDeployment.options(num_replicas=args.num_replicas).bind(
            checkpoint_dir=str(args.checkpoint),
            threshold=args.threshold,
        )
    )

    base = f"http://{args.host}:{args.port}"
    print(f"\nServer running at {base}")
    print(f"  Dashboard  : {base}/")
    print(f"  POST       : {base}/predict")
    print(f"  POST       : {base}/predict_batch")
    print(f"  GET        : {base}/health")
    print("\nPress Ctrl-C to stop.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        serve.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    main()
