"""
scripts/run_serve.py

Online inference server: FastAPI + uvicorn.

Usage
-----
python scripts/run_serve.py --checkpoint saved_models/my_run/best_model

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
GET  /health            → {"status": "ok", "device": str}
"""

import argparse
from importlib.resources import files
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from arxiv_paper_discovery.predictor import ArticleTagger


def _load_dashboard_html() -> str | None:
    try:
        return files("arxiv_paper_discovery.web").joinpath("dashboard.html").read_text(encoding="utf-8")
    except Exception:
        return None


class PredictRequest(BaseModel):
    title: str
    abstract: str = ""


class PredictResponse(BaseModel):
    title: str
    predicted_tags: list[str]
    tag_probabilities: dict[str, float]


def create_app(checkpoint: Path) -> FastAPI:
    tagger = ArticleTagger(checkpoint)
    dashboard_html = _load_dashboard_html()

    app = FastAPI(title="arxiv paper tagger", version="1.0")

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        if dashboard_html is not None:
            return HTMLResponse(dashboard_html)
        return HTMLResponse("<h3>dashboard.html not found</h3>", status_code=404)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "device": tagger.device}

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        result = tagger.predict(request.title, request.abstract)
        return PredictResponse(
            title=request.title,
            predicted_tags=result["tags"],
            tag_probabilities=result["probabilities"],
        )

    @app.post("/predict_batch", response_model=list[PredictResponse])
    def predict_batch(requests: list[PredictRequest]) -> list[PredictResponse]:
        if not requests:
            raise HTTPException(status_code=422, detail="Request list is empty.")
        results = tagger.predict(
            title=[r.title for r in requests],
            abstract=[r.abstract for r in requests],
        )
        return [
            PredictResponse(
                title=req.title,
                predicted_tags=res["tags"],
                tag_probabilities=res["probabilities"],
            )
            for req, res in zip(requests, results)
        ]

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the arxiv paper tagger via FastAPI.")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to a HuggingFace model directory saved via save_pretrained().")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Host to bind to (default: 127.0.0.1). Use 0.0.0.0 to expose externally.")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to listen on (default: 8000).")
    args = parser.parse_args()

    app = create_app(args.checkpoint)

    base = f"http://{args.host}:{args.port}"
    print(f"\nServer running at {base}")
    print(f"  Dashboard  : {base}/")
    print(f"  POST       : {base}/predict")
    print(f"  POST       : {base}/predict_batch")
    print(f"  GET        : {base}/health")
    print("\nPress Ctrl-C to stop.\n")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
