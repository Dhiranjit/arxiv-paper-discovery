# label_taxonomy.py
# Maps raw arXiv labels → 27 grouped categories for multi-label classification.
# Multi-hot encoding uses the deduplicated group indices.

from __future__ import annotations
from typing import Optional

# fmt: off
LABEL_TO_GROUP: dict[str, str] = {
    # ── Machine Learning ──────────────────────────────────────────────
    "cs.LG": "Machine Learning", "stat.ML": "Machine Learning",
    "cs.AI": "Machine Learning", "cs.DL":   "Machine Learning",

    # ── Computer Vision ───────────────────────────────────────────────
    "cs.CV": "Computer Vision",

    # ── NLP / Computational Linguistics ───────────────────────────────
    "cs.CL": "Natural Language Processing", "cmp-lg": "Natural Language Processing",

    # ── Robotics ──────────────────────────────────────────────────────
    "cs.RO": "Robotics",

    # ── Security & Cryptography ───────────────────────────────────────
    "cs.CR": "Security and Cryptography",

    # ── Systems & Control ─────────────────────────────────────────────
    "cs.SY":  "Systems and Control", "eess.SY": "Systems and Control",
    "eess.SP":"Systems and Control", "eess.IV": "Systems and Control",
    "eess.AS":"Systems and Control", "cs.DC":   "Systems and Control",
    "cs.NI":  "Systems and Control", "cs.AR":   "Systems and Control",
    "cs.PF":  "Systems and Control", "cs.ET":   "Systems and Control",
    "cs.SC":  "Systems and Control", "cs.OS":   "Systems and Control",
    "cs.MS":  "Systems and Control",

    # ── Information Theory ────────────────────────────────────────────
    "cs.IT": "Information Theory", "math.IT": "Information Theory",

    # ── HCI & Social Computing ────────────────────────────────────────
    "cs.HC": "Human Computer Interaction", "cs.SI": "Human Computer Interaction",
    "cs.CY": "Human Computer Interaction", "cs.MM": "Human Computer Interaction",
    "cs.SD": "Human Computer Interaction", "cs.IR": "Human Computer Interaction",

    # ── CS Theory & Algorithms ────────────────────────────────────────
    "cs.DS": "CS Theory and Algorithms", "cs.CC": "CS Theory and Algorithms",
    "cs.LO": "CS Theory and Algorithms", "cs.DM": "CS Theory and Algorithms",
    "cs.GT": "CS Theory and Algorithms", "cs.FL": "CS Theory and Algorithms",
    "cs.CG": "CS Theory and Algorithms", "cs.PL": "CS Theory and Algorithms",
    "cs.NA": "CS Theory and Algorithms", "cs.CE": "CS Theory and Algorithms",
    "cs.MA": "CS Theory and Algorithms", "cs.DB": "CS Theory and Algorithms",
    "cs.GR": "CS Theory and Algorithms", "cs.SE": "CS Theory and Algorithms",
    "cs.NE": "CS Theory and Algorithms",

    # ── CS Other ──────────────────────────────────────────────────────
    "cs.GL": "Computer Science Other", "cs.OH": "Computer Science Other",
    "comp-gas": "Computer Science Other",

    # ── High Energy Physics ───────────────────────────────────────────
    "hep-ph": "High Energy Physics", "hep-th": "High Energy Physics",
    "hep-ex": "High Energy Physics", "hep-lat": "High Energy Physics",

    # ── Quantum Physics ───────────────────────────────────────────────
    "quant-ph": "Quantum Physics",

    # ── Gravitational Physics ─────────────────────────────────────────
    "gr-qc": "Gravitational Physics",

    # ── Nuclear Physics ───────────────────────────────────────────────
    "nucl-th": "Nuclear Physics", "nucl-ex": "Nuclear Physics",

    # ── Astrophysics ──────────────────────────────────────────────────
    "astro-ph":    "Astrophysics", "astro-ph.GA": "Astrophysics",
    "astro-ph.CO": "Astrophysics", "astro-ph.SR": "Astrophysics",
    "astro-ph.HE": "Astrophysics", "astro-ph.IM": "Astrophysics",
    "astro-ph.EP": "Astrophysics",

    # ── Condensed Matter Physics ──────────────────────────────────────
    "cond-mat.mtrl-sci":  "Condensed Matter Physics",
    "cond-mat.mes-hall":  "Condensed Matter Physics",
    "cond-mat.str-el":    "Condensed Matter Physics",
    "cond-mat.stat-mech": "Condensed Matter Physics",
    "cond-mat.supr-con":  "Condensed Matter Physics",
    "cond-mat.soft":      "Condensed Matter Physics",
    "cond-mat.dis-nn":    "Condensed Matter Physics",
    "cond-mat.quant-gas": "Condensed Matter Physics",
    "cond-mat.other":     "Condensed Matter Physics",
    "cond-mat":           "Condensed Matter Physics",
    "mtrl-th":            "Condensed Matter Physics",
    "supr-con":           "Condensed Matter Physics",

    # ── Physics Other ─────────────────────────────────────────────────
    "physics.optics":   "Physics Other", "physics.flu-dyn":  "Physics Other",
    "physics.comp-ph":  "Physics Other", "physics.chem-ph":  "Physics Other",
    "physics.soc-ph":   "Physics Other", "physics.ins-det":  "Physics Other",
    "physics.atom-ph":  "Physics Other", "physics.app-ph":   "Physics Other",
    "physics.bio-ph":   "Physics Other", "physics.plasm-ph": "Physics Other",
    "physics.data-an":  "Physics Other", "physics.gen-ph":   "Physics Other",
    "physics.class-ph": "Physics Other", "physics.med-ph":   "Physics Other",
    "physics.acc-ph":   "Physics Other", "physics.geo-ph":   "Physics Other",
    "physics.ao-ph":    "Physics Other", "physics.space-ph": "Physics Other",
    "physics.hist-ph":  "Physics Other", "physics.ed-ph":    "Physics Other",
    "physics.atm-clus": "Physics Other", "physics.pop-ph":   "Physics Other",
    "nlin.CD":  "Physics Other", "nlin.SI":  "Physics Other",
    "nlin.PS":  "Physics Other", "nlin.AO":  "Physics Other",
    "nlin.CG":  "Physics Other", "chao-dyn": "Physics Other",
    "patt-sol": "Physics Other", "adap-org": "Physics Other",
    "atom-ph":  "Physics Other", "plasm-ph": "Physics Other",
    "ao-sci":   "Physics Other", "acc-phys": "Physics Other",
    "chem-ph":  "Physics Other", "funct-an": "Physics Other",

    # ── Mathematical Physics ──────────────────────────────────────────
    "math.MP": "Mathematical Physics", "math-ph": "Mathematical Physics",

    # ── Statistics & Probability ──────────────────────────────────────
    "math.PR": "Statistics and Probability", "stat.ME": "Statistics and Probability",
    "stat.TH": "Statistics and Probability", "math.ST": "Statistics and Probability",
    "stat.AP": "Statistics and Probability", "stat.CO": "Statistics and Probability",
    "stat.OT": "Statistics and Probability", "bayes-an": "Statistics and Probability",

    # ── Optimization & Numerical Methods ──────────────────────────────
    "math.OC": "Optimization and Numerical Methods",
    "math.NA": "Optimization and Numerical Methods",

    # ── Pure Mathematics ──────────────────────────────────────────────
    "math.CO": "Pure Mathematics", "math.AG": "Pure Mathematics",
    "math.AP": "Pure Mathematics", "math.NT": "Pure Mathematics",
    "math.DG": "Pure Mathematics", "math.DS": "Pure Mathematics",
    "math.FA": "Pure Mathematics", "math.RT": "Pure Mathematics",
    "math.GT": "Pure Mathematics", "math.GR": "Pure Mathematics",
    "math.CA": "Pure Mathematics", "math.QA": "Pure Mathematics",
    "math.RA": "Pure Mathematics", "math.AT": "Pure Mathematics",
    "math.LO": "Pure Mathematics", "math.AC": "Pure Mathematics",
    "math.OA": "Pure Mathematics", "math.SP": "Pure Mathematics",
    "math.SG": "Pure Mathematics", "math.CT": "Pure Mathematics",
    "math.MG": "Pure Mathematics", "math.CV": "Pure Mathematics",
    "math.KT": "Pure Mathematics", "math.GN": "Pure Mathematics",
    "math.GM": "Pure Mathematics", "math.HO": "Pure Mathematics",
    "alg-geom": "Pure Mathematics", "q-alg":   "Pure Mathematics",
    "dg-ga":    "Pure Mathematics", "solv-int": "Pure Mathematics",

    # ── Quantitative Biology ──────────────────────────────────────────
    "q-bio.QM": "Quantitative Biology", "q-bio.PE": "Quantitative Biology",
    "q-bio.NC": "Quantitative Biology", "q-bio.BM": "Quantitative Biology",
    "q-bio.MN": "Quantitative Biology", "q-bio.GN": "Quantitative Biology",
    "q-bio.TO": "Quantitative Biology", "q-bio.CB": "Quantitative Biology",
    "q-bio.SC": "Quantitative Biology", "q-bio.OT": "Quantitative Biology",
    "q-bio":    "Quantitative Biology",

    # ── Quantitative Finance & Economics ──────────────────────────────
    "q-fin.EC": "Quantitative Finance and Economics",
    "q-fin.ST": "Quantitative Finance and Economics",
    "q-fin.MF": "Quantitative Finance and Economics",
    "q-fin.CP": "Quantitative Finance and Economics",
    "q-fin.GN": "Quantitative Finance and Economics",
    "q-fin.RM": "Quantitative Finance and Economics",
    "q-fin.PM": "Quantitative Finance and Economics",
    "q-fin.PR": "Quantitative Finance and Economics",
    "q-fin.TR": "Quantitative Finance and Economics",
    "econ.GN":  "Quantitative Finance and Economics",
    "econ.EM":  "Quantitative Finance and Economics",
    "econ.TH":  "Quantitative Finance and Economics",
}
# fmt: on

# Sorted canonical group list — index = position in multi-hot vector
GROUPS: list[str] = sorted({
    # Computer Science
    "Machine Learning",
    "Computer Vision",
    "Natural Language Processing",
    "Robotics",
    "Security and Cryptography",
    "Systems and Control",
    "Information Theory",
    "Human Computer Interaction",
    "CS Theory and Algorithms",
    "Computer Science Other",
    # Physics
    "High Energy Physics",
    "Quantum Physics",
    "Gravitational Physics",
    "Nuclear Physics",
    "Astrophysics",
    "Condensed Matter Physics",
    "Physics Other",
    # Mathematics
    "Mathematical Physics",
    "Statistics and Probability",
    "Optimization and Numerical Methods",
    "Pure Mathematics",
    # Interdisciplinary
    "Quantitative Biology",
    "Quantitative Finance and Economics",
})

GROUP_TO_IDX: dict[str, int] = {g: i for i, g in enumerate(GROUPS)}
NUM_CLASSES = len(GROUPS)

GROUP_TO_IDX: dict[str, int] = {g: i for i, g in enumerate(GROUPS)}
NUM_CLASSES = len(GROUPS)  # 23 — fits under target; expand cs.other if signal warrants it


def labels_to_multihot(raw_labels: list[str]) -> list[int]:
    """Convert a paper's raw arXiv labels to a multi-hot vector.

    Handles:
    - Unknown labels → silently skipped (maps to nothing, not 'other')
    - Cross-listed duplicates → deduplicated via set
    - Returns a dense int list of length NUM_CLASSES
    """
    groups = {
        LABEL_TO_GROUP[lbl]
        for lbl in raw_labels
        if lbl in LABEL_TO_GROUP
    }
    vec = [0] * NUM_CLASSES
    for g in groups:
        vec[GROUP_TO_IDX[g]] = 1
    return vec


def multihot_to_groups(vec: list[int]) -> list[str]:
    """Decode a multi-hot vector back to group names."""
    return [GROUPS[i] for i, v in enumerate(vec) if v == 1]

