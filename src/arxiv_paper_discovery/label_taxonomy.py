"""
Maps raw arXiv category codes to 25 taxonomy labels for multi-label classification.

Exports:
    CATEGORY_TO_LABEL   dict mapping arXiv category codes to taxonomy label names
    LABEL_TO_IDX        dict mapping taxonomy label names to class indices
    IDX_TO_LABEL        dict mapping class indices to taxonomy label names
    NUM_CLASSES         total number of taxonomy labels (25)

    categories_to_labels(categories)  arXiv codes → label names
    labels_to_multihot(group_names)   label names → multi-hot vector
    multihot_to_labels(vec)           multi-hot vector → label names
"""


# fmt: off
CATEGORY_TO_LABEL: dict[str, str] = {
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

    # ── Signal Processing & Control ───────────────────────────────────
    "cs.SY":  "Signal Processing and Control", "eess.SY": "Signal Processing and Control",
    "eess.SP":"Signal Processing and Control", "eess.IV": "Signal Processing and Control",
    "eess.AS":"Signal Processing and Control",

    # ── Computer Systems & Networking ─────────────────────────────────
    "cs.DC": "Computer Systems and Networking", "cs.NI": "Computer Systems and Networking",
    "cs.AR": "Computer Systems and Networking", "cs.PF": "Computer Systems and Networking",
    "cs.ET": "Computer Systems and Networking", "cs.SC": "Computer Systems and Networking",
    "cs.OS": "Computer Systems and Networking", "cs.MS": "Computer Systems and Networking",

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

    # ── Applied & Interdisciplinary Physics ───────────────────────────
    "physics.bio-ph":  "Applied and Interdisciplinary Physics",
    "physics.chem-ph": "Applied and Interdisciplinary Physics",
    "physics.med-ph":  "Applied and Interdisciplinary Physics",
    "physics.soc-ph":  "Applied and Interdisciplinary Physics",
    "physics.geo-ph":  "Applied and Interdisciplinary Physics",
    "physics.ao-ph":   "Applied and Interdisciplinary Physics",
    "physics.space-ph":"Applied and Interdisciplinary Physics",
    "physics.data-an": "Applied and Interdisciplinary Physics",
    "chem-ph": "Applied and Interdisciplinary Physics",
    "ao-sci":  "Applied and Interdisciplinary Physics",

    # ── Nonlinear Dynamics ────────────────────────────────────────────
    "nlin.CD":  "Nonlinear Dynamics", "nlin.SI":  "Nonlinear Dynamics",
    "nlin.PS":  "Nonlinear Dynamics", "nlin.AO":  "Nonlinear Dynamics",
    "nlin.CG":  "Nonlinear Dynamics", "chao-dyn": "Nonlinear Dynamics",
    "patt-sol": "Nonlinear Dynamics", "adap-org": "Nonlinear Dynamics",

    # ── Physics Other ─────────────────────────────────────────────────
    "physics.optics":   "Physics Other", "physics.flu-dyn":  "Physics Other",
    "physics.comp-ph":  "Physics Other", "physics.ins-det":  "Physics Other",
    "physics.atom-ph":  "Physics Other", "physics.app-ph":   "Physics Other",
    "physics.plasm-ph": "Physics Other", "physics.gen-ph":   "Physics Other",
    "physics.class-ph": "Physics Other", "physics.acc-ph":   "Physics Other",
    "physics.hist-ph":  "Physics Other", "physics.ed-ph":    "Physics Other",
    "physics.atm-clus": "Physics Other", "physics.pop-ph":   "Physics Other",
    "atom-ph":  "Physics Other", "plasm-ph": "Physics Other",
    "acc-phys": "Physics Other",

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
    "funct-an": "Pure Mathematics",

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
LABELS: list[str] = sorted({
    # Computer Science
    "Machine Learning",
    "Computer Vision",
    "Natural Language Processing",
    "Robotics",
    "Security and Cryptography",
    "Signal Processing and Control",
    "Computer Systems and Networking",
    "Information Theory",
    "Human Computer Interaction",
    "CS Theory and Algorithms",
    # Physics
    "High Energy Physics",
    "Quantum Physics",
    "Gravitational Physics",
    "Nuclear Physics",
    "Astrophysics",
    "Condensed Matter Physics",
    "Applied and Interdisciplinary Physics",
    "Nonlinear Dynamics",
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

LABEL_TO_IDX: dict[str, int] = {g: i for i, g in enumerate(LABELS)}
IDX_TO_LABEL: dict[int, str] = {i: g for g, i in LABEL_TO_IDX.items()}
NUM_CLASSES = len(LABELS)  # 25


def categories_to_labels(categories: list[str]) -> list[str]:
    """
    Convert raw arXiv categories to a deduplicated list of taxonomy label names.
    """
    return list({CATEGORY_TO_LABEL[cat] for cat in categories if cat in CATEGORY_TO_LABEL})


def labels_to_multihot(group_names: list[str]) -> list[float]:
    """
    Convert a list of taxonomy group names to a multi-hot vector.
    Returns a dense float list of length NUM_CLASSES.
    """
    vec = [0.0] * NUM_CLASSES
    for g in group_names:
        if g in LABEL_TO_IDX:
            vec[LABEL_TO_IDX[g]] = 1.0
    return vec


def multihot_to_labels(vec: list[int]) -> list[str]:
    """Decode a multi-hot vector back to taxonomy group names."""
    return [LABELS[i] for i, v in enumerate(vec) if v == 1]

