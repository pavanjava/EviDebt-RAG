import math

import numpy as np
from sklearn.linear_model import LogisticRegression

from sure_rag.verifier import Verifier

LABELS = ["support", "refute", "neutral"]
LABEL_TO_DECISION = {"support": "Supported", "refute": "Refuted", "neutral": "Insufficient"}


def _entropy(dist: list[float]) -> float:
    return -sum(p * math.log(p) for p in dist if p > 0) / math.log(3)


def compute_z(verifier: Verifier, question: str, claim: str, passages: list[dict]) -> np.ndarray:
    probs = np.array(verifier.predict(question, claim, passages))  # [n_passages, 3]
    mean_dist = probs.mean(axis=0)
    max_sup, max_ref = probs[:, 0].max(), probs[:, 1].max()
    mean_neu = probs[:, 2].mean()
    entropy = _entropy(mean_dist.tolist())
    disagreement = float(probs[:, 0].var())
    conflict = float(min(max_sup, max_ref))
    return np.array(
        [mean_dist[0], mean_dist[1], mean_dist[2], max_sup, max_ref, mean_neu, entropy, disagreement, conflict]
    )


def compute_u(z: np.ndarray) -> float:
    entropy, disagreement, conflict = z[6], z[7], z[8]
    coverage_deficit = 1 - max(z[0], z[1], z[2])
    return float(np.mean([entropy, disagreement, conflict, coverage_deficit]))


def fit_aggregator(examples: list[dict], verifier: Verifier) -> LogisticRegression:
    """examples: [{question, claim, passages, label}] with a full multi-passage
    evidence set per example (see pairs.build_question_examples)."""
    X, y = [], []
    for ex in examples:
        z = compute_z(verifier, ex["question"], ex["claim"], ex["passages"])
        X.append(z)
        y.append(ex["label"])
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


def decide(model: LogisticRegression, z: np.ndarray, tau_sr: float, beta_sr: float):
    """Returns (decision, selective_score)."""
    pi = dict(zip(model.classes_, model.predict_proba([z])[0]))
    u = compute_u(z)
    s = pi.get("support", 0.0) - beta_sr * u
    label = max(pi, key=pi.get)
    decision = LABEL_TO_DECISION[label]
    return decision, s


def calibrate(records: list[dict]) -> tuple[float, float]:
    """records: [{pi_support, u, decision, correct}] from running the pipeline over D_tune.
    Grid-search tau_SR, beta_SR maximizing accuracy on accepted answers at >=50% coverage."""
    best = (0.5, 1.0, -1.0)
    for beta_sr in [0.0, 0.5, 1.0, 1.5, 2.0]:
        scored = [(r["pi_support"] - beta_sr * r["u"], r["decision"], r["correct"]) for r in records]
        for tau_sr in [round(t, 2) for t in np.arange(0.0, 1.01, 0.05)]:
            accepted = [c for s, d, c in scored if d == "Supported" and s >= tau_sr]
            coverage = len(accepted) / len(scored)
            if coverage < 0.5 or not accepted:
                continue
            accuracy = sum(accepted) / len(accepted)
            if accuracy > best[2]:
                best = (tau_sr, beta_sr, accuracy)
    return best[0], best[1]
