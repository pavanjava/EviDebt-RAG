import math
from statistics import pvariance

from evidebt_rag.verifier import Verifier


def compute_signals(verifier: Verifier, question: str, obligation: dict, support_floor: float = 0.3):
    """s_i (noisy-OR support), r_i (max contradiction), u_i (entropy + disagreement)
    over P_i = obligation['passages'] (eq 4-6). If empty: s=0, r=0, u=1."""
    passages = obligation["passages"]
    if not passages:
        return 0.0, 0.0, 1.0

    probs = verifier.predict(question, obligation["statement"], passages)  # [[p_sup,p_ref,p_neu], ...]
    sups = [p[0] for p in probs]
    refs = [p[1] for p in probs]

    s_i = 1.0 - math.prod(1 - p for p in sups)
    r_i = max(refs)

    mean = [sum(p[k] for p in probs) / len(probs) for k in range(3)]
    entropy = -sum(m * math.log(m) for m in mean if m > 0) / math.log(3)
    disagreement = pvariance(sups) * 4 if len(sups) > 1 else 0.0
    u_i = 0.5 * (entropy + min(disagreement, 1.0))

    return s_i, r_i, u_i


def is_resolved(obligation: dict, theta_s: float, theta_r: float, theta_u: float) -> bool:
    return obligation["s"] >= theta_s and obligation["r"] <= theta_r and obligation["u"] <= theta_u


def blocked_count(obligations: list[dict], theta_s: float, theta_r: float, theta_u: float) -> int:
    resolved = [is_resolved(o, theta_s, theta_r, theta_u) for o in obligations]
    count = 0
    for o in obligations:
        if not o.get("active", True):
            continue
        if any(not resolved[p] for p in o["parents"] if obligations[p].get("active", True)):
            count += 1
    return count


def compute_debt(obligations: list[dict], cfg: dict) -> tuple[float, float]:
    """D_t, normalised D_tilde_t (eq 7-8). Decomposed-away (inactive) obligations are
    excluded; their sub-obligations carry the debt instead."""
    alpha, beta, gamma, lam = cfg["alpha"], cfg["beta"], cfg["gamma"], cfg["lambda_"]
    theta_s, theta_r, theta_u = cfg["theta_s"], cfg["theta_r"], cfg["theta_u"]

    active = [o for o in obligations if o.get("active", True)]
    m_t = blocked_count(obligations, theta_s, theta_r, theta_u)
    n = len(active)

    d_t = sum(alpha * (1 - o["s"]) + beta * o["r"] + gamma * o["u"] for o in active) + lam * m_t
    denom = (alpha + beta + gamma) * n + lam * n
    d_tilde = d_t / denom if denom > 0 else 1.0
    return d_t, min(max(d_tilde, 0.0), 1.0)
