import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.generate import ollama_generate, openai_generate
from evidebt_rag.debt import compute_debt, compute_signals
from evidebt_rag.obligations import decompose as decompose_obligation


def _generate(gen_name: str, config: dict, prompt: str) -> str:
    if gen_name == "openai":
        cfg = config["generators"]["openai"]
        return openai_generate(prompt, cfg["model"])
    cfg = config["generators"]["ollama"]
    return ollama_generate(prompt, cfg["model"], cfg["url"])


class RhoTracker:
    """Beta(1,1) weak-prior success-rate estimate per action, updated online."""

    def __init__(self, actions: list[str]):
        self.counts = {a: [1.0, 1.0] for a in actions}  # [alpha, beta]

    def mean(self, action: str) -> float:
        a, b = self.counts[action]
        return a / (a + b)

    def update(self, action: str, success: bool):
        idx = 0 if success else 1
        self.counts[action][idx] += 1


def unresolved(obligations: list[dict], theta_s: float, theta_r: float, theta_u: float, idx: int) -> bool:
    o = obligations[idx]
    return not (o["s"] >= theta_s and o["r"] <= theta_r and o["u"] <= theta_u)


def unblocked(obligations: list[dict], theta_s: float, theta_r: float, theta_u: float, idx: int) -> bool:
    o = obligations[idx]
    return all(
        not obligations[p].get("active", True) or not unresolved(obligations, theta_s, theta_r, theta_u, p)
        for p in o["parents"]
    )


def candidate_action(obligations: list[dict], rho: RhoTracker, cfg: dict, blacklist: set = frozenset()):
    """Returns (action_name, target_idx, delta_hat, cost) for the best of
    {retrieve, reformulate, decompose, verify}, or None if nothing is applicable."""
    theta_s, theta_r, theta_u = cfg["theta_s"], cfg["theta_r"], cfg["theta_u"]
    alpha, beta, gamma = cfg["alpha"], cfg["beta"], cfg["gamma"]
    max_attempts = cfg["max_attempts"]

    best = None

    def consider(candidate):
        nonlocal best
        if best is None or (candidate[2] / (candidate[3] + 1e-6)) > (best[2] / (best[3] + 1e-6)):
            best = candidate

    retrieve_candidates = [
        i
        for i, o in enumerate(obligations)
        if o.get("active", True)
        and o["n_queried"] == 0
        and unresolved(obligations, theta_s, theta_r, theta_u, i)
        and unblocked(obligations, theta_s, theta_r, theta_u, i)
        and ("retrieve", i) not in blacklist
    ]
    def deficiency(i):
        # how far obligation i is from resolved, on either the support or uncertainty axis
        # (not just support) -- a high-s/high-u obligation still badly needs work.
        return alpha * (1 - obligations[i]["s"]) + gamma * obligations[i]["u"]

    if retrieve_candidates:
        i = max(retrieve_candidates, key=deficiency)
        delta = rho.mean("retrieve") * deficiency(i)
        consider(("retrieve", i, delta, cfg["c_ret"]))

    reformulate_candidates = [
        i
        for i, o in enumerate(obligations)
        if o.get("active", True)
        and 1 <= o["n_queried"] < max_attempts
        and unresolved(obligations, theta_s, theta_r, theta_u, i)
        and unblocked(obligations, theta_s, theta_r, theta_u, i)
        and ("reformulate", i) not in blacklist
    ]
    if reformulate_candidates:
        i = max(reformulate_candidates, key=deficiency)
        delta = rho.mean("reformulate") * deficiency(i)
        consider(("reformulate", i, delta, cfg["c_reform"]))

    decompose_candidates = [
        i
        for i, o in enumerate(obligations)
        if o.get("active", True)
        and o["n_queried"] >= max_attempts
        and unresolved(obligations, theta_s, theta_r, theta_u, i)
        and ("decompose", i) not in blacklist
    ]
    if decompose_candidates:
        i = max(decompose_candidates, key=deficiency)
        delta = rho.mean("decompose") * deficiency(i)
        consider(("decompose", i, delta, cfg["c_decomp"]))

    counter_candidates = [
        i
        for i, o in enumerate(obligations)
        if o.get("active", True)
        and o["n_queried"] >= 1
        and o["r"] > theta_r / 2
        and ("counter_retrieve", i) not in blacklist
    ]
    if counter_candidates:
        i = max(counter_candidates, key=lambda i: obligations[i]["r"])
        delta = rho.mean("counter_retrieve") * beta * obligations[i]["r"]
        consider(("counter_retrieve", i, delta, cfg["c_counter"]))

    pool_titles = {p["title"] for o in obligations for p in o["passages"]}
    verify_candidates = [
        i
        for i, o in enumerate(obligations)
        if o.get("active", True)
        and o["u"] > theta_u
        and len(o["passages"]) >= 2
        and ("verify", i) not in blacklist
        and pool_titles - {p["title"] for p in o["passages"]}  # only if the pool has new evidence to rescan
    ]
    if verify_candidates:
        i = max(verify_candidates, key=lambda i: gamma * obligations[i]["u"])
        delta = rho.mean("verify") * gamma * obligations[i]["u"]
        consider(("verify", i, delta, cfg["c_ver"] * len(obligations[i]["passages"])))

    return best


def _pool(obligations: list[dict]) -> list[dict]:
    seen, pool = set(), []
    for o in obligations:
        for p in o["passages"]:
            if p["title"] not in seen:
                seen.add(p["title"])
                pool.append(p)
    return pool


def _short_answer(obligation: dict, gen_name: str, config: dict) -> str:
    if obligation.get("answer") is not None:
        return obligation["answer"]
    context = "\n".join(f"[{i+1}] {p['text']}" for i, p in enumerate(obligation["passages"]))
    prompt = (
        f"Answer the question using only the passages below. Give a short, direct answer.\n\n"
        f"Passages:\n{context}\n\nQuestion: {obligation['statement']}\nAnswer:"
    )
    answer = _generate(gen_name, config, prompt) if context else "unknown"
    obligation["answer"] = answer
    return answer


def _build_query(obligation: dict, obligations: list[dict], gen_name: str, config: dict) -> str:
    if not obligation["parents"]:
        return obligation["statement"]

    facts = []
    for p in obligation["parents"]:
        parent = obligations[p]
        facts.append(f"- {parent['statement']}: {_short_answer(parent, gen_name, config)}")

    prompt = (
        "Rewrite the question below into a self-contained search query, replacing any pronoun "
        "or vague reference ('that', 'this', 'it', 'the person', etc.) with the specific fact "
        "it refers to. Reply with only the rewritten query.\n\n"
        f"Known facts:\n{chr(10).join(facts)}\n\nQuestion: {obligation['statement']}\nRewritten:"
    )
    return _generate(gen_name, config, prompt).strip()


def _build_reformulate_query(obligation: dict, gen_name: str, config: dict) -> str:
    found = "; ".join(p["title"] for p in obligation["passages"]) or "nothing useful"
    prompt = (
        "A search for the question below did not find enough evidence. Passages found so far: "
        f"{found}. Propose a different search query likely to find better evidence. "
        "Reply with only the query.\n\n"
        f"Question: {obligation['statement']}\nQuery:"
    )
    return _generate(gen_name, config, prompt).strip()


def _build_counter_query(obligation: dict, gen_name: str, config: dict) -> str:
    answer = _short_answer(obligation, gen_name, config)
    prompt = (
        f"Find evidence that confirms or refutes the following claim. "
        f"Reply with only a search query.\n\n"
        f"Claim: {obligation['statement']} Answer: {answer}\nQuery:"
    )
    return _generate(gen_name, config, prompt).strip()


def run_episode(
    question: str,
    obligations: list[dict],
    retrieve_fn,
    verifier,
    rho: RhoTracker,
    cfg: dict,
    gen_name: str = None,
    config: dict = None,
):
    """Runs the closed loop (doc §5.12, Retrieve/Verify only) until Stop, budget, or Abstain.
    obligations: seeded list of {type, statement, parents} — mutated in place with
    passages/n_queried/s/r/u. Returns (obligations, trajectory, outcome) where outcome
    is 'stop', 'abstain_budget', or 'abstain_kappa'.
    gen_name/config: when given, a dependent obligation's retrieval query has its parent's
    resolved answer substituted in before retrieving (fixes unresolved anaphora like "that film")."""
    for o in obligations:
        o["passages"] = []
        o["n_queried"] = 0
        o["s"], o["r"], o["u"] = 0.0, 0.0, 1.0
        o["answer"] = None
        o.setdefault("active", True)

    for o in obligations:
        o["s"], o["r"], o["u"] = compute_signals(verifier, question, o)
    _, d_tilde = compute_debt(obligations, cfg)

    trajectory = [{"step": 0, "action": None, "d_tilde": d_tilde}]
    blacklist: set = set()

    for step in range(1, cfg["budget"] + 1):
        if d_tilde <= cfg["tau_stop"]:
            return obligations, trajectory, "stop"

        best = candidate_action(obligations, rho, cfg, blacklist)
        if best is None:
            return obligations, trajectory, "abstain_kappa"

        action, target, delta, cost = best
        ratio = delta / (cost + 1e-6)
        if ratio < cfg["kappa"]:
            return obligations, trajectory, "abstain_kappa"

        obligation = obligations[target]
        if action in ("retrieve", "reformulate"):
            if action == "reformulate" and gen_name and config:
                query = _build_reformulate_query(obligation, gen_name, config)
            elif gen_name and config:
                query = _build_query(obligation, obligations, gen_name, config)
            else:
                query = obligation["statement"]
            new_passages = retrieve_fn(query, cfg["retrieve_k"])
            existing_titles = {p["title"] for p in obligation["passages"]}
            obligation["passages"].extend(p for p in new_passages if p["title"] not in existing_titles)
            obligation["n_queried"] += 1
        elif action == "counter_retrieve":  # targeted search for confirming/refuting evidence
            query = _build_counter_query(obligation, gen_name, config) if gen_name and config else obligation["statement"]
            new_passages = retrieve_fn(query, cfg["retrieve_k"])
            existing_titles = {p["title"] for p in obligation["passages"]}
            obligation["passages"].extend(p for p in new_passages if p["title"] not in existing_titles)
        elif action == "verify":  # rescan against the full accumulated pool
            obligation["passages"] = _pool(obligations)
        else:  # decompose: split the stuck obligation into 2 sub-obligations
            subs = decompose_obligation(obligation, gen_name, config) if gen_name and config else []
            if not subs:
                blacklist.add(("decompose", target))
                trajectory.append({"step": step, "action": "decompose_noop", "target": target, "d_tilde": d_tilde})
                continue
            for sub in subs:
                sub["parents"] = list(obligation["parents"])
                sub["passages"] = []
                sub["n_queried"] = 0
                sub["s"], sub["r"], sub["u"] = compute_signals(verifier, question, sub)
                sub["answer"] = None
            new_indices = list(range(len(obligations), len(obligations) + len(subs)))
            obligations.extend(subs)
            for o in obligations:
                if target in o["parents"]:
                    o["parents"] = [p for p in o["parents"] if p != target] + new_indices
            obligation["active"] = False
            _, new_d_tilde = compute_debt(obligations, cfg)
            rho.update("decompose", success=(d_tilde - new_d_tilde) > cfg["eps_p"])
            d_tilde = new_d_tilde
            trajectory.append({"step": step, "action": "decompose", "target": target, "d_tilde": d_tilde})
            continue

        obligation["s"], obligation["r"], obligation["u"] = compute_signals(verifier, question, obligation)
        _, new_d_tilde = compute_debt(obligations, cfg)

        rho.update(action, success=(d_tilde - new_d_tilde) > cfg["eps_p"])
        d_tilde = new_d_tilde
        trajectory.append({"step": step, "action": action, "target": target, "d_tilde": d_tilde})

    return obligations, trajectory, "abstain_budget"
