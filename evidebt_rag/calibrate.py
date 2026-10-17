import argparse
import copy
import itertools
import random
import sys
from pathlib import Path

import yaml
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_train_split
from common.qdrant_index import QdrantIndex
from evidebt_rag.controller import RhoTracker, run_episode
from evidebt_rag.debt import compute_debt
from evidebt_rag.obligations import extract_obligations
from evidebt_rag.tier_a import PoolIndex, build_conditions
from evidebt_rag.verifier import Verifier

CONDITIONS = ["Full", "Partial", "Hard-insufficient", "Irrelevant", "Refuted"]

THETA_S_GRID = [0.6, 0.7, 0.8]
THETA_R_GRID = [0.2, 0.3, 0.4]
THETA_U_GRID = [0.5, 0.6, 0.7]


def run(config: dict, n_questions: int, gen_name: str):
    """Runs the controller once per D_tune question/condition (using config.yaml's current
    thresholds to gather evidence), then cheaply re-scores ordering accuracy / AUROC for each
    (theta_s, theta_r, theta_u) in the grid directly from the gathered signals -- avoids
    re-running the full retrieve/reformulate/decompose loop per grid point."""
    ds, _, _ = load_train_split(config["dataset"]["seed"], config["dataset"]["d_tune_size"])
    rng = random.Random(config["dataset"]["seed"])
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    dtune_indices = indices[: config["dataset"]["d_tune_size"]]
    sample = dtune_indices[:n_questions]

    embed_model = QdrantIndex(config["qdrant"], config["embedding"]["model"]).model
    verifier = Verifier(seeds=config["sure_rag"]["seeds"])
    cfg = dict(config["evidebt_rag"], retrieve_k=5)
    rho = RhoTracker(["retrieve", "reformulate", "decompose", "counter_retrieve", "verify"])

    records = []
    for idx in sample:
        ex = ds[idx]
        question = ex["question"]
        obligations_seed = extract_obligations(question, gen_name, config)
        conditions = build_conditions(ex, rng)
        row = {}
        for cond_name in CONDITIONS:
            pool_index = PoolIndex(embed_model, conditions[cond_name])
            obligations = copy.deepcopy(obligations_seed)
            obligations, _, _ = run_episode(
                question, obligations, pool_index.retrieve, verifier, rho, cfg, gen_name, config
            )
            # keep the full list (including inactive/decomposed obligations) so "parents"
            # indices, which point into this list, stay valid -- compute_debt already
            # skips inactive obligations internally
            row[cond_name] = obligations
        records.append(row)
    print(f"gathered evidence for {len(records)} questions x {len(CONDITIONS)} conditions")

    best = None
    for theta_s, theta_r, theta_u in itertools.product(THETA_S_GRID, THETA_R_GRID, THETA_U_GRID):
        grid_cfg = dict(cfg, theta_s=theta_s, theta_r=theta_r, theta_u=theta_u)
        d_tildes = []
        for row in records:
            d_tildes.append({c: compute_debt(row[c], grid_cfg)[1] for c in CONDITIONS})

        ordering_correct = sum(
            1
            for d in d_tildes
            if d["Full"] < d["Partial"] < d["Hard-insufficient"] <= d["Irrelevant"]
        )
        ordering_accuracy = ordering_correct / len(d_tildes)

        non_full = [max(d["Partial"], d["Hard-insufficient"], d["Irrelevant"]) for d in d_tildes]
        labels = [0] * len(d_tildes) + [1] * len(d_tildes)
        scores = [d["Full"] for d in d_tildes] + non_full
        auroc = roc_auc_score(labels, scores)

        result = (ordering_accuracy, auroc, theta_s, theta_r, theta_u)
        print(f"theta_s={theta_s} theta_r={theta_r} theta_u={theta_u} "
              f"ordering_accuracy={ordering_accuracy:.3f} auroc={auroc:.3f}")
        if best is None or result[:2] > best[:2]:
            best = result

    print(f"\nBEST: theta_s={best[2]} theta_r={best[3]} theta_u={best[4]} "
          f"ordering_accuracy={best[0]:.3f} auroc={best[1]:.3f}")
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--generator", default="openai", choices=["ollama", "openai"])
    args = parser.parse_args()

    with open(Path(__file__).resolve().parent.parent / "config.yaml") as f:
        config = yaml.safe_load(f)

    run(config, args.n, args.generator)
