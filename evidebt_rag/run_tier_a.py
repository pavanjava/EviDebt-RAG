import argparse
import copy
import json
import random
import sys
from pathlib import Path

import yaml
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_common_examples
from common.qdrant_index import QdrantIndex
from evidebt_rag.controller import RhoTracker, run_episode
from evidebt_rag.obligations import extract_obligations
from evidebt_rag.tier_a import PoolIndex, build_conditions
from evidebt_rag.verifier import Verifier

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "evidebt_rag" / "tier_a"
CONDITIONS = ["Full", "Partial", "Hard-insufficient", "Irrelevant", "Refuted"]


def run(config: dict, n_questions: int, gen_name: str):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ds, sample_indices = load_common_examples(config["dataset"]["seed"], n_questions)
    print(f"Tier A: {len(sample_indices)} questions x {len(CONDITIONS)} conditions")

    embed_model = QdrantIndex(config["qdrant"], config["embedding"]["model"]).model
    verifier = Verifier(seeds=config["sure_rag"]["seeds"])
    cfg = dict(config["evidebt_rag"], retrieve_k=5)
    rng = random.Random(config["dataset"]["seed"])
    rho = RhoTracker(["retrieve", "reformulate", "decompose", "counter_retrieve", "verify"])

    records = []
    for idx in tqdm(sample_indices, desc="Tier A"):
        ex = ds[idx]
        question = ex["question"]
        obligations_seed = extract_obligations(question, gen_name, config)
        conditions = build_conditions(ex, rng)

        row = {"id": ex["id"], "question": question}
        for cond_name in CONDITIONS:
            passages = conditions[cond_name]
            pool_index = PoolIndex(embed_model, passages)
            obligations = copy.deepcopy(obligations_seed)
            obligations, trajectory, outcome = run_episode(
                question, obligations, pool_index.retrieve, verifier, rho, cfg, gen_name, config
            )
            active = [o for o in obligations if o.get("active", True)]
            r_component = sum(o["r"] for o in active) / len(active)
            row[cond_name] = {"d_tilde": trajectory[-1]["d_tilde"], "r_mean": r_component}

        records.append(row)

    with open(RESULTS_DIR / "trajectories.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    ordering_correct = 0
    for r in records:
        d = {c: r[c]["d_tilde"] for c in CONDITIONS}
        if d["Full"] < d["Partial"] < d["Hard-insufficient"] <= d["Irrelevant"]:
            ordering_correct += 1
    ordering_accuracy = ordering_correct / len(records)

    non_full_scores = [
        max(r["Partial"]["d_tilde"], r["Hard-insufficient"]["d_tilde"], r["Irrelevant"]["d_tilde"])
        for r in records
    ]
    auroc_labels = [0] * len(records) + [1] * len(records)
    auroc_scores = [r["Full"]["d_tilde"] for r in records] + non_full_scores
    auroc = roc_auc_score(auroc_labels, auroc_scores)

    refuted_higher = sum(1 for r in records if r["Refuted"]["r_mean"] > r["Full"]["r_mean"])
    refuted_contradiction_rate = refuted_higher / len(records)

    summary = {
        "n": len(records),
        "ordering_accuracy": ordering_accuracy,
        "auroc_full_vs_nonfull": auroc,
        "refuted_contradiction_rate": refuted_contradiction_rate,
        "mean_d_tilde": {c: sum(r[c]["d_tilde"] for r in records) / len(records) for c in CONDITIONS},
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--generator", default="openai", choices=["ollama", "openai"])
    args = parser.parse_args()

    with open(Path(__file__).resolve().parent.parent / "config.yaml") as f:
        config = yaml.safe_load(f)

    n = args.n or config["dataset"]["d_common_size"]
    run(config, n, args.generator)
