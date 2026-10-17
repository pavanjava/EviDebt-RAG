import argparse
import random
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_train_split
from sure_rag.pairs import balance_triples, build_triples
from sure_rag.verifier import train_seed


def run(config: dict, epochs: int, sample_questions: int | None):
    sr_cfg = config["sure_rag"]
    ds, _, pool = load_train_split(config["dataset"]["seed"], config["dataset"]["d_tune_size"])
    rng = random.Random(config["dataset"]["seed"])

    if sample_questions:
        pool = pool[:sample_questions]

    triples = build_triples(ds, pool, rng)
    triples = balance_triples(triples, sr_cfg["pairs_per_class"], rng)

    split = int(len(triples) * 0.9)
    train_triples, val_triples = triples[:split], triples[split:]
    print(f"triples: {len(triples)} (train {len(train_triples)}, val {len(val_triples)})")

    for seed in sr_cfg["seeds"]:
        print(f"--- seed {seed} ---")
        train_seed(seed, train_triples, val_triples, sr_cfg["verifier_model"], epochs=epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--sample-questions", type=int, default=None)
    args = parser.parse_args()

    with open(Path(__file__).resolve().parent.parent / "config.yaml") as f:
        config = yaml.safe_load(f)

    run(config, args.epochs, args.sample_questions)
