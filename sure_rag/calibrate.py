import argparse
import json
import random
import sys
from pathlib import Path

import joblib
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_pooled_corpus_and_questions, load_train_split
from common.eval import exact_match
from common.generate import openai_generate
from common.qdrant_index import QdrantIndex
from sure_rag.aggregate import calibrate, compute_u, compute_z, fit_aggregator
from sure_rag.pairs import build_question_examples
from sure_rag.verifier import Verifier

OUT_DIR = Path(__file__).resolve().parent
CALIB_K = 10
DECISION = {"support": "Supported", "refute": "Refuted", "neutral": "Insufficient"}


def build_prompt(question: str, passages: list[dict]) -> str:
    context = "\n".join(f"[{i+1}] {p['text']}" for i, p in enumerate(passages))
    return (
        f"Answer the question using only the passages below. "
        f"Give a short, direct answer with no explanation.\n\n"
        f"Passages:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


def run(config: dict, sample_questions: int | None):
    sr_cfg = config["sure_rag"]
    ds, dtune, pool = load_train_split(config["dataset"]["seed"], config["dataset"]["d_tune_size"])
    rng = random.Random(config["dataset"]["seed"])

    if sample_questions:
        dtune = dtune[:sample_questions]

    agg_pool = pool[: sr_cfg["aggregator_questions"]]
    examples = build_question_examples(ds, agg_pool, rng)
    print(f"fitting aggregator on {len(examples)} question-level examples")

    verifier = Verifier(seeds=sr_cfg["seeds"])
    aggregator = fit_aggregator(examples, verifier)
    joblib.dump(aggregator, OUT_DIR / "aggregator.joblib")

    corpus, _ = load_pooled_corpus_and_questions(
        config["dataset"]["seed"], config["dataset"]["d_common_size"]
    )
    index = QdrantIndex(config["qdrant"], config["embedding"]["model"])
    index.build(corpus)

    openai_cfg = config["generators"]["openai"]
    records = []
    for q in tqdm(dtune, desc="calibrating on D_tune"):
        passages = index.retrieve(q["question"], CALIB_K)
        prompt = build_prompt(q["question"], passages)
        answer = openai_generate(prompt, openai_cfg["model"])

        z = compute_z(verifier, q["question"], answer, passages)
        u = compute_u(z)
        pi = dict(zip(aggregator.classes_, aggregator.predict_proba([z])[0]))
        pi_support = pi.get("support", 0.0)
        label = max(pi, key=pi.get)
        decision = DECISION[label]
        correct = exact_match(answer, q["answer"])
        records.append({"pi_support": pi_support, "u": u, "decision": decision, "correct": correct})

    tau_sr, beta_sr = calibrate(records)
    coverage = sum(1 for r in records if r["decision"] == "Supported") / len(records)
    print(f"calibrated: tau_sr={tau_sr}, beta_sr={beta_sr}, raw Supported-rate={coverage:.3f}")

    with open(OUT_DIR / "thresholds.json", "w") as f:
        json.dump({"tau_sr": tau_sr, "beta_sr": beta_sr}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-questions", type=int, default=None)
    args = parser.parse_args()

    with open(Path(__file__).resolve().parent.parent / "config.yaml") as f:
        config = yaml.safe_load(f)

    run(config, args.sample_questions)
