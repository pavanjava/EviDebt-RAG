import argparse
import json
import sys
from pathlib import Path

import joblib
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_pooled_corpus_and_questions
from common.eval import exact_match, f1_score
from common.generate import ollama_generate, openai_generate
from common.qdrant_index import QdrantIndex
from sure_rag.aggregate import compute_u, compute_z, decide
from sure_rag.verifier import Verifier

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "sure_rag"
MODEL_DIR = Path(__file__).resolve().parent


def build_prompt(question: str, passages: list[dict]) -> str:
    context = "\n".join(f"[{i+1}] {p['text']}" for i, p in enumerate(passages))
    return (
        f"Answer the question using only the passages below. "
        f"Give a short, direct answer with no explanation.\n\n"
        f"Passages:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


def generate(gen_name: str, config: dict, prompt: str) -> str:
    if gen_name == "openai":
        cfg = config["generators"]["openai"]
        return openai_generate(prompt, cfg["model"])
    cfg = config["generators"]["ollama"]
    return ollama_generate(prompt, cfg["model"], cfg["url"])


def run(config: dict, n_questions: int, generators: list[str], k0_values: list[int]):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    corpus, questions = load_pooled_corpus_and_questions(
        seed=config["dataset"]["seed"], d_common_size=n_questions
    )
    print(f"pooled corpus: {len(corpus)} paragraphs, D_common: {len(questions)} questions")

    index = QdrantIndex(config["qdrant"], config["embedding"]["model"])
    index.build(corpus)

    sr_cfg = config["sure_rag"]
    verifier = Verifier(seeds=sr_cfg["seeds"])
    aggregator = joblib.load(MODEL_DIR / "aggregator.joblib")
    thresholds = json.loads((MODEL_DIR / "thresholds.json").read_text())
    tau_sr, beta_sr = thresholds["tau_sr"], thresholds["beta_sr"]
    k_step, t_max = sr_cfg["k_step"], sr_cfg["t_max"]

    for gen_name in generators:
        for k0 in k0_values:
            records = []
            for q in tqdm(questions, desc=f"{gen_name} k0={k0}"):
                k = k0
                for round_i in range(t_max):
                    passages = index.retrieve(q["question"], k)
                    prompt = build_prompt(q["question"], passages)
                    answer = generate(gen_name, config, prompt)

                    z = compute_z(verifier, q["question"], answer, passages)
                    decision, s = decide(aggregator, z, tau_sr, beta_sr)

                    if decision != "Insufficient" or round_i == t_max - 1:
                        break
                    k += k_step

                abstained = not (decision == "Supported" and s >= tau_sr)
                final_answer = None if abstained else answer
                em = 0.0 if abstained else exact_match(answer, q["answer"])
                f1 = 0.0 if abstained else f1_score(answer, q["answer"])

                records.append(
                    {
                        "id": q["id"],
                        "question": q["question"],
                        "gold_answer": q["answer"],
                        "final_k": k,
                        "rounds": round_i + 1,
                        "decision": decision,
                        "score": s,
                        "abstained": abstained,
                        "generated_answer": final_answer,
                        "em": em,
                        "f1": f1,
                    }
                )

            out_file = RESULTS_DIR / f"{gen_name}_k{k0}.jsonl"
            with open(out_file, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            answered = [r for r in records if not r["abstained"]]
            coverage = len(answered) / len(records)
            summary = {
                "generator": gen_name,
                "k0": k0,
                "n": len(records),
                "coverage": coverage,
                "em_overall": sum(r["em"] for r in records) / len(records),
                "f1_overall": sum(r["f1"] for r in records) / len(records),
                "em_answered": sum(r["em"] for r in answered) / len(answered) if answered else 0.0,
                "f1_answered": sum(r["f1"] for r in answered) / len(answered) if answered else 0.0,
            }
            with open(RESULTS_DIR / f"{gen_name}_k{k0}_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="override D_common size")
    parser.add_argument(
        "--generators", nargs="+", default=["ollama", "openai"], choices=["ollama", "openai"]
    )
    parser.add_argument("--k", nargs="+", type=int, default=[5, 10], help="K0 starting depth")
    args = parser.parse_args()

    with open(Path(__file__).resolve().parent.parent / "config.yaml") as f:
        config = yaml.safe_load(f)

    n = args.n or config["dataset"]["d_common_size"]
    run(config, n, args.generators, args.k)
