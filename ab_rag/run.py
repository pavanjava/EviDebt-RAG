import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import CrossEncoder
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_pooled_corpus_and_questions
from common.eval import exact_match, f1_score, normalize_answer
from common.generate import ollama_generate, openai_generate_with_prob
from common.qdrant_index import QdrantIndex

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "ab_rag"


def build_prompt(question: str, passages: list[dict]) -> str:
    context = "\n".join(f"[{i+1}] {p['text']}" for i, p in enumerate(passages))
    return (
        f"Answer the question using only the passages below. "
        f"Give a short, direct answer with no explanation.\n\n"
        f"Passages:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


def generate_with_s1(gen_name: str, config: dict, prompt: str) -> tuple[str, float]:
    if gen_name == "openai":
        cfg = config["generators"]["openai"]
        return openai_generate_with_prob(prompt, cfg["model"])

    cfg = config["generators"]["ollama"]
    answers = [ollama_generate(prompt, cfg["model"], cfg["url"]) for _ in range(3)]
    counts = Counter(normalize_answer(a) for a in answers)
    modal_norm, modal_count = counts.most_common(1)[0]
    answer = next(a for a in answers if normalize_answer(a) == modal_norm)
    return answer, modal_count / 3


def compute_s2(model, answer: str, passages: list[dict]) -> float:
    a_vec = model.encode(f"query: {answer}", normalize_embeddings=True)
    e_vecs = model.encode([f"passage: {p['text']}" for p in passages], normalize_embeddings=True)
    e_vec = np.mean(e_vecs, axis=0)
    e_vec = e_vec / np.linalg.norm(e_vec)
    cos = float(np.dot(a_vec, e_vec))
    return 0.5 * (cos + 1)


def compute_s3(reranker: CrossEncoder, question: str, passages: list[dict]) -> float:
    scores = [float(s) for s in reranker.predict([(question, p["text"]) for p in passages])]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return 0.0
    normed = [(s - lo) / (hi - lo) for s in scores]
    return statistics.pvariance(normed)


def run(config: dict, n_questions: int, generators: list[str], k0_values: list[int]):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    corpus, questions = load_pooled_corpus_and_questions(
        seed=config["dataset"]["seed"], d_common_size=n_questions
    )
    print(f"pooled corpus: {len(corpus)} paragraphs, D_common: {len(questions)} questions")

    index = QdrantIndex(config["qdrant"], config["embedding"]["model"])
    index.build(corpus)
    reranker = CrossEncoder(config["reranker"]["model"])

    ab_cfg = config["ab_rag"]
    tau, k_step, t_max = ab_cfg["tau"], ab_cfg["k_step"], ab_cfg["t_max"]
    w1, w2, w3 = ab_cfg["weights"]

    for gen_name in generators:
        for k0 in k0_values:
            records = []
            for q in tqdm(questions, desc=f"{gen_name} k0={k0}"):
                k = k0
                for round_i in range(t_max):
                    passages = index.retrieve(q["question"], k)
                    prompt = build_prompt(q["question"], passages)
                    answer, s1 = generate_with_s1(gen_name, config, prompt)
                    s2 = compute_s2(index.model, answer, passages)
                    s3 = compute_s3(reranker, q["question"], passages)
                    conf = min(max(w1 * s1 + w2 * s2 + w3 * s3, 0.0), 1.0)
                    if conf >= tau or round_i == t_max - 1:
                        break
                    k += k_step

                em = exact_match(answer, q["answer"])
                f1 = f1_score(answer, q["answer"])
                records.append(
                    {
                        "id": q["id"],
                        "question": q["question"],
                        "gold_answer": q["answer"],
                        "final_k": k,
                        "rounds": round_i + 1,
                        "s1": s1,
                        "s2": s2,
                        "s3": s3,
                        "conf": conf,
                        "generated_answer": answer,
                        "em": em,
                        "f1": f1,
                    }
                )

            out_file = RESULTS_DIR / f"{gen_name}_k{k0}.jsonl"
            with open(out_file, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            summary = {
                "generator": gen_name,
                "k0": k0,
                "n": len(records),
                "em": sum(r["em"] for r in records) / len(records),
                "f1": sum(r["f1"] for r in records) / len(records),
                "avg_rounds": sum(r["rounds"] for r in records) / len(records),
                "avg_final_k": sum(r["final_k"] for r in records) / len(records),
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
