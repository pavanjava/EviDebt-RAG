import argparse
import json
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import load_pooled_corpus_and_questions
from common.eval import exact_match, f1_score
from common.generate import ollama_generate, openai_generate
from common.qdrant_index import QdrantIndex

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "pyrag"
MAX_SUBQ = 4
BOOST_K = 5


def generate(gen_name: str, config: dict, prompt: str) -> str:
    if gen_name == "ollama":
        cfg = config["generators"]["ollama"]
        return ollama_generate(prompt, cfg["model"], cfg["url"])
    cfg = config["generators"]["openai"]
    return openai_generate(prompt, cfg["model"])


def decompose(question: str, gen_name: str, config: dict) -> list[str]:
    prompt = (
        "Break this question into the minimal ordered list of simple sub-questions "
        "needed to answer it. One sub-question per line, no numbering.\n\n"
        f"Question: {question}\nSub-questions:"
    )
    out = generate(gen_name, config, prompt)
    subqs = [l.strip("-* ").strip() for l in out.splitlines() if l.strip()]
    return subqs[:MAX_SUBQ] or [question]


def answer_subq(subq: str, passages: list[dict], gen_name: str, config: dict) -> str:
    context = "\n".join(f"[{i+1}] {p['text']}" for i, p in enumerate(passages))
    prompt = (
        "Answer the question using only the passages below. "
        "If the passages don't contain the answer, reply exactly: unknown\n\n"
        f"Passages:\n{context}\n\nQuestion: {subq}\nAnswer:"
    )
    return generate(gen_name, config, prompt)


def synthesize(question: str, subq_answers: list[tuple[str, str]], gen_name: str, config: dict) -> str:
    facts = "\n".join(f"- {sq}: {a}" for sq, a in subq_answers)
    prompt = (
        f"Given these facts:\n{facts}\n\n"
        f"Answer the question with a short, direct answer, no explanation.\n"
        f"Question: {question}\nAnswer:"
    )
    return generate(gen_name, config, prompt)


def run(config: dict, n_questions: int, generators: list[str], k_values: list[int]):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    corpus, questions = load_pooled_corpus_and_questions(
        seed=config["dataset"]["seed"], d_common_size=n_questions
    )
    print(f"pooled corpus: {len(corpus)} paragraphs, D_common: {len(questions)} questions")

    index = QdrantIndex(config["qdrant"], config["embedding"]["model"])
    index.build(corpus)

    for gen_name in generators:
        for k in k_values:
            records = []
            for q in tqdm(questions, desc=f"{gen_name} k={k}"):
                subqs = decompose(q["question"], gen_name, config)
                subq_answers = []
                retries = 0
                for subq in subqs:
                    passages = index.retrieve(subq, k)
                    ans = answer_subq(subq, passages, gen_name, config)
                    if ans.strip().lower() == "unknown":
                        retries += 1
                        passages = index.retrieve(subq, k + BOOST_K)
                        ans = answer_subq(subq, passages, gen_name, config)
                    subq_answers.append((subq, ans))

                answer = synthesize(q["question"], subq_answers, gen_name, config)

                em = exact_match(answer, q["answer"])
                f1 = f1_score(answer, q["answer"])
                records.append(
                    {
                        "id": q["id"],
                        "question": q["question"],
                        "gold_answer": q["answer"],
                        "sub_questions": subqs,
                        "sub_answers": [a for _, a in subq_answers],
                        "retries": retries,
                        "generated_answer": answer,
                        "em": em,
                        "f1": f1,
                    }
                )

            out_file = RESULTS_DIR / f"{gen_name}_k{k}.jsonl"
            with open(out_file, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            summary = {
                "generator": gen_name,
                "k": k,
                "n": len(records),
                "em": sum(r["em"] for r in records) / len(records),
                "f1": sum(r["f1"] for r in records) / len(records),
                "avg_retries": sum(r["retries"] for r in records) / len(records),
            }
            with open(RESULTS_DIR / f"{gen_name}_k{k}_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="override D_common size")
    parser.add_argument(
        "--generators", nargs="+", default=["ollama", "openai"], choices=["ollama", "openai"]
    )
    parser.add_argument("--k", nargs="+", type=int, default=[5, 10])
    args = parser.parse_args()

    with open(Path(__file__).resolve().parent.parent / "config.yaml") as f:
        config = yaml.safe_load(f)

    n = args.n or config["dataset"]["d_common_size"]
    run(config, n, args.generators, args.k)
