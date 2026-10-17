

/?.≤/import argparse
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

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "vanilla_rag"


def build_prompt(question: str, passages: list[dict]) -> str:
    context = "\n".join(f"[{i+1}] {p['text']}" for i, p in enumerate(passages))
    return (
        f"Answer the question using only the passages below. "
        f"Give a short, direct answer with no explanation.\n\n"
        f"Passages:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


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
                passages = index.retrieve(q["question"], k)
                prompt = build_prompt(q["question"], passages)

                if gen_name == "ollama":
                    cfg = config["generators"]["ollama"]
                    answer = ollama_generate(prompt, cfg["model"], cfg["url"])
                else:
                    cfg = config["generators"]["openai"]
                    answer = openai_generate(prompt, cfg["model"])

                em = exact_match(answer, q["answer"])
                f1 = f1_score(answer, q["answer"])
                records.append(
                    {
                        "id": q["id"],
                        "question": q["question"],
                        "gold_answer": q["answer"],
                        "retrieved_titles": [p["title"] for p in passages],
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
