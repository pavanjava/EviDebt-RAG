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
from evidebt_rag.controller import RhoTracker, run_episode
from evidebt_rag.debt import is_resolved
from evidebt_rag.obligations import extract_obligations
from evidebt_rag.verifier import Verifier

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "evidebt_rag" / "tier_b"


def generate(gen_name: str, config: dict, prompt: str) -> str:
    if gen_name == "openai":
        cfg = config["generators"]["openai"]
        return openai_generate(prompt, cfg["model"])
    cfg = config["generators"]["ollama"]
    return ollama_generate(prompt, cfg["model"], cfg["url"])


def build_final_prompt(question: str, obligations: list[dict], cfg: dict) -> tuple[str, list[str]]:
    resolved = [
        o
        for o in obligations
        if o.get("active", True) and is_resolved(o, cfg["theta_s"], cfg["theta_r"], cfg["theta_u"])
    ]
    # fall back to everything gathered so far (like Vanilla RAG) rather than answering
    # from nothing when the controller didn't fully resolve any obligation
    source = resolved if resolved else [o for o in obligations if o.get("active", True)]

    passages = []
    seen = set()
    for o in source:
        for p in o["passages"]:
            if p["title"] not in seen:
                seen.add(p["title"])
                passages.append(p)

    context = "\n".join(f"[{i+1}] {p['text']}" for i, p in enumerate(passages))
    prompt = (
        f"Answer the question using only the passages below. "
        f"Give a short, direct answer with no explanation.\n\n"
        f"Passages:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    return prompt, [p["title"] for p in passages]


def run(config: dict, n_questions: int, generators: list[str], k_values: list[int]):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    corpus, questions = load_pooled_corpus_and_questions(
        seed=config["dataset"]["seed"], d_common_size=n_questions
    )
    print(f"pooled corpus: {len(corpus)} paragraphs, D_common: {len(questions)} questions")

    index = QdrantIndex(config["qdrant"], config["embedding"]["model"])
    index.build(corpus)
    verifier = Verifier(seeds=config["sure_rag"]["seeds"])
    ed_cfg = config["evidebt_rag"]

    for gen_name in generators:
        for k in k_values:
            cfg = dict(ed_cfg, retrieve_k=k)
            rho = RhoTracker(["retrieve", "reformulate", "decompose", "counter_retrieve", "verify"])
            records = []

            for q in tqdm(questions, desc=f"{gen_name} k={k}"):
                obligations = extract_obligations(q["question"], gen_name, config)
                obligations, trajectory, outcome = run_episode(
                    q["question"], obligations, index.retrieve, verifier, rho, cfg, gen_name, config
                )

                abstained = outcome != "stop"
                prompt, cited = build_final_prompt(q["question"], obligations, cfg)
                answer = generate(gen_name, config, prompt)
                em = exact_match(answer, q["answer"])
                f1 = f1_score(answer, q["answer"])

                n_retrieve = sum(1 for t in trajectory if t.get("action") == "retrieve")
                n_verify = sum(1 for t in trajectory if t.get("action") == "verify")

                records.append(
                    {
                        "id": q["id"],
                        "question": q["question"],
                        "gold_answer": q["answer"],
                        "n_obligations": len(obligations),
                        "outcome": outcome,
                        "steps": len(trajectory) - 1,
                        "n_retrieve": n_retrieve,
                        "n_verify": n_verify,
                        "final_d_tilde": trajectory[-1]["d_tilde"],
                        "cited_titles": cited,
                        "abstained": abstained,
                        "generated_answer": answer,
                        "em": em,
                        "f1": f1,
                    }
                )

            out_file = RESULTS_DIR / f"{gen_name}_k{k}.jsonl"
            with open(out_file, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            # every question now gets an answer (fallback to gathered evidence on abstain);
            # "stop_rate" tracks how often the controller fully resolved obligations vs fell back
            stopped = [r for r in records if not r["abstained"]]
            summary = {
                "generator": gen_name,
                "k": k,
                "n": len(records),
                "coverage": 1.0,
                "stop_rate": len(stopped) / len(records),
                "em_overall": sum(r["em"] for r in records) / len(records),
                "f1_overall": sum(r["f1"] for r in records) / len(records),
                "em_answered": sum(r["em"] for r in stopped) / len(stopped) if stopped else 0.0,
                "f1_answered": sum(r["f1"] for r in stopped) / len(stopped) if stopped else 0.0,
                "avg_steps": sum(r["steps"] for r in records) / len(records),
                "avg_retrieve": sum(r["n_retrieve"] for r in records) / len(records),
                "avg_verify": sum(r["n_verify"] for r in records) / len(records),
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
