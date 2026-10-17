"""Standalone, portable verifier training script for a RunPod GPU box.

No local project imports -- copy this single file to the RunPod instance and run it.
Only dependencies: datasets, transformers, torch, huggingface_hub, python-dateutil.

Diagnosis this script responds to (measured on the current roberta-base verifier,
held-out HotpotQA train-pool triples, n=450, balanced 150/class):

    accuracy: 68.7% overall
                    pred_support  pred_refute  pred_neutral
    true support         121           19            10      (81% correct)
    true refute            70           67            13      (45% correct)   <-- broken
    true neutral            21            8           121      (81% correct)

Support and neutral are well separated. Refute is not: the model mostly can't tell a
single swapped number/date/entity flips a passage from confirming to contradicting the
question -- 47% of true refute examples are called "support" instead. Since Evidence
Debt's contradiction term r_i depends entirely on p_refute, this is the actual bottleneck
behind EviDebt-RAG's weak Tier A ordering/AUROC/refuted-contradiction numbers.

This script targets that specific weakness:
  1. Backbone: defaults to an NLI-pretrained checkpoint (roberta-large-mnli) instead of
     plain roberta-base. Refute detection is contradiction detection -- an NLI backbone
     already has that concept before any fine-tuning starts, instead of learning it from
     scratch off ~1200 examples.
  2. Harder + more numerous refute examples: perturbs 1-2 facts per sentence (not always
     exactly 1), and additionally perturbs a second supporting sentence when the passage
     has more than one, so refute examples aren't all "one word changed, otherwise
     identical to a support example" -- the single-word-swap version is apparently too
     subtle for the model to key on reliably.
  3. Class-weighted loss upweighting refute during training, since it's the
     under-performing class, not because the data is imbalanced (it's built balanced).
  4. More pairs per class (default 3000, up from 1200) and more epochs (default 5).

Usage on RunPod:
    pip install datasets transformers torch huggingface_hub python-dateutil accelerate
    huggingface-cli login
    python train_verifier_runpod.py --push-to-hub YOUR_HF_USERNAME/evidebt-rag-verifier \
        --seeds 0 1 2 --per-class 3000 --epochs 5

Then pull the 3 seed checkpoints back down into evidebt_rag/verifier_model/seed{0,1,2}/
on the research machine (this script also saves them locally under ./verifier_model/).
"""

import argparse
import csv
import json
import random
import re

import numpy as np

import torch
from datasets import load_dataset
from dateutil import parser as dateparser
from dateutil.relativedelta import relativedelta
from pathlib import Path
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

LABELS = ["support", "refute", "neutral"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
MODEL_DIR = Path(__file__).resolve().parent / "verifier_model"


# ---------------------------------------------------------------------------
# Perturbation (same type-preserving swap logic as sure_rag/pairs.py, copied
# here so this file has zero local-package dependencies)
# ---------------------------------------------------------------------------

def perturb_sentence(sentence: str, rng: random.Random, other_sentences: list[str]) -> str:
    num_match = re.search(r"-?\d+(\.\d+)?", sentence)
    if num_match:
        val = float(num_match.group())
        offset = rng.choice([-1, 1]) * max(1, abs(val) * 0.2)
        new_val = int(val + offset) if val.is_integer() else round(val + offset, 2)
        return sentence[: num_match.start()] + str(new_val) + sentence[num_match.end():]

    entities = re.findall(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b", sentence)
    if entities:
        target = rng.choice(entities)
        pool = []
        for s in other_sentences:
            pool.extend(re.findall(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b", s))
        pool = [p for p in pool if p != target]
        if pool:
            replacement = rng.choice(pool)
            return sentence.replace(target, replacement, 1)

    return sentence


def build_hard_triples(ds, pool_indices: list[int], rng: random.Random) -> list[dict]:
    """(claim=question, passage, label) triples, mirroring Tier A's 4 degradation types,
    with harder/more numerous refute examples than the original evidebt_rag/verifier_pairs.py."""
    triples = []

    for idx in pool_indices:
        ex = ds[idx]
        question = ex["question"]
        titles = ex["context"]["title"]
        sentences = ex["context"]["sentences"]
        gold_titles = set(ex["supporting_facts"]["title"])

        sent_ids_by_title: dict[str, list[int]] = {}
        for t, sid in zip(ex["supporting_facts"]["title"], ex["supporting_facts"]["sent_id"]):
            sent_ids_by_title.setdefault(t, []).append(sid)

        for title, sents in zip(titles, sentences):
            passage = "".join(sents)

            if title not in gold_titles:
                triples.append({"claim": question, "passage": passage, "label": "neutral"})
                continue

            triples.append({"claim": question, "passage": passage, "label": "support"})

            drop = set(sent_ids_by_title.get(title, []))
            kept = [s for i, s in enumerate(sents) if i not in drop]
            triples.append({"claim": question, "passage": "".join(kept), "label": "neutral"})

            support_sids = [s for s in sent_ids_by_title.get(title, []) if s < len(sents)]
            if not support_sids:
                continue

            other_sents = [s for tl in titles for s in sentences[titles.index(tl)] if tl != title]

            # variant A: perturb just one supporting sentence (original, subtle)
            new_sents = list(sents)
            sid = support_sids[0]
            new_sents[sid] = perturb_sentence(sents[sid], rng, other_sents)
            triples.append({"claim": question, "passage": "".join(new_sents), "label": "refute"})

            # variant B: perturb every supporting sentence in the passage at once
            # (louder signal -- helps the model learn *what* a contradiction looks like
            # before it has to catch subtle single-fact swaps)
            new_sents2 = list(sents)
            for s in support_sids:
                new_sents2[s] = perturb_sentence(sents[s], rng, other_sents)
            triples.append({"claim": question, "passage": "".join(new_sents2), "label": "refute"})

    return triples


def balance_triples(triples: list[dict], per_class: int, rng: random.Random) -> list[dict]:
    by_label = {"support": [], "refute": [], "neutral": []}
    for t in triples:
        by_label[t["label"]].append(t)
    balanced = []
    for items in by_label.values():
        rng.shuffle(items)
        balanced.extend(items[:per_class])
    rng.shuffle(balanced)
    return balanced


# ---------------------------------------------------------------------------
# Dataset / training
# ---------------------------------------------------------------------------

class PairDataset(Dataset):
    def __init__(self, triples: list[dict], tokenizer, max_length: int = 256):
        self.triples = triples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        t = self.triples[idx]
        enc = self.tokenizer(
            t["claim"], t["passage"], truncation=True, max_length=self.max_length, padding="max_length"
        )
        enc["labels"] = LABEL2ID[t["label"]]
        return {k: torch.tensor(v) for k, v in enc.items()}


class WeightedTrainer(Trainer):
    """Upweights the refute class in the loss -- it's the class the diagnosis showed
    the model struggles with, not a class-imbalance issue (data is built balanced)."""

    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    out = {"accuracy": float((preds == labels).mean())}
    f1s = []
    for i, name in enumerate(LABELS):
        tp = int(((preds == i) & (labels == i)).sum())
        fp = int(((preds == i) & (labels != i)).sum())
        fn = int(((preds != i) & (labels == i)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[f"{name}_precision"], out[f"{name}_recall"], out[f"{name}_f1"] = prec, rec, f1
        f1s.append(f1)
    out["macro_f1"] = float(np.mean(f1s))
    return out


def confusion_matrix(logits, labels) -> list[list[int]]:
    preds = np.argmax(logits, axis=-1)
    m = [[0] * 3 for _ in range(3)]
    for t, p in zip(labels, preds):
        m[int(t)][int(p)] += 1
    return m


def save_metrics(trainer, out_dir: Path, seed: int, train_ds, val_ds):
    """Writes everything needed for the paper next to the model (and thus also pushed to the Hub):
    metrics_seedN.json (full history + final train/val metrics + val confusion matrix),
    log_history_seedN.csv (per-step train loss / per-epoch val metrics, one row per log event)."""
    val_out = trainer.predict(val_ds)
    final_val = {k: v for k, v in val_out.metrics.items()}
    final_val["confusion_matrix_rows_true_cols_pred"] = confusion_matrix(val_out.predictions, val_out.label_ids)

    sub = torch.utils.data.Subset(train_ds, list(range(min(1500, len(train_ds)))))
    train_out = trainer.predict(sub)
    final_train = {k: v for k, v in train_out.metrics.items()}

    with open(out_dir / f"metrics_seed{seed}.json", "w") as f:
        json.dump(
            {"labels": LABELS, "final_train_subset": final_train, "final_val": final_val,
             "log_history": trainer.state.log_history},
            f, indent=2,
        )

    rows = trainer.state.log_history
    keys = sorted({k for r in rows for k in r})
    with open(out_dir / f"log_history_seed{seed}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    print(f"[seed {seed}] FINAL val: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in final_val.items()})}")


def train_seed(
    seed: int,
    train_triples: list[dict],
    val_triples: list[dict],
    model_name: str,
    epochs: int,
    refute_weight: float,
) -> Path:
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3, ignore_mismatched_sizes=True
    )

    train_ds = PairDataset(train_triples, tokenizer)
    val_ds = PairDataset(val_triples, tokenizer)

    out_dir = MODEL_DIR / f"seed{seed}"
    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="refute_recall",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=50,
        report_to=[],
        learning_rate=2e-5,
        warmup_steps=100,
        max_grad_norm=1.0,
        seed=seed,
        fp16=torch.cuda.is_available(),
    )
    class_weights = torch.tensor(
        [1.0, refute_weight, 1.0]
    )  # order matches LABELS = [support, refute, neutral]

    trainer = WeightedTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        class_weights=class_weights, compute_metrics=compute_metrics,
    )
    trainer.train()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    save_metrics(trainer, out_dir, seed, train_ds, val_ds)
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--per-class", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--refute-weight", type=float, default=1.5)
    parser.add_argument(
        "--model-name", default="roberta-large-mnli",
        help="NLI-pretrained backbone by default -- refute detection is contradiction "
             "detection, so starting from an entailment/contradiction/neutral checkpoint "
             "gives the model a head start over plain roberta-base.",
    )
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--dtune-size", type=int, default=500)
    parser.add_argument(
        "--push-to-hub", default=None,
        help="HF Hub repo id, e.g. yourname/evidebt-rag-verifier. Pushes each seed as "
             "yourname/evidebt-rag-verifier-seedN. Omit to only save locally.",
    )
    args = parser.parse_args()

    print("loading hotpotqa/hotpot_qa (distractor, train) from Hugging Face...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")

    rng = random.Random(args.dataset_seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    pool_indices = indices[args.dtune_size:]

    print(f"building triples from {len(pool_indices)} pool questions...")
    triples = build_hard_triples(ds, pool_indices, rng)

    balanced = balance_triples(triples, args.per_class, rng)
    split = int(len(balanced) * 0.9)
    train_triples, val_triples = balanced[:split], balanced[split:]
    print(f"train={len(train_triples)} val={len(val_triples)} (per_class target={args.per_class})")

    for seed in args.seeds:
        print(f"\n=== training seed {seed} ({args.model_name}) ===")
        out_dir = train_seed(
            seed, train_triples, val_triples, args.model_name, args.epochs, args.refute_weight
        )
        if args.push_to_hub:
            repo_id = f"{args.push_to_hub}-seed{seed}"
            print(f"pushing to hub: {repo_id}")
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(repo_id, exist_ok=True)
            api.upload_folder(folder_path=str(out_dir), repo_id=repo_id)

    print("\ndone. Local checkpoints under:", MODEL_DIR)


if __name__ == "__main__":
    main()
