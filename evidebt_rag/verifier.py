from pathlib import Path

import torch
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


def train_seed(
    seed: int, train_triples: list[dict], val_triples: list[dict], model_name: str, epochs: int = 3
):
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    train_ds = PairDataset(train_triples, tokenizer)
    val_ds = PairDataset(val_triples, tokenizer)

    out_dir = MODEL_DIR / f"seed{seed}"
    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        report_to=[],
        learning_rate=2e-5,
        warmup_steps=50,
        max_grad_norm=1.0,
        seed=seed,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds)
    trainer.train()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)


class Verifier:
    """Ensemble of fine-tuned seed checkpoints, trained specifically on EviDebt-RAG's
    per-obligation, per-passage judgment task (not reused from SURE-RAG).
    h_theta(q, c, e) -> [p_sup, p_ref, p_neu]."""

    def __init__(self, seeds: list[int], device: str | None = None):
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR / f"seed{seeds[0]}"))
        self.models = []
        for s in seeds:
            m = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR / f"seed{s}"))
            m.to(self.device).eval()
            self.models.append(m)
        self._cache = {}

    def predict(self, question: str, claim: str, passages: list[dict]) -> list[list[float]]:
        results = []
        to_score = []
        keys = []
        for p in passages:
            key = (question, claim, p["title"])
            keys.append(key)
            if key not in self._cache:
                to_score.append(p)

        if to_score:
            text_a = [claim] * len(to_score)
            text_b = [p["text"] for p in to_score]
            enc = self.tokenizer(
                text_a, text_b, truncation=True, max_length=256, padding=True, return_tensors="pt"
            ).to(self.device)

            probs_sum = None
            with torch.no_grad():
                for m in self.models:
                    logits = m(**enc).logits
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    probs_sum = probs if probs_sum is None else probs_sum + probs
            probs_avg = probs_sum / len(self.models)

            for p, probs in zip(to_score, probs_avg):
                self._cache[(question, claim, p["title"])] = probs.tolist()

        for key in keys:
            results.append(self._cache[key])
        return results
