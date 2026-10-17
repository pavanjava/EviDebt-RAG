import random

import numpy as np

from sure_rag.pairs import perturb_sentence


class PoolIndex:
    """Small in-memory e5-similarity retrieval over a fixed per-question passage pool
    (Tier A: EviDebt-RAG "retrieves within the pool", doc §2.2/§3.1)."""

    def __init__(self, model, passages: list[dict]):
        self.model = model
        self.passages = passages
        texts = [f"passage: {p['text']}" for p in passages]
        self.vectors = model.encode(texts, normalize_embeddings=True) if texts else np.zeros((0, 768))

    def retrieve(self, query: str, k: int) -> list[dict]:
        if not self.passages:
            return []
        qvec = self.model.encode(f"query: {query}", normalize_embeddings=True)
        scores = self.vectors @ qvec
        top = np.argsort(-scores)[:k]
        return [self.passages[i] for i in top]


def build_conditions(ex, rng: random.Random) -> dict[str, list[dict]]:
    titles = ex["context"]["title"]
    sentences = ex["context"]["sentences"]
    gold_titles = set(ex["supporting_facts"]["title"])

    all_passages = [{"title": t, "text": "".join(s)} for t, s in zip(titles, sentences)]
    gold = [p for p in all_passages if p["title"] in gold_titles]
    distractors = [p for p in all_passages if p["title"] not in gold_titles]

    sent_ids_by_title = {}
    for t, sid in zip(ex["supporting_facts"]["title"], ex["supporting_facts"]["sent_id"]):
        sent_ids_by_title.setdefault(t, []).append(sid)

    hard_insufficient_gold = []
    for t, s in zip(titles, sentences):
        if t not in gold_titles:
            continue
        drop = set(sent_ids_by_title.get(t, []))
        kept = [sent for i, sent in enumerate(s) if i not in drop]
        hard_insufficient_gold.append({"title": t, "text": "".join(kept)})

    full = gold + distractors

    partial = [p for p in full if p["title"] != gold[0]["title"]] if gold else list(full)

    hard_insufficient = hard_insufficient_gold + distractors

    irrelevant = list(distractors)

    refuted_passage = None
    if gold_titles:
        target_title = ex["supporting_facts"]["title"][0]
        target_sid = ex["supporting_facts"]["sent_id"][0]
        target_sents = sentences[titles.index(target_title)]
        if target_sid < len(target_sents):
            other_sents = [s for tl in titles for s in sentences[titles.index(tl)] if tl != target_title]
            perturbed_sent = perturb_sentence(target_sents[target_sid], rng, other_sents)
            new_sents = list(target_sents)
            new_sents[target_sid] = perturbed_sent
            refuted_passage = {"title": target_title + " (refuted)", "text": "".join(new_sents)}
    refuted = full + ([refuted_passage] if refuted_passage else [])

    return {
        "Full": full,
        "Partial": partial,
        "Hard-insufficient": hard_insufficient,
        "Irrelevant": irrelevant,
        "Refuted": refuted,
    }
