import random

from sure_rag.pairs import perturb_sentence


def build_obligation_triples(ds, pool_indices: list[int], rng: random.Random) -> list[dict]:
    """Training pairs for the obligation-level verifier: (question, passage, label),
    constructed from HotpotQA's gold/supporting-fact structure to mirror the exact
    4 evidence-quality distinctions Tier A tests (full / hard-insufficient / irrelevant /
    refuted)."""
    triples = []

    for idx in pool_indices:
        ex = ds[idx]
        question = ex["question"]
        titles = ex["context"]["title"]
        sentences = ex["context"]["sentences"]
        gold_titles = set(ex["supporting_facts"]["title"])

        sent_ids_by_title = {}
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

            sid = sent_ids_by_title.get(title, [None])[0]
            if sid is not None and sid < len(sents):
                other_sents = [s for tl in titles for s in sentences[titles.index(tl)] if tl != title]
                perturbed_sent = perturb_sentence(sents[sid], rng, other_sents)
                new_sents = list(sents)
                new_sents[sid] = perturbed_sent
                triples.append({"claim": question, "passage": "".join(new_sents), "label": "refute"})

    return triples
