import random

from datasets import load_dataset


def load_pooled_corpus_and_questions(seed: int, d_common_size: int):
    """Load HotpotQA dev-distractor split: dedup paragraphs into a pooled corpus,
    sample D_common questions with a fixed seed."""
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

    corpus = {}  # title -> paragraph text
    for ex in ds:
        titles = ex["context"]["title"]
        sentences = ex["context"]["sentences"]
        for title, sents in zip(titles, sentences):
            if title not in corpus:
                corpus[title] = "".join(sents)

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    sample_indices = indices[:d_common_size]

    questions = [
        {"id": ds[i]["id"], "question": ds[i]["question"], "answer": ds[i]["answer"]}
        for i in sample_indices
    ]

    return corpus, questions


def load_common_examples(seed: int, d_common_size: int):
    """Same D_common sample as load_pooled_corpus_and_questions, but returns the raw
    dataset + indices (for Tier A, which needs full context/supporting_facts)."""
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    sample_indices = indices[:d_common_size]

    return ds, sample_indices


def load_train_split(seed: int, dtune_size: int):
    """HotpotQA train split, shuffled once with a fixed seed: the first dtune_size
    indices form D_tune, the rest are the pool available for pair-label construction."""
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    dtune_indices = indices[:dtune_size]
    pool_indices = indices[dtune_size:]

    dtune = [
        {"id": ds[i]["id"], "question": ds[i]["question"], "answer": ds[i]["answer"]}
        for i in dtune_indices
    ]

    return ds, dtune, pool_indices


def load_dtune(seed: int, size: int):
    _, dtune, _ = load_train_split(seed, size)
    return dtune
