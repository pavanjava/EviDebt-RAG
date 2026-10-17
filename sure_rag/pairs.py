import random
import re

from dateutil import parser as dateparser
from dateutil.relativedelta import relativedelta


def _is_yesno(answer: str) -> bool:
    return answer.strip().lower() in ("yes", "no")


def _is_numeric(answer: str) -> bool:
    return bool(re.fullmatch(r"-?\d+(\.\d+)?", answer.strip()))


def _is_date(answer: str) -> bool:
    try:
        dateparser.parse(answer, fuzzy=False)
        return bool(re.search(r"\d", answer))
    except (ValueError, OverflowError):
        return False


def perturb_answer(answer: str, rng: random.Random, other_answers: list[str]) -> str:
    """Type-preserving swap for a Refute pair. No negation prefixes."""
    if _is_yesno(answer):
        return "no" if answer.strip().lower() == "yes" else "yes"

    if _is_numeric(answer):
        val = float(answer)
        offset = rng.choice([-1, 1]) * max(1, abs(val) * 0.2)
        new_val = val + offset
        return str(int(new_val)) if val.is_integer() else str(round(new_val, 2))

    if _is_date(answer):
        try:
            dt = dateparser.parse(answer, fuzzy=False)
            shifted = dt + relativedelta(years=rng.choice([-2, -1, 1, 2]))
            return shifted.strftime("%d %B %Y")
        except (ValueError, OverflowError):
            pass

    return rng.choice(other_answers)


def perturb_sentence(sentence: str, rng: random.Random, other_sentences: list[str]) -> str:
    """Type-preserving swap of one fact inside a sentence (for Tier A's Refuted condition).
    No negation prefixes. Tries number -> date-year -> capitalized-entity swap, in that order."""
    num_match = re.search(r"-?\d+(\.\d+)?", sentence)
    if num_match:
        val = float(num_match.group())
        offset = rng.choice([-1, 1]) * max(1, abs(val) * 0.2)
        new_val = int(val + offset) if val.is_integer() else round(val + offset, 2)
        return sentence[: num_match.start()] + str(new_val) + sentence[num_match.end() :]

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


def build_triples(ds, pool_indices: list[int], rng: random.Random) -> list[dict]:
    """(question, claim, passage, label) triples: label in {support, refute, neutral}."""
    all_answers = [ds[i]["answer"] for i in pool_indices]
    triples = []

    for idx in pool_indices:
        ex = ds[idx]
        question = ex["question"]
        answer = ex["answer"]
        gold_titles = set(ex["supporting_facts"]["title"])
        titles = ex["context"]["title"]
        sentences = ex["context"]["sentences"]

        distractors = []
        for title, sents in zip(titles, sentences):
            passage = "".join(sents)
            if title in gold_titles:
                triples.append(
                    {"question": question, "claim": answer, "passage": passage, "label": "support"}
                )
                perturbed = perturb_answer(answer, rng, all_answers)
                triples.append(
                    {"question": question, "claim": perturbed, "passage": passage, "label": "refute"}
                )
            else:
                distractors.append(passage)

        for passage in distractors:
            triples.append(
                {"question": question, "claim": answer, "passage": passage, "label": "neutral"}
            )

    return triples


def build_question_examples(ds, pool_indices: list[int], rng: random.Random) -> list[dict]:
    """Aggregator training examples with a full multi-passage evidence set per question
    (matching the K-passage evidence sets used at inference), unlike build_triples'
    single-passage pairs used for verifier training."""
    examples = []
    all_answers = [ds[i]["answer"] for i in pool_indices]

    for pos, idx in enumerate(pool_indices):
        ex = ds[idx]
        question = ex["question"]
        answer = ex["answer"]
        titles = ex["context"]["title"]
        sentences = ex["context"]["sentences"]
        evidence = [{"title": t, "text": "".join(s)} for t, s in zip(titles, sentences)]

        examples.append({"question": question, "claim": answer, "passages": evidence, "label": "support"})

        perturbed = perturb_answer(answer, rng, all_answers)
        examples.append({"question": question, "claim": perturbed, "passages": evidence, "label": "refute"})

        other_idx = pool_indices[(pos + 1) % len(pool_indices)]
        other_ex = ds[other_idx]
        other_titles = other_ex["context"]["title"]
        other_sentences = other_ex["context"]["sentences"]
        other_evidence = [{"title": t, "text": "".join(s)} for t, s in zip(other_titles, other_sentences)]
        examples.append({"question": question, "claim": answer, "passages": other_evidence, "label": "neutral"})

    return examples


def balance_triples(triples: list[dict], per_class: int, rng: random.Random) -> list[dict]:
    by_label = {"support": [], "refute": [], "neutral": []}
    for t in triples:
        by_label[t["label"]].append(t)

    balanced = []
    for label, items in by_label.items():
        rng.shuffle(items)
        balanced.extend(items[:per_class])

    rng.shuffle(balanced)
    return balanced
