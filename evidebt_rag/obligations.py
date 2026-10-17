import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.generate import ollama_generate, openai_generate

TYPES = {"bridge", "attribute", "comparison", "temporal", "constraint"}

PROMPT = """Break the question into the minimal set of typed obligations (atomic sub-requirements) needed to answer it.

Output one obligation per line in this exact format:
type|statement|parents

- type is one of: bridge, attribute, comparison, temporal, constraint
- statement is the obligation as a short, self-contained question or claim
- parents is a comma-separated list of earlier line numbers this obligation depends on, or "none"

Example for "What year was the founder of the company that made the iPhone born?":
bridge|Which company made the iPhone?|none
attribute|Who founded that company?|1
attribute|What year was that person born?|2

Question: {question}
Obligations:"""


def generate(gen_name: str, config: dict, prompt: str) -> str:
    if gen_name == "openai":
        cfg = config["generators"]["openai"]
        return openai_generate(prompt, cfg["model"])
    cfg = config["generators"]["ollama"]
    return ollama_generate(prompt, cfg["model"], cfg["url"])


def _parse(raw: str) -> list[dict]:
    obligations = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 3:
            continue
        otype, statement, parents_raw = parts
        otype = otype.lower()
        if otype not in TYPES or not statement:
            continue
        parents = []
        if parents_raw.lower() != "none":
            for p in parents_raw.split(","):
                p = p.strip()
                if p.isdigit():
                    parents.append(int(p) - 1)
        obligations.append({"type": otype, "statement": statement, "parents": parents, "active": True})
    return obligations


def _break_cycles(obligations: list[dict]) -> list[dict]:
    n = len(obligations)
    state = [0] * n  # 0=unvisited, 1=in-progress, 2=done

    def dfs(i, path):
        state[i] = 1
        for p in list(obligations[i]["parents"]):
            if p >= n or p == i:
                obligations[i]["parents"].remove(p)
                continue
            if state[p] == 1:
                obligations[i]["parents"].remove(p)
                continue
            if state[p] == 0:
                dfs(p, path)
        state[i] = 2

    for i in range(n):
        if state[i] == 0:
            dfs(i, [])
    return obligations


def extract_obligations(question: str, gen_name: str, config: dict) -> list[dict]:
    raw = generate(gen_name, config, PROMPT.format(question=question))
    obligations = _parse(raw)
    if not obligations:
        obligations = [{"type": "constraint", "statement": question, "parents": [], "active": True}]
    return _break_cycles(obligations)


DECOMPOSE_PROMPT = """The obligation below is proving hard to resolve. Split it into exactly \
two simpler sub-obligations that together cover it.

Output one per line in this exact format:
type|statement

- type is one of: bridge, attribute, comparison, temporal, constraint
- statement is a short, self-contained question or claim

Obligation: {statement}
Sub-obligations:"""


def decompose(obligation: dict, gen_name: str, config: dict) -> list[dict]:
    """One LLM call splitting a stuck obligation into 2 simpler sub-obligations
    (same pipe format as extraction, no parents field). Empty list on parse failure."""
    raw = generate(gen_name, config, DECOMPOSE_PROMPT.format(statement=obligation["statement"]))
    subs = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 2:
            continue
        otype, statement = parts
        otype = otype.lower()
        if otype not in TYPES or not statement:
            continue
        subs.append({"type": otype, "statement": statement, "active": True})
    return subs[:2]
