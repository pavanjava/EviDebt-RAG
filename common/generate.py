import math
import os

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def ollama_generate(prompt: str, model: str, url: str) -> str:
    resp = requests.post(url, json={"model": model, "prompt": prompt, "stream": False})
    resp.raise_for_status()
    return resp.json()["response"].strip()


def openai_generate(prompt: str, model: str) -> str:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip()


def openai_generate_with_prob(prompt: str, model: str) -> tuple[str, float]:
    """Returns (answer, mean token probability) using OpenAI's logprobs."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        logprobs=True,
    )
    choice = resp.choices[0]
    logprobs = [t.logprob for t in choice.logprobs.content]
    mean_prob = sum(math.exp(lp) for lp in logprobs) / len(logprobs) if logprobs else 0.0
    return choice.message.content.strip(), mean_prob
