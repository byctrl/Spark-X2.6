"""Minimal Spark-X2.6-4B health-information chatbot (HER Hack-Astron #7).

Public-interest use case: accessible health information. Answers are plain
language and always redirect the person to a qualified professional.

Run against a locally served model (vLLM / SGLang OpenAI-compatible endpoint):

    python examples/health_chatbot/health_chatbot.py
    python examples/health_chatbot/health_chatbot.py "What is a fever?"
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.3

SYSTEM_PROMPT = (
    "You are a patient-education assistant powered by Spark-X2.6-4B. "
    "Explain health topics in short, plain, respectful language for a person "
    "with no medical training. "
    "You must NOT diagnose, name a likely disease as a conclusion, recommend "
    "prescription drugs or dosages, or tell the person to delay care. "
    "Always: (1) summarize what the described signs can mean in general terms, "
    "(2) list warning signs that need urgent in-person care, "
    "(3) suggest which kind of professional or service to contact and what to "
    "prepare, (4) remind the person that this is general information, not a "
    "medical assessment. "
    "If the person describes an emergency (chest pain, trouble breathing, heavy "
    "bleeding, fainting, thoughts of self-harm), tell them to contact local "
    "emergency services right away. "
    "Keep the answer under 250 words and use short paragraphs or bullet points."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and now I have a dry cough and a "
    "mild fever in the evening. What should I know before I talk to a doctor?"
)


def build_messages(question):
    """Build the chat messages sent to Spark-X2.6-4B."""
    text = (question or "").strip()
    if not text:
        raise ValueError("question must be a non-empty string")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def _post_json(url, payload, timeout=120.0):
    """POST JSON using only the standard library."""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def chat(question, base_url=None, model=MODEL_ID, transport=None, timeout=120.0):
    """Send one question to Spark-X2.6-4B and return the plain-text answer."""
    base = (base_url or DEFAULT_BASE_URL).rstrip("/")
    url = base + "/chat/completions"
    payload = {
        "model": model,
        "messages": build_messages(question),
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
    }
    post = transport or _post_json
    try:
        response = post(url, payload, timeout)
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not reach Spark-X2.6-4B at %s. Serve the model locally with "
            "an OpenAI-compatible server (vLLM or SGLang) or set SPARK_BASE_URL. "
            "Error: %s" % (url, exc)
        ) from exc
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "Unexpected response from Spark-X2.6-4B: %r" % (response,)
        ) from exc
    answer = (content or "").strip()
    if not answer:
        raise RuntimeError("Spark-X2.6-4B returned an empty answer")
    return answer


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Spark-X2.6-4B health-information chatbot"
    )
    parser.add_argument("question", nargs="?", default=DEMO_QUESTION)
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    parser.add_argument("--model", default=MODEL_ID, help="served model name")
    args = parser.parse_args(argv)
    print(chat(args.question, base_url=args.base_url, model=args.model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
