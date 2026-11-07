"""Minimal health-information chatbot built on Spark-X2.6-4B.

Public-interest demo for HER Hack-Astron #7 ("AI for Good"): an inclusive
health-knowledge assistant.

Scope and safety
----------------
The assistant explains health topics in plain language. It does not diagnose,
prescribe, or replace a clinician: the system prompt forbids it, and every
answer ends with a pointer to professional care. Model output is treated as
unverified text, never as medical advice or ground truth.

The application talks to any OpenAI-compatible Spark-X2.6 endpoint
(llama.cpp, vLLM, SGLang, Spark-MLX-LLM). Standard library only.

Usage
-----
    export SPARK_BASE_URL=http://127.0.0.1:30000/v1
    python src/health_chatbot.py
    python src/health_chatbot.py --question "Batuk kering dan demam setelah travelling"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.3
DEFAULT_BASE_URL = "http://127.0.0.1:30000/v1"
DEFAULT_TIMEOUT = 180

SYSTEM_PROMPT = """You are a patient-education assistant for a public health help desk.

Rules:
- Use simple, everyday language and short paragraphs.
- Explain what a symptom or term generally means, and which questions the
  person could ask a clinician.
- Never diagnose, never name a prescription medicine, never give dosages, and
  never tell the person to stop or change a treatment.
- Never claim certainty. Say clearly when something needs professional review.
- End every answer with: "This is general information, not a diagnosis. Please
  contact a qualified health worker about your own situation."
- If the question suggests an emergency (chest pain, trouble breathing, heavy
  bleeding, fainting, thoughts of self-harm), start by telling the person to
  seek emergency care now.
- Do not ask for or repeat names, addresses, ID numbers, or other personal data.
"""

DEMO_QUESTION = (
    "Saya baru pulang dari perjalanan dan sekarang batuk kering serta demam "
    "ringan selama tiga hari. Apa yang sebaiknya saya perhatikan sebelum ke klinik?"
)


def build_messages(question: str) -> "list[dict[str, str]]":
    """Build the bounded chat messages sent to Spark-X2.6-4B."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]


def build_payload(messages):
    """Build the OpenAI-compatible request body for Spark-X2.6-4B."""
    return {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "stream": False,
    }


def extract_answer(response):
    """Pull the assistant text out of a chat-completions response."""
    if not isinstance(response, dict):
        raise ValueError("model response is not a JSON object")
    choices = response.get("choices")
    if not choices:
        raise ValueError("model response has no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("model returned an empty answer")
    return content.strip()


def _http_transport(base_url, api_key, timeout):
    url = base_url.rstrip("/") + "/chat/completions"

    def transport(messages):
        body = json.dumps(build_payload(messages)).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = "Bearer " + api_key
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError("Spark endpoint returned HTTP %s" % exc.code) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "cannot reach Spark endpoint at %s: %s" % (url, exc.reason)
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("Spark endpoint returned invalid JSON") from exc

    return transport


def generate(question, base_url=None, api_key=None, timeout=DEFAULT_TIMEOUT, transport=None):
    """Ask Spark-X2.6-4B one health question and return its plain-language answer."""
    messages = build_messages(question)
    if transport is None:
        transport = _http_transport(
            base_url or os.environ.get("SPARK_BASE_URL", DEFAULT_BASE_URL),
            api_key if api_key is not None else os.environ.get("SPARK_API_KEY"),
            timeout,
        )
    return extract_answer(transport(messages))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Spark-X2.6-4B health-information chatbot (educational, not medical advice)"
    )
    parser.add_argument("--question", default=DEMO_QUESTION)
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible Spark endpoint")
    args = parser.parse_args(argv)

    print("Q:", args.question)
    try:
        answer = generate(args.question, base_url=args.base_url)
    except (ValueError, RuntimeError) as exc:
        print("error:", exc, file=sys.stderr)
        return 1
    print("\nA:", answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
