"""Minimal health-education chatbot built on Spark-X2.6-4B.

Runs against any OpenAI-compatible endpoint serving
``XHToken/Spark-X2.6-4B`` (a local Ollama / vLLM / SGLang / Spark-MLX-LLM
server, for example).  The assistant is deliberately bounded: it explains
health information in plain language and points people to clinicians.  It
does not diagnose, does not prescribe, and does not replace professional
care -- that boundary lives in ``SYSTEM_PROMPT`` and in the coordinator /
clinician review step around the app.

Usage::

    SPARK_BASE_URL=http://127.0.0.1:30000/v1 python src/health_chatbot.py

Only the Python standard library is used; no extra dependency is added.
"""

import argparse
import json
import os
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024
DEFAULT_BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
REQUEST_TIMEOUT = 120

SYSTEM_PROMPT = (
    "You are a patient-education assistant for people who may have limited "
    "health literacy or limited access to care.\n"
    "Rules you must follow:\n"
    "1. Use short sentences and plain, everyday language. Avoid medical jargon; "
    "if a term is unavoidable, explain it in one clause.\n"
    "2. Do not diagnose, do not name a likely disease as a conclusion, and do "
    "not recommend prescription medicines or dosages.\n"
    "3. Explain what the described signs can mean in general, what the person "
    "can safely do meanwhile, and which questions to ask a clinician.\n"
    "4. Always tell the person to see a qualified clinician for diagnosis and "
    "treatment, and state clearly which warning signs mean emergency care is "
    "needed right away.\n"
    "5. Do not invent facts, sources, phone numbers or statistics. If you are "
    "unsure, say so and recommend asking a clinician.\n"
    "6. If the message suggests an emergency (for example trouble breathing, "
    "chest pain, fainting, heavy bleeding, confusion), lead with the advice to "
    "seek emergency help now.\n"
    "End with one short line reminding the reader that this is general "
    "information and not medical advice."
)

DEMO_QUESTION = (
    "I came back from a two-week trip three days ago and now I have a dry cough "
    "and a fever of 38.4 C. What should I know before I see a doctor?"
)

DISCLAIMER = (
    "This is general health information, not medical advice. "
    "Contact a qualified clinician for diagnosis and treatment."
)


def build_messages(question, history=None):
    """Build the chat messages sent to Spark for ``question``.

    ``history`` is an optional list of earlier ``{"role", "content"}`` turns
    (``user`` / ``assistant`` only) so follow-up questions keep context.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role") if isinstance(turn, dict) else None
        content = turn.get("content") if isinstance(turn, dict) else None
        if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
            raise ValueError(
                "each history turn needs role=user|assistant and non-empty content"
            )
        messages.append({"role": role, "content": content.strip()})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def extract_content(data):
    """Pull the assistant text out of an OpenAI-style response."""
    try:
        choice = data["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "Unexpected response from the Spark endpoint: %r" % (data,)
        ) from exc

    if isinstance(content, list):
        content = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict)
        )
    content = (content or "").strip()
    if not content:
        finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
        raise RuntimeError(
            "Spark returned an empty answer (finish_reason=%r)" % (finish_reason,)
        )
    return content


def _post_json(payload, base_url=None, timeout=REQUEST_TIMEOUT):
    """POST ``payload`` to the OpenAI-compatible chat endpoint."""
    url = (base_url or DEFAULT_BASE_URL).rstrip("/") + "/chat/completions"
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def chat(
    question,
    history=None,
    base_url=None,
    model=MODEL_ID,
    max_new_tokens=MAX_NEW_TOKENS,
    timeout=REQUEST_TIMEOUT,
    transport=None,
):
    """Ask Spark-X2.6-4B ``question`` and return its plain-language answer.

    ``transport`` is an optional callable taking the request payload and
    returning the decoded response dict; it exists so tests (and other
    front-ends) can run without a live model server.
    """
    messages = build_messages(question, history)
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": 0.2,
    }

    if transport is None:
        def transport(request_payload):
            return _post_json(request_payload, base_url=base_url, timeout=timeout)

    return extract_content(transport(payload))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Ask Spark-X2.6-4B a plain-language health question."
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="question text; defaults to DEMO_QUESTION",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args(argv)

    question = " ".join(args.question).strip() or DEMO_QUESTION
    answer = chat(question, base_url=args.base_url, model=args.model)
    print(answer)
    print()
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
