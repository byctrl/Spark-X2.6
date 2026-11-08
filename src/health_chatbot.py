"""Minimal health-information chatbot built on Spark-X2.6-4B.

Submission shape for HER Hack-Astron #7 (Spark-X2.6 Open Source AI for
Good Challenge), health + accessibility track: an inclusive, plain-language
patient-education chatbot for people who cannot easily read formal medical
text.

What the model does here:
    Spark-X2.6-4B is used as the text-generation engine behind the assistant.
    The application sends a bounded system prompt plus the user's question to
a local OpenAI-compatible Spark-X2.6 endpoint, and returns the model's
plain-language explanation. The application never diagnoses or prescribes;
it explains, and routes the user to a clinician or emergency service.

Run (after starting a local OpenAI-compatible Spark-X2.6 server, e.g. the
Spark-MLX-LLM / SGLang / vLLM OpenAI-compatible endpoint):

    python src/health_chatbot.py --demo
    python src/health_chatbot.py --question "What is oral rehydration solution?"

Environment:
    SPARK_BASE_URL  default http://127.0.0.1:30000/v1
    SPARK_API_KEY   default "EMPTY" (local servers usually ignore it)

Only the Python standard library is used, so the demo has no extra
install step beyond the model runtime itself.
"""

import argparse
import json
import os
import urllib.error
import urllib.request

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

BASE_URL = os.environ.get("SPARK_BASE_URL", "http://127.0.0.1:30000/v1")
API_KEY = os.environ.get("SPARK_API_KEY", "EMPTY")

SYSTEM_PROMPT = (
    "You are a patient-education assistant. You help people understand health "
    "information in plain language, including people with low literacy, "
    "limited local-language access, or no reliable internet."
    "\n\nRules you must follow:"
    "\n1. Use short sentences and everyday words. Explain any medical term you "
    "must use."
    "\n2. You never diagnose and you never prescribe: no diagnosis, no "
    "medication names with doses, no treatment plans. Give general education "
    "only."
    "\n3. Always tell the user when to seek professional care, and say clearly "
    "that a qualified clinician decides."
    "\n4. If details are missing, ask one short clarifying question instead of "
    "guessing."
    "\n5. Say plainly when you are unsure or when the information depends on "
    "local guidance."
    "\n6. Do not ask for or repeat names, ID numbers, addresses, or other "
    "identifying personal data."
    "\n7. Answer in the language the user wrote in. Keep the answer under 200 "
    "words unless the user asks for more."
)

DEMO_QUESTION = (
    "I came back from a two-week trip and still have a dry cough and a low "
    "fever in the evenings. What could this be, and when should I see a doctor?"
)

EMERGENCY_KEYWORDS = (
    "chest pain",
    "can't breathe",
    "cannot breathe",
    "difficulty breathing",
    "severe bleeding",
    "unconscious",
    "suicidal",
    "stroke",
    "seizure",
    "胸痛",
    "呼吸困难",
    "大出血",
    "昏迷",
    "自杀",
    "中风",
    "抽搻",
)

EMERGENCY_NOTICE = (
    "⚠️ This may be an emergency. Contact your local emergency number now "
    "(for example 120 in mainland China, 911 in the US, 112 in the EU). "
    "This assistant cannot provide emergency care."
)


def build_messages(question, history=None):
    """Build the chat messages sent to Spark-X2.6-4B."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role") if isinstance(turn, dict) else None
        content = turn.get("content") if isinstance(turn, dict) else None
        if role not in ("user", "assistant") or not isinstance(content, str):
            raise ValueError(
                "history turns must look like {'role': 'user'|'assistant', "
                "'content': str}"
            )
        messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": question.strip()})
    return messages


def build_payload(
    question,
    history=None,
    model=MODEL_ID,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=0.3,
):
    """Build the OpenAI-compatible request body for the Spark endpoint."""
    return {
        "model": model,
        "messages": build_messages(question, history=history),
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "stream": False,
    }


def parse_response(data):
    """Extract assistant text from an OpenAI-compatible response body."""
    if isinstance(data, str):
        return data
    if not isinstance(data, dict):
        raise RuntimeError("Unexpected response type: %r" % (type(data),))

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(first.get("text"), str):
                return first["text"]

    if isinstance(data.get("content"), str):
        return data["content"]

    raise RuntimeError("Response has no message content: %s" % (str(data)[:300],))


def emergency_notice(question):
    """Return a fixed emergency notice when the question carries red flags."""
    text = (question or "").lower()
    for keyword in EMERGENCY_KEYWORDS:
        if keyword in text:
            return EMERGENCY_NOTICE
    return ""


def _post_json(url, payload, api_key, timeout=120.0):
    """POST JSON to the local Spark-X2.6 OpenAI-compatible endpoint."""
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("Authorization", "Bearer " + (api_key or "EMPTY"))

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(
            "Spark endpoint returned HTTP %s: %s" % (exc.code, detail)
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Cannot reach the Spark-X2.6 endpoint at %s (%s). Start a local "
            "OpenAI-compatible server for %s and retry."
            % (url, exc.reason, MODEL_ID)
        ) from exc


def generate_reply(
    question,
    history=None,
    http_post=None,
    base_url=None,
    api_key=None,
    model=MODEL_ID,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=0.3,
):
    """Ask Spark-X2.6-4B for a plain-language health explanation."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    post = http_post or _post_json
    url = (base_url or BASE_URL).rstrip("/") + "/chat/completions"
    payload = build_payload(
        question,
        history=history,
        model=model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    data = post(url, payload, api_key if api_key is not None else API_KEY)
    answer = parse_response(data).strip()

    notice = emergency_notice(question)
    if notice:
        return notice + "\n\n" + answer
    return answer


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Spark-X2.6-4B plain-language health information chatbot demo"
    )
    parser.add_argument("--question", help="user question to answer")
    parser.add_argument(
        "--demo", action="store_true", help="run the bundled demo scenario instead"
    )
    parser.add_argument("--base-url", default=BASE_URL, help="Spark OpenAI-compatible base URL")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args(argv)

    question = DEMO_QUESTION if args.demo else args.question
    if not question:
        parser.error("pass --question or --demo")
        return 2

    answer = generate_reply(
        question, base_url=args.base_url, max_new_tokens=args.max_new_tokens
    )
    print("Q:", question)
    print()
    print("A:", answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
