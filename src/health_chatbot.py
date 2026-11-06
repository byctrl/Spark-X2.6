"""Minimal Spark-X2.6-4B health-information chatbot (HER Hack-Astron #7).

Public-interest demo: inclusive, plain-language health education.

Safety boundary: the model is framed as a patient-education assistant, it is
told not to diagnose or prescribe, and every reply is passed through
finalize_reply() so a clinician referral is always present.

Usage:
    python src/health_chatbot.py
    python src/health_chatbot.py --ask "your question"
    python src/health_chatbot.py --plan

The model is loaded lazily so tests do not need transformers or a checkpoint.
"""

import argparse
import json

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant. You explain health topics in short, "
    "plain sentences that a person with no medical training can understand. "
    "Rules you must always follow:\n"
    "1. You are not a doctor. Never give a diagnosis and never name a medicine, "
    "dose or treatment plan.\n"
    "2. Explain what the described symptoms usually mean in general terms, and "
    "say plainly when something needs urgent professional care.\n"
    "3. Ask at most two short clarifying questions if key facts are missing.\n"
    "4. Point the reader to a qualified clinician, clinic or local emergency "
    "service as the next step.\n"
    "5. Never ask for a name, address, ID number or other identifying detail.\n"
    "6. If the question is outside health education, say so briefly and stop."
)

DEMO_QUESTION = (
    "I came back from a two-week trip three days ago. Since yesterday I have a "
    "dry cough and a fever I measured at 38.4 C. I can breathe normally and "
    "there is no blood in the cough. What should I know, and when should I "
    "see someone?"
)

DISCLAIMER = (
    "This is general health education, not medical advice or a diagnosis. "
    "Please contact a qualified clinician, your local clinic, or an emergency "
    "service if your symptoms are severe or getting worse."
)

_DISCLAIMER_MARKER = "not medical advice"


def build_messages(question, history=None):
    """Build an OpenAI-style chat message list for the model."""
    question = (question or "").strip()
    if not question:
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def render_prompt(messages, tokenizer=None):
    """Render messages with the model chat template, or a readable fallback."""
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    parts = []
    for message in messages:
        parts.append("<|" + message["role"] + "|>\n" + message["content"])
    parts.append("<|assistant|>")
    return "\n\n".join(parts) + "\n"


def finalize_reply(text):
    """Guarantee a non-empty reply that always carries the disclaimer."""
    text = (text or "").strip()
    if not text:
        return "I could not produce an answer for this question. " + DISCLAIMER
    if _DISCLAIMER_MARKER not in text.lower():
        text = text + "\n\n" + DISCLAIMER
    return text


def load_generator(model_id=MODEL_ID):
    """Load a text-generation pipeline. Imported lazily on purpose."""
    from transformers import pipeline

    return pipeline("text-generation", model=model_id, device_map="auto")


def generate_reply(question, history=None, generator=None, model_id=MODEL_ID,
                   max_new_tokens=MAX_NEW_TOKENS):
    """Ask Spark-X2.6-4B for a plain-language answer about a health question."""
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")

    messages = build_messages(question, history)
    if generator is None:
        generator = load_generator(model_id)

    prompt = render_prompt(messages, getattr(generator, "tokenizer", None))
    output = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        return_full_text=False,
    )

    return finalize_reply(_extract_text(output))


def _extract_text(output):
    """Pull generated text out of the common pipeline output shapes."""
    if isinstance(output, list):
        if not output:
            return ""
        output = output[0]
        if isinstance(output, list):
            if not output:
                return ""
            output = output[0]
    if isinstance(output, dict):
        return output.get("generated_text", "") or ""
    return str(output)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Spark-X2.6-4B health-information chatbot"
    )
    parser.add_argument(
        "--ask", default=None, help="question to ask (default: demo question)"
    )
    parser.add_argument(
        "--model", default=MODEL_ID, help="model id or local path"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=MAX_NEW_TOKENS,
        help="generation budget"
    )
    parser.add_argument(
        "--plan", action="store_true",
        help="print the request payload and exit without loading the model"
    )
    args = parser.parse_args(argv)

    question = args.ask or DEMO_QUESTION
    messages = build_messages(question)

    if args.plan:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "max_new_tokens": args.max_new_tokens,
                    "messages": messages,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    reply = generate_reply(
        question,
        model_id=args.model,
        max_new_tokens=args.max_new_tokens,
    )
    print(reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
