"""Minimal Spark-X2.6-4B health-information chatbot (HER Hack-Astron #7).

Public-interest demo: plain-language health education for people with limited
access to clinicians. The model never diagnoses and never prescribes; it
explains and points the user to professional care.

Usage:
    python src/health_chatbot.py --question "..."
"""

import argparse

MODEL_ID = "XHToken/Spark-X2.6-4B"
MAX_NEW_TOKENS = 1024

SYSTEM_PROMPT = (
    "You are a patient-education assistant for a public-interest health line. "
    "Explain health topics in plain language at a 6th-grade reading level. "
    "Never diagnose the user and never name a likely disease as their condition. "
    "Never recommend prescription drugs or dosages. You do not replace a clinician. "
    "Always state when the user should seek professional help, and list concrete "
    "emergency warning signs when relevant. If the question is unclear, ask one "
    "short clarifying question. Keep answers under 200 words."
)

DEMO_QUESTION = (
    "I came back from a trip two weeks ago and since then I have a dry cough "
    "and a mild fever in the evenings. I do not have a doctor nearby. "
    "What should I know and when should I get help?"
)


def build_messages(question, history=None):
    """Build the chat messages sent to Spark-X2.6-4B.

    The system prompt is always first and the new user question is always last,
    so earlier turns cannot replace the safety rules.
    """
    question = question.strip() if isinstance(question, str) else ""
    if not question:
        raise ValueError("question must be a non-empty string")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in history or []:
        role = turn.get("role") if isinstance(turn, dict) else None
        content = turn.get("content") if isinstance(turn, dict) else None
        content = content.strip() if isinstance(content, str) else ""
        if role not in ("user", "assistant") or not content:
            raise ValueError(
                "history turns must be dicts with role user/assistant "
                "and non-empty content"
            )
        messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": question})
    return messages


def format_prompt(messages):
    """Render messages as a plain chat prompt fallback."""
    lines = []
    for message in messages:
        lines.append("<|{}|>\n{}".format(message["role"], message["content"]))
    lines.append("<|assistant|>\n")
    return "\n".join(lines)


def answer(question, history=None, generator=None):
    """Return the plain-language answer for one question.

    `generator` is a callable taking the message list and returning text; it is
    injected in tests so no model download is needed.
    """
    messages = build_messages(question, history=history)
    if generator is None:
        generator = load_generator()
    text = generator(messages)
    return (text or "").strip()


def load_generator(model_id=MODEL_ID, max_new_tokens=MAX_NEW_TOKENS):
    """Load Spark-X2.6-4B and return a generate(messages) -> str callable."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "health_chatbot needs transformers and torch: "
            "pip install transformers torch"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )

    def generate(messages):
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = format_prompt(messages)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        new_tokens = output[0][inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    return generate


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Spark-X2.6-4B plain-language health information demo"
    )
    parser.add_argument("--question", default=DEMO_QUESTION)
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args(argv)

    generate = load_generator(model_id=args.model)
    print(answer(args.question, generator=generate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
