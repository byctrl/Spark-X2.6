'''Minimal Spark-X2.6-4B health-information chatbot (HER Hack-Astron #7).

Public-interest demo: inclusive, plain-language health education for people who
may not have easy access to a clinician. The model runs locally; this script
never diagnoses, never prescribes, and always points the user to professional
care.

Usage:
    python src/health_chatbot.py --demo
    python src/health_chatbot.py --question 'I have a cough and fever after travel'
'''

from __future__ import annotations

import argparse
from typing import Callable, Dict, List, Optional

MODEL_ID = 'XHToken/Spark-X2.6-4B'
MAX_NEW_TOKENS = 1024

DISCLAIMER = (
    'This is general health information, not a diagnosis or a prescription. '
    'Please consult a qualified clinician about your own situation.'
)

SYSTEM_PROMPT = (
    'You are a patient-education assistant for a public-interest health service. '
    'Explain health topics in simple, everyday language and short paragraphs. '
    'You must NOT diagnose, prescribe, or give dosages, and you must not claim to '
    'replace a clinician. Always encourage the user to see a qualified health '
    'professional, and clearly flag any warning signs that need urgent care. '
    'If information is missing or uncertain, say so instead of guessing.'
)

DEMO_QUESTION = (
    'I have had a cough and a mild fever for three days, since I came back from a '
    'trip. What should I know, and when should I see a doctor?'
)

_CACHE: Dict[str, object] = {}


def build_messages(
    question: str, history: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, str]]:
    '''Build the chat messages sent to Spark-X2.6-4B.'''
    messages: List[Dict[str, str]] = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    if history:
        for turn in history:
            role = turn.get('role')
            content = turn.get('content')
            if role in ('user', 'assistant') and isinstance(content, str):
                messages.append({'role': role, 'content': content})
    messages.append({'role': 'user', 'content': question.strip()})
    return messages


def build_prompt(messages: List[Dict[str, str]], tokenizer: object = None) -> str:
    '''Render messages with the model chat template, or a plain fallback.'''
    template = getattr(tokenizer, 'chat_template', None)
    if template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    parts = ['<|{0}|>\n{1}'.format(m['role'], m['content']) for m in messages]
    parts.append('<|assistant|>')
    return '\n\n'.join(parts)


def postprocess(text: str) -> str:
    '''Trim the reply and guarantee the safety disclaimer is present.'''
    body = (text or '').strip()
    if not body:
        body = 'I do not have enough reliable information to answer that.'
    if DISCLAIMER.lower() not in body.lower():
        body = body + '\n\n' + DISCLAIMER
    return body


def _load_model(model_id: str = MODEL_ID):
    '''Load Spark-X2.6-4B once and reuse it.'''
    if model_id in _CACHE:
        return _CACHE[model_id]
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto')
    model.eval()
    _CACHE[model_id] = (model, tokenizer)
    return model, tokenizer


def _run_local(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    '''Generate one reply with the local Spark-X2.6-4B model.'''
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = output[0][inputs['input_ids'].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def generate_reply(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    model_id: str = MODEL_ID,
    max_new_tokens: int = MAX_NEW_TOKENS,
    generate_fn: Optional[Callable[[str], str]] = None,
) -> str:
    '''Answer a health question; generate_fn allows offline testing.'''
    if not question or not question.strip():
        raise ValueError('question must be a non-empty string')
    messages = build_messages(question, history)
    prompt = build_prompt(messages) if generate_fn is not None else None
    if generate_fn is None:
        model, tokenizer = _load_model(model_id)
        prompt = build_prompt(messages, tokenizer)
        raw = _run_local(model, tokenizer, prompt, max_new_tokens)
    else:
        raw = generate_fn(prompt)
    return postprocess(raw)


def main(argv: Optional[List[str]] = None) -> int:
    '''Command-line entry point for the demo.'''
    parser = argparse.ArgumentParser(description='Spark-X2.6-4B health chatbot')
    parser.add_argument('--question', default=None, help='health question')
    parser.add_argument('--demo', action='store_true', help='run the demo question')
    args = parser.parse_args(argv)

    question = args.question or (DEMO_QUESTION if args.demo else None)
    if not question:
        parser.error('provide --question or --demo')

    print('Q: ' + question)
    print('')
    print(generate_reply(question))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
