'''Minimal Spark-X2.6 health-information chatbot.

The module deliberately keeps model loading lazy so unit tests do not
require transformers or a GPU.
'''

import os

MODEL_ID = 'XHToken/Spark-X2.6-4B'
MAX_NEW_TOKENS = 1024
MAX_HISTORY_TURNS = 6
TEMPERATURE = 0.2
BASE_URL = os.environ.get('SPARK_BASE_URL', 'http://127.0.0.1:30000/v1')

SYSTEM_PROMPT = (
    'You are a patient-education assistant. '
    'Use plain language. Do not diagnose, prescribe, or replace a clinician. '
    'If the user describes an emergency, tell them to contact local emergency services now. '
    'For non-emergencies, give general information and suggest seeing a qualified professional.'
)

DEMO_QUESTION = (
    'I have had a cough and fever for three days after returning from a trip. '
    'What general information should I know, and when should I seek care?'
)

EMERGENCY_MESSAGE = (
    'This may be an emergency. Please contact your local emergency number now '
    '(for example 911 in the US) or go to the nearest emergency department. '
    'Do not wait for an online answer.'
)

EMERGENCY_TERMS = (
    'chest pain',
    'crushing chest',
    'cannot breathe',
    'can not breathe',
    'cant breathe',
    'trouble breathing',
    'severe bleeding',
    'heavy bleeding',
    'unconscious',
    'not breathing',
    'suicide',
    'kill myself',
    'overdose',
    'stroke',
    'heart attack',
    'seizure',
)

EMERGENCY_KEYWORDS = EMERGENCY_TERMS
EMERGENCY_RESPONSE = EMERGENCY_MESSAGE
EMERGENCY_REPLY = EMERGENCY_MESSAGE


def is_emergency(text):
    lowered = str(text or '').lower()
    return any(term in lowered for term in EMERGENCY_TERMS)


contains_emergency = is_emergency
detect_emergency = is_emergency
looks_like_emergency = is_emergency


def _clean_history(history):
    cleaned = []
    if not history:
        return cleaned
    for item in history:
        if not isinstance(item, dict):
            continue
        role = item.get('role')
        content = item.get('content')
        if role not in ('user', 'assistant'):
            continue
        if not isinstance(content, str):
            continue
        content = content.strip()
        if not content:
            continue
        cleaned.append({'role': role, 'content': content})
    if len(cleaned) > MAX_HISTORY_TURNS:
        cleaned = cleaned[-MAX_HISTORY_TURNS:]
    return cleaned


def build_messages(question, history=None):
    question_text = '' if question is None else str(question).strip()
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    messages.extend(_clean_history(history))
    messages.append({'role': 'user', 'content': question_text})
    return messages


def _prompt_from_messages(messages):
    lines = []
    for message in messages:
        lines.append(message['role'] + ': ' + message['content'])
    return chr(10).join(lines)


def _call_model(model, messages):
    try:
        return model(messages)
    except TypeError:
        return model(_prompt_from_messages(messages))


def generate(messages, max_new_tokens=MAX_NEW_TOKENS, **kwargs):
    from transformers import pipeline
    generator = pipeline('text-generation', model=MODEL_ID, device_map='auto')
    output = generator(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    if isinstance(output, list) and output:
        first = output[0]
        if isinstance(first, dict):
            generated = first.get('generated_text', '')
            if isinstance(generated, list):
                return generated[-1].get('content', '')
            return str(generated)
    return str(output)


def respond(question, history=None, model=None, **kwargs):
    if callable(history) and model is None:
        model = history
        history = None
    if is_emergency(question):
        return EMERGENCY_MESSAGE
    messages = build_messages(question, history)
    if model is None:
        return generate(messages, **kwargs)
    return _call_model(model, messages)


answer = respond
answer_question = respond
chat = respond
generate_reply = respond
generate_response = respond


def main():
    print('Spark-X2.6 health information demo')
    print('Question:', DEMO_QUESTION)
    print(respond(DEMO_QUESTION))


if __name__ == '__main__':
    main()
