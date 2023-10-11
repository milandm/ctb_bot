import re
import base64
import json
from string import Template
from collections import defaultdict

def get_base64_string(string_key):
    bytes_s = string_key.encode('utf-8')
    base64_bytes = base64.b64encode(bytes_s)
    base64_string = base64_bytes.decode('utf-8')
    return base64_string

def extract_human_ai_conversation_from_string(text):
    if not text:
        return []

    text = text.strip()
    text = f"\n{text}"
    print(text)
    # Split into separate 'Human' and 'AI' messages based on '\nHuman: ' or '\nAI: '
    human_messages = re.split('\nHuman:', text)

    # For each 'Human' message, further split into 'Human' and 'AI' parts
    parsed_messages = []
    for human_message in human_messages:
        parts = re.split('\nAI:', human_message)
        human_text = parts[0].replace('Human: ', '').strip()
        if human_text and len(human_text) > 0:
            parsed_messages.append({'type': 'HumanMessage', 'text': human_text})
        if len(parts) > 1:
            ai_text = parts[1].replace('AI: ', '').strip()
            if ai_text and len(ai_text) > 0:
                parsed_messages.append({'type': 'AIMessage', 'text': f'ANSWER: \n- {ai_text}'})

    return parsed_messages


def get_questions_list_from_text_bot_api_buffer(text):
    if not text:
        return []

    text = text.strip()
    text = f"\n{text}"
    print(text)
    # Split into separate 'Human' and 'AI' messages based on '\nHuman: ' or '\nAI: '
    human_messages = re.split('\nHuman:', text)

    # For each 'Human' message, further split into 'Human' and 'AI' parts
    parsed_messages = []
    for human_message in human_messages:
        parts = re.split('\nAI:', human_message)
        human_text = parts[0].replace('Human: ', '').strip()
        if human_text and len(human_text) > 0:
            parsed_messages.append(human_text)
    return parsed_messages

def get_questions_list_from_text_bot_api_structured_buffer(history_response_buffer):
    if not history_response_buffer:
        return []

    # For each 'Human' message, further split into 'Human' and 'AI' parts
    parsed_messages = []
    for human_message in history_response_buffer:
        if human_message['type'] == 'HumanMessage':
            parsed_messages.append(human_message['text'])
    return parsed_messages


@staticmethod
def load_file(file_name: str):
    with open(file_name, "r") as file:
        meditations_json = json.load(file)
        return meditations_json

@staticmethod
def prepare_template(template: str, key_value_to_change:dict) -> str:
    prompt_template = Template(template)
    try:
        prepared_prompt = prompt_template.safe_substitute(key_value_to_change)
    except KeyError as e:
        print(e)
    except ValueError as e:
        print(e)

    # mapping = defaultdict(str, key_value_to_change)
    # prepared_prompt = template.format_map(mapping=mapping)
    return prepared_prompt


