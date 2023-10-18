import sys
import os
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.text_splitter import MarkdownHeaderTextSplitter
# "/content/drive/My Drive/kodi_bot"
from text_bot.views.models import DocumentSplit


import sys
import os
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chains import RetrievalQAWithSourcesChain, MapReduceDocumentsChain, StuffDocumentsChain
from langchain.prompts import PromptTemplate

from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain

from langchain.chains.summarize import load_summarize_chain

from langchain.chains import (
LLMBashChain,
LLMChain,
RetrievalQA,
SimpleSequentialChain
)






import re
import base64
import json
from string import Template
from collections import defaultdict
import os
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from pathlib import Path

def remove_quotes(s):
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    return s

def extract_single_value_openai_content(openai_response_content: str, extract_key: str):
    # Define a pattern and find match
    value = openai_response_content.replace(extract_key,"")
    return value


def parse_openai_response(openai_response):
    openai_response = json.loads(openai_response)
    openai_response.get("choices")[0].get("message").get("content")
# {
#   "id": "chatcmpl-898dSJDcr3PPCtfcX3FA8WHU8o6hy",
#   "object": "chat.completion",
#   "created": 1697188838,
#   "model": "gpt-3.5-turbo-0613",
#   "choices": [
#     {
#       "index": 0,
#       "message": {
#         "role": "assistant",
#         "content": "TITLE: Uputstvo za popunjavanje kvartalnog izve\u0161taja o toku sprovo\u0111enja klini\u010dkog ispitivanja"
#       },
#       "finish_reason": "stop"
#     }
#   ],
#   "usage": {
#     "prompt_tokens": 446,
#     "completion_tokens": 38,
#     "total_tokens": 484
#   }
# }



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

def get_proper_file_loader(file_path):
    print("get_proper_file_loader "+ str(file_path))
    loader = None
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx') or file_path.endswith('.doc'):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path)
    return loader

def load_documents(documents_folder_path):
    documents = list()
    for file_path in os.listdir(documents_folder_path):
        file = Path(file_path)
        base_path = os.path.dirname(file.absolute())
        try:
            loader = get_proper_file_loader(base_path+"/"+documents_folder_path+file.name)
            if loader:
                document_pages = loader.load()
                for page in document_pages:
                    page.metadata["source"] = file.name
                # print("document: "+str(document))
                # print("filename: "+str(file.name))
                documents.append(document_pages)
        except Exception as e:
            print(e)
    return documents

