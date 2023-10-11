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


def load_documents(documents_folder_path):
    documents = []
    for file in os.listdir(documents_folder_path):
        if file.endswith(".pdf"):
            pdf_path = "./documents/" + file
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        elif file.endswith('.docx') or file.endswith('.doc'):
            doc_path = "./documents/" + file
            loader = Docx2txtLoader(doc_path)
            documents.extend(loader.load())
        elif file.endswith('.txt'):
            text_path = "./documents/" + file
            loader = TextLoader(text_path)
            documents.extend(loader.load())
    return documents
