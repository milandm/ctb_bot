import sys

sys.path.append("..")

import json

import numpy as np

from tqdm import tqdm

from text_bot.nlp_model.config import DATA
from text_bot.nlp_model.nlp_model import NlpModel
from text_bot.utils import load_documents

# SENTENCE_MIN_LENGTH = 15
SENTENCE_MIN_LENGTH = 2

from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from text_bot.views.models import CTDocument, CTDocumentSplit, CTDocumentPage

from text_bot.nlp_model.prompt_creator import PromptCreator
from text_bot.ai_utils import get_mmr_cosine_sorted_docs

MAX_CHUNK_SIZE = 1000
MAX_CHUNK_OVERLAP_SIZE = 500
MAX_PAGE_SIZE = 5500

class ChatManager:

    def __init__(self, nlp_model :NlpModel):
        self.model = nlp_model
        self.prompt_creator = PromptCreator(nlp_model)

    def send_user_query(self, current_query: str, history_key:str = "") -> dict:
        print("current_query "+current_query)
        # get_chat_history = self.get_chat_history(history_key)
        query_embedding = self.model.get_embedding(current_query)
        documents = CTDocumentSplit.objects.query_embedding_by_distance(query_embedding)
        documents_list = list(documents)
        doc_for_prompt = get_mmr_cosine_sorted_docs(query_embedding, documents)

        print("not ranked ")
        for doc in documents:
            print(doc.split_text)

        print("ranked ")
        for doc in doc_for_prompt:
            print(doc.split_text)
        # self.prompt_creator.get_answer(query_embedding, doc_for_prompt)


    def get_chat_history(self, history_key: str) -> dict:
        pass


    def get_user_history(self) -> dict:
        pass