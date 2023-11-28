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
from text_bot.views.models import (CTDocument,
                                   CTDocumentSplit,
                                   CTDocumentPage,
                                   CTDocumentSection,
                                   CTDocumentSubsection,
                                   CTDocumentSectionText,
                                   CTDocumentSubsectionText)

from text_bot.nlp_model.prompt_creator import PromptCreator
from text_bot.ai_utils import get_mmr_cosine_sorted_docs

MAX_CHUNK_SIZE = 1000
MAX_CHUNK_OVERLAP_SIZE = 500
MAX_PAGE_SIZE = 5500

class ChatManager:

    def __init__(self, nlp_model :NlpModel):
        self.model = nlp_model
        self.prompt_creator = PromptCreator(nlp_model)

    # def send_user_query(self, current_query: str, history_key:str = "") -> dict:
    #     print(" ")
    #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ")
    #     print("current_query "+current_query)
    #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ")
    #     print(" ")
    #     # get_chat_history = self.get_chat_history(history_key)
    #     query_embedding = self.model.get_embedding(current_query)
    #     documents = CTDocumentSplit.objects.query_embedding_by_distance(query_embedding)
    #     documents_list = list(documents)
    #     doc_for_prompt = get_mmr_cosine_sorted_docs(query_embedding, documents)
    #
    #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ")
    #     print("not ranked ")
    #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ")
    #     print(" ")
    #     for doc in documents:
    #         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ")
    #         print(doc.split_text)
    #         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ")
    #
    #     print(" ")
    #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ")
    #     print("ranked ")
    #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ")
    #     print(" ")
    #     for doc in doc_for_prompt:
    #         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ")
    #         print(doc.split_text)
    #         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ")
    #     # self.prompt_creator.get_answer(query_embedding, doc_for_prompt)


    def send_user_query(self, current_query: str) -> dict:
        print(" ")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ")
        print("current_query "+current_query)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ")
        print(" ")

        three_question_statements_list = self.prompt_creator.get_three_question_statements(current_query)

        embedded_three_question_statements_list = self.model.get_embeddings(three_question_statements_list)

        sections_dict = dict()
        for question_statement_embedded in embedded_three_question_statements_list:
            sections = CTDocumentSection.objects.query_embedding_by_distance(question_statement_embedded)
            subsections = CTDocumentSubsection.objects.query_embedding_by_distance(question_statement_embedded)
            section_full_texts = CTDocumentSectionText.objects.query_embedding_by_distance(question_statement_embedded)
            subsection_full_texts = CTDocumentSubsectionText.objects.query_embedding_by_distance(question_statement_embedded)

            for section in sections:
                sections_dict[section.ct_document.id] = section.section_text_value
            for subsection in subsections:
                sections_dict[subsection.ct_document_section.ct_document.id] = subsection.ct_document_section.section_text_value
            for section in section_full_texts:
                sections_dict[section.ct_document_section.ct_document.id] = section.section_text
            for subsection in subsection_full_texts:
                sections_dict[subsection.ct_document_subsection.ct_document_section.ct_document.id] = subsection.ct_document_subsection.ct_document_section.section_text_value

        question_related_info_list = list()
        for key, section_text in sections_dict.items():
            question_related_info = self.prompt_creator.get_question_related_informations(current_query, section_text)
            question_related_info_list.append(question_related_info)

        json_data = json.dumps(question_related_info_list)
        return json_data
        # documents_list = list(documents)
        # doc_for_prompt = get_mmr_cosine_sorted_docs(query_embedding, documents)



    def get_text_compression(self, documents_split_txt):
        text_split_compression = self.prompt_creator.get_document_text_compression(documents_split_txt)
        text_split_compression_check = self.prompt_creator.get_document_text_compression_check(documents_split_txt, text_split_compression)

        if text_split_compression_check and "YES" in text_split_compression_check:
            return text_split_compression
        else:
            return text_split_compression_check


    def get_chat_history(self, history_key: str) -> dict:
        pass


    def get_user_history(self) -> dict:
        pass