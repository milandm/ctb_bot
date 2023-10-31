import sys

sys.path.append("..")

import json

import numpy as np

from tqdm import tqdm

from text_bot.nlp_model.config import DATA
from text_bot.nlp_model.nlp_model import NlpModel
from text_bot.utils import load_documents, remove_quotes, extract_single_value_openai_content

# SENTENCE_MIN_LENGTH = 15
SENTENCE_MIN_LENGTH = 2

from text_bot.nlp_model.prompt_template_creator import \
    PromptTemplateCreator, \
    SYSTEM_MSG_TITLE, \
    TITLE_EXTRACT_KEY, \
    DOCUMENT_SYSTEM_MSG_COMPRESSION_V3, \
    DOCUMENT_COMPRESSION_EXTRACT_KEY, \
    DOCUMENT_SYSTEM_MSG_COMPRESSION_CHECK_V1


class PromptCreator:

    def __init__(self, nlp_model: NlpModel):
        self.model = nlp_model
        self.prompt_template_creator = PromptTemplateCreator()


    def get_document_title(self, first_documents_split_txt: str):
        title_extract_prompt = self.prompt_template_creator.get_title_extract_prompt(first_documents_split_txt)
        document_title_openai_response = self.model.send_prompt(SYSTEM_MSG_TITLE, title_extract_prompt)
        document_title_content = document_title_openai_response.get("choices")[0].get("message").get("content")
        document_title = extract_single_value_openai_content(document_title_content, TITLE_EXTRACT_KEY)
        return document_title

    def get_query_based_text_compression(self, query, documents_split_txt: str):
        text_compression_prompt = self.prompt_template_creator.get_document_text_compression_prompt(documents_split_txt)
        text_compression_openai_response = self.model.send_prompt(DOCUMENT_SYSTEM_MSG_COMPRESSION_V3, text_compression_prompt)
        print(str(text_compression_openai_response))
        text_compression_content = text_compression_openai_response.get("choices")[0].get("message").get("content")
        text_compression = extract_single_value_openai_content(text_compression_content, DOCUMENT_COMPRESSION_EXTRACT_KEY)
        text_compression = remove_quotes(text_compression)
        return text_compression

    def get_document_text_compression(self, documents_split_txt: str):
        documents_split_txt = self.clean_documents_split(documents_split_txt)
        text_compression_prompt = self.prompt_template_creator.get_document_text_compression_prompt(documents_split_txt)
        text_compression_openai_response = self.model.send_prompt(DOCUMENT_SYSTEM_MSG_COMPRESSION_V3, text_compression_prompt)
        print(str(text_compression_openai_response))
        text_compression_content = text_compression_openai_response.get("choices")[0].get("message").get("content")
        text_compression = extract_single_value_openai_content(text_compression_content, DOCUMENT_COMPRESSION_EXTRACT_KEY)
        text_compression = remove_quotes(text_compression)
        return text_compression

    def clean_documents_split(self, documents_split_txt):
        documents_split_txt = documents_split_txt.replace("page_content=", "")
        return documents_split_txt

    def clean_openai_response_content(self, response_content):
        response_content_txt = response_content.replace("```", "")
        return response_content_txt

    def get_document_text_compression_check(self, documents_split_txt: str, previous_response: str):
        documents_split_txt = self.clean_documents_split(documents_split_txt)
        text_compression_check_prompt = self.prompt_template_creator.get_document_text_compression_check_prompt(documents_split_txt, previous_response)
        text_compression_check_openai_response = self.model.send_prompt(DOCUMENT_SYSTEM_MSG_COMPRESSION_CHECK_V1, text_compression_check_prompt)
        print(str(text_compression_check_openai_response))
        text_compression_check_content = text_compression_check_openai_response.get("choices")[0].get("message").get("content")

        text_compression_check_content_txt = self.clean_openai_response_content(text_compression_check_content)
        text_compression_check = text_compression_check_content_txt
        try:
            text_compression_check_content_json = json.loads(text_compression_check_content_txt)
            text_compression_check = text_compression_check_content_json.get("new_response")
        except Exception as e:
            print(e)
        text_compression_check = remove_quotes(text_compression_check)
        return text_compression_check