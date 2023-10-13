import sys

sys.path.append("..")

import json

import numpy as np

from tqdm import tqdm

from text_bot.nlp_model.config import DATA
from text_bot.nlp_model.nlp_model import NlpModel
from text_bot.utils import load_documents, extract_value_openai_content

BOOK_FILENAME = "Marcus_Aurelius_Antoninus_-_His_Meditations_concerning_himselfe"

# SENTENCE_MIN_LENGTH = 15
SENTENCE_MIN_LENGTH = 2

from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from text_bot.views.models import DocumentSplit
from text_bot.nlp_model.prompt_template_creator import \
    PromptTemplateCreator, \
    SYSTEM_MSG_TITLE, \
    TITLE_EXTRACT_KEY, \
    SYSTEM_MSG_COMPRESSION_V2, \
    COMPRESSION_EXTRACT_KEY

class VectorizeDocumentsEngine:

    def __init__(self, nlp_model :NlpModel):
        self.model = nlp_model
        self.prompt_template_creator = PromptTemplateCreator()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
        # self.text_splitter = MarkdownHeaderTextSplitter(chunk_size=1000, chunk_overlap=500)


    def get_documents_splits(self):
        documents = load_documents("documents/")
        for document_pages in documents:

            documents_splits = self.text_splitter.split_documents(document_pages)
            document_title  = self.get_document_title(documents_splits[0])

            for documents_split in documents_splits:

                text_split_compression = self.get_text_compression(documents_split)
                filename = documents_split.metadata["source"]
                page = documents_split.metadata["page"]
                text = documents_split.page_content

                print("Document title: ", document_title)
                print("Document filename: ", filename)
                print("Document text: ", text)
                print("Document page: ", page)
                print("Document text_split_compression: ", text_split_compression)

                embedding = self.model.get_embeddings(text)

                DocumentSplit.objects.create(
                        filename=filename,
                        title=document_title,
                        text=text,
                        embedding=embedding,
                        page = page,
                        text_compression = text_split_compression)

    def get_document_title(self, first_documents_split):
        title_extract_prompt = self.prompt_template_creator.get_title_extract_prompt(first_documents_split)
        document_title_openai_response = self.model.send_prompt(SYSTEM_MSG_TITLE, title_extract_prompt)
        document_title_content = document_title_openai_response.get("choices")[0].get("message").get("content")
        document_title = extract_value_openai_content(TITLE_EXTRACT_KEY, document_title_content)
        return document_title


    def get_text_compression(self, documents_split):
        text_compression_prompt = self.prompt_template_creator.get_text_compression_prompt(documents_split)
        text_compression_openai_response = self.model.send_prompt(SYSTEM_MSG_COMPRESSION_V2, text_compression_prompt)
        print(str(text_compression_openai_response))
        text_compression_content = text_compression_openai_response.get("choices")[0].get("message").get("content")
        text_compression = extract_value_openai_content(COMPRESSION_EXTRACT_KEY, text_compression_content)

        return text_compression


