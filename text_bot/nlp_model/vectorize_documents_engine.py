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


class VectorizeDocumentsEngine:

    def __init__(self, nlp_model :NlpModel):
        self.model = nlp_model
        self.prompt_creator = PromptCreator(nlp_model)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
        # self.text_splitter = MarkdownHeaderTextSplitter(chunk_size=1000, chunk_overlap=500)


    def get_documents_splits(self):
        documents = load_documents("documents/")
        for document_pages in documents:

            documents_splits = self.text_splitter.split_documents(document_pages)
            document_title  = self.prompt_creator.get_document_title(documents_splits[0].page_content)
            document_filename = document_pages[0].metadata.get("source", "")

            print("Document title: ", document_title)
            print("Document filename: ", document_filename)

            ct_document = CTDocument.objects.create(
                document_version="1",
                document_title = document_title,
                document_filename = document_filename)

            for i, document_page in enumerate(document_pages):
                CTDocumentPage.objects.create(
                    ct_document=ct_document,
                    document_page_text = document_page.page_content,
                    document_page = i)


            for documents_split in documents_splits:

                document_page = documents_split.metadata.get("page",0)
                split_text = documents_split.page_content
                split_text_compression = self.get_text_compression(documents_split.page_content)

                print("Document title: ", document_title)
                print("Document filename: ", document_filename)
                print("Document page: ", document_page)
                print("Split text: ", split_text)
                print("Split text compression: ", split_text_compression)

                embedding = self.model.get_embedding(split_text)

                CTDocumentSplit.objects.create(
                    ct_document=ct_document,
                    document_title = document_title,
                    document_filename = document_filename,
                    document_page = document_page,
                    split_text = split_text,
                    split_text_compression = split_text_compression,
                    embedding = embedding)


    def get_text_compression(self, documents_split_txt):
        text_split_compression = self.prompt_creator.get_text_compression(documents_split_txt)
        text_split_compression_check = self.prompt_creator.get_text_compression_check(documents_split_txt, text_split_compression)

        if text_split_compression_check and "YES" in text_split_compression_check:
            return text_split_compression
        else:
            return text_split_compression_check