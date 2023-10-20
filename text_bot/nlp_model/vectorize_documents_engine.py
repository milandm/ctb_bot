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

MAX_CHUNK_SIZE = 1000
MAX_CHUNK_OVERLAP_SIZE = 500
MAX_PAGE_SIZE = 5500

class VectorizeDocumentsEngine:

    def __init__(self, nlp_model :NlpModel):
        self.model = nlp_model
        self.prompt_creator = PromptCreator(nlp_model)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_CHUNK_SIZE, chunk_overlap=MAX_CHUNK_OVERLAP_SIZE)
        self.pages_splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_PAGE_SIZE, chunk_overlap=0)
        # self.text_splitter = MarkdownHeaderTextSplitter(chunk_size=1000, chunk_overlap=500)


    def load_documents_to_db(self):
        documents = load_documents("documents/")
        for document_pages in documents:

            document_pages_formatted = self.get_document_split_pages(document_pages)
            documents_splits = self.text_splitter.split_documents(document_pages_formatted)

            ct_document = self.get_document(documents_splits)
            self.add_document_pages(ct_document, document_pages_formatted)
            self.add_document_splits(ct_document, documents_splits)


    def get_text_compression(self, documents_split_txt):
        text_split_compression = self.prompt_creator.get_text_compression(documents_split_txt)
        text_split_compression_check = self.prompt_creator.get_text_compression_check(documents_split_txt, text_split_compression)

        if text_split_compression_check and "YES" in text_split_compression_check:
            return text_split_compression
        else:
            return text_split_compression_check


    def get_document_split_pages(self, document_pages):
        for document_page in document_pages:
            if len(document_page.page_content) > MAX_PAGE_SIZE:
                return self.pages_splitter.split_documents(document_pages)
        return document_pages



    def get_document(self, documents_splits):
        document_filename = documents_splits[0].metadata.get("source", "")
        ct_document = CTDocument.objects.filter(document_filename=document_filename)
        if not ct_document:
            ct_document = self.create_new_document(documents_splits)
        return ct_document


    def create_new_document(self, documents_splits):
        document_title = self.prompt_creator.get_document_title(documents_splits[0].page_content)
        document_filename = documents_splits[0].metadata.get("source", "")

        print("Document title: ", document_title)
        print("Document filename: ", document_filename)

        ct_document = CTDocument.objects.create(
            document_version="1",
            document_title=document_title,
            document_filename=document_filename)

        return ct_document

    def add_document_pages(self, ct_document, document_pages):
        old_document_pages = ct_document.document_pages.all()
        if len(old_document_pages) < len(document_pages):
            ct_document.document_pages.all().delete()
            for i, document_page in enumerate(document_pages):
                print("Document page content: ", document_page.page_content)
                CTDocumentPage.objects.create(
                    ct_document=ct_document,
                    document_page_text=document_page.page_content,
                    document_page=i)

    def add_document_splits(self, ct_document, documents_splits):
        for documents_split in documents_splits:
            document_page = documents_split.metadata.get("page", 0)
            split_text = documents_split.page_content
            split_text_compression = self.get_text_compression(documents_split.page_content)

            print("Document title: ", ct_document.document_title)
            print("Document filename: ", ct_document.document_filename)
            print("Document page: ", document_page)
            print("Split text: ", split_text)
            print("Split text compression: ", split_text_compression)

            embedding = self.model.get_embedding(split_text)

            CTDocumentSplit.objects.create(
                ct_document=ct_document,
                document_title=ct_document.document_title,
                document_filename=ct_document.document_filename,
                document_page=document_page,
                split_text=split_text,
                split_text_compression=split_text_compression,
                embedding=embedding)