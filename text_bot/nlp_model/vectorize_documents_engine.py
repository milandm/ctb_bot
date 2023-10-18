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
from text_bot.views.models import DocumentSplit

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

            for documents_split in documents_splits:

                text_split_compression = self.get_text_compression(documents_split.page_content)
                filename = documents_split.metadata.get("source","")
                # page = documents_split.metadata["page"]page
                page = documents_split.metadata.get("page",0)
                text = documents_split.page_content

                print("Document title: ", document_title)
                print("Document filename: ", filename)
                print("Document text: ", text)
                print("Document page: ", page)
                print("Document text_split_compression: ", text_split_compression)

                embedding = self.model.get_embedding(text)

                DocumentSplit.objects.create(
                        filename=filename,
                        title=document_title,
                        text=text,
                        embedding=embedding,
                        page = page,
                        text_compression = text_split_compression)

    def get_text_compression(self, documents_split_txt):
        text_split_compression = self.prompt_creator.get_text_compression(documents_split_txt)
        text_split_compression_check = self.prompt_creator.get_text_compression_check(documents_split_txt, text_split_compression)

        if text_split_compression_check and "YES" in text_split_compression_check:
            return text_split_compression
        else:
            return text_split_compression_check