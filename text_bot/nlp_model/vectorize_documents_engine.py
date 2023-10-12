import sys

sys.path.append("..")

import json

import numpy as np

from tqdm import tqdm

from text_bot.nlp_model.config import DATA
from text_bot.nlp_model.nlp_model import NlpModel
from text_bot.utils import load_documents

BOOK_FILENAME = "Marcus_Aurelius_Antoninus_-_His_Meditations_concerning_himselfe"

# SENTENCE_MIN_LENGTH = 15
SENTENCE_MIN_LENGTH = 2

from langchain.text_splitter import RecursiveCharacterTextSplitter
from text_bot.views.models import DocumentSplit
from text_bot.nlp_model.prompt_template_creator import PromptTemplateCreator, SYSTEM_MSG_TITLE

class VectorizeDocumentsEngine:

    def __init__(self, nlp_model :NlpModel):
        self.model = nlp_model
        self.prompt_template_creator = PromptTemplateCreator()

    def get_documents_splits(self, document):
        # MarkdownHeaderTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
        documents_splits = text_splitter.split_documents(document)
        return documents_splits

    def get_documents_splits(self):
        documents = load_documents("documents/")
        for document_page in documents:
            filename = document_page.metadata["source"]
            page = document_page.metadata["page"]
            document = document_page.page_content
            print("document: "+str(document))
            documents_splits = self.get_documents_splits(document)
            document_title = self.get_document_title(documents_splits[0])

            for documents_split in documents_splits:

                print("Document title: ", document_title)
                print("Document filename: ", filename)
                print("Document text: ", documents_split)
                print("Document page: ", page)

                embedding = self.model.get_embeddings(documents_split)

                DocumentSplit.objects.create(
                        filename=filename,
                        title=document_title,
                        text=documents_split,
                        embedding=embedding,
                        page = page)

    def get_document_title(self, first_documents_split):
        title_extract_prompt = self.prompt_template_creator.get_title_extract_prompt(first_documents_split)
        title = self.model.send_prompt(SYSTEM_MSG_TITLE, title_extract_prompt)
        return title


