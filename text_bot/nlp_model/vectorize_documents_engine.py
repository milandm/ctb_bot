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
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)

    def get_documents_splits(self):
        documents = load_documents("documents/")
        for document_pages in documents:

            documents_splits = self.text_splitter.split_documents(document_pages)
            document_title = self.get_document_title(documents_splits[0])

            for documents_split in documents_splits:

                filename = documents_split.metadata["source"]
                page = documents_split.metadata["page"]
                text = documents_split.page_content

                print("Document title: ", document_title)
                print("Document filename: ", filename)
                print("Document text: ", text)
                print("Document page: ", page)

                embedding = self.model.get_embeddings(text)

                DocumentSplit.objects.create(
                        filename=filename,
                        title=document_title,
                        text=text,
                        embedding=embedding,
                        page = page)

    def get_document_title(self, first_documents_split):
        title_extract_prompt = self.prompt_template_creator.get_title_extract_prompt(first_documents_split)
        title = self.model.send_prompt(SYSTEM_MSG_TITLE, title_extract_prompt)
        return title


