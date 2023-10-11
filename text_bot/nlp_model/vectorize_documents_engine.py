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
from text_bot.nlp_model.prompt_template_creator import PromptTemplateCreator

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
        documents = load_documents("/content/drive/My Drive/kodi_bot")
        for document in documents:
            documents_splits = self.get_documents_splits(document)
            document_title = self.get_document_title(documents_splits[0])
            for documents_split in documents_splits:
                DocumentSplit.objects.create(
                        document_filename=document_filename,
                        document_title=document_title,
                        document_text=documents_split,
                        embedding=self.model.get_embeddings(documents_split))

    def get_document_title(self, first_documents_split):
        title_extract_prompt = self.prompt_template_creator.get_title_extract_prompt(first_documents_split)
        title = self.model.send_prompt(title_extract_prompt)
        return title


