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
from text_bot.views.models import CTDocument, \
    CTDocumentSplit, \
    CTDocumentPage, \
    CTDocumentSection, \
    CTDocumentSectionTitle, \
    CTDocumentSectionText, \
    CTDocumentSectionReferences, \
    CTDocumentSectionTopics, \
    CTDocumentSubsection,\
    CTDocumentSubsectionTitle,\
    CTDocumentSubsectionText,\
    CTDocumentSubsectionReferences,\
    CTDocumentSubsectionTopics


from text_bot.nlp_model.prompt_creator import PromptCreator

MAX_CHUNK_SIZE = 500
MAX_CHUNK_OVERLAP_SIZE = 250
MAX_SEMANTIC_CHUNK_SIZE = 1000
MAX_SEMANTIC_CHUNK_OVERLAP_SIZE = 500
MAX_PAGE_SIZE = 5500

HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
]

class VectorizeDocumentsEngine:


    def __init__(self, nlp_model :NlpModel):
        self.model = nlp_model
        self.prompt_creator = PromptCreator(nlp_model)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_CHUNK_SIZE, chunk_overlap=MAX_CHUNK_OVERLAP_SIZE)
        self.semantic_text_splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_SEMANTIC_CHUNK_SIZE, chunk_overlap=MAX_SEMANTIC_CHUNK_OVERLAP_SIZE)
        self.pages_splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_PAGE_SIZE, chunk_overlap=0)
        self.markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)


    def load_documents_to_db(self):
        documents = load_documents("documents/")
        for document_pages in documents:

            document_pages_formatted = self.get_document_split_pages(document_pages)

            md_header_splits = self.markdown_splitter.split_text(document_pages_formatted)
            documents_splits = self.text_splitter.split_documents(md_header_splits)

            self.add_document_pages(document_pages_formatted)
            self.add_document_splits(documents_splits)

    def load_semantic_document_chunks_to_db(self):
        documents = load_documents("documents/")
        for document_pages in documents:

            document_pages_formatted = self.get_document_split_pages(document_pages)
            documents_splits = self.semantic_text_splitter.split_documents(document_pages_formatted)

            self.add_document_pages(document_pages_formatted)
            self.add_semantic_document_splits(documents_splits)

    def splits_already_added_to_db(self, ct_document, documents_splits):
        old_document_splits = ct_document.document_splits.all()
        return len(old_document_splits) >= len(documents_splits)

    def get_text_compression(self, documents_split_txt):
        text_split_compression = self.prompt_creator.get_document_text_compression(documents_split_txt)
        text_split_compression_check = self.prompt_creator.get_document_text_compression_check(documents_split_txt, text_split_compression)

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

        ct_document = None
        try:
            ct_document = CTDocument.objects.get(document_filename=document_filename)
        except CTDocument.DoesNotExist as e:
            print(e)

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

    def add_document_pages(self, document_pages):
        ct_document = self.get_document(document_pages)
        if not self.pages_already_added_to_db(ct_document, document_pages):
            ct_document.document_pages.all().delete()
            for i, document_page in enumerate(document_pages):
                print("Document page content: ", document_page.page_content)
                CTDocumentPage.objects.create(
                    ct_document=ct_document,
                    document_page_text=document_page.page_content,
                    document_page_number=i)

    def pages_already_added_to_db(self, ct_document, document_pages):
        old_document_pages = ct_document.document_pages.all()
        return len(old_document_pages) >= len(document_pages)

    def add_document_splits(self, documents_splits):
        ct_document = self.get_document(documents_splits)
        if not self.splits_already_added_to_db(ct_document, documents_splits):
            ct_document.document_splits.all().delete()
            for i, documents_split in enumerate(documents_splits):
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
                    split_number=i,
                    embedding=embedding)

    def add_semantic_document_splits(self, documents_splits):
        ct_document = self.get_document(documents_splits)
        if not self.splits_already_added_to_db(ct_document, documents_splits):
            ct_document.document_splits.all().delete()

            previous_last_semantic_chunk = ""
            for i, documents_split in enumerate(documents_splits):
                document_page = documents_split.metadata.get("page", 0)
                split_text = documents_split.page_content

                semantic_sections_json_list = self.prompt_creator.get_document_semantic_text_chunks(split_text,
                                                                                                  previous_last_semantic_chunk)

                if not semantic_sections_json_list:
                    continue

                previous_last_semantic_chunk = semantic_sections_json_list[-1].get("subsection_list", [])[-1]

                for i, raw_semantic_section_json in enumerate(semantic_sections_json_list):
                    print("semantic_section_json: ", str(semantic_section_json))

                    semantic_subsections_json_list = semantic_section_json.get("subsection_list",[])

                    semantic_section_json = CTDocumentSection.objects.prepare_json(raw_semantic_section_json, i)
                    ct_document_section = CTDocumentSection.objects.create_from_json(semantic_section_json, ct_document, document_page)
                    ct_document_section_title = CTDocumentSectionTitle.objects.create_from_json(semantic_section_json, ct_document_section)
                    ct_document_section_text = CTDocumentSectionText.objects.create_from_json(semantic_section_json, ct_document_section)
                    ct_document_section_references = CTDocumentSectionReferences.objects.create_from_json(semantic_section_json, ct_document_section)
                    ct_document_section_topics = CTDocumentSectionTopics.objects.create_from_json(semantic_section_json, ct_document_section)

                    for j, raw_semantic_subsection_json in enumerate(semantic_subsections_json_list):

                        semantic_subsection_json = CTDocumentSubsection.objects.prepare_json(raw_semantic_subsection_json, j)
                        ct_document_subsection = CTDocumentSubsection.objects.create_from_json(semantic_subsection_json, ct_document_section)
                        ct_document_subsection_title = CTDocumentSubsectionTitle.objects.create_from_json(semantic_subsection_json, ct_document_subsection)
                        ct_document_subsection_text = CTDocumentSubsectionText.objects.create_from_json(semantic_subsection_json, ct_document_subsection)
                        ct_document_subsection_references = CTDocumentSubsectionReferences.objects.create_from_json(semantic_subsection_json, ct_document_subsection)
                        ct_document_subsection_topics = CTDocumentSubsectionTopics.objects.create_from_json(semantic_subsection_json, ct_document_subsection)



