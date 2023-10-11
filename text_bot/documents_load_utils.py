import sys
import os
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.text_splitter import MarkdownHeaderTextSplitter
# "/content/drive/My Drive/kodi_bot"

def load_documents(documents_folder_path):
    documents = []
    for file in os.listdir(documents_folder_path):
        if file.endswith(".pdf"):
            pdf_path = "./documents/" + file
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        elif file.endswith('.docx') or file.endswith('.doc'):
            doc_path = "./documents/" + file
            loader = Docx2txtLoader(doc_path)
            documents.extend(loader.load())
        elif file.endswith('.txt'):
            text_path = "./documents/" + file
            loader = TextLoader(text_path)
            documents.extend(loader.load())
    return documents


def get_documents_splits(documents):
    # MarkdownHeaderTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    documents_splits = text_splitter.split_documents(documents)
    return documents_splits