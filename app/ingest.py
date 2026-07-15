from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = "./chroma_db"

def ingest_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    return len(chunks)

def get_vectorstore():
    embeddings = OpenAIEmbeddings()
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )