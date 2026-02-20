"""
Ingestion module - functions for loading and processing documents.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import config


def load_pdfs(pdf_files):
    """
    Load PDF files and return documents with metadata.

    Args:
        pdf_files: List of PDF file paths

    Returns:
        List of documents with metadata
    """
    all_documents = []

    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        for doc in docs:
            doc.metadata["resource"] = pdf_path.split("/")[-1].replace(".pdf", "")

        all_documents.extend(docs)

    print(f"Loaded {len(all_documents)} pages total.")
    return all_documents


def split_documents(documents):
    """
    Split documents into chunks for embedding.

    Args:
        documents: List of documents to split

    Returns:
        List of chunked documents
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunked_documents = text_splitter.split_documents(documents)
    print(f"Split into {len(chunked_documents)} chunks.")
    return chunked_documents

def save_documents_to_db(documents):
    """
    Create embeddings and save documents to the vector database.

    Args:
        documents: List of chunked documents to save

    Returns:
        The Chroma database object (in case you need it later)
    """

    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    db = Chroma.from_documents(
        documents,
        embeddings,
        persist_directory=config.DB_PATH,
    )
    print(f"✅ Database saved to {config.DB_PATH} with {len(documents)} chunks!")
    return db