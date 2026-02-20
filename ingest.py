from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import config

# 1. Load All PDF Files
pdf_files = [
    "./resources/gida_hijyeni_yonetmeligi.pdf",
    "./resources/toplu_tuketim_yerleri_hijyen_uygulama_kilavuzu.pdf"
    # "./resources/isyeri_acma_ve_calisma_ruhsatlarina_iliskin_yonetmelik.pdf",
]

all_documents=[]

for pdf_path in pdf_files:
  print(f"Loading: {pdf_path}")
  loader = PyPDFLoader(pdf_path)
  docs=loader.load()

  # Add resource metadata for each chunk
  for doc in docs:
    doc.metadata["resource"] = pdf_path.split("/")[-1].replace(".pdf", "")

  all_documents.extend(docs)

print(f"Loaded {len(all_documents)} pages total.")

# 2. Split into chunks
text_splitter= RecursiveCharacterTextSplitter(
  chunk_size=config.CHUNK_SIZE, 
  chunk_overlap=config.CHUNK_OVERLAP
)
documents=text_splitter.split_documents(all_documents)

print(f"Split into {len(documents)} chunks.")

# 3. Create embeddings and save to disk
embeddings=OpenAIEmbeddings(model=config.EMBEDDING_MODEL)

db = Chroma.from_documents(
  documents,
  embeddings,
  persist_directory=config.DB_PATH
)

print(f"✅ Database saved to ./chroma_db with {len(documents)} chunks!")