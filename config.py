"""
Configuration file - stores all setting in one place.
This makes it easy to change values without searching through multiple files.
"""

# OpenAI Configuration
EMBEDDING_MODEL="text-embedding-3-small"
LLM_MODEL="gpt-4o"
LLM_TEMPERATURE=0

# Database Configuration
DB_PATH="./chroma_db"

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# RAG Configuration
DEFAULT_TOP_K=3 # Number of results to retrieve