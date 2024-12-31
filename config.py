# config.py

# Central configuration for the Generative AI Math project

MODEL_NAME = "gpt2"  # or "EleutherAI/gpt-neo-1.3B", "facebook/galactica-125m", etc.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DB_DIR = "chroma_db"  # folder for local Chroma vector store
DOCS_PATH = "data/sample_docs.txt"

# Generation parameters
MAX_LENGTH = 150
TOP_P = 0.9
