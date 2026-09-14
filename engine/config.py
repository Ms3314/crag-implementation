"""Configuration and environment setup for the CRAG demo."""

import os

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# --- Pipeline settings -----------------------------------------------------
QUESTION = "What is Information retreival systems ????"
PDF_PATHS = ["documents/book1.pdf"]
INDEX_DIR = "faiss_index"
REBUILD_INDEX = False       # True = always re-embed; False = reuse faiss_index/ if present

# Chunking is in CHARACTERS. bge-large-en-v1.5 only handles 512 tokens
# (~4 chars/token), so keep chunk_size under ~2000 to avoid silent truncation.
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
EMBED_BATCH = 128           # texts per embedding request
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "hf").lower()

# --- Models ----------------------------------------------------------------
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
HF_EMBED_MODEL = "BAAI/bge-large-en-v1.5"
OPENAI_EMBED_MODEL = "text-embedding-3-large"
OLLAMA_EMBED_MODEL = "nomic-embed-text-v2-moe:latest"

# --- Corrective-RAG thresholds ---------------------------------------------
UPPER_TH = 0.8
LOWER_TH = 0.4
RERANK_TOP_K = 6            # keep the K best-scoring strips (ms-marco scores are not 0..1)
