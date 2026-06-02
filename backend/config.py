"""Centralized configuration for Tradebulls Financial RAG System."""

import os
from dotenv import load_dotenv

load_dotenv()


# ── API Keys ──────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ── Qdrant ────────────────────────────────────────────────────────────
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_URL = os.getenv("QDRANT_URL", "")          # set for Qdrant Cloud
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")  # set for Qdrant Cloud

# ── LangSmith ─────────────────────────────────────────────────────────
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "tradebulls-financial-rag")

# ── Data Sources ──────────────────────────────────────────────────────
YOUTUBE_VIDEO_ID = os.getenv("YOUTUBE_VIDEO_ID", "e9B2F3m0CL8")
WEB_SCRAPE_URL = os.getenv("WEB_SCRAPE_URL", "https://www.tradebulls.in/about-us")
PDF_DIRECTORY = os.getenv("PDF_DIRECTORY", "documents")

# ── Model Config ──────────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.2
COHERE_RERANK_MODEL = "rerank-english-v3.0"

# ── Retrieval Config ──────────────────────────────────────────────────
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
RETRIEVAL_TOP_K = 12       # chunks before reranking
RERANK_TOP_N = 4           # chunks after reranking
HYDE_ENABLED = True
RRF_CONSTANT = 60          # RRF fusion constant

# ── Agentic Config ────────────────────────────────────────────────────
MAX_RETRIES = 2             # max regeneration loops
RELEVANCE_THRESHOLD = 0.5  # min relevance score for grading
