"""
Qdrant vector store with session-isolated collections.

Each user session gets its own collection: tradebulls_{session_id}

Connection priority:
  1. QDRANT_URL + QDRANT_API_KEY  → Qdrant Cloud (production)
  2. No URL set                   → in-memory Qdrant (Streamlit Cloud demo)
"""

import uuid

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from backend.config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_URL,
    QDRANT_API_KEY,
    EMBEDDING_MODEL,
    RETRIEVAL_TOP_K,
)

# Singleton client — one connection shared across all calls in this process.
# In-memory client must be a singleton or collections disappear between calls.
_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        if QDRANT_URL:
            _client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY or None,
            )
            print(f"[VectorStore] Connected to Qdrant Cloud: {QDRANT_URL}")
        else:
            _client = QdrantClient(":memory:")
            print("[VectorStore] Using in-memory Qdrant (data resets on restart)")
    return _client


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def get_collection_name(session_id: str) -> str:
    return f"tradebulls_{session_id}"


def create_session_id() -> str:
    return str(uuid.uuid4())[:8]


def create_vector_store(
    documents: list[Document],
    session_id: str,
) -> QdrantVectorStore:
    collection_name = get_collection_name(session_id)
    embeddings = get_embeddings()
    client = get_qdrant_client()

    collections = [c.name for c in client.get_collections().collections]
    if collection_name not in collections:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=1536,  # text-embedding-3-small dimension
                distance=Distance.COSINE,
            ),
        )
        print(f"[VectorStore] Created collection: {collection_name}")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    vector_store.add_documents(documents)

    print(f"[VectorStore] Indexed {len(documents)} chunks into {collection_name}")
    return vector_store


def get_existing_vector_store(session_id: str) -> QdrantVectorStore | None:
    collection_name = get_collection_name(session_id)
    client = get_qdrant_client()

    collections = [c.name for c in client.get_collections().collections]
    if collection_name not in collections:
        return None

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=get_embeddings(),
    )


def get_dense_retriever(vector_store: QdrantVectorStore, top_k: int = RETRIEVAL_TOP_K):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )


def delete_session(session_id: str) -> bool:
    collection_name = get_collection_name(session_id)
    client = get_qdrant_client()
    try:
        client.delete_collection(collection_name)
        print(f"[VectorStore] Deleted collection: {collection_name}")
        return True
    except Exception as e:
        print(f"[VectorStore] Failed to delete {collection_name}: {e}")
        return False


def list_sessions() -> list[str]:
    client = get_qdrant_client()
    sessions = []
    for col in client.get_collections().collections:
        if col.name.startswith("tradebulls_"):
            sessions.append(col.name.replace("tradebulls_", ""))
    return sessions
