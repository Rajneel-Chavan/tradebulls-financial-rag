"""
Qdrant vector store with session-isolated collections.

Each user session gets its own collection: tradebulls_{session_id}
Supports CacheBackedEmbeddings to avoid re-embedding duplicates.
"""

import uuid

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain.storage import InMemoryByteStore
from langchain.embeddings import CacheBackedEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from backend.config import (
    QDRANT_HOST,
    QDRANT_PORT,
    EMBEDDING_MODEL,
    RETRIEVAL_TOP_K,
)


def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client instance."""
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def get_cached_embeddings() -> CacheBackedEmbeddings:
    """Get CacheBackedEmbeddings to avoid re-embedding same content."""
    base_embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    store = InMemoryByteStore()

    return CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=base_embeddings,
        document_embedding_cache=store,
        namespace=EMBEDDING_MODEL,
    )


def get_collection_name(session_id: str) -> str:
    """Generate session-isolated collection name."""
    return f"tradebulls_{session_id}"


def create_session_id() -> str:
    """Generate a new session ID."""
    return str(uuid.uuid4())[:8]


def create_vector_store(
    documents: list[Document],
    session_id: str,
) -> QdrantVectorStore:
    """
    Create a Qdrant vector store for a specific session.

    Each session gets isolated storage — different documents
    per conversation thread.
    """
    collection_name = get_collection_name(session_id)
    embeddings = get_cached_embeddings()
    client = get_qdrant_client()

    # Create collection if it doesn't exist
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

    # Build Qdrant vector store from documents
    vector_store = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
        force_recreate=False,
    )

    print(
        f"[VectorStore] Indexed {len(documents)} chunks "
        f"into {collection_name}"
    )
    return vector_store


def get_existing_vector_store(session_id: str) -> QdrantVectorStore | None:
    """Reconnect to an existing session's vector store."""
    collection_name = get_collection_name(session_id)
    client = get_qdrant_client()

    collections = [c.name for c in client.get_collections().collections]

    if collection_name not in collections:
        return None

    embeddings = get_cached_embeddings()

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )


def get_dense_retriever(vector_store: QdrantVectorStore, top_k: int = RETRIEVAL_TOP_K):
    """Get dense similarity search retriever."""
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )


def delete_session(session_id: str) -> bool:
    """Delete a session's vector store collection."""
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
    """List all active Tradebulls sessions."""
    client = get_qdrant_client()
    collections = client.get_collections().collections

    sessions = []
    for col in collections:
        if col.name.startswith("tradebulls_"):
            session_id = col.name.replace("tradebulls_", "")
            sessions.append(session_id)

    return sessions
