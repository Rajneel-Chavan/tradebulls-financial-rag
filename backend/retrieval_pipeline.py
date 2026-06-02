"""
Advanced Retrieval Pipeline for Tradebulls Financial RAG.

Pipeline stages:
  1. HyDE — generate hypothetical financial analyst answer, embed that instead
  2. RRF Fusion — 3 parallel retrievers (dense, BM25, multi-query) fused
  3. Cohere Reranking — cross-encoder reranks top candidates

This is the core differentiator from vanilla RAG.
"""

from __future__ import annotations

import cohere
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from backend.config import (
    COHERE_API_KEY,
    COHERE_RERANK_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    EMBEDDING_MODEL,
    RETRIEVAL_TOP_K,
    RERANK_TOP_N,
    HYDE_ENABLED,
    RRF_CONSTANT,
)


# ── HyDE: Hypothetical Document Embeddings ────────────────────────────

HYDE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a senior financial analyst at a trading firm. "
        "Write a detailed paragraph that would appear in a financial "
        "research report answering the following question. "
        "Use professional financial language. Do NOT add disclaimers.",
    ),
    ("human", "{query}"),
])


def generate_hypothetical_answer(query: str) -> str:
    """
    HyDE: Generate a hypothetical answer as a financial analyst would write it.
    The hypothesis is embedded instead of the raw query for better retrieval.
    """
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.7)
    chain = HYDE_PROMPT | llm | StrOutputParser()
    hypothesis = chain.invoke({"query": query})
    return hypothesis


def embed_hyde_query(query: str) -> list[float]:
    """Embed the HyDE hypothesis for similarity search."""
    hypothesis = generate_hypothetical_answer(query)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return embeddings.embed_query(hypothesis)


# ── Multi-Query Generation ────────────────────────────────────────────

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a financial research assistant. "
        "Generate 3 alternative versions of the given query "
        "to retrieve relevant financial documents. "
        "Each version should approach the question from a different angle. "
        "Return ONLY the 3 queries, one per line. No numbering.",
    ),
    ("human", "{query}"),
])


def generate_multi_queries(query: str) -> list[str]:
    """Generate 3 alternative query formulations for multi-query retrieval."""
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.5)
    chain = MULTI_QUERY_PROMPT | llm | StrOutputParser()
    result = chain.invoke({"query": query})
    queries = [q.strip() for q in result.strip().split("\n") if q.strip()]
    return queries[:3]


# ── BM25 Retriever ────────────────────────────────────────────────────

def create_bm25_retriever(
    documents: list[Document],
    top_k: int = RETRIEVAL_TOP_K,
) -> BM25Retriever:
    """Create BM25 sparse retriever from document chunks."""
    retriever = BM25Retriever.from_documents(documents, k=top_k)
    return retriever


# ── Reciprocal Rank Fusion ────────────────────────────────────────────

def reciprocal_rank_fusion(
    result_lists: list[list[Document]],
    k: int = RRF_CONSTANT,
) -> list[Document]:
    """
    Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    RRF score = Σ (1 / (k + rank_i)) for each document across all lists.
    Higher k reduces the impact of high-ranking outliers.
    """
    doc_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list):
            doc_key = doc.page_content[:200]  # use content prefix as key
            if doc_key not in doc_map:
                doc_map[doc_key] = doc
                doc_scores[doc_key] = 0.0
            doc_scores[doc_key] += 1.0 / (k + rank + 1)

    # Sort by RRF score descending
    sorted_keys = sorted(doc_scores, key=lambda x: doc_scores[x], reverse=True)

    fused_docs = []
    for key in sorted_keys:
        doc = doc_map[key]
        doc.metadata["rrf_score"] = round(doc_scores[key], 6)
        fused_docs.append(doc)

    return fused_docs


# ── Cohere Cross-Encoder Reranking ────────────────────────────────────

def cohere_rerank(
    query: str,
    documents: list[Document],
    top_n: int = RERANK_TOP_N,
) -> list[Document]:
    """
    Rerank documents using Cohere cross-encoder.
    Cross-encoder considers query-document pair jointly, much more accurate
    than bi-encoder similarity search alone.
    """
    if not documents:
        return []

    client = cohere.Client(api_key=COHERE_API_KEY)

    doc_texts = [doc.page_content for doc in documents]

    response = client.rerank(
        model=COHERE_RERANK_MODEL,
        query=query,
        documents=doc_texts,
        top_n=min(top_n, len(documents)),
    )

    reranked_docs = []
    for result in response.results:
        doc = documents[result.index]
        doc.metadata["rerank_score"] = round(result.relevance_score, 4)
        reranked_docs.append(doc)

    return reranked_docs


# ── Full Advanced Retrieval Pipeline ──────────────────────────────────

def advanced_retrieve(
    query: str,
    vector_store: Qdrant,
    all_chunks: list[Document],
    top_k: int = RETRIEVAL_TOP_K,
    rerank_top_n: int = RERANK_TOP_N,
) -> list[Document]:
    """
    Full advanced retrieval pipeline:
      1. HyDE query expansion (if enabled)
      2. Three parallel retrievers: dense, BM25, multi-query
      3. RRF fusion across all results
      4. Cohere cross-encoder reranking

    Returns the final top_n most relevant documents.
    """
    result_lists = []

    # ── Retriever 1: Dense (with optional HyDE) ──────────────────────
    if HYDE_ENABLED:
        try:
            hyde_embedding = embed_hyde_query(query)
            dense_results = vector_store.similarity_search_by_vector(
                embedding=hyde_embedding, k=top_k
            )
            print(f"  [Retrieval] Dense (HyDE): {len(dense_results)} docs")
        except Exception as e:
            print(f"  [Retrieval] HyDE failed, using standard dense: {e}")
            dense_retriever = vector_store.as_retriever(
                search_kwargs={"k": top_k}
            )
            dense_results = dense_retriever.invoke(query)
    else:
        dense_retriever = vector_store.as_retriever(
            search_kwargs={"k": top_k}
        )
        dense_results = dense_retriever.invoke(query)
        print(f"  [Retrieval] Dense (standard): {len(dense_results)} docs")

    result_lists.append(dense_results)

    # ── Retriever 2: BM25 Sparse ─────────────────────────────────────
    try:
        bm25_retriever = create_bm25_retriever(all_chunks, top_k=top_k)
        bm25_results = bm25_retriever.invoke(query)
        result_lists.append(bm25_results)
        print(f"  [Retrieval] BM25: {len(bm25_results)} docs")
    except Exception as e:
        print(f"  [Retrieval] BM25 failed: {e}")

    # ── Retriever 3: Multi-Query ─────────────────────────────────────
    try:
        alt_queries = generate_multi_queries(query)
        multi_query_results = []

        dense_retriever = vector_store.as_retriever(
            search_kwargs={"k": top_k // 3}
        )

        for alt_q in alt_queries:
            results = dense_retriever.invoke(alt_q)
            multi_query_results.extend(results)

        result_lists.append(multi_query_results)
        print(
            f"  [Retrieval] Multi-Query ({len(alt_queries)} queries): "
            f"{len(multi_query_results)} docs"
        )
    except Exception as e:
        print(f"  [Retrieval] Multi-Query failed: {e}")

    # ── RRF Fusion ───────────────────────────────────────────────────
    fused_docs = reciprocal_rank_fusion(result_lists)
    print(f"  [Retrieval] RRF Fusion: {len(fused_docs)} unique docs")

    # ── Cohere Reranking ─────────────────────────────────────────────
    try:
        reranked_docs = cohere_rerank(
            query=query,
            documents=fused_docs[:top_k],  # rerank top candidates
            top_n=rerank_top_n,
        )
        print(
            f"  [Retrieval] Cohere Reranked: {len(reranked_docs)} docs "
            f"(scores: {[d.metadata.get('rerank_score', 0) for d in reranked_docs]})"
        )
        return reranked_docs

    except Exception as e:
        print(f"  [Retrieval] Cohere reranking failed, using RRF top: {e}")
        return fused_docs[:rerank_top_n]
