"""
Smart chunking strategies for Tradebulls financial documents.

Uses two strategies:
  - SemanticChunker for financial PDFs (respects topic boundaries)
  - RecursiveCharacterTextSplitter for FAQs/web/YouTube (character-based)
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from backend.config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL


def get_recursive_splitter() -> RecursiveCharacterTextSplitter:
    """Standard recursive splitter for FAQ, web, YouTube content."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["→", "\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def get_semantic_splitter() -> SemanticChunker:
    """Semantic splitter for financial PDFs — splits on meaning boundaries."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=75,
    )


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Apply appropriate chunking strategy per source type.

    - PDF documents → SemanticChunker (respects topic boundaries)
    - FAQ/Web/YouTube → RecursiveCharacterTextSplitter
    """
    recursive_splitter = get_recursive_splitter()

    pdf_docs = []
    other_docs = []

    for doc in documents:
        source_type = doc.metadata.get("source_type", "unknown")
        if source_type == "pdf":
            pdf_docs.append(doc)
        else:
            other_docs.append(doc)

    all_chunks = []

    # Chunk PDFs with semantic splitter
    if pdf_docs:
        try:
            semantic_splitter = get_semantic_splitter()
            pdf_chunks = semantic_splitter.split_documents(pdf_docs)

            for i, chunk in enumerate(pdf_chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["chunking_strategy"] = "semantic"

            all_chunks.extend(pdf_chunks)
            print(f"[Chunker] PDF semantic chunks: {len(pdf_chunks)}")

        except Exception as e:
            print(f"[Chunker] Semantic chunking failed, falling back to recursive: {e}")
            pdf_chunks = recursive_splitter.split_documents(pdf_docs)
            for i, chunk in enumerate(pdf_chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["chunking_strategy"] = "recursive_fallback"
            all_chunks.extend(pdf_chunks)

    # Chunk other sources with recursive splitter
    if other_docs:
        other_chunks = recursive_splitter.split_documents(other_docs)

        for i, chunk in enumerate(other_chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunking_strategy"] = "recursive"

        all_chunks.extend(other_chunks)
        print(f"[Chunker] Other recursive chunks: {len(other_chunks)}")

    print(f"[Chunker] Total chunks produced: {len(all_chunks)}")
    return all_chunks
