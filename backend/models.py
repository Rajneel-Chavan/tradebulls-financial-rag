"""Pydantic models and LangGraph state definitions."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_core.documents import Document


# ── Query Route ───────────────────────────────────────────────────────

class QueryRoute(BaseModel):
    """Route a user query to the most appropriate pipeline."""

    route: Literal[
        "direct_answer",
        "retrieve",
        "financial_fact_check",
    ] = Field(
        description=(
            "direct_answer — simple factual questions not needing document retrieval. "
            "retrieve — questions needing financial document context. "
            "financial_fact_check — market claims needing live verification."
        )
    )
    reasoning: str = Field(description="Brief explanation for chosen route.")


# ── Document Grading ──────────────────────────────────────────────────

class DocumentGrade(BaseModel):
    """Grade whether a retrieved document is relevant to the query."""

    relevant: Literal["yes", "no"] = Field(
        description="Is this document relevant to the user query? 'yes' or 'no'."
    )


# ── Generation Grading ────────────────────────────────────────────────

class GenerationGrade(BaseModel):
    """Grade the quality of a generated answer."""

    is_grounded: Literal["yes", "no"] = Field(
        description="Is the answer grounded in the provided documents?"
    )
    answers_query: Literal["yes", "no"] = Field(
        description="Does the answer actually address the user's question?"
    )


# ── Guardrail Check ───────────────────────────────────────────────────

class GuardrailResult(BaseModel):
    """Result of input/output guardrail check."""

    is_safe: bool = Field(description="Whether the content passes guardrail check.")
    reason: str = Field(default="", description="Reason if blocked.")
    modified_content: str = Field(
        default="", description="Modified content with disclaimers if needed."
    )


# ── LangGraph State ───────────────────────────────────────────────────

class AgenticRAGState(TypedDict):
    """State flowing through the LangGraph agentic RAG pipeline."""

    # Input
    query: str
    session_id: str

    # Routing
    route: str
    route_reasoning: str

    # Retrieval
    documents: list[Document]
    web_results: list[Document]

    # Generation
    generation: str
    retry_count: int

    # Grading
    doc_relevance_score: float
    is_grounded: bool
    answers_query: bool

    # Guardrails
    input_safe: bool
    guardrail_message: str

    # Metadata
    sources_used: list[str]
