"""
LangGraph Agentic RAG Pipeline for Tradebulls Financial Intelligence.

The agent autonomously:
  1. Routes queries (direct answer / retrieve / financial fact-check)
  2. Retrieves using advanced pipeline (HyDE + RRF + Cohere reranking)
  3. Grades retrieved documents for relevance
  4. Falls back to web search if docs are irrelevant (CRAG)
  5. Generates an answer
  6. Grades its own answer (Self-RAG: grounded? answers query?)
  7. Loops back with rewritten query if quality fails (max 2 retries)

Flow:
  input_guardrail → route_query → [retrieve / direct / fact_check]
    → grade_documents → [web_search if irrelevant]
    → generate → grade_generation → [rewrite + retry OR output_guardrail]
    → END
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END

from backend.config import LLM_MODEL, LLM_TEMPERATURE, MAX_RETRIES
from backend.models import (
    AgenticRAGState,
    QueryRoute,
    DocumentGrade,
    GenerationGrade,
)
from backend.guardrails import (
    check_input_guardrail,
    check_output_guardrail,
)
from backend.retrieval_pipeline import advanced_retrieve


# ── Shared LLM Instance ──────────────────────────────────────────────

def get_llm(temperature: float = LLM_TEMPERATURE) -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL, temperature=temperature)


# ═══════════════════════════════════════════════════════════════════════
# NODE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════


def input_guardrail_node(state: AgenticRAGState) -> dict:
    """Check if the user query is safe to process."""
    print("\n🛡️  [Node] Input Guardrail")
    result = check_input_guardrail(state["query"])

    if not result.is_safe:
        print(f"  ❌ BLOCKED: {result.reason}")
        return {
            "input_safe": False,
            "guardrail_message": result.modified_content,
            "generation": result.modified_content,
        }

    print("  ✅ Query is safe")
    return {"input_safe": True, "guardrail_message": ""}


def route_query_node(state: AgenticRAGState) -> dict:
    """Route the query to the appropriate pipeline."""
    print("\n🔀 [Node] Route Query")

    llm = get_llm(temperature=0)
    structured_llm = llm.with_structured_output(QueryRoute)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a financial query router for Tradebulls Securities.
Route the query to the best pipeline:

- "direct_answer": Simple factual questions about trading concepts, platform features,
  or general finance knowledge that don't need document retrieval.
  Examples: "What is a stop loss?", "How to open a demat account?"

- "retrieve": Questions that need information from Tradebulls research reports,
  market analysis PDFs, or company-specific data.
  Examples: "What does Tradebulls recommend for Nifty?", "What's in the latest report?"

- "financial_fact_check": Market claims that need live verification against
  current data plus document context.
  Examples: "Is the market bullish right now?", "Has Nifty broken support today?"
""",
        ),
        ("human", "{query}"),
    ])

    chain = prompt | structured_llm
    route_result = chain.invoke({"query": state["query"]})

    print(f"  Route: {route_result.route}")
    print(f"  Reason: {route_result.reasoning}")

    return {
        "route": route_result.route,
        "route_reasoning": route_result.reasoning,
    }


def retrieve_node(state: AgenticRAGState) -> dict:
    """Retrieve documents using the advanced pipeline (HyDE + RRF + Cohere)."""
    print("\n📚 [Node] Advanced Retrieve")

    # vector_store and all_chunks are injected via graph config
    vector_store = state.get("_vector_store")
    all_chunks = state.get("_all_chunks", [])

    if vector_store is None:
        print("  ⚠️  No vector store available")
        return {"documents": [], "sources_used": ["none"]}

    documents = advanced_retrieve(
        query=state["query"],
        vector_store=vector_store,
        all_chunks=all_chunks,
    )

    sources = list(set(
        doc.metadata.get("source_type", "unknown") for doc in documents
    ))

    print(f"  Retrieved {len(documents)} documents from sources: {sources}")
    return {"documents": documents, "sources_used": sources}


def grade_documents_node(state: AgenticRAGState) -> dict:
    """Grade each retrieved document for relevance (Self-RAG pattern)."""
    print("\n📝 [Node] Grade Documents")

    llm = get_llm(temperature=0)
    structured_llm = llm.with_structured_output(DocumentGrade)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a financial document relevance grader. "
            "Given a user question about financial markets or trading, "
            "determine if the retrieved document contains information "
            "relevant to answering the question. "
            "Respond with 'yes' if relevant, 'no' if not.",
        ),
        (
            "human",
            "Question: {query}\n\nDocument:\n{document}",
        ),
    ])

    relevant_docs = []
    for i, doc in enumerate(state.get("documents", [])):
        grade = (prompt | structured_llm).invoke({
            "query": state["query"],
            "document": doc.page_content,
        })
        if grade.relevant == "yes":
            relevant_docs.append(doc)
            print(f"  Doc {i + 1}: ✅ relevant")
        else:
            print(f"  Doc {i + 1}: ❌ not relevant")

    total = len(state.get("documents", []))
    relevant_count = len(relevant_docs)
    score = relevant_count / max(total, 1)

    print(f"  Relevance: {relevant_count}/{total} = {score:.2f}")

    return {
        "documents": relevant_docs,
        "doc_relevance_score": score,
    }


def web_search_node(state: AgenticRAGState) -> dict:
    """CRAG: Fallback to web search when document retrieval is insufficient."""
    print("\n🌐 [Node] Web Search Fallback (CRAG)")

    search = TavilySearchResults(max_results=3)
    search_query = f"Tradebulls {state['query']} Indian stock market"

    try:
        results = search.invoke(search_query)
        web_docs = [
            Document(
                page_content=r.get("content", ""),
                metadata={
                    "source_type": "web_search",
                    "url": r.get("url", ""),
                    "title": r.get("title", ""),
                },
            )
            for r in results
            if r.get("content")
        ]

        print(f"  Found {len(web_docs)} web results")

        # Merge with any existing relevant documents
        existing_docs = state.get("documents", [])
        all_docs = existing_docs + web_docs

        sources = list(set(
            doc.metadata.get("source_type", "unknown") for doc in all_docs
        ))

        return {
            "documents": all_docs,
            "web_results": web_docs,
            "sources_used": sources,
        }

    except Exception as e:
        print(f"  ⚠️  Web search failed: {e}")
        return {"web_results": [], "sources_used": state.get("sources_used", [])}


def generate_node(state: AgenticRAGState) -> dict:
    """Generate answer from retrieved context."""
    print("\n💬 [Node] Generate")

    documents = state.get("documents", [])

    if not documents:
        return {
            "generation": (
                "I couldn't find relevant information in the Tradebulls "
                "knowledge base or web search to answer this question. "
                "Please try rephrasing or ask about a different topic."
            )
        }

    context = "\n\n---\n\n".join(doc.page_content for doc in documents)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a knowledgeable financial assistant for Tradebulls Securities.
Answer the user's question using ONLY the provided context from financial reports,
research documents, and market data.

Rules:
- Be specific with numbers, levels, targets, and dates from the context
- If the context mentions specific stock recommendations, include entry/exit levels
- If the context is from different time periods, mention the dates
- If you can't fully answer from the context, say what you can and note limitations
- Use professional financial language

Context:
{context}""",
        ),
        ("human", "{query}"),
    ])

    llm = get_llm()
    chain = prompt | llm | StrOutputParser()

    generation = chain.invoke({
        "context": context,
        "query": state["query"],
    })

    print(f"  Generated {len(generation)} chars")
    return {"generation": generation}


def direct_answer_node(state: AgenticRAGState) -> dict:
    """Answer simple factual questions without retrieval."""
    print("\n💬 [Node] Direct Answer")

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a knowledgeable financial assistant for Tradebulls Securities. "
            "Answer the user's question about general financial concepts, "
            "trading platform features, or basic market knowledge. "
            "Be concise and accurate. Use professional language.",
        ),
        ("human", "{query}"),
    ])

    llm = get_llm()
    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke({"query": state["query"]})

    print(f"  Generated {len(generation)} chars (direct)")
    return {
        "generation": generation,
        "sources_used": ["direct_knowledge"],
    }


def grade_generation_node(state: AgenticRAGState) -> dict:
    """Self-RAG: Grade the generated answer for groundedness and relevance."""
    print("\n🔍 [Node] Grade Generation (Self-RAG)")

    llm = get_llm(temperature=0)
    structured_llm = llm.with_structured_output(GenerationGrade)

    documents = state.get("documents", [])
    context = "\n\n".join(doc.page_content for doc in documents) if documents else ""

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a quality checker for financial AI responses.
Given the source documents and the generated answer, evaluate:

1. is_grounded: Is the answer supported by the provided documents?
   Answer 'yes' if all claims are traceable to the documents.
   Answer 'no' if the answer contains hallucinated information.

2. answers_query: Does the answer actually address the user's question?
   Answer 'yes' if the question is meaningfully answered.
   Answer 'no' if the answer is off-topic or superficial.""",
        ),
        (
            "human",
            "Question: {query}\n\nDocuments:\n{context}\n\nGenerated Answer:\n{generation}",
        ),
    ])

    try:
        grade = (prompt | structured_llm).invoke({
            "query": state["query"],
            "context": context,
            "generation": state.get("generation", ""),
        })

        print(f"  Grounded: {grade.is_grounded}")
        print(f"  Answers query: {grade.answers_query}")

        return {
            "is_grounded": grade.is_grounded == "yes",
            "answers_query": grade.answers_query == "yes",
        }

    except Exception as e:
        print(f"  ⚠️  Grading failed: {e}, assuming pass")
        return {"is_grounded": True, "answers_query": True}


def rewrite_query_node(state: AgenticRAGState) -> dict:
    """Rewrite the query for a better retrieval attempt."""
    print("\n✏️  [Node] Rewrite Query")

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a financial query optimizer. "
            "The original query did not retrieve good results. "
            "Rewrite it to be more specific and likely to match "
            "content in Tradebulls financial research reports. "
            "Keep the same intent but use different financial terminology.",
        ),
        ("human", "Original query: {query}\n\nRewrite this query:"),
    ])

    llm = get_llm(temperature=0.3)
    chain = prompt | llm | StrOutputParser()
    new_query = chain.invoke({"query": state["query"]})

    retry_count = state.get("retry_count", 0) + 1

    print(f"  Original: {state['query']}")
    print(f"  Rewritten: {new_query}")
    print(f"  Retry: {retry_count}/{MAX_RETRIES}")

    return {
        "query": new_query,
        "retry_count": retry_count,
    }


def output_guardrail_node(state: AgenticRAGState) -> dict:
    """Apply output guardrail and add financial disclaimers."""
    print("\n🛡️  [Node] Output Guardrail")

    result = check_output_guardrail(
        response=state.get("generation", ""),
        query=state.get("query", ""),
    )

    if not result.is_safe:
        print(f"  ❌ Output blocked: {result.reason}")
        return {"generation": result.modified_content}

    print("  ✅ Output passed")
    return {"generation": result.modified_content}


# ═══════════════════════════════════════════════════════════════════════
# CONDITIONAL EDGES
# ═══════════════════════════════════════════════════════════════════════


def route_after_guardrail(state: AgenticRAGState) -> str:
    """After input guardrail: continue or end."""
    if not state.get("input_safe", True):
        return "end"
    return "route_query"


def route_after_routing(state: AgenticRAGState) -> str:
    """After routing: which pipeline to use."""
    route = state.get("route", "retrieve")

    if route == "direct_answer":
        return "direct_answer"
    elif route == "financial_fact_check":
        return "retrieve"  # retrieve first, then we'll also web search
    else:
        return "retrieve"


def route_after_grading_docs(state: AgenticRAGState) -> str:
    """After document grading: use docs or fallback to web."""
    score = state.get("doc_relevance_score", 0)
    route = state.get("route", "")

    # If fact-checking or low relevance, also do web search
    if route == "financial_fact_check" or score < 0.5:
        return "web_search"

    return "generate"


def route_after_grading_gen(state: AgenticRAGState) -> str:
    """After generation grading: accept, retry, or force accept."""
    is_grounded = state.get("is_grounded", True)
    answers_query = state.get("answers_query", True)
    retry_count = state.get("retry_count", 0)

    if is_grounded and answers_query:
        return "output_guardrail"

    if retry_count >= MAX_RETRIES:
        print(f"  ⚠️  Max retries ({MAX_RETRIES}) reached, accepting answer")
        return "output_guardrail"

    return "rewrite_query"


# ═══════════════════════════════════════════════════════════════════════
# BUILD THE GRAPH
# ═══════════════════════════════════════════════════════════════════════


def build_rag_graph() -> StateGraph:
    """
    Build the complete agentic RAG graph.

    Flow:
      input_guardrail → route_query → [retrieve / direct_answer]
        → grade_documents → [web_search / generate]
        → generate → grade_generation → [rewrite+retry / output_guardrail]
        → END
    """
    graph = StateGraph(AgenticRAGState)

    # ── Add Nodes ─────────────────────────────────────────────────────
    graph.add_node("input_guardrail", input_guardrail_node)
    graph.add_node("route_query", route_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("direct_answer", direct_answer_node)
    graph.add_node("grade_documents", grade_documents_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("generate", generate_node)
    graph.add_node("grade_generation", grade_generation_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("output_guardrail", output_guardrail_node)

    # ── Set Entry Point ───────────────────────────────────────────────
    graph.set_entry_point("input_guardrail")

    # ── Add Edges ─────────────────────────────────────────────────────

    # Input guardrail → route or end
    graph.add_conditional_edges(
        "input_guardrail",
        route_after_guardrail,
        {
            "route_query": "route_query",
            "end": END,
        },
    )

    # Route → retrieve or direct answer
    graph.add_conditional_edges(
        "route_query",
        route_after_routing,
        {
            "retrieve": "retrieve",
            "direct_answer": "direct_answer",
        },
    )

    # Direct answer → output guardrail → END
    graph.add_edge("direct_answer", "output_guardrail")

    # Retrieve → grade documents
    graph.add_edge("retrieve", "grade_documents")

    # Grade documents → web search or generate
    graph.add_conditional_edges(
        "grade_documents",
        route_after_grading_docs,
        {
            "web_search": "web_search",
            "generate": "generate",
        },
    )

    # Web search → generate
    graph.add_edge("web_search", "generate")

    # Generate → grade generation
    graph.add_edge("generate", "grade_generation")

    # Grade generation → output guardrail or rewrite
    graph.add_conditional_edges(
        "grade_generation",
        route_after_grading_gen,
        {
            "output_guardrail": "output_guardrail",
            "rewrite_query": "rewrite_query",
        },
    )

    # Rewrite → retrieve again (loop)
    graph.add_edge("rewrite_query", "retrieve")

    # Output guardrail → END
    graph.add_edge("output_guardrail", END)

    return graph


def compile_rag_graph():
    """Compile the graph for execution."""
    graph = build_rag_graph()
    return graph.compile()


def run_query(
    query: str,
    session_id: str,
    vector_store=None,
    all_chunks: list[Document] | None = None,
) -> dict:
    """
    Run a query through the full agentic RAG pipeline.

    Returns the final state with generation, sources, and metadata.
    """
    app = compile_rag_graph()

    initial_state: AgenticRAGState = {
        "query": query,
        "session_id": session_id,
        "route": "",
        "route_reasoning": "",
        "documents": [],
        "web_results": [],
        "generation": "",
        "retry_count": 0,
        "doc_relevance_score": 0.0,
        "is_grounded": False,
        "answers_query": False,
        "input_safe": True,
        "guardrail_message": "",
        "sources_used": [],
        # Inject dependencies
        "_vector_store": vector_store,
        "_all_chunks": all_chunks or [],
    }

    print("\n" + "=" * 60)
    print(f"🚀 AGENTIC RAG QUERY: {query}")
    print("=" * 60)

    final_state = app.invoke(initial_state)

    print(f"\n{'=' * 60}")
    print(f"✅ COMPLETE | Route: {final_state.get('route', 'N/A')}")
    print(f"   Sources: {final_state.get('sources_used', [])}")
    print(f"   Retries: {final_state.get('retry_count', 0)}")
    print(f"   Grounded: {final_state.get('is_grounded', 'N/A')}")
    print("=" * 60)

    return final_state
