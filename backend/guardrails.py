"""
Financial domain guardrails for Tradebulls RAG System.

Input Guardrails:
  - Block personalized investment advice requests
  - Block requests for guaranteed returns predictions
  - Block requests for insider trading information

Output Guardrails:
  - Add financial disclaimers to market-related responses
  - Block outputs making guaranteed profit claims
  - Ensure compliance with SEBI advisory guidelines
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.config import LLM_MODEL
from backend.models import GuardrailResult


# ── Input Guardrail ───────────────────────────────────────────────────

INPUT_GUARDRAIL_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a compliance checker for a financial services platform.
Evaluate if the user query is safe to process through a financial RAG system.

BLOCK the query if it:
1. Asks for personalized investment advice ("Should I invest my savings in X?")
2. Asks for guaranteed returns or profit predictions ("Will X stock give 100% returns?")
3. Asks for insider trading information or market manipulation strategies
4. Asks for help with tax evasion or illegal financial activities

ALLOW the query if it:
1. Asks factual questions about markets, stocks, or financial concepts
2. Asks about information from financial reports or research documents
3. Asks about trading platform features or processes
4. Asks about general market trends or analysis from reports

Respond with ONLY one of:
SAFE: <reason>
BLOCKED: <reason>""",
    ),
    ("human", "{query}"),
])


def check_input_guardrail(query: str) -> GuardrailResult:
    """Check if user query passes input guardrail."""
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    chain = INPUT_GUARDRAIL_PROMPT | llm | StrOutputParser()

    try:
        result = chain.invoke({"query": query})
        result = result.strip()

        if result.startswith("BLOCKED"):
            reason = result.replace("BLOCKED:", "").strip()
            return GuardrailResult(
                is_safe=False,
                reason=reason,
                modified_content=(
                    "I can't provide personalized investment advice or guaranteed "
                    "return predictions. I can help you with factual information from "
                    "Tradebulls research reports, market analysis, and trading platform "
                    "features. Please rephrase your question."
                ),
            )

        return GuardrailResult(is_safe=True, reason="Query is safe to process.")

    except Exception as e:
        # Fail open — allow query if guardrail check itself fails
        return GuardrailResult(
            is_safe=True,
            reason=f"Guardrail check failed, allowing query: {e}",
        )


# ── Output Guardrail ──────────────────────────────────────────────────

FINANCIAL_DISCLAIMER = (
    "\n\n---\n"
    "*Disclaimer: This information is sourced from Tradebulls research reports "
    "and is for educational purposes only. It does not constitute investment "
    "advice. Please consult a SEBI-registered investment advisor before making "
    "any investment decisions. Past performance is not indicative of future results.*"
)

OUTPUT_GUARDRAIL_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a compliance checker for financial content.
Evaluate if this AI-generated response contains:
1. Guaranteed profit/return claims (e.g., "you WILL make money")
2. Direct buy/sell commands presented as certainties (not analysis)
3. Claims about insider information
4. Misleading financial statistics

Respond with ONLY:
SAFE — if the content is compliant
MODIFY — if the content needs a disclaimer added
BLOCK — if the content makes dangerous financial claims""",
    ),
    ("human", "Response to check:\n{response}"),
])


def check_output_guardrail(response: str, query: str) -> GuardrailResult:
    """Check if generated response passes output guardrail."""
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    chain = OUTPUT_GUARDRAIL_PROMPT | llm | StrOutputParser()

    try:
        result = chain.invoke({"response": response}).strip()

        if result.startswith("BLOCK"):
            return GuardrailResult(
                is_safe=False,
                reason="Response contained unsafe financial claims.",
                modified_content=(
                    "I found relevant information but my response contained "
                    "claims that don't meet financial compliance standards. "
                    "Please consult the original Tradebulls research reports "
                    "directly for detailed analysis."
                ),
            )

        if result.startswith("MODIFY") or _is_market_related(query):
            return GuardrailResult(
                is_safe=True,
                reason="Disclaimer added.",
                modified_content=response + FINANCIAL_DISCLAIMER,
            )

        return GuardrailResult(
            is_safe=True,
            reason="Response is compliant.",
            modified_content=response,
        )

    except Exception as e:
        # On failure, add disclaimer as precaution
        return GuardrailResult(
            is_safe=True,
            reason=f"Output check failed, adding disclaimer: {e}",
            modified_content=response + FINANCIAL_DISCLAIMER,
        )


def _is_market_related(query: str) -> bool:
    """Quick keyword check if query is market/stock related."""
    market_keywords = [
        "stock", "nifty", "sensex", "market", "trade", "buy", "sell",
        "bullish", "bearish", "target", "support", "resistance",
        "report", "analysis", "recommendation", "price", "sector",
        "portfolio", "investment", "mutual fund", "derivative", "option",
        "future", "intraday", "swing", "breakdown", "breakout",
    ]
    query_lower = query.lower()
    return any(kw in query_lower for kw in market_keywords)
