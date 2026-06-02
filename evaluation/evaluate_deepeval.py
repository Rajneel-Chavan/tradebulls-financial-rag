"""
DeepEval Evaluation Pipeline for Tradebulls Financial RAG.

Metrics:
  - AnswerRelevancyMetric: Does the answer address the question?
  - FaithfulnessMetric: Is the answer grounded in context?
  - ContextualRelevancyMetric: Are retrieved contexts relevant?
  - HallucinationMetric: Does the answer hallucinate?

Usage:
  python -m evaluation.evaluate_deepeval
"""

import json
import os
import sys
from datetime import datetime

from deepeval import evaluate as deep_evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.rag_graph import run_query
from backend.data_loader import load_all_sources
from backend.chunker import chunk_documents
from backend.vector_store import create_vector_store, create_session_id


def load_golden_qa(path: str = "evaluation/golden_qa.json") -> list[dict]:
    """Load golden QA pairs."""
    with open(path, "r") as f:
        return json.load(f)


def run_evaluation():
    """Run full DeepEval evaluation pipeline."""
    print("\n" + "=" * 60)
    print("DEEPEVAL EVALUATION PIPELINE")
    print("=" * 60)

    # Load and index data
    print("\n[1/4] Loading and indexing data...")
    documents = load_all_sources()
    chunks = chunk_documents(documents)
    session_id = create_session_id()
    vector_store = create_vector_store(chunks, session_id)

    # Load golden QA
    print("\n[2/4] Loading golden QA pairs...")
    golden_qa = load_golden_qa()

    eval_qa = [
        qa for qa in golden_qa
        if qa.get("category") != "guardrail_test"
        and qa.get("category") != "direct_knowledge"
    ]

    print(f"  Evaluating on {len(eval_qa)} QA pairs")

    # Define metrics
    metrics = [
        AnswerRelevancyMetric(threshold=0.7),
        FaithfulnessMetric(threshold=0.7),
        ContextualRelevancyMetric(threshold=0.7),
        HallucinationMetric(threshold=0.5),
    ]

    # Run queries and build test cases
    print("\n[3/4] Running queries and building test cases...")
    test_cases = []

    for i, qa in enumerate(eval_qa):
        print(f"\n  Query {i + 1}/{len(eval_qa)}: {qa['question'][:60]}...")

        result = run_query(
            query=qa["question"],
            session_id=session_id,
            vector_store=vector_store,
            all_chunks=chunks,
        )

        docs = result.get("documents", [])
        retrieval_context = [doc.page_content for doc in docs] if docs else [""]

        test_case = LLMTestCase(
            input=qa["question"],
            actual_output=result.get("generation", ""),
            expected_output=qa["ground_truth"],
            retrieval_context=retrieval_context,
        )
        test_cases.append(test_case)

    # Run evaluation
    print("\n[4/4] Running DeepEval metrics...")
    eval_results = deep_evaluate(test_cases, metrics)

    # Collect scores
    metric_scores = {}
    for metric in metrics:
        scores = []
        for tc in test_cases:
            try:
                metric.measure(tc)
                scores.append(metric.score)
            except Exception:
                scores.append(0.0)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        metric_name = metric.__class__.__name__
        metric_scores[metric_name] = round(avg_score, 4)

    # Print results
    print("\n" + "=" * 60)
    print("DEEPEVAL EVALUATION RESULTS")
    print("=" * 60)
    for name, score in metric_scores.items():
        print(f"  {name}: {score:.4f}")
    print("=" * 60)

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "num_questions": len(eval_qa),
        "metrics": metric_scores,
        "per_question": [
            {
                "question": eval_qa[i]["question"],
                "category": eval_qa[i].get("category", ""),
                "answer_preview": test_cases[i].actual_output[:200]
                if test_cases[i].actual_output
                else "",
            }
            for i in range(len(eval_qa))
        ],
    }

    output_path = "evaluation/deepeval_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")
    return metric_scores


if __name__ == "__main__":
    run_evaluation()
