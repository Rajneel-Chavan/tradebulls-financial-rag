"""
RAGAS Evaluation Pipeline for Tradebulls Financial RAG.

Metrics:
  - Faithfulness: Is the answer grounded in the retrieved context?
  - Answer Relevancy: Does the answer address the question?
  - Context Precision: Are the retrieved documents relevant?
  - Context Recall: Did retrieval capture all needed information?

Usage:
  python -m evaluation.evaluate_ragas
"""

import json
import os
import sys
from datetime import datetime

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.rag_graph import run_query
from backend.data_loader import load_all_sources
from backend.chunker import chunk_documents
from backend.vector_store import create_vector_store, create_session_id


def load_golden_qa(path: str = "evaluation/golden_qa.json") -> list[dict]:
    """Load golden QA pairs for evaluation."""
    with open(path, "r") as f:
        return json.load(f)


def run_evaluation():
    """Run full RAGAS evaluation pipeline."""
    print("\n" + "=" * 60)
    print("RAGAS EVALUATION PIPELINE")
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

    # Filter out guardrail test cases
    eval_qa = [
        qa for qa in golden_qa
        if qa.get("category") != "guardrail_test"
        and qa.get("category") != "direct_knowledge"
    ]

    print(f"  Evaluating on {len(eval_qa)} QA pairs")

    # Run queries and collect results
    print("\n[3/4] Running queries through pipeline...")
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for i, qa in enumerate(eval_qa):
        print(f"\n  Query {i + 1}/{len(eval_qa)}: {qa['question'][:60]}...")

        result = run_query(
            query=qa["question"],
            session_id=session_id,
            vector_store=vector_store,
            all_chunks=chunks,
        )

        questions.append(qa["question"])
        answers.append(result.get("generation", ""))

        # Extract context from retrieved documents
        docs = result.get("documents", [])
        context_list = [doc.page_content for doc in docs] if docs else [""]
        contexts.append(context_list)

        ground_truths.append(qa["ground_truth"])

    # Build RAGAS dataset
    print("\n[4/4] Computing RAGAS metrics...")
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    # Run evaluation
    results = evaluate(
        dataset=eval_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    # Print results
    print("\n" + "=" * 60)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Faithfulness:      {results['faithfulness']:.4f}")
    print(f"  Answer Relevancy:  {results['answer_relevancy']:.4f}")
    print(f"  Context Precision: {results['context_precision']:.4f}")
    print(f"  Context Recall:    {results['context_recall']:.4f}")
    print("=" * 60)

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "num_questions": len(eval_qa),
        "metrics": {
            "faithfulness": float(results["faithfulness"]),
            "answer_relevancy": float(results["answer_relevancy"]),
            "context_precision": float(results["context_precision"]),
            "context_recall": float(results["context_recall"]),
        },
        "per_question": [
            {
                "question": questions[i],
                "answer": answers[i][:200],
                "ground_truth": ground_truths[i][:200],
            }
            for i in range(len(questions))
        ],
    }

    output_path = "evaluation/ragas_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")
    return results


if __name__ == "__main__":
    run_evaluation()
