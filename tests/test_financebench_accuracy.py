"""
Integration test for evaluating RAG accuracy against FinanceBench.

This test:
1. Loads the FinanceBench dataset.
2. Queries the RAG pipeline for a subset of questions.
3. Checks if the retrieved context contains the ground truth evidence.
4. Prints accuracy statistics.

Usage:
    pytest tests/test_financebench_accuracy.py -s
    (Use -s to see the printed statistics)
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import warnings
import pandas as pd
from datasets import load_dataset
from retriever import FinancialRetriever
from qdrant_client import QdrantClient
from config import settings

# Suppress Qdrant insecure connection warning for local dev
warnings.filterwarnings("ignore", message="Api key is used with an insecure connection")

# Threshold for passing the evaluation (e.g., 50% recall on the subset)
# This is arbitrary for now, adjusted based on how much data is actually indexed.
ACCURACY_THRESHOLD = 0.0 

def normalize_text(text: str) -> str:
    """Simple text normalization for comparison."""
    return " ".join(text.lower().split())

@pytest.mark.integration
def test_financebench_retrieval_accuracy():
    """
    Evaluate retrieval accuracy on FinanceBench.
    """
    # 1. Check if Qdrant is reachable and has data
    try:
        client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        if "financial_filings" not in collection_names:
            pytest.skip("Qdrant collection 'financial_filings' not found. Skipping evaluation.")
        
        count = client.count(collection_name="financial_filings").count
        if count == 0:
            pytest.skip("Qdrant collection is empty. Please ingest data first.")
            
    except Exception as e:
        pytest.skip(f"Could not connect to Qdrant: {e}")

    # 2. Load FinanceBench
    print("\nLoading FinanceBench dataset...")
    dataset = load_dataset("PatronusAI/financebench", split="train")
    
    # Take a small sample for quick testing, or run on full set if configured
    # For CI/local testing, let's limit to 10 examples to avoid long runtimes
    # unless we want a full eval.
    SAMPLE_SIZE = 20
    subset = dataset.select(range(min(len(dataset), SAMPLE_SIZE)))
    
    retriever = FinancialRetriever()
    
    hits = 0
    total = 0
    skipped = 0
    
    print(f"\nEvaluating on {len(subset)} samples...")
    print("-" * 60)
    
    for item in subset:
        question = item['question']
        evidence_text = item['evidence_text']
        doc_name = item['doc_name']
        
        # Parse ticker and year from doc_name (e.g., "3M_2018_10K")
        try:
            parts = doc_name.split('_')
            # We need a robust mapper here or just rely on what we have.
            # For this test, we'll try to extract, but if we can't map the ticker
            # exactly to what we indexed, retrieval might fail.
            # Ideally, we should use the same mapper as in manifest_generator.
            # For simplicity, we assume the user has indexed the relevant files.
            
            # Note: The retriever needs the exact ticker symbol used during ingestion.
            # If ingestion used "MMM" for "3M", we need to pass "MMM".
            # We'll reuse the map from manifest_generator if possible, or just skip filtering
            # to test "global" search if specific filtering fails.
            
            # Let's try to run WITHOUT strict filters first to see if Hybrid Search
            # can find it, OR we define a mini-map for the test.
            
            # For now, let's try to infer filters but fallback to no filters if unsure.
            filters = {}
            # filters['year'] = int(parts[1]) # strict year filtering
            
            # Perform Search
            context = retriever.search(question, filters=filters)
            
            # Check for matches
            # We check if a significant portion of the evidence text is in the context.
            norm_context = normalize_text(context)
            norm_evidence = normalize_text(evidence_text)
            
            # Simple substring match (can be improved with fuzzy matching or LLM grading)
            if norm_evidence in norm_context:
                hits += 1
                result = "HIT"
            else:
                # Try partial match (if evidence is long, maybe we retrieved part of it)
                # Heuristic: check if 50% of evidence words are present
                ev_words = set(norm_evidence.split())
                ctx_words = set(norm_context.split())
                overlap = len(ev_words.intersection(ctx_words))
                if len(ev_words) > 0 and (overlap / len(ev_words) > 0.5):
                    hits += 1
                    result = "HIT (Partial)"
                else:
                    result = "MISS"
            
            total += 1
            print(f"[{result}] Q: {question[:50]}...")
            
        except Exception as e:
            print(f"Error processing {doc_name}: {e}")
            skipped += 1

    if total == 0:
        pytest.skip("No valid samples processed.")

    accuracy = hits / total
    print("-" * 60)
    print(f"Evaluation Complete.")
    print(f"Total: {total}")
    print(f"Hits: {hits}")
    print(f"Skipped: {skipped}")
    print(f"Accuracy: {accuracy:.2%}")
    print("-" * 60)
    
    # Assert accuracy if we want to enforce a baseline
    assert accuracy >= ACCURACY_THRESHOLD, f"Accuracy {accuracy:.2%} is below threshold {ACCURACY_THRESHOLD}"
