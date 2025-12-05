"""
Unit tests for the Forensic RAG Pipeline.

These tests verify the logic of the ingestion and retrieval components without
making real API calls. We mock LlamaParse and QdrantClient to ensure isolation.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from unittest.mock import MagicMock, patch
from llama_index.core.schema import Document, TextNode
from ingest import FinancialIngestionPipeline
from retriever import FinancialRetriever
from config import settings

# Sample 10-K Markdown with a table
SAMPLE_MARKDOWN = """
# Item 6. Selected Financial Data

The following table sets forth selected financial data for the last five years.

| Year | Revenue | Net Income |
|------|---------|------------|
| 2023 | $100B   | $20B       |
| 2022 | $90B    | $18B       |

End of table.
"""

@pytest.fixture
def mock_settings():
    """Mock settings to avoid environment variable validation errors during tests."""
    with patch("config.settings") as mock_settings:
        mock_settings.LLAMA_CLOUD_API_KEY = "mock_llama_key"
        mock_settings.QDRANT_API_KEY = "mock_qdrant_key"
        mock_settings.QDRANT_URL = "http://mock-qdrant"
        mock_settings.OPENAI_API_KEY = "mock_openai_key"
        yield mock_settings

@pytest.fixture
def mock_qdrant_client():
    """Mock QdrantClient."""
    with patch("ingest.QdrantClient") as mock_client_cls:
        mock_instance = MagicMock()
        mock_client_cls.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_llama_parse():
    """Mock LlamaParse."""
    with patch("ingest.LlamaParse") as mock_parse_cls:
        mock_instance = MagicMock()
        mock_parse_cls.return_value = mock_instance
        yield mock_instance

def test_ingest_filing(mock_settings, mock_qdrant_client, mock_llama_parse):
    """
    Test that ingest_filing correctly parses markdown and identifies tables.
    """
    # Setup mock return value for load_data
    mock_doc = Document(text=SAMPLE_MARKDOWN)
    mock_llama_parse.load_data.return_value = [mock_doc]

    pipeline = FinancialIngestionPipeline()
    
    # Run ingestion
    nodes = pipeline.ingest_filing("dummy.pdf", "AAPL", 2023)

    # Verification
    assert len(nodes) > 0
    
    # Check if metadata was attached
    assert nodes[0].metadata["ticker"] == "AAPL"
    assert nodes[0].metadata["year"] == 2023

    # Check if MarkdownElementNodeParser split the text
    # We expect at least one node containing the table or text
    # Since we used MarkdownElementNodeParser, it should handle the table.
    # Note: Without an LLM, the parser might just treat it as text or simple split,
    # but we verify that we got nodes back and the pipeline ran.
    # In a real scenario with LLM, we'd check for specific table nodes.
    content = nodes[0].get_content()
    assert "Selected Financial Data" in content or "Year" in content

def test_index_documents(mock_settings, mock_qdrant_client):
    """
    Test that index_documents calls Qdrant with correct parameters.
    """
    pipeline = FinancialIngestionPipeline()
    nodes = [TextNode(text="test node", metadata={"ticker": "AAPL"})]

    # Mock VectorStoreIndex to avoid real indexing logic which requires embeddings
    with patch("ingest.VectorStoreIndex") as mock_index_cls:
        with patch("ingest.QdrantVectorStore") as mock_store_cls:
            pipeline.index_documents(nodes)
            
            # Verify QdrantVectorStore was initialized with hybrid mode
            mock_store_cls.assert_called_once()
            _, kwargs = mock_store_cls.call_args
            assert kwargs["enable_hybrid"] is True
            assert kwargs["collection_name"] == "financial_filings"

def test_retriever_search(mock_settings):
    """
    Test that the retriever applies metadata filters correctly.
    """
    with patch("retriever.QdrantClient") as mock_qdrant_cls:
        with patch("retriever.QdrantVectorStore") as mock_store_cls:
            with patch("retriever.VectorStoreIndex") as mock_index_cls:
                # Setup mock retriever
                mock_index_instance = MagicMock()
                mock_index_cls.from_vector_store.return_value = mock_index_instance
                mock_retriever = MagicMock()
                mock_index_instance.as_retriever.return_value = mock_retriever
                
                # Mock retrieve result
                mock_node = TextNode(text="Retrieved content")
                mock_retriever.retrieve.return_value = [mock_node]

                retriever = FinancialRetriever()
                result = retriever.search("query", filters={"year": 2023})

                # Verify result
                assert "Retrieved content" in result

                # Verify filters were passed
                mock_index_instance.as_retriever.assert_called_once()
                _, kwargs = mock_index_instance.as_retriever.call_args
                
                # Check alpha
                assert kwargs["alpha"] == 0.5
                
                # Check filters
                filters = kwargs["filters"]
                assert filters is not None
                # Depending on implementation, filters might be a MetadataFilters object
                # We can check if it contains our filter
                assert len(filters.filters) == 1
                assert filters.filters[0].key == "year"
                assert filters.filters[0].value == 2023
