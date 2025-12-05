"""
Ingestion pipeline for the Forensic RAG system.

This module handles the extraction of data from SEC 10-K filings using LlamaParse
and indexing into Qdrant with Hybrid Search enabled.

Security & Best Practices:
- Data Sanitization: Ensure that the text extracted from filings does not contain
  inadvertent PII before indexing, although 10-Ks are public.
- Rate Limiting: The pipeline should handle 429 errors from LlamaCloud gracefully.
  (Note: LlamaIndex often handles some retries, but explicit handling is better).
"""

import os
import warnings
from typing import List, Dict, Any
from llama_parse import LlamaParse
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.core.schema import BaseNode, TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from qdrant_client import QdrantClient

# Suppress Qdrant insecure connection warning for local dev
warnings.filterwarnings("ignore", message="Api key is used with an insecure connection")

from config import settings, PARSING_INSTRUCTION


class FinancialIngestionPipeline:
    """
    Pipeline to ingest financial documents, parse them preserving tables,
    and index them into Qdrant.
    """

    def __init__(self):
        """
        Initialize the pipeline components.
        """
        # Initialize Qdrant Client
        # Security: We use the API key and URL from validated settings.
        self.qdrant_client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )

    def ingest_filing(self, file_path: str, ticker: str, year: int) -> List[BaseNode]:
        """
        Parse a 10-K filing and return structured nodes.

        Args:
            file_path: Path to the PDF/document file.
            ticker: Stock ticker symbol (e.g., "AAPL").
            year: Filing year (e.g., 2023).

        Returns:
            List[BaseNode]: A list of parsed nodes (text and table nodes).
        """
        # Security: Validate metadata before processing to prevent injection or malformed data.
        if not ticker.isalnum():
            raise ValueError("Invalid ticker symbol. Must be alphanumeric.")
        if not (1900 <= year <= 2100):
            raise ValueError("Invalid year provided.")

        print(f"Starting ingestion for {ticker} {year}...")

        # Initialize LlamaParse
        # Rate Limiting: LlamaParse client handles some retries, but for production
        # we might wrap this in a tenacity retry block.
        parser = LlamaParse(
            api_key=settings.LLAMA_CLOUD_API_KEY,
            result_type="markdown",
            system_prompt=PARSING_INSTRUCTION,
        )

        # Load and parse the document
        # This returns a list of Document objects (usually one per file)
        documents = parser.load_data(file_path)

        # Add metadata to documents
        # Security: Explicitly setting metadata ensures we can filter strictly later.
        for doc in documents:
            doc.metadata["ticker"] = ticker.upper()
            doc.metadata["year"] = year

        # Use MarkdownElementNodeParser to split text while keeping tables intact.
        # This is crucial for financial data where tables contain the "forensic" details.
        node_parser = MarkdownElementNodeParser(
            llm=None, # We can pass an LLM here for table summarization if needed
            num_workers=8
        )
        
        # Get nodes from documents
        nodes = node_parser.get_nodes_from_documents(documents)
        
        # Ensure metadata is propagated to all nodes
        for node in nodes:
            node.metadata["ticker"] = ticker.upper()
            node.metadata["year"] = year

        return nodes

    def index_documents(self, nodes: List[BaseNode], collection_name: str = "financial_filings"):
        """
        Index the processed nodes into Qdrant.

        Args:
            nodes: List of nodes to index.
            collection_name: Name of the Qdrant collection.
        """
        print(f"Indexing {len(nodes)} nodes into Qdrant collection '{collection_name}'...")

        # Critical: Enable hybrid_mode=True (Dense + Sparse vectors)
        # This requires the vector store to be configured for hybrid search.
        vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=collection_name,
            enable_hybrid=True, # Enables sparse vectors for keyword search
            batch_size=20, # Adjust based on rate limits/performance
        )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create the index (upsert documents)
        # Note: In a real production system, we might check if the collection exists
        # and create it with specific vector params if not handled by LlamaIndex.
        VectorStoreIndex(
            nodes,
            storage_context=storage_context,
        )
        
        print("Indexing complete.")
