"""
Retriever module for the Forensic RAG system.

This module implements the retrieval logic, ensuring that queries are strictly
scoped using metadata filters and leveraging hybrid search (dense + sparse vectors)
for high precision.

Security & Best Practices:
- Metadata Filtering: We strictly enforce filters (e.g., year, ticker) to prevent
  data leakage across different entities or time periods.
- Least Privilege: The retriever should ideally use a read-only API key if supported
  by the vector DB, though here we use the main key from settings.
"""

import warnings
from typing import Dict, List, Any
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from qdrant_client import QdrantClient

# Suppress Qdrant insecure connection warning for local dev
warnings.filterwarnings("ignore", message="Api key is used with an insecure connection")

from config import settings


class FinancialRetriever:
    """
    Query engine for financial documents using Hybrid Search.
    """

    def __init__(self, collection_name: str = "financial_filings"):
        """
        Initialize the retriever.

        Args:
            collection_name: Name of the Qdrant collection to query.
        """
        self.collection_name = collection_name
        self.qdrant_client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=collection_name,
            enable_hybrid=True, # Critical for Hybrid Search
        )
        # We don't need to rebuild the index here, just load the storage context
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store
        )

    def search(self, query: str, filters: Dict[str, Any] = None) -> str:
        """
        Search for relevant context using Hybrid Search and Metadata Filtering.

        Args:
            query: The user's natural language query.
            filters: A dictionary of strict filters (e.g., {"ticker": "AAPL", "year": 2023}).

        Returns:
            str: A consolidated string of retrieved context nodes.
        """
        # Construct MetadataFilters
        # Security: strictly isolate search scope.
        metadata_filters = None
        if filters:
            filter_list = [
                ExactMatchFilter(key=k, value=v) for k, v in filters.items()
            ]
            metadata_filters = MetadataFilters(filters=filter_list)

        # Configure the retriever
        # Alpha=0.5 balances dense (semantic) and sparse (keyword) search.
        retriever = self.index.as_retriever(
            vector_store_kwargs={"hybrid_top_k": 5},
            alpha=0.5,
            filters=metadata_filters,
            similarity_top_k=5
        )

        # Execute retrieval
        nodes = retriever.retrieve(query)

        # Format results
        # In a full RAG pipeline, these nodes would be passed to the LLM.
        # Here we return the text content for inspection/usage.
        context_str = "\n\n".join([node.get_content() for node in nodes])
        
        return context_str
