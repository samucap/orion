"""
Configuration module for the Financial RAG Pipeline.

This module handles environment variable validation and defines global constants.
It uses pydantic_settings to ensure all required API keys and URLs are present
and correctly formatted before the application starts.

Security & Best Practices:
- Fail Fast: The application will crash immediately if critical secrets are missing,
  preventing runtime errors later.
- Least Privilege: When generating API keys (e.g., Qdrant), ensure they have
  only the necessary permissions (read/write for specific collections) rather than
  admin access.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Explicitly load .env file into os.environ so that libraries like LlamaIndex
# (which rely on os.environ for keys like OPENAI_API_KEY) can find them.
load_dotenv()

class Settings(BaseSettings):
    """
    Application settings and environment variable validation.
    """

    # LlamaCloud is used for parsing complex financial documents (tables).
    # Rate Limiting: Handle 429 errors from LlamaCloud by implementing exponential backoff
    # in the calling code (ingest.py).
    LLAMA_CLOUD_API_KEY: str = Field(..., description="API Key for LlamaCloud/LlamaParse")

    # Qdrant is the vector database for hybrid search.
    # Security: Ensure QDRANT_API_KEY has restricted scopes if possible.
    QDRANT_API_KEY: str = Field(..., description="API Key for Qdrant Cloud/Instance")
    QDRANT_URL: str = Field(..., description="URL for Qdrant instance")

    # OpenAI is used for embeddings and LLM generation.
    OPENAI_API_KEY: str = Field(..., description="API Key for OpenAI")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


# Global parsing instruction for LlamaParse to ensure high-quality table extraction.
# This prompt is critical for the "Forensic" aspect of the pipeline.
PARSING_INSTRUCTION = (
    "Extract financial tables as Markdown, preserving headers and row-column structure."
)

# Instantiate settings to trigger validation immediately upon import.
try:
    settings = Settings()
except Exception as e:
    # Data Sanitization: Ensure we don't log the actual values of the missing keys if possible,
    # though Pydantic's default error message is usually safe (shows field name, not value).
    raise RuntimeError(f"Configuration validation failed: {e}") from e
