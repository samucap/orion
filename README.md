# Forensic RAG Pipeline

A "Forensic" Retrieval-Augmented Generation (RAG) pipeline for financial analysis. It ingests SEC 10-K filings, preserving table structures, and enables high-precision retrieval using Hybrid Search.

## Prerequisites

- **Python 3.10 - 3.12** (Note: Python 3.14 is currently incompatible with some dependencies)
- **Docker & Docker Compose** (for local Qdrant)
- **LlamaCloud API Key** (for parsing)
- **OpenAI API Key** (for embeddings)

## Setup

1.  **Clone the repository** (if applicable) or navigate to the project directory.

2.  **Start Local Qdrant**:
    This project uses a local Qdrant instance secured with an API key.
    ```bash
    docker-compose up -d
    ```

3.  **Create a Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

4.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Configure Environment Variables**:
    Create a `.env` file in the root directory. You can use the default key from `docker-compose.yml` for local development.
    ```env
    LLAMA_CLOUD_API_KEY=your_llama_cloud_key
    # Default key set in docker-compose.yml is 'default_secure_key'
    QDRANT_API_KEY=default_secure_key
    QDRANT_URL=http://localhost:6333
    OPENAI_API_KEY=your_openai_key
    ```

## Usage

The project includes a CLI `main.py` for easy interaction.

### 1. Ingest a Filing
Extract data from a PDF (e.g., a 10-K filing) and index it into Qdrant.

```bash
python main.py ingest --file path/to/filing.pdf --ticker AAPL --year 2023
```

### 2. Search
Query the indexed data with strict metadata filtering.

```bash
python main.py search --query "What was the net income in 2023?" --ticker AAPL --year 2023
```

## End-to-End Workflow

1.  **Start Infrastructure**:
    ```bash
    docker-compose up -d
    ```

2.  **Download Data**:
    Fetch 10-K filings referenced in FinanceBench.
    ```bash
    python manifest_generator.py --email your.email@example.com --output_dir data/
    ```

3.  **Ingest Data**:
    Process the downloaded PDFs and index them into Qdrant.
    ```bash
    # Example for a single file
    python main.py ingest --file data/3M_2018_10K.pdf --ticker MMM --year 2018

    # OR Batch Ingest all downloaded filings
    python batch_ingest.py --data_dir data/
    ```

4.  **Evaluate Accuracy**:
    Run the integration test to measure performance against FinanceBench.
    ```bash
    pytest tests/test_financebench_accuracy.py -s
    ```

## Running Tests

To run the unit tests:

```bash
pytest tests/test_pipeline.py
```

## Project Structure

- `config.py`: Configuration and environment validation.
- `ingest.py`: Pipeline for parsing (LlamaParse) and indexing (Qdrant).
- `retriever.py`: Search engine with metadata filtering and hybrid search.
- `main.py`: CLI entry point.
- `tests/`: Unit tests.
