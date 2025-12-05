"""
CLI Entry point for the Forensic RAG Pipeline.

Usage:
    python main.py ingest --file path/to/10k.pdf --ticker AAPL --year 2023
    python main.py search --query "What is the revenue?" --ticker AAPL --year 2023
"""

import argparse
import sys
from ingest import FinancialIngestionPipeline
from retriever import FinancialRetriever

def handle_ingest(args):
    """Handle the ingest command."""
    print(f"Initializing ingestion pipeline for {args.ticker} {args.year}...")
    try:
        pipeline = FinancialIngestionPipeline()
        nodes = pipeline.ingest_filing(args.file, args.ticker, args.year)
        pipeline.index_documents(nodes)
        print("Ingestion and indexing completed successfully.")
    except Exception as e:
        print(f"Error during ingestion: {e}", file=sys.stderr)
        sys.exit(1)

def handle_search(args):
    """Handle the search command."""
    print(f"Searching for: '{args.query}'...")
    filters = {}
    if args.ticker:
        filters["ticker"] = args.ticker
    if args.year:
        filters["year"] = args.year
    
    try:
        retriever = FinancialRetriever()
        result = retriever.search(args.query, filters=filters)
        print("\n=== Search Results ===\n")
        print(result)
        print("\n======================\n")
    except Exception as e:
        print(f"Error during search: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Forensic RAG Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest Subcommand
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a 10-K filing")
    ingest_parser.add_argument("--file", required=True, help="Path to the PDF/document file")
    ingest_parser.add_argument("--ticker", required=True, help="Stock ticker symbol (e.g., AAPL)")
    ingest_parser.add_argument("--year", required=True, type=int, help="Filing year (e.g., 2023)")
    ingest_parser.set_defaults(func=handle_ingest)

    # Search Subcommand
    search_parser = subparsers.add_parser("search", help="Search the knowledge base")
    search_parser.add_argument("--query", required=True, help="Natural language query")
    search_parser.add_argument("--ticker", help="Filter by ticker")
    search_parser.add_argument("--year", type=int, help="Filter by year")
    search_parser.set_defaults(func=handle_search)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)

if __name__ == "__main__":
    main()
