"""
Batch Ingestion Script.

This script iterates through the directory structure created by sec-edgar-downloader
(or a flat directory) and ingests all found 10-K filings into the RAG pipeline.

It automatically infers:
- Ticker: From the directory name.
- Year: From the Accession Number (e.g., 0000320193-23-000077 -> 2023).

Usage:
    python batch_ingest.py --data_dir data/
"""

import argparse
import os
import glob
import re
import warnings
from tqdm import tqdm
from ingest import FinancialIngestionPipeline

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="urllib3")

def get_year_from_accession(accession_number: str) -> int:
    """
    Extract year from SEC Accession Number (format: CIK-YY-SEQUENCE).
    Example: 0000320193-23-000077 -> 2023
    """
    match = re.search(r'\d{10}-(\d{2})-\d{6}', accession_number)
    if match:
        yy = int(match.group(1))
        # Heuristic: 50-99 is 1950-1999, 00-49 is 2000-2049
        return 1900 + yy if yy > 50 else 2000 + yy
    return None

def batch_ingest(data_dir: str):
    """
    Walk the data directory and ingest filings.
    """
    pipeline = FinancialIngestionPipeline()
    
    # Pattern for sec-edgar-downloader: data/sec-edgar-filings/TICKER/10-K/ACCESSION/primary-document.html
    # We also support .txt or .pdf if they exist.
    search_pattern = os.path.join(data_dir, "**", "*")
    
    print(f"Scanning {data_dir} for filings...")
    
    # First pass: Collect all valid files
    valid_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            # Support both primary-document and full-submission
            is_valid_file = (
                file.lower().startswith("primary-document") or 
                file.lower().startswith("full-submission")
            ) and file.lower().endswith((".html", ".xml", ".txt", ".pdf"))
            
            if is_valid_file:
                file_path = os.path.join(root, file)
                valid_files.append(file_path)

    print(f"Found {len(valid_files)} potential filings. Starting ingestion...")

    count = 0
    # Use tqdm for progress bar
    for file_path in tqdm(valid_files, desc="Ingesting Filings", unit="file"):
        try:
            # Try to infer metadata from path
            # Expected path: .../TICKER/10-K/ACCESSION/primary-document.html
            parts = file_path.split(os.sep)
            
            # Go backwards from file
            # parts[-1] = file
            # parts[-2] = accession (0000320193-23-000077)
            # parts[-3] = doc_type (10-K)
            # parts[-4] = ticker (AAPL)
            
            try:
                accession = parts[-2]
                doc_type = parts[-3]
                ticker = parts[-4]
            except IndexError:
                tqdm.write(f"Skipping {file_path}: Unexpected directory structure.")
                continue
            
            if doc_type != "10-K":
                continue
                
            year = get_year_from_accession(accession)
            if not year:
                tqdm.write(f"Skipping {file_path}: Could not infer year from {accession}")
                continue
                
            # tqdm.write(f"Processing {ticker} {year}...")
            
            try:
                nodes = pipeline.ingest_filing(file_path, ticker, year)
                pipeline.index_documents(nodes)
                count += 1
            except Exception as e:
                tqdm.write(f"Failed to ingest {file_path}: {e}")
                
        except Exception as e:
            tqdm.write(f"Error processing {file_path}: {e}")

    print(f"Batch ingestion complete. Successfully processed {count} filings.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Ingestion for SEC Filings")
    parser.add_argument("--data_dir", default="data", help="Root directory containing filings")
    
    args = parser.parse_args()
    
    batch_ingest(args.data_dir)
