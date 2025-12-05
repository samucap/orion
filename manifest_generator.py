"""
Manifest Generator for FinanceBench.

This script loads the FinanceBench dataset from HuggingFace, determines the required
10-K filings, and downloads them using the sec-edgar-downloader.

Usage:
    python manifest_generator.py --email your.email@example.com --output_dir data/
"""

import argparse
import os
from datasets import load_dataset
from sec_edgar_downloader import Downloader

def get_financebench_requirements():
    """
    Extracts unique document requirements (Ticker, Year) from the FinanceBench dataset.
    """
    print("Loading FinanceBench dataset from HuggingFace...")
    # 1. Load the open-source subset of FinanceBench
    dataset = load_dataset("PatronusAI/financebench", split="train")
    
    # 2. Extract unique document identifiers
    # The 'doc_name' field usually looks like "3M_2018_10K"
    required_docs = set(dataset['doc_name'])
    
    download_queue = []
    
    # 3. Map Company Names to Tickers
    ticker_map = {
        "3M": "MMM",
        "ADOBE": "ADBE",
        "AES": "AES",
        "AMAZON": "AMZN",
        "AMD": "AMD",
        "AMCOR": "AMCR",
        "AMERICANEXPRESS": "AXP",
        "AMGEN": "AMGN",
        "APPLE": "AAPL",
        "AT&T": "T",
        "BESTBUY": "BBY",
        "BLOCK": "SQ",
        "BOEING": "BA",
        "BOOKING": "BKNG",
        "COCACOLA": "KO",
        "COSTCO": "COST",
        "CVSHEALTH": "CVS",
        "GENERALMOTORS": "GM",
        "GOOGLE": "GOOGL",
        "HOMEDEPOT": "HD",
        "HONEYWELL": "HON",
        "HP": "HPQ",
        "INTEL": "INTC",
        "JNJ": "JNJ",
        "JPMORGAN": "JPM",
        "LOCKHEEDMARTIN": "LMT",
        "LOWES": "LOW",
        "MASTERCARD": "MA",
        "MCDONALDS": "MCD",
        "META": "META",
        "MICROSOFT": "MSFT",
        "NETFLIX": "NFLX",
        "NIKE": "NKE",
        "NVIDIA": "NVDA",
        "ORACLE": "ORCL",
        "PAYPAL": "PYPL",
        "PEPSICO": "PEP",
        "PFIZER": "PFE",
        "SALESFORCE": "CRM",
        "STARBUCKS": "SBUX",
        "TARGET": "TGT",
        "TESLA": "TSLA",
        "ULTA": "ULTA",
        "UPS": "UPS",
        "VERIZON": "VZ",
        "VISA": "V",
        "WALMART": "WMT",
        "WALTDISNEY": "DIS",
        "WELLSFARGO": "WFC"
    }
    
    print(f"Found {len(required_docs)} unique documents referenced.")
    
    for doc_id in required_docs:
        # Parse "3M_2018_10K" -> company="3M", year="2018"
        try:
            parts = doc_id.split('_')
            company_name = parts[0]
            year = int(parts[1])
            
            if company_name in ticker_map:
                download_queue.append({
                    "ticker": ticker_map[company_name],
                    "year": year,
                    "doc_type": "10-K"
                })
            else:
                print(f"Warning: No ticker mapping for {company_name}")
        except Exception as e:
            print(f"Error parsing doc_id {doc_id}: {e}")
            
    return download_queue

def download_filings(queue, output_dir, email):
    """
    Downloads filings using sec-edgar-downloader.
    """
    dl = Downloader("OrionFinancialAI", email, output_dir)
    
    total = len(queue)
    print(f"Starting download of {total} filings...")
    
    for i, item in enumerate(queue):
        ticker = item['ticker']
        year = item['year']
        
        print(f"[{i+1}/{total}] Downloading 10-K for {ticker} ({year})...")
        
        try:
            # Download 10-K filings filed in the specified year
            # Note: 10-Ks for a fiscal year are often filed in the *following* calendar year.
            # FinanceBench 'year' usually refers to the fiscal year or the filing year.
            # We will try to fetch filings from that year.
            # limit=1 gets the latest one found in that range if we don't specify exact dates.
            # To be precise, we might need date ranges, but for now let's try getting 
            # filings filed in that year.
            
            # sec-edgar-downloader 'after' and 'before' format: YYYY-MM-DD
            after_date = f"{year}-01-01"
            before_date = f"{year}-12-31"
            
            dl.get("10-K", ticker, after=after_date, before=before_date, download_details=False)
            
        except Exception as e:
            print(f"Failed to download {ticker} {year}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinanceBench Manifest Downloader")
    parser.add_argument("--email", required=True, help="User email for SEC EDGAR access (User-Agent)")
    parser.add_argument("--output_dir", default="data", help="Directory to save filings")
    
    args = parser.parse_args()
    
    queue = get_financebench_requirements()
    download_filings(queue, args.output_dir, args.email)
