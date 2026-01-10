import re

def extract_ticker(query) -> str:
    """Extract ticker from user message - supports both $TSLA and TSLA formats"""
    
    
    ticker_match = re.search(r'\$([A-Z]{1,5})\b', query)
    
    if not ticker_match:
        ticker_match = re.search(r'\b([A-Z]{1,5})\b', query)
    
    ticker = ticker_match.group(1).upper() if ticker_match else None
    assert ticker 

    return ticker 