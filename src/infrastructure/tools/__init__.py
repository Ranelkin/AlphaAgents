from typing import List, Callable
from .web_search.web_search import search_web
from .yahoo import retrieve_yahoo_data
from .sec_filings import create_tenk_filing_repl, create_tenq_filing_repl

# List of generally available tools for multiple agents via mcp 
TOOLS: List[Callable] = [
    search_web,
    retrieve_yahoo_data,
]

