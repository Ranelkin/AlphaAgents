from typing import List, Callable
from .web_search.web_search import search_web
from .yahoo import retrieve_yahoo_data
from .sec_filings import query_tenk_filing, query_tenq_filing, create_tenk_filing_repl, create_tenq_filing_repl

TOOLS: List[Callable] = [
    search_web,
    retrieve_yahoo_data,
    query_tenk_filing,
    query_tenq_filing
]

