import os 
from edgar import * 
from langchain_experimental.utilities import PythonREPL
from langchain.tools import Tool
from ....util.log_config import setup_logging

logger = setup_logging('tool.SEC_filings')

EMAIL = os.environ.get('EMAIL')
COMPANY = os.environ.get('COMPANY')
assert EMAIL


set_identity(EMAIL)
logger.info(f'Set email: {EMAIL}')

def create_tenk_filing_repl(ticker: str): 
    company = Company(ticker)
    assert company.is_company, f"No company found for {ticker}"
    filing = company.get_filings(form='10-K').latest(1)
    assert filing 
    restricted_env = {
        "filing": filing,
        "print": print,
        "str": str,
        "len": len,
        "dir": dir,       
        "help": help,     
        "type": type,       
       
    }  
    # Python REPL does not have an __init__ method so one has to set 
    # the globals after initiation 
    repl = PythonREPL()
    repl.globals = restricted_env
    repl.locals = restricted_env
    return repl 
    
def create_tenq_filing_repl(ticker: str): 
    company = Company(ticker)
    assert company.is_company, f"No company found for {ticker}"
    filing = company.get_filings(form='10-Q').latest(1)
    assert filing 
    restricted_env = {
        "filing": filing,
        "print": print,
        "str": str,
        "len": len,
        "dir": dir,       
        "help": help,     
        "type": type,       
       
    }  
 
    repl = PythonREPL()
    repl.globals = restricted_env
    repl.locals = restricted_env
    return repl 
    
