import os 
from edgar import * 
from langchain_experimental.utilities import PythonREPL
from langchain.tools import Tool
from ....util.log_config import setup_logging

logger = setup_logging('tool.SEC_filings')

EMAIL = os.environ.get('EMAIL')
COMPANY = os.environ.get('COMPANY')
assert EMAIL

logger.info(f'Set email: {EMAIL}')
set_identity(EMAIL)

def create_tenk_filing_repl(ticker: str): 
    company = Company(ticker)
    assert company.is_company, f"No company found for {ticker}"
    filing = company.get_filings(form='10-K').latest(1)
    restricted_env = {
        "filing": filing,
        "print": print,
        "str": str,
        "len": len,
        "dir": dir,       
        "help": help,     
        "type": type,       
       
    }  
    repl = PythonREPL(globals=restricted_env, locals=restricted_env)

    description = """This a python shell equiped with the edgar company 10-K Filing.
    Use this to explore the filing. Start with: print(filing.to_context()).
    You can use dir(filing), help(filing.obj), etc. to discover available methods."""    
    
    return Tool(
        name="python_10_K_repl",
        description=description,
        func=repl.run,
    )
   
def create_tenq_filing_repl(ticker: str): 
    company = Company(ticker)
    assert company.is_company, f"No company found for {ticker}"
    filing = company.get_filings(form='10-Q').latest(1)
    restricted_env = {
        "filing": filing,
        "print": print,
        "str": str,
        "len": len,
        "dir": dir,       
        "help": help,     
        "type": type,       
       
    }  
    repl =  PythonREPL(globals=restricted_env, locals=restricted_env)
    
    description = """This a python shell equiped with the edgar company 10-Q Filing.
    Use this to explore the filing. Start with: print(filing.to_context()).
    You can use dir(filing), help(filing.obj), etc. to discover available methods."""    
    
    return Tool(
        name="python_10_Q_repl",
        description=description,
        func=repl.run,
    )
    

