import os

from edgar import Company, set_identity
from langchain_experimental.utilities import PythonREPL


def _require_identity():
    email = os.environ.get("EMAIL")
    if not email:
        raise RuntimeError("EMAIL must be set before fetching SEC filings")
    set_identity(email)


def _filing_repl(ticker, form):
    _require_identity()

    company = Company(ticker)
    if not company.is_company:
        raise ValueError(f"No company found for {ticker}")

    filing = company.get_filings(form=form).latest(1)
    if not filing:
        raise LookupError(f"No {form} filing found for {ticker}")

    namespace = {
        "filing": filing,
        "print": print,
        "str": str,
        "len": len,
        "dir": dir,
        "help": help,
        "type": type,
    }
    repl = PythonREPL()
    repl.globals = namespace
    repl.locals = namespace

    return repl


def create_tenk_filing_repl(ticker):
    return _filing_repl(ticker, "10-K")


def create_tenq_filing_repl(ticker):
    return _filing_repl(ticker, "10-Q")
