import re

_PREFIXED_TICKER = re.compile(r"\$([A-Z]{1,5})\b")
_PLAIN_TICKER = re.compile(r"\b([A-Z]{1,5})\b")


def extract_ticker(query):
    match = _PREFIXED_TICKER.search(query) or _PLAIN_TICKER.search(query)
    if match is None:
        raise ValueError("No ticker symbol found")
    return match.group(1)
