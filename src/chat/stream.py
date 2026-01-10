# Stream generator for a nice UI experience 
import time
from _collections_abc import Generator

def chat_stream(query: str) -> Generator[str]:
    for word in query.split(" "):
        yield word + " "
        time.sleep(0.02)
