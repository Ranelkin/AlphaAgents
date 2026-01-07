"""Not yet fully implemented. But one should have the ability to decide which provider one wants
Currently the only provider in use is OpenAI. 
"""

from .providers.openai import llm 

def remote_llm(): 
    return llm 
