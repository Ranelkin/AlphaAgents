import os
from dotenv import load_dotenv
from .local import AutoLLM
from .remote import remote_llm 
load_dotenv()

LOCAL = os.environ.get('LOCAL', False)

llm = None 

if LOCAL: 
    llm = AutoLLM()
else: 
    llm = remote_llm
    
