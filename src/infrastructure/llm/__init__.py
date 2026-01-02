import os
from .local import AutoLLM
from .remote import remote_llm 
from ...util.log_config import setup_logging

logger = setup_logging('infrastructure.llm')

LOCAL = os.environ.get('LOCAL', False)

llm = None 

if LOCAL=='true': 
    logger.info('Initiliaze local llm')
    llm = AutoLLM()
else: 
    logger.info('Initiliaze remote llm')
    llm = remote_llm
    
