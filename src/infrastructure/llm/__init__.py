import os
from langchain_core.messages.utils import trim_messages
from langchain_core.language_models import BaseChatModel
from .local import AutoLLM
from .remote import remote_llm 
from ...util.log_config import setup_logging

logger = setup_logging('infrastructure.llm')

def _initialize_llm() -> BaseChatModel:
    LOCAL = os.environ.get('LOCAL', False)
    if LOCAL == 'true':
        logger.info('Initialize local llm')
        return AutoLLM()
    else:
        logger.info('Initialize remote llm')
        return remote_llm()

llm: BaseChatModel = _initialize_llm()

def pre_model_hook(state):
    trimmed = trim_messages(
        state["messages"],
        strategy="last",
        max_tokens=10000,  
        token_counter=llm.get_num_tokens_from_messages,
        start_on="human",
        end_on=("human", "tool")
    )
    return {"llm_input_messages": trimmed}