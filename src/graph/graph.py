from src.types import ConversationState
from langgraph.graph import StateGraph, START, END
from src.util.log_config import setup_logging
from src.infrastructure.llm import llm 
from .util import round_robin

logger = setup_logging('graph.graph')

def mediatior_node(state: ConversationState)-> ConversationState: 
    """Initializes round robin conversation and summarizes
    the result of the different analysts"""
    
    logger.info('mediator node invoked')
    
    messages = state['messages']
    last = messages[-1].content if messages else ''
    
    mediator_prompt = f"""You are a helpful assistant skilled at coordinating a group
    of other agents to solve a task. You make sure that every agent in
    the group chat has a chance to speak at least twice. When all agents
    provide their analysis, consolidate inputs of all agent into a report.
    Reply "TERMINATE" at the end when everything is done. 
    """
    mediator_messages = [{"role": "user", "content": mediator_prompt}]
    response = llm.invoke(mediator_prompt)
    
    state['messages'].append(response)
    return state 



def create_main_graph() -> StateGraph: 

    builder = StateGraph(ConversationState)
    
    return builder 
    
main_graph = create_main_graph()