from src.graph.state import ConversationState
from langgraph.graph import StateGraph, START, END
from src.util.log_config import setup_logging
from src.infrastructure.llm import llm 
logger = setup_logging('graph.graph')

def mediatior_node(state: ConversationState)-> ConversationState: 
    """Initializes round robin conversation and summarizes
    the result of the different analysts"""
    
    logger.info('mediator node invoked')
    
    messages = state['messages']
    last = messages[-1].content if messages else ''
    
    mediator_prompt = f""""You are a helpful assistant skilled at coordinating a group
    of other agents to solve a task. You make sure that every agent in
    the group chat has a chance to speak at least twice. When all agents
    provide their analysis, consolidate inputs of all agent into a report.
    Reply "TERMINATE" at the end when everything is done. 
    """"
    mediator_messages = [{"role": "user", "content": mediator_prompt}]
    response = llm.invoke(mediator_prompt)
    
    return response


def round_robin(state: ConversationState) -> Literal["fundamental", "technical", "sentiment", "mediator", "end"]:
    """Routes between investment agent steps for multi-round discussion"""
    round_num = state.get("discussion_round", 0)
    
    # Round 0: fundamental, 1: technical, 2: sentiment
    # Round 3: fundamental, 4: technical, 5: sentiment
    # Round 6: mediator
    
    if round_num == 6:
        return "mediator"
    elif round_num > 6:
        return "end"
    
    assign = round_num % 3
    if assign == 0:
        return "fundamental"
    elif assign == 1:
        return "technical"
    else:
        return "sentiment"
        


def create_main_graph() -> StateGraph: 

    builder = StateGraph(ConversationState)
    
main_graph = create_main_graph()