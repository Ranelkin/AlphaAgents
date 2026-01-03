from langgraph.prebuilt import create_react_agent
from src.types import ConversationState
from src.infrastructure.llm import llm 
from src.util.log_config import setup_logging
from src.infrastructure.mcp import get_mcp_manager

logger = setup_logging('Fundamental Agent')

def fundamental_agent_node(state: ConversationState) -> ConversationState:
    logger.info('Fundamental agent invoked')
    
    session = get_mcp_manager()
    tools = session.get_tools()
    tool_names = ", ".join([t.name for t in tools])
    
    ticker = state.get('ticker')
    fundamental_analysis = state.get('fundamental_analysis', '')
    
    # System prompt from paper
    # The format appendix is help for the ReAct agent loop 
    system_prompt = f"""As a fundamental financial equity
    analyst your primary responsibility is to analyze the most
    recent 10K/ 10Q report provided for a company. You have access to a
    powerful tool that can help you extract relevant information
    from the 10K. Your analysis should be based solely on the
    information that you retrieve using this tool. You can interact
    with this tool using natural language queries. The tool will
    understand your requests and return relevant text snippets
    and data points from the 10K/ 10Q document. Keep checking if you
    have answered the users question to avoid looping.
    Available tools: {tool_names}
    Company: {ticker}

    Use the following format:
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question"""
        
    agent_graph = create_react_agent(
        llm, 
        tools,
        state_modifier=system_prompt
    )
    
    if fundamental_analysis == '': 
        input_text = f"Analyze the fundamentals for {ticker}"
    else: 
        technical_analysis = state.get('technical_analysis')
        sentiment_analysis = state.get('sentiment_analysis')
        assert technical_analysis and sentiment_analysis, "Missing peer analyses"
        
        input_text = f"""Adjust your evaluation by taking your previous analysis and the analysis on the stock 
        of your peers into consideration when creating a final evalutation of the stock.
        Return a recomendation BUY/SELL and your rate of conviction from 1 to 10.
        Your previous analysis: {fundamental_analysis}
        technical agent analyis: {technical_analysis}
        sentinemnt agent analysis: {sentiment_analysis}"""
            
    try:
        result = agent_graph.invoke(
            {"messages": [{"role": "user", "content": input_text}]},
            {"recursion_limit": 30}
        )
        output = result["messages"][-1].content
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        output = f"Analysis incomplete due to error: {e}"
    
    state['fundamental_analysis'] = output
    state['discussion_round'] = state.get('discussion_round', 0) + 1
    state['next_agent'] = 'technical_agent'
    return state