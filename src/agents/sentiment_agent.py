from langgraph.prebuilt import create_react_agent
from langchain.tools import Tool

from src.types import ConversationState
from src.infrastructure.llm import llm, pre_model_hook
from src.infrastructure.tools import retrieve_yahoo_data
from src.util.log_config import setup_logging

logger = setup_logging('Sentiment Analysis')

def sentiment_agent_node(state: ConversationState): 

    ticker = state.get('ticker')
    yahoo_data = retrieve_yahoo_data(ticker)

    def get_sentiment_data(ticker: str) -> str: 
        """Returns sentiment data formatted as a string for the LLM"""
        sentiment_info = yahoo_data['sentiment']
        return f"""
        Current sentiment data for {ticker}
        - Mean Sentiment score: {sentiment_info['mean']}
        - News: {sentiment_info['news']}
        - Analyst price targets: {sentiment_info['price_targets']}
        """

    tools = [
        Tool(
            name = 'sentiment_data',
            func = get_sentiment_data,
            description = "Sentiment data for the stock to analyze"
        )
    ]
    tool_names = [t.name for t in tools]
    
    sentiment_analysis = state.get('sentiment_analysis', '')

    system_prompt = f"""As a sentiment equity analyst your pri-
        mary responsibility is to analyze the financial news, analyst
        ratings and disclosures related to the underlying security;
        and analyze its implication and sentiment for investors or
        stakeholders.

        Available tools: {tool_names}
        Company: {ticker}
        Your final analysis MUST include:
        - A summary 
        - BUY/SELL recommendation + conviction score
        """
    
    agent_graph = create_react_agent(
        llm,
        tools,
        prompt=system_prompt,
        pre_model_hook=pre_model_hook
    )

    if sentiment_analysis == '': 
        input_text= f"""Analyze valutation trends for {ticker}"""
    else:
        valuation_analysis = state.get('valuation_analysis')
        fundamental_analysis = state.get('fundamental_analysis')
        assert sentiment_analysis and fundamental_analysis, 'Missing peer analysis'      

        input_text = f"""Adjust your evaluation by taking your previous analyis
        of the stock of your peers into consideration when creating a final evaluation f the stock. 
        Return a recomendation with a short justification BUY/SELL and your rate of conviction from 1 to 10. 
        Your previous analysis: {sentiment_analysis}
        fundamental agent analyis: {fundamental_analysis}
        valuation agent analysis: {valuation_analysis}""" 

    
    try:
        final_messages = None 

        for event in agent_graph.stream(
            {"messages": [{"role": "user", "content": input_text}]},
            {"recursion_limit": 50},
            stream_mode="values" 
        ):
            logger.info(f"Agent step: {event}")
            if "messages" in event:
                final_messages = event["messages"]
        
        output = final_messages[-1].content if final_messages else "No output"
        
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        output = f"Analysis incomplete due to error: {e}"
    
    state['sentiment_analysis'] = output
    state['discussion_round'] = state.get('discussion_round', 0) + 1
    state['messages'] = [{"role": "assistant", "content": output}] #Truncate messages 
    return state