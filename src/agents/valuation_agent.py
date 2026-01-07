from langgraph.prebuilt import create_react_agent
from langchain.tools import Tool

from src.types import ConversationState
from src.infrastructure.llm import llm, pre_model_hook
from src.infrastructure.tools import retrieve_yahoo_data
from src.util.log_config import setup_logging

logger = setup_logging('Valuation Agent')



def valuation_agent_node(state: ConversationState): 
    logger.info('Valuation Agent invoked')

    ticker = state.get('ticker')
    yahoo_data = retrieve_yahoo_data(ticker)
    
    #Wrapper functions
    def get_price_data(ticker) -> str:
        """Returns price data formatted as a string for the LLM"""
        price_info = yahoo_data['price']
        return f"""
        Current Price Data for {ticker}:
        - Open: ${price_info['open']:.2f}
        - Close: ${price_info['close']:.2f}
        - High: ${price_info['high']:.2f}
        - Low: ${price_info['low']:.2f}
        
        Daily Data:
        {price_info['day'].to_string()}
        
        Monthly Data (last 20 days):
        {price_info['month'].to_string()}
        
        Yearly trend available with {len(price_info['year'])} data points
        """

    def get_volume_data(ticker) -> str:
        """Returns volume data formatted as a string for the LLM"""
        volume_info = yahoo_data['volume']
        return f"""
        Volume Data for {ticker}:
        
        Today: {volume_info['1d'].values[0]:,} shares
        
        Monthly Average: {volume_info['1mo'].mean():,.0f} shares
        Monthly Max: {volume_info['1mo'].max():,} shares
        
        Yearly Average: {volume_info['1y'].mean():,.0f} shares
        Yearly Max: {volume_info['1y'].max():,} shares
        """

    tools = [
        Tool(
            name="price_data",
            func = get_price_data,
            description="""Price data for the stock to analyze, tool doesnt take any input"""
        ),
        Tool(
            name="volume_data",
            func = get_volume_data,
            description = """Volume data for the stock to analyze, tool doesnt take any input"""
        )
    ]

    
    tool_names = [t.name for t in tools]
    valuation_analysis = state.get('valuation_analysis', '')
    
    system_prompt = f"""As a valuation equity analyst, your pri-
            mary responsibility is to analyze the valuation trends of a
            given asset or portfolio over an extended time horizon. To com-
            plete the task, you must analyze the historical valuation data
            of the asset or portfolio provided, identify trends and patterns
            in valuation metrics over time, and interpret the implications
            of these trends for investors or stakeholders

            Available tools: {tool_names}
            Company: {ticker}
        """

    agent_graph = create_react_agent(
        llm,
        tools,
        prompt=system_prompt,
        pre_model_hook=pre_model_hook
    )

    if valuation_analysis == '': 
        input_text= f"""Analyze valutation trends for {ticker}"""
    else:
        sentiment_analysis = state.get('sentiment_analysis')
        fundamental_analysis = state.get('fundamental_analysis')
        assert sentiment_analysis and fundamental_analysis, 'Missing peer analysis'      

        input_text = f"""Adjust your evaluation by taking your previous analyis
        of the stock of your peers into consideration when creating a final evaluation f the stock. 
        Return a recomendation with a short justification BUY/SELL and your rate of conviction from 1 to 10. 
        Your previous analysis: {valuation_analysis}
        fundamental agent analyis: {fundamental_analysis}
        sentiment agent analysis: {sentiment_analysis}""" 

    
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
    
    state['valuation_analysis'] = output
    state['discussion_round'] = state.get('discussion_round', 0) + 1
    state['next_agent'] = 'sentiment_agent'
    state['messages'] = [{"role": "assistant", "content": output}] #Truncate messages 
    return state