from langgraph.prebuilt import create_react_agent
from langchain.tools import Tool
from langchain_core.messages.utils import trim_messages

from src.types import ConversationState
from src.infrastructure.llm import llm 
from src.util.log_config import setup_logging
from src.infrastructure.tools import  create_tenk_filing_repl, create_tenq_filing_repl

logger = setup_logging('Fundamental Agent')

def pre_model_hook(state):
    trimmed = trim_messages(
        state["messages"],
        strategy="last",
        max_tokens=10000,  # Adjust based on model limit
        token_counter=llm.get_num_tokens_from_messages,
        start_on="human",
        end_on=("human", "tool")
    )
    return {"llm_input_messages": trimmed}

def fundamental_agent_node(state: ConversationState) -> ConversationState:
    logger.info('Fundamental agent invoked')
    
    ticker = state.get('ticker')
    
    # Initialize REPL tools
    tenk_repl = create_tenk_filing_repl(ticker)
    tenq_repl = create_tenq_filing_repl(ticker)

    tools = [
        Tool(
            name="tenk_repl",
            func=tenk_repl.run,
            description="""Query 10-K filing with Python code. 
            Input: Python code string. start with print(filing.to_context())"""
        ),
        Tool(
            name="tenq_repl", 
            func=tenq_repl.run,
            description="""Query 10-Q filing with Python code. 
            Input: Python code string. start with print(filing.to_context())"""
        )
    ]

    tool_names = [t.name for t in tools]
    fundamental_analysis = state.get('fundamental_analysis', '')
    
    # System prompt from paper
    # The format appendix is help for the ReAct agent loop 
    system_prompt = f"""s a fundamental financial equity
        analyst your primary responsibility is to analyze the most
        recent 10K report provided for a company. You have access to a
        powerful tool that can help you extract relevant information
        from the 10K. Your analysis should be based solely on the
        information that you retrieve using this tool. You can interact
        with this tool using python commands. The tool will
        will return relevant text snippets
        and data points from the 10K document. Keep checking if you
        have answered the users question to avoid looping.
        
        After gathering key sections (Business, Risk Factors, MD&A, Financial Statements), stop tool use and provide the full fundamental analysis.
        You have access to Python REPL tools with pre-loaded SEC filing objects.
        START by running: print(filing.to_context())
        Then use methods that are listed to navigate the filing. 
        DO NOT provide qualitative speculation - extract actual data from the filing.
        DO NOT ask the user if they want you to proceed - just do the analysis.

        Available tools: {tool_names}
        Company: {ticker}

        Your final analysis MUST include:
        - Revenue trend (3 years with % growth)
        - Net income and margins
        - BUY/SELL + conviction score
        ..."""
    
    
    agent_graph = create_react_agent(
        llm, 
        tools,
        prompt=system_prompt,
        pre_model_hook=pre_model_hook
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
        final_messages = None 
        
        for event in agent_graph.stream(
            {"messages": [{"role": "user", "content": input_text}]},
            {"recursion_limit": 75},
            stream_mode="values" 
        ):
            logger.info(f"Agent step: {event}")
            if "messages" in event:
                final_messages = event["messages"]
        
        output = final_messages[-1].content if final_messages else "No output"
        
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        output = f"Analysis incomplete due to error: {e}"
    
    state['fundamental_analysis'] = output
    state['discussion_round'] = state.get('discussion_round', 0) + 1
    state['next_agent'] = 'technical_agent'
    state['messages'] = [{"role": "assistant", "content": output}] #Truncate messages 
    return state