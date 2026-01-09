from autogen import AssistantAgent, GroupChat, GroupChatManager
from src.tools import retrieve_yahoo_data
from src.tools.sec_filings import create_tenk_filing_repl, create_tenq_filing_repl
from src.util.log_config import setup_logging
from src.config import llm_config

logger = setup_logging('autogen_agents')

def create_agents(ticker: str):
    """Create the three specialized agents for a given ticker"""
    
    yahoo_data = retrieve_yahoo_data(ticker)
    news = yahoo_data['sentiment'].get('news', [])
    fundamental_agent = AssistantAgent(
        name="Fundamental_Analyst",
        system_message=f"""As a fundamental financial equity
        analyst your primary responsibility is to analyze the most
        recent 10K report provided for a company {ticker}. You have access to a
        powerful tool that can help you extract relevant information
        from the 10K. Your analysis should be based solely on the
        information that you retrieve using this tool. You can interact
        with this tool using python commands. The tool will
        will return relevant text snippets
        and data points from the 10K document. Keep checking if you
        have answered the users question to avoid looping.
        
        FIRST ROUND: Provide independent analysis with BUY/SELL/HOLD + conviction (1-10).
        SECOND ROUND: Adjust based on peer analyses if needed.""",
        llm_config=llm_config,
        function_map={
            "tenk_repl": lambda code: create_tenk_filing_repl(ticker).run(code),
            "tenq_repl": lambda code: create_tenq_filing_repl(ticker).run(code)
        }
    )
    
    valuation_agent = AssistantAgent(
        name="Valuation_Analyst",
        system_message=f"""As a valuation equity analyst, your pri-
        mary responsibility is to analyze the valuation trends of a
        given asset or portfolio over an extended time horizon. To com-
        plete the task, you must analyze the historical valuation data
        of the asset or portfolio provided, identify trends and patterns
        in valuation metrics over time, and interpret the implications
        of these trends for investors or stakeholders.
        Price data available:
        - Open: ${yahoo_data['price']['open']:.2f}
        - Close: ${yahoo_data['price']['close']:.2f}
        - High: ${yahoo_data['price']['high']:.2f}
        - Low: ${yahoo_data['price']['low']:.2f}
        
        FIRST ROUND: Provide independent analysis with BUY/SELL/HOLD + conviction (1-10).
        SECOND ROUND: Adjust based on peer analyses if needed.""",
        llm_config=llm_config
    )

    sentiment_agent = AssistantAgent(
        name="Sentiment_Analyst",
        system_message=f"""As a sentiment equity analyst your pri-
        mary responsibility is to analyze the financial news, analyst
        ratings and disclosures related to the underlying security;
        and analyze its implication and sentiment for investors or
        stakeholders.
        
        Sentiment data:
        - Mean sentiment: {yahoo_data['sentiment']['mean']:.2f}
        - Recent news: {yahoo_data['sentiment']['news']} articles
        - Analyst targets: {yahoo_data['sentiment']['price_targets']}
        
        FIRST ROUND: Provide independent analysis with BUY/SELL/HOLD + conviction (1-10).
        SECOND ROUND: Adjust based on peer analyses if needed.""", 
        llm_config=llm_config
    )
    
    return [fundamental_agent, valuation_agent, sentiment_agent]


def run_debate(ticker: str, mode: str = "debate"):
    """
    Multi-agent analysis for a stock
    mode: "debate" or "collaboration" 
    """
    logger.info(f"Starting {mode} mode analysis for {ticker}")
    
    agents = create_agents(ticker)
    
    if mode == "debate":
        # Debate mode: Round-robin, 2 rounds each
        groupchat = GroupChat(
            agents=agents, # type: ignore
            messages=[],
            max_round=6,  # 3 agents * 2 rounds
            speaker_selection_method="round_robin",
            allow_repeat_speaker=False
        ) 
        
        manager = GroupChatManager(
            groupchat=groupchat,
            llm_config=llm_config,
            system_message="""You are a helpful assistant skilled at coordinating a group
            of other agents to solve a task. You make sure that every agent in
            the group chat has a chance to speak at least twice. Each agent can
            not decide for the whole group. They are tasked with coming to a
            consensus. You must invoke all agents before deciding to Terminate.
            Reply "TERMINATE" at the end when everything is done."""
        )
    # collaboration    
    else:  
        groupchat = GroupChat(
            agents=agents, # type: ignore
            messages=[],
            max_round=4, 
            speaker_selection_method="round_robin"
        )
        
        manager = GroupChatManager(
            groupchat=groupchat,
            llm_config=llm_config,
            system_message="""You are a helpful assistant skilled at coordinating a group
            of other agents to solve a task. You make sure that every agent in
            the group chat has a chance to speak at least twice. When all agents
            provide their analysis, consolidate inputs of all agent into a report.
            Reply "TERMINATE" at the end when everything is done."""
        )

    # Concersation start     
    initial_message = f"Analyze {ticker} stock. Each analyst provide your recommendation."
    
    result = agents[0].initiate_chat(
        manager,
        message=initial_message
    )
    
    return result
