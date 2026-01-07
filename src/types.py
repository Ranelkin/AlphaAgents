from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from typing import Annotated

class ConversationState(TypedDict):
    """Conversation State contains the discussion of the agents
        messages: Annotated[list, add_messages]
        next_agent: str | None
        ticker: str | None
        company_data: str | None
        fundamental_analysis: str | None
        valuation_analysis: str | None
        discussion_round: int | None
        final_recommendation: str | None
        documentation_results: dict | None
        search_results: str | None
        sentiment_analysis: str
    """
    messages: Annotated[list, add_messages]
    next_agent: str 
    prompt: str
    ticker: str
    company_data: str 
    fundamental_analysis: str 
    valuation_analysis: str 
    sentiment_analysis: str
    discussion_round: int 
    final_recommendation: str 
    documentation_results: dict | None
    search_results: str | None
     


