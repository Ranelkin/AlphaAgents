from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from typing import Annotated

class ConversationState(TypedDict):
    """Conversation State contains the discussion of the agents"""
    messages: Annotated[list, add_messages]
    next_agent: str | None
    ticker: str | None
    company_data: str | None
    fundamental_analysis: str | None
    technical_analysis: str | None
    discussion_round: int | None
    final_recommendation: str | None
    test_results: dict | None
    documentation_results: dict | None
    search_results: str | None
