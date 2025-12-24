
def round_robin(state: MainState) -> Literal["fundamental", "technical", "sentiment", "mediator", "end"]:
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
