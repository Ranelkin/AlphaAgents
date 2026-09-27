def r_annualized(cumulative_return, trading_days):
    return (1 + cumulative_return) ** (252 / trading_days) - 1
