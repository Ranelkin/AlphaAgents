import math

TRADING_DAYS = 252


def annualized_return(cumulative_return, periods):
    if periods <= 0:
        return 0.0
    return (1.0 + cumulative_return) ** (TRADING_DAYS / periods) - 1.0


def annualized_volatility(daily_returns):
    if len(daily_returns) < 2:
        return 0.0

    mean = sum(daily_returns) / len(daily_returns)
    variance = sum((value - mean) ** 2 for value in daily_returns) / (
        len(daily_returns) - 1
    )
    return math.sqrt(variance) * math.sqrt(TRADING_DAYS)


def sharpe_ratio(daily_returns, risk_free_rate_annual=0.0):
    if len(daily_returns) < 2:
        return 0.0
    daily_rf = risk_free_rate_annual / TRADING_DAYS
    excess_returns = [value - daily_rf for value in daily_returns]

    vol = annualized_volatility(excess_returns)
    if vol == 0:
        return 0.0
    mean_excess = sum(excess_returns) / len(excess_returns)
    return mean_excess * TRADING_DAYS / vol


def cumulative_return_from_prices(prices):
    if len(prices) < 2 or prices[0] == 0:
        return 0.0
    return prices[-1] / prices[0] - 1.0


def daily_returns_from_prices(prices):
    if len(prices) < 2:
        return []
    returns = []
    for previous, current in zip(prices, prices[1:]):
        if previous == 0:
            returns.append(0.0)
        else:
            returns.append(current / previous - 1.0)

    return returns


def rolling_sharpe(daily_returns, window, risk_free_rate_annual=0.0):
    if window <= 1 or len(daily_returns) < window:
        return []
    values = []
    for start in range(0, len(daily_returns) - window + 1):
        chunk = daily_returns[start : start + window]
        values.append(sharpe_ratio(chunk, risk_free_rate_annual=risk_free_rate_annual))

    return values


def _ranks_with_ties(values):
    order = sorted(range(len(values)), key=lambda index: values[index])

    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        tail = position
        while (
            tail + 1 < len(order) and values[order[tail + 1]] == values[order[position]]
        ):
            tail += 1

        average_rank = (position + tail) / 2.0 + 1.0
        for index in order[position : tail + 1]:
            ranks[index] = average_rank
        position = tail + 1
    return ranks


def spearman_correlation(xs, ys):
    """Return Spearman's rho, using average ranks for ties."""
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    rank_x = _ranks_with_ties(xs)
    rank_y = _ranks_with_ties(ys)

    mean_x = sum(rank_x) / len(rank_x)

    mean_y = sum(rank_y) / len(rank_y)
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(rank_x, rank_y))
    var_x = sum((a - mean_x) ** 2 for a in rank_x)
    var_y = sum((b - mean_y) ** 2 for b in rank_y)
    if var_x == 0 or var_y == 0:
        return None
    return cov / math.sqrt(var_x * var_y)


_T_CRIT_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    20: 2.086,
    25: 2.060,
    30: 2.042,
}


def mean_ci95(values):
    """Return the average and its 95% interval."""
    number_of_values = len(values)

    if number_of_values == 0:
        return 0.0, 0.0, 0.0
    mean = sum(values) / number_of_values
    if number_of_values < 2:
        return mean, mean, mean
    sample_standard_deviation = math.sqrt(
        sum((sample_value - mean) ** 2 for sample_value in values)
        / (number_of_values - 1)
    )
    degrees_of_freedom = number_of_values - 1
    available_degrees = [k for k in _T_CRIT_95 if k >= degrees_of_freedom]
    critical_t_value = _T_CRIT_95[min(available_degrees)] if available_degrees else 1.96
    interval_half_width = (
        critical_t_value * sample_standard_deviation / math.sqrt(number_of_values)
    )
    return mean, mean - interval_half_width, mean + interval_half_width


def holm_bonferroni(p_values):
    """Adjust the p-values and keep missing values as they are."""
    indexed = [
        (original_index, p_value)
        for original_index, p_value in enumerate(p_values)
        if p_value is not None
    ]
    adjusted = [None] * len(p_values)
    number_of_tests = len(indexed)
    running = 0.0
    for rank, (original_index, p_value) in enumerate(
        sorted(indexed, key=lambda item: item[1])
    ):
        running = max(running, (number_of_tests - rank) * p_value)
        adjusted[original_index] = min(1.0, running)
    return adjusted


def max_drawdown(prices):
    if not prices:
        return 0.0
    peak = prices[0]
    max_dd = 0.0
    for price in prices:
        peak = max(peak, price)

        if peak == 0:
            continue
        drawdown = price / peak - 1.0

        max_dd = min(max_dd, drawdown)
    return max_dd
