from __future__ import annotations

import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph

from src.model_context import init_agent_model
from src.util.log_config import setup_logging

from .metrics import annualized_return, annualized_volatility, daily_returns_from_prices
from .schemas import AgentDecision, StockRunResult

logger = setup_logging("experiments.inference")

AGENTS = ("fundamental", "sentiment", "valuation")
PROMPT_VERSION = "v5"

_TIE_ORDER = {"HOLD": 2, "SELL": 1, "BUY": 0}


class ModelSettings(TypedDict):
    agent_temperature: float
    synthesizer_temperature: float
    llm_seed: int | None


def _allowed_signals(config):
    if config.decision_mode == "buy_sell_hold":
        return ("BUY", "SELL", "HOLD")
    return ("BUY", "SELL")


def _parse_json(text):
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if match:
        text = match.group(1).strip()
    return json.loads(text)


def model_settings() -> ModelSettings:
    seed = os.getenv("LLM_SEED")
    return {
        "agent_temperature": float(os.getenv("AGENT_TEMPERATURE", "0.7")),
        "synthesizer_temperature": float(os.getenv("SYNTHESIZER_TEMPERATURE", "0.2")),
        "llm_seed": int(seed) if seed else None,
    }


def _init_model(config, temperature=None):
    settings = model_settings()
    if temperature is None:
        temperature = settings["agent_temperature"]
    return init_agent_model(
        config.model,
        config.provider,
        temperature=temperature,
        seed=settings["llm_seed"],
    )


def _response_provenance(response):
    meta = getattr(response, "response_metadata", {}) or {}
    return {
        key: meta.get(key)
        for key in ("system_fingerprint", "model_name")
        if meta.get(key)
    }


def _risk_instructions(risk_profile):
    if risk_profile == "risk_averse":
        return (
            "You advise a risk-averse investor: weight downside risk, volatility, "
            "fragile balance sheets, and crowded narratives more heavily than upside. "
            "Express caution by recommending SELL on stocks whose risk is not clearly "
            "compensated, and BUY on stocks with resilient fundamentals - caution is "
            "expressed through your selection, not by avoiding commitment."
        )
    if risk_profile == "risk_seeking":
        return (
            "You advise a risk-seeking investor: pursue maximum total return and accept "
            "high volatility and concentrated bets when the upside justifies it. Still "
            "recommend SELL on stocks with weak expected return - risk appetite is not "
            "indiscriminate buying, and you must discriminate within the universe."
        )
    return (
        "You advise a risk-neutral investor: optimize total return at acceptable risk, "
        "weighing upside and downside symmetrically. Do not add an artificial safety bias."
    )


def _tooling_flags(tooling_mode):
    if tooling_mode == "minimal":
        return False, False
    if tooling_mode == "no_rag":
        return True, False
    if tooling_mode == "no_news":
        return False, True
    return True, True


SECTIONED_FILING_QUERIES = {
    "cash_flow_income": "cash flow operating income net income revenue earnings",
    "ops_margin": "operations gross margin operating margin cost of revenue efficiency",
    "concerns": "risk factors litigation impairment going concern weakness uncertainty",
    "progress": "strategic objectives outlook guidance milestones progress execution",
}


def _top_filing_chunks(manifest, ticker, query, top_k):
    snapshot = manifest.stocks[ticker]
    query_terms = {term for term in re.findall(r"[a-zA-Z]{3,}", query.lower())}
    scored = []

    for chunk in snapshot.filings:
        words = set(re.findall(r"[a-zA-Z]{3,}", chunk.text.lower()))

        overlap = len(query_terms & words)
        if overlap:
            label = f"{chunk.form_type} {chunk.chunk_id}: {chunk.text[:500]}"
            scored.append((float(overlap), label))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [text for _, text in scored[:top_k]]


def _window_return(values):
    if len(values) < 2 or values[0] == 0:
        return 0.0
    return values[-1] / values[0] - 1.0


def _cross_section(manifest):
    """Make the stock comparison table from older price data."""
    rows = []

    for ticker in manifest.universe:
        snapshot = manifest.stocks.get(ticker)

        if snapshot is None:
            continue

        closes = [bar.close for bar in snapshot.pre_window_prices]

        rows.append(
            {
                "ticker": ticker,
                "one_month_return": _window_return(closes[-21:]),
                "three_month_return": _window_return(closes[-63:]),
                "annualized_volatility": annualized_volatility(
                    daily_returns_from_prices(closes)
                ),
                "latest_close": closes[-1] if closes else None,
            }
        )
    rows.sort(key=lambda row: -row["three_month_return"])
    for rank, row in enumerate(rows, start=1):
        row["momentum_rank"] = rank
    return rows


def _build_dossier(manifest, ticker, config):
    snapshot = manifest.stocks[ticker]
    closes = [bar.close for bar in snapshot.pre_window_prices]
    volumes = [bar.volume for bar in snapshot.pre_window_prices]
    one_month_slice = closes[-21:] if len(closes) >= 21 else closes
    three_month_slice = closes[-63:] if len(closes) >= 63 else closes

    window_return = _window_return

    def avg(items):
        return sum(items) / len(items) if items else 0.0

    allow_news, allow_filings = _tooling_flags(config.tooling_mode)
    news_items = (
        [
            {
                "title": article.title,
                "summary": article.summary,
                "sentiment": article.sentiment,
                "source": article.source,
                "published_at": article.published_at,
            }
            for article in snapshot.news
        ]
        if allow_news
        else [{"title": article.title} for article in snapshot.news]
    )
    filing_sections = {}
    if allow_filings:
        for section, query in SECTIONED_FILING_QUERIES.items():
            filing_sections[section] = _top_filing_chunks(
                manifest,
                ticker,
                query,
                max(2, config.filing_top_k // len(SECTIONED_FILING_QUERIES)),
            )
    return {
        "ticker": ticker,
        "news_source": snapshot.news_source,
        "tooling_mode": config.tooling_mode,
        "analyst_price_targets": snapshot.analyst_price_targets if allow_news else {},
        "valuation_window_months": config.valuation_window_months,
        "pre_window_start": (
            snapshot.pre_window_prices[0].date if snapshot.pre_window_prices else None
        ),
        "pre_window_end": (
            snapshot.pre_window_prices[-1].date if snapshot.pre_window_prices else None
        ),
        "latest_close": closes[-1] if closes else None,
        "one_month_return": window_return(one_month_slice),
        "three_month_return": window_return(three_month_slice),
        "average_volume": avg(volumes),
        "latest_volume": volumes[-1] if volumes else None,
        "news_count": len(news_items),
        "news_items": news_items,
        "filings_available": bool(snapshot.filings) and allow_filings,
        "filing_sections": filing_sections,
        "universe_cross_section": [
            {**row, "is_this_stock": row["ticker"] == ticker}
            for row in _cross_section(manifest)
        ],
    }


AGENT_DOSSIER_FIELDS = {
    "fundamental": (
        "ticker",
        "tooling_mode",
        "valuation_window_months",
        "filings_available",
        "filing_sections",
        "universe_cross_section",
    ),
    "sentiment": (
        "ticker",
        "tooling_mode",
        "news_source",
        "news_count",
        "news_items",
        "analyst_price_targets",
        "universe_cross_section",
    ),
    "valuation": (
        "ticker",
        "tooling_mode",
        "valuation_window_months",
        "pre_window_start",
        "pre_window_end",
        "latest_close",
        "one_month_return",
        "three_month_return",
        "average_volume",
        "latest_volume",
        "universe_cross_section",
    ),
}


def _scoped_dossier(full_dossier, agent_name):
    fields = AGENT_DOSSIER_FIELDS.get(agent_name)
    if not fields:
        return {
            key: value for key, value in full_dossier.items() if not key.startswith("_")
        }
    return {field: full_dossier[field] for field in fields if field in full_dossier}


def _make_agent_tools(manifest, ticker, config, agent_name):
    snapshot = manifest.stocks[ticker]
    allow_news, allow_filings = _tooling_flags(config.tooling_mode)
    tools = []
    if agent_name == "fundamental":
        if allow_filings and snapshot.filings:

            @tool
            def search_filings(query):
                """Search the frozen 10-K and 10-Q text."""
                chunks = _top_filing_chunks(
                    manifest, ticker, query, config.filing_top_k
                )

                return "\n\n".join(chunks) if chunks else "No matching filing sections."

            tools.append(search_filings)

        @tool
        def price_history():
            """Return the last 63 pre-cutoff price and volume observations."""
            bars = snapshot.pre_window_prices[-63:]
            return json.dumps(
                [
                    {"date": bar.date, "close": bar.close, "volume": bar.volume}
                    for bar in bars
                ]
            )

        tools.append(price_history)
    elif agent_name == "valuation":

        @tool
        def compute_annualized_return(cumulative_return, num_trading_days):
            """Annualize a cumulative return over a number of trading days."""
            return annualized_return(cumulative_return, num_trading_days)

        @tool
        def compute_annualized_volatility(daily_returns):
            """Annualize a series of daily returns."""
            return annualized_volatility(daily_returns)

        tools.extend([compute_annualized_return, compute_annualized_volatility])
    return tools


def _agent_system_prompt(agent_name, config, has_tools=False):
    domain = {
        "fundamental": (
            "business quality, financial resilience, capital allocation, growth durability"
        ),
        "sentiment": "news flow, narrative momentum, analyst targets, and market tone",
        "valuation": (
            "historical price and volume windows, risk-adjusted upside, and valuation discipline"
        ),
    }[agent_name]
    allow_news, allow_filings = _tooling_flags(config.tooling_mode)

    if allow_news and allow_filings:
        tool_mode_note = (
            "Use the article summaries and the four sectioned filing snippets "
            "(cash_flow_income, ops_margin, concerns, progress) as tool outputs. "
            "Reference available evidence explicitly."
        )
    elif allow_news:
        tool_mode_note = "Use news evidence; filings are not available in this mode."
    elif allow_filings:
        tool_mode_note = (
            "Use sectioned filing evidence; news is not available in this mode."
        )
    else:
        tool_mode_note = (
            "Use only price/volume statistics from the dossier; do not invent "
            "unavailable evidence."
        )
    tools_note = (
        "You may call the provided tools to gather or verify evidence before answering. "
        "ONLY the tools explicitly provided exist - never call any other tool name; "
        "peer comparison must use the universe_cross_section values as given. "
        if has_tools
        else ""
    )
    allowed = _allowed_signals(config)
    if len(allowed) == 2:
        signal_note = (
            "You must commit: signal must be either BUY or SELL - there is no HOLD option. "
            "BUY means you expect this stock to outperform the equal-weight average of the "
            "experiment's stock universe over the evaluation window; SELL means you expect "
            "it to underperform. Roughly half of any universe underperforms its own average, "
            "so judge this stock relative to its peers - a strong company can still be a "
            "SELL if its peers are stronger. The dossier field universe_cross_section lists "
            "every stock in the universe with its recent price statistics (the stock under "
            "analysis is flagged is_this_stock); rank this stock against those peers, not "
            "in isolation. The peer statistics are already computed and pre-sorted by "
            "three-month momentum (momentum_rank 1 = best); use them directly. "
        )
    else:
        signal_note = "signal must be one of BUY, SELL, HOLD. "

    return (
        f"You are the {agent_name} analyst in a multi-agent equity research experiment. "
        f"Focus on {domain}. {_risk_instructions(config.risk_profile)} {tool_mode_note} "
        f"{tools_note}"
        "Return JSON only with keys: signal, conviction, thesis, evidence, risk_flags, "
        "expected_excess_return_pct. "
        f"{signal_note}conviction must be an integer 1-10. "
        "expected_excess_return_pct is your point estimate of this stock's return minus "
        "the equal-weight universe average over the evaluation window, in percentage "
        "points (e.g. 3.5 or -2.0; negative if it underperforms, consistent with your signal). "
        "evidence and risk_flags must be arrays of short strings."
    )


def _decision_from_payload(payload, allowed):
    signal = str(payload.get("signal", "")).strip().upper()
    if signal not in allowed:
        return None
    try:
        conviction = int(payload.get("conviction", 5))
    except (TypeError, ValueError):
        conviction = 5
    conviction = min(10, max(1, conviction))
    try:
        raw_score = payload.get("expected_excess_return_pct")

        score = float(raw_score) if raw_score is not None else None
    except (TypeError, ValueError):
        score = None
    return AgentDecision(
        signal=signal,
        conviction=conviction,
        thesis=str(payload.get("thesis", "")),
        evidence=[str(item) for item in payload.get("evidence", []) or []],
        risk_flags=[str(item) for item in payload.get("risk_flags", []) or []],
        expected_excess_return_pct=score,
    )


def _invoke_decision(
    model, agent_name, full_dossier, config, extra_context="", manifest=None
):
    tools = (
        _make_agent_tools(manifest, full_dossier["ticker"], config, agent_name)
        if manifest is not None
        else []
    )
    system = _agent_system_prompt(agent_name, config, has_tools=bool(tools))
    scoped = _scoped_dossier(full_dossier, agent_name)
    human_text = json.dumps(scoped, indent=2)
    if extra_context:
        human_text += (
            "\n\nPeer context (do not copy phrasing, form your own view):\n"
            + extra_context
        )
    messages = [SystemMessage(content=system), HumanMessage(content=human_text)]

    bound = model.bind_tools(tools) if tools else model
    tool_map = {t.name: t for t in tools}

    tool_log = []

    response = bound.invoke(messages)
    iterations = 0
    while (
        getattr(response, "tool_calls", None)
        and iterations < config.max_tool_iterations
    ):
        messages.append(response)
        for call in response.tool_calls:
            tool_fn = tool_map.get(call["name"])
            try:
                if tool_fn is None:
                    output = (
                        f"unknown tool {call['name']}. The only tools that exist are: "
                        f"{', '.join(tool_map) or 'none'}. Do not call {call['name']} again; "
                        "use the data already provided and answer now with the required JSON."
                    )
                else:
                    output = tool_fn.invoke(call["args"])
            except Exception as exc:  # surface tool errors to the model
                output = f"tool error: {exc}"
            tool_log.append(
                {
                    "name": call["name"],
                    "args": call["args"],
                    "output": str(output)[:2000],
                }
            )
            messages.append(
                ToolMessage(content=str(output), tool_call_id=call.get("id", ""))
            )
        iterations += 1
        response = bound.invoke(messages)
    if getattr(response, "tool_calls", None):
        logger.warning(
            "%s agent hit tool-iteration bound (%s)",
            agent_name,
            config.max_tool_iterations,
        )

    allowed = _allowed_signals(config)

    def _try_parse(raw_value):
        text = raw_value if isinstance(raw_value, str) else str(raw_value)
        try:
            payload = _parse_json(text)
        except json.JSONDecodeError:
            payload = {}
        return text, _decision_from_payload(payload, allowed)

    content, decision = _try_parse(response.content)

    retried = False
    if decision is None:
        retried = True
        messages.append(response)
        messages.append(
            HumanMessage(
                content=(
                    "Your previous reply could not be used. Respond with JSON only: "
                    '{"signal": "'
                    + '" or "'.join(allowed)
                    + '", "conviction": <int 1-10>, '
                    '"thesis": "<string>", "evidence": [], "risk_flags": [], '
                    '"expected_excess_return_pct": <float>}. '
                    "No prose, no markdown fences."
                )
            )
        )

        response = bound.invoke(messages)
        content, decision = _try_parse(response.content)
        if decision is None:
            logger.warning(
                "%s agent produced no usable decision after retry", agent_name
            )
    trace = {
        "agent": agent_name,
        "prompt_version": PROMPT_VERSION,
        "system_prompt": system,
        "human_prompt": human_text,
        "tool_calls": tool_log,
        "response": content,
        "parse_retried": retried,
        "parse_failed": decision is None,
        "parsed": asdict(decision) if decision is not None else None,
        "provenance": _response_provenance(response),
    }
    return decision, trace


def _invoke_synthesizer(model, config, dossier, recommendations):
    allowed = _allowed_signals(config)
    if len(allowed) == 2:
        signal_note = (
            "signal must be either BUY or SELL - there is no HOLD option. "
            "BUY means the stock is expected to outperform the equal-weight average of the "
            "experiment's stock universe over the evaluation window; SELL means underperform. "
            "Judge relative to peers, not in isolation. "
        )
    else:
        signal_note = "signal must be one of BUY, SELL, HOLD. "
    system = (
        "You are the portfolio construction judge in a reproducible experiment. "
        f"{_risk_instructions(config.risk_profile)} You are given the specialist views "
        "(fundamental, sentiment, valuation) and a compact dossier. Identify points of "
        "agreement AND disagreement between specialists. If they disagree, record the "
        "dissenting view in `risk_flags`. Do NOT simply average; weigh the evidence. "
        "Return JSON only with keys: signal, conviction, thesis, evidence, risk_flags, "
        "expected_excess_return_pct. "
        f"{signal_note}"
        "expected_excess_return_pct is your point estimate of the stock's return minus "
        "the equal-weight universe average over the evaluation window, in percentage points."
    )
    human_payload = {
        "dossier": {
            key: value for key, value in dossier.items() if not key.startswith("_")
        },
        "specialist_recommendations": {
            name: asdict(decision) for name, decision in recommendations.items()
        },
    }

    response = model.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=json.dumps(human_payload, indent=2)),
        ]
    )
    content = (
        response.content if isinstance(response.content, str) else str(response.content)
    )

    try:
        payload = _parse_json(content)
    except json.JSONDecodeError:
        payload = {}
    decision = _decision_from_payload(payload, allowed)
    if decision is None:
        logger.warning(
            "Synthesizer produced no usable decision; falling back to majority vote"
        )

        return _majority_vote(recommendations)
    trace = {
        "agent": "synthesizer",
        "prompt_version": PROMPT_VERSION,
        "synthesis_method": "llm_synthesizer",
        "system_prompt": system,
        "human_prompt": json.dumps(human_payload, indent=2),
        "response": content,
        "parsed": asdict(decision),
        "provenance": _response_provenance(response),
    }
    return decision, trace


def _signals_agree(recommendations):
    signals = {decision.signal for decision in recommendations.values()}
    return len(recommendations) == len(AGENTS) and len(signals) == 1


def _majority_vote(recommendations):
    if not recommendations:
        return None, {
            "agent": "synthesizer",
            "prompt_version": PROMPT_VERSION,
            "synthesis_method": "no_valid_decisions",
            "votes": {},
            "parsed": None,
        }
    convictions_by_signal = {}
    for decision in recommendations.values():
        convictions_by_signal.setdefault(decision.signal, []).append(
            decision.conviction
        )

    def rank(signal):
        convictions = convictions_by_signal[signal]
        return (len(convictions), sum(convictions), _TIE_ORDER.get(signal, -1))

    winner = max(convictions_by_signal, key=rank)

    winning = convictions_by_signal[winner]
    dissent = [
        f"{name}: {decision.signal} (conviction {decision.conviction}) - {decision.thesis[:200]}"
        for name, decision in recommendations.items()
        if decision.signal != winner
    ]
    winner_scores = [
        rec.expected_excess_return_pct
        for rec in recommendations.values()
        if rec.signal == winner and rec.expected_excess_return_pct is not None
    ]
    decision = AgentDecision(
        signal=winner,
        conviction=round(sum(winning) / len(winning)),
        expected_excess_return_pct=(
            sum(winner_scores) / len(winner_scores) if winner_scores else None
        ),
        thesis=(
            "No consensus within the debate round budget; deterministic majority vote "
            "with conviction-weighted tie-breaking."
        ),
        evidence=[
            f"{name}: {rec.signal} (conviction {rec.conviction})"
            for name, rec in recommendations.items()
        ],
        risk_flags=dissent,
    )
    trace = {
        "agent": "synthesizer",
        "prompt_version": PROMPT_VERSION,
        "synthesis_method": "majority_vote_fallback",
        "votes": {name: asdict(rec) for name, rec in recommendations.items()},
        "parsed": asdict(decision),
    }

    return decision, trace


def _parallel_round(model, config, dossier, peer_context="", manifest=None):
    decisions = {}
    traces = []

    def _call(agent_name):
        return agent_name, _invoke_decision(
            model,
            agent_name,
            dossier,
            config,
            extra_context=peer_context,
            manifest=manifest,
        )

    with ThreadPoolExecutor(max_workers=len(AGENTS)) as pool:
        for agent_name, (decision, trace) in pool.map(_call, AGENTS):
            if decision is not None:
                decisions[agent_name] = decision
            traces.append(trace)
    return decisions, traces


class AgentState(TypedDict):
    ticker: str
    round: int
    messages: list[Any]
    recommendations: dict[str, AgentDecision]
    consensus: AgentDecision | None
    traces: list[dict[str, Any]]


_REVISE_INSTRUCTION = (
    "The previous round is complete. Review the recommendations from all three "
    "specialists below. Challenge or support their reasoning from YOUR own "
    "domain perspective (do not adopt their phrasing). Adjust your view only if "
    "the evidence in your scoped dossier supports doing so."
)


def _build_debate_graph(manifest, config, model, synth_model, dossier):
    def debate_round(state):
        peer_context = ""
        if state["round"] >= 1:
            panel = json.dumps(
                {
                    name: asdict(decision)
                    for name, decision in state["recommendations"].items()
                },
                indent=2,
            )

            peer_context = _REVISE_INSTRUCTION + "\n\nPrevious round panel:\n" + panel
        recommendations, traces = _parallel_round(
            model, config, dossier, peer_context=peer_context, manifest=manifest
        )
        return {
            "round": state["round"] + 1,
            "recommendations": recommendations,
            "messages": state["messages"] + [trace["response"] for trace in traces],
            "traces": state["traces"] + traces,
        }

    def synthesize(state):
        if state["recommendations"]:
            consensus, trace = _invoke_synthesizer(
                synth_model, config, dossier, state["recommendations"]
            )
        else:
            consensus, trace = _majority_vote(state["recommendations"])

        return {"consensus": consensus, "traces": state["traces"] + [trace]}

    def route(state):
        if state["round"] >= config.debate_min_rounds and _signals_agree(
            state["recommendations"]
        ):
            return "synthesize"
        if state["round"] >= config.debate_max_rounds:
            return "synthesize"
        return "debate_round"

    graph = StateGraph(AgentState)
    graph.add_node("debate_round", debate_round)
    graph.add_node("synthesize", synthesize)

    graph.set_entry_point("debate_round")
    graph.add_conditional_edges(
        "debate_round",
        route,
        {"synthesize": "synthesize", "debate_round": "debate_round"},
    )
    graph.add_edge("synthesize", END)
    return graph.compile()


def _run_debate(manifest, config, model, synth_model, dossier):
    graph = _build_debate_graph(manifest, config, model, synth_model, dossier)
    final = graph.invoke(
        {
            "ticker": dossier["ticker"],
            "round": 0,
            "messages": [],
            "recommendations": {},
            "consensus": None,
            "traces": [],
        }
    )
    consensus_reached = _signals_agree(final["recommendations"])
    if final["consensus"] is None:
        synthesis_method = "no_valid_decisions"
    else:
        synthesis_method = final["traces"][-1].get("synthesis_method", "unknown")
    metadata = {
        "rounds": final["round"],
        "consensus_reached": consensus_reached,
        "synthesis_method": synthesis_method,
    }
    return final["consensus"], final["recommendations"], final["traces"], metadata


def _signed_score(decision):
    if decision.expected_excess_return_pct is not None:
        return float(decision.expected_excess_return_pct)
    sign = {"BUY": 1.0, "SELL": -1.0}.get(decision.signal, 0.0)
    return sign * float(decision.conviction)


def _relabel(stock_results, ordered, buy_count, rule):
    for rank, ticker in enumerate(ordered, start=1):
        result = stock_results[ticker]
        result.metadata["raw_signal"] = result.consensus.signal
        result.metadata["universe_rank"] = rank
        result.metadata["decision_rule_applied"] = rule
        result.consensus = replace(
            result.consensus, signal="BUY" if rank <= buy_count else "SELL"
        )


def _apply_median_split(stock_results):
    scored = sorted(
        (
            (_signed_score(result.consensus), ticker)
            for ticker, result in stock_results.items()
            if result.consensus is not None
        ),
        key=lambda item: (-item[0], item[1]),
    )
    ordered = [ticker for _, ticker in scored]
    for score, ticker in scored:
        stock_results[ticker].metadata["cross_sectional_score"] = score
    _relabel(stock_results, ordered, len(ordered) // 2, "median_split")


def _invoke_universe_ranker(synth_model, config, manifest, stock_results):
    usable = [
        ticker
        for ticker in manifest.universe
        if stock_results.get(ticker) and stock_results[ticker].consensus is not None
    ]
    lines = {}
    for ticker in usable:
        consensus = stock_results[ticker].consensus
        snapshot = manifest.stocks[ticker]
        closes = [bar.close for bar in snapshot.pre_window_prices]
        lines[ticker] = {
            "signal": consensus.signal,
            "conviction": consensus.conviction,
            "expected_excess_return_pct": consensus.expected_excess_return_pct,
            "thesis": consensus.thesis[:200],
            "one_month_return": _window_return(closes[-21:]),
            "three_month_return": _window_return(closes[-63:]),
        }
    system = (
        "You are the portfolio construction judge in a reproducible experiment. "
        f"{_risk_instructions(config.risk_profile)} You are given the consensus view for "
        "every stock in the universe. Rank ALL tickers from strongest to weakest expected "
        "performance relative to the equal-weight universe average over the evaluation "
        'window. Return JSON only: {"ranking": [<ticker>, ...]} listing every ticker '
        "exactly once, best first. No prose, no markdown fences."
    )
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=json.dumps(lines, indent=2)),
    ]

    def _parse_ranking(text):
        try:
            ranking = _parse_json(text).get("ranking")
        except json.JSONDecodeError:
            return None

        if isinstance(ranking, list) and sorted(str(t) for t in ranking) == sorted(
            usable
        ):
            return [str(t) for t in ranking]
        return None

    response = synth_model.invoke(messages)
    content = (
        response.content if isinstance(response.content, str) else str(response.content)
    )

    ordering = _parse_ranking(content)

    retried = False

    if ordering is None:
        retried = True
        messages.append(response)
        messages.append(
            HumanMessage(
                content=(
                    "Your previous reply could not be used. Return JSON only: "
                    '{"ranking": [...]} containing each of these tickers exactly once: '
                    + ", ".join(usable)
                    + ". No prose, no markdown fences."
                )
            )
        )
        response = synth_model.invoke(messages)
        content = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        ordering = _parse_ranking(content)
    trace = {
        "agent": "universe_ranker",
        "prompt_version": PROMPT_VERSION,
        "system_prompt": system,
        "human_prompt": json.dumps(lines, indent=2),
        "response": content,
        "parse_retried": retried,
        "parse_failed": ordering is None,
        "parsed": {"ranking": ordering} if ordering is not None else None,
        "provenance": _response_provenance(response),
    }
    return ordering, trace


def _apply_decision_rule(stock_results, config, manifest, synth_model):
    if config.decision_rule == "agent":
        return
    if config.decision_rule == "median_split":
        _apply_median_split(stock_results)

        return

    usable = [
        ticker
        for ticker in manifest.universe
        if stock_results.get(ticker) and stock_results[ticker].consensus is not None
    ]
    if not usable:
        return
    ordering, trace = _invoke_universe_ranker(
        synth_model, config, manifest, stock_results
    )
    stock_results[usable[0]].prompt_trace.append(trace)
    if ordering is None:
        logger.warning(
            "Universe ranker produced no valid permutation; falling back to median split"
        )
        _apply_median_split(stock_results)
        return
    buy_count = (
        config.rank_buy_count
        if config.rank_buy_count is not None
        else len(ordering) // 2
    )
    _relabel(stock_results, ordering, buy_count, "llm_rank_topk")


def _run_baseline(manifest, config):
    # Start with an empty dictionary.
    stock_results = {}

    for ticker in manifest.universe:
        if config.workflow_mode == "baseline_buy":
            signal = "BUY"
        else:
            random_number_generator = random.Random(
                f"{manifest.manifest_hash}:{ticker}"
            )

            signal = random_number_generator.choice(_allowed_signals(config))
        decision = AgentDecision(
            signal=signal,
            conviction=5,
            thesis=f"{config.workflow_mode} baseline (no model)",
        )
        stock_results[ticker] = StockRunResult(
            ticker=ticker,
            workflow_mode=config.workflow_mode,
            risk_profile=config.risk_profile,
            tooling_mode=config.tooling_mode,
            recommendations={config.workflow_mode: decision},
            consensus=decision,
            metadata={
                "decision_status": "ok",
                "baseline": True,
                "decision_mode": config.decision_mode,
            },
        )
    return stock_results


def run_inference(manifest, config):
    if config.workflow_mode in ("baseline_random", "baseline_buy"):
        return _run_baseline(manifest, config)
    model = _init_model(config)
    synth_model = _init_model(
        config, temperature=model_settings()["synthesizer_temperature"]
    )
    stock_results = {}
    for ticker in manifest.universe:
        dossier = _build_dossier(manifest, ticker, config)
        recommendations = {}
        traces = []
        workflow_metadata = {"rounds": 1}

        if config.workflow_mode == "single_agent":
            decision, trace = _invoke_decision(
                model, "fundamental", dossier, config, manifest=manifest
            )
            if decision is not None:
                recommendations["single_agent"] = decision
            traces.append(trace)
            consensus = decision
        elif config.workflow_mode == "collaboration":
            recommendations, round_traces = _parallel_round(
                model, config, dossier, peer_context="", manifest=manifest
            )
            traces.extend(round_traces)

            if recommendations:
                consensus, syn_trace = _invoke_synthesizer(
                    synth_model, config, dossier, recommendations
                )
            else:
                consensus, syn_trace = _majority_vote(recommendations)
            traces.append(syn_trace)
        else:  # debate
            consensus, recommendations, traces, workflow_metadata = _run_debate(
                manifest, config, model, synth_model, dossier
            )

        parse_failures = sum(1 for trace in traces if trace.get("parse_failed"))
        if consensus is None:
            logger.warning(
                "No usable decision for %s (%d parse failures); excluded from portfolios",
                ticker,
                parse_failures,
            )
        agreement_count = sum(
            1
            for decision in recommendations.values()
            if consensus is not None and decision.signal == consensus.signal
        )
        stock_results[ticker] = StockRunResult(
            ticker=ticker,
            workflow_mode=config.workflow_mode,
            risk_profile=config.risk_profile,
            tooling_mode=config.tooling_mode,
            recommendations=recommendations,
            consensus=consensus,
            prompt_trace=traces,
            metadata={
                "news_articles_used": len(manifest.stocks[ticker].news),
                "filing_chunks_used": len(manifest.stocks[ticker].filings),
                "consensus_agreement_count": agreement_count,
                "prompt_version": PROMPT_VERSION,
                "decision_mode": config.decision_mode,
                "decision_rule": config.decision_rule,
                "parse_failures": parse_failures,
                "decision_status": "ok" if consensus is not None else "failed",
                "scoped_dossier": True,
                **workflow_metadata,
            },
        )
        logger.info("Completed %s inference for %s", config.workflow_mode, ticker)
    _apply_decision_rule(stock_results, config, manifest, synth_model)

    return stock_results
