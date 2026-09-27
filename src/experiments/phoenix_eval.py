import json
import os
import re

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from src.util.log_config import setup_logging

logger = setup_logging("experiments.phoenix_eval")


JUDGE_SYSTEM = (
    "You are an impartial evaluator scoring an analyst LLM's response. Given the "
    "system prompt, the human prompt (which contains the available evidence), and "
    "the analyst's response, produce two scores in [0,1]:\n"
    "- faithfulness: 1 if every factual claim in the response is supported by the "
    "human prompt evidence; 0 if claims are fabricated or contradict the evidence.\n"
    "- relevance: 1 if the response directly addresses the analyst's task; 0 if it "
    "drifts off-topic.\n"
    'Return ONLY a JSON object: {"faithfulness": <float>, "relevance": '
    '<float>, "comment": <one short sentence>}.'
)


def _judge_model():
    name = (
        os.getenv("JUDGE_MODEL")
        or os.getenv("MODEL")
        or "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    )
    provider = os.getenv("JUDGE_PROVIDER") or os.getenv("PROVIDER") or "together"
    return init_chat_model(name, model_provider=provider, max_retries=2, timeout=60)


def _parse_judge(text):
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _score_trace(model, trace):
    human = (
        f"AGENT: {trace.get('agent')}\n\n"
        f"SYSTEM PROMPT:\n{trace.get('system_prompt', '')}\n\n"
        f"HUMAN PROMPT:\n{trace.get('human_prompt', '')}\n\n"
        f"RESPONSE:\n{trace.get('response', '')}"
    )
    try:
        response = model.invoke(
            [SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=human)]
        )
        content = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        parsed = _parse_judge(content)
        return {
            "agent": trace.get("agent"),
            "faithfulness": (
                float(parsed["faithfulness"]) if "faithfulness" in parsed else None
            ),
            "relevance": float(parsed["relevance"]) if "relevance" in parsed else None,
            "comment": parsed.get("comment"),
        }
    except Exception as exc:
        logger.warning("Judge failed for agent=%s: %s", trace.get("agent"), exc)
        return {
            "agent": trace.get("agent"),
            "faithfulness": None,
            "relevance": None,
            "comment": str(exc),
        }


def evaluate_stock_results(stock_results, *, max_per_run=None):
    if os.getenv("DISABLE_JUDGE", "false").lower() == "true":
        return {
            "enabled": False,
            "reason": "DISABLE_JUDGE=true",
            "per_trace": [],
            "per_agent": {},
        }
    try:
        model = _judge_model()
    except Exception as exc:
        logger.warning("Judge model unavailable; skipping eval: %s", exc)
        return {"enabled": False, "reason": str(exc), "per_trace": [], "per_agent": {}}

    per_trace = []
    # Start with an empty dictionary.
    by_agent = {}
    count = 0
    for ticker, result in stock_results.items():
        for trace in result.prompt_trace:
            if max_per_run is not None and count >= max_per_run:
                break

            count += 1
            scored = _score_trace(model, trace)
            scored["ticker"] = ticker
            per_trace.append(scored)
            by_agent.setdefault(scored["agent"] or "unknown", []).append(scored)

    def mean(values):
        clean = [v for v in values if v is not None]

        return sum(clean) / len(clean) if clean else None

    per_agent = {
        agent: {
            "faithfulness_mean": mean([item["faithfulness"] for item in items]),
            "relevance_mean": mean([item["relevance"] for item in items]),
            "n": len(items),
        }
        for agent, items in by_agent.items()
    }
    return {"enabled": True, "per_trace": per_trace, "per_agent": per_agent}
