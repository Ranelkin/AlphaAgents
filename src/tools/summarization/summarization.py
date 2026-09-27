import os
from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from src.util.log_config import setup_logging

logger = setup_logging("tool.summarization")


REFLECTION_PROMPT = (
    "You are a financial news summarization tool used inside an agentic equity-research "
    "system. For the news content provided, perform three steps inline:\n"
    "1. SUMMARIZE: produce a 2-3 sentence factual summary.\n"
    "2. CRITIQUE: in one sentence, name what your summary missed or overstated.\n"
    "3. REFINE: emit the final summary (2-4 sentences) followed by an informed "
    "recommendation on whether to invest in the underlying stock based on this article alone.\n"
    "Return ONLY the refined output (do not include the words SUMMARIZE/CRITIQUE/REFINE)."
)


@lru_cache(maxsize=1)
def _summary_model():
    model_name = (
        os.getenv("SUMMARIZER_MODEL")
        or os.getenv("MODEL")
        or "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    )

    provider = os.getenv("SUMMARIZER_PROVIDER") or os.getenv("PROVIDER") or "together"
    logger.info("Initializing summarizer model=%s provider=%s", model_name, provider)
    return init_chat_model(
        model_name, model_provider=provider, max_retries=3, timeout=60
    )


def _truncate_fallback(text, limit=240):
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def summarize(text, max_chars=3000):
    """Summarize the news text and shorten it if the model fails."""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
    if os.getenv("DISABLE_LLM_SUMMARY", "false").lower() == "true":
        return _truncate_fallback(cleaned)
    try:
        model = _summary_model()
        response = model.invoke(
            [SystemMessage(content=REFLECTION_PROMPT), HumanMessage(content=cleaned)]
        )
        content = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        return content.strip() or _truncate_fallback(cleaned)
    except Exception as exc:
        logger.warning("LLM summarizer failed (%s); using truncate fallback.", exc)
        return _truncate_fallback(cleaned)


def summarize_optional(text):
    return summarize(text or "")
