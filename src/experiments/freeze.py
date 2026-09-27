import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from edgar import Company, set_identity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.util.log_config import setup_logging

from .news_store import NewsStore, month_bounds
from .schemas import (
    ExperimentManifest,
    FilingChunk,
    NewsArticle,
    PriceBar,
    StockSnapshot,
    to_primitive,
    utc_now,
)

logger = setup_logging("experiments.freeze")
analyzer = SentimentIntensityAnalyzer()


class MissingPriceDataError(RuntimeError):
    def __init__(self, tickers, detail):
        self.tickers = list(tickers)

        super().__init__(f"No usable price data for {self.tickers}: {detail}")


def _yf_symbol(ticker):
    return ticker.replace(".", "-")


def _safe_summary(text):
    trimmed = (text or "").strip()
    if not trimmed:
        return ""
    try:
        from src.tools.summarization import summarize

        return summarize(trimmed[:3000])
    except Exception as exc:
        logger.warning("Summarization fallback used: %s", exc)
        if len(trimmed) <= 240:
            return trimmed
        return trimmed[:237] + "..."


def _normalize_date(value):
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _history_to_price_bars(history):
    if history.empty:
        return []
    frame = history.reset_index()
    date_col = frame.columns[0]
    bars = []
    for _, row in frame.iterrows():
        bars.append(
            PriceBar(
                date=_normalize_date(row[date_col]),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row["Volume"]),
            )
        )
    return bars


def _published_on_or_before(published_at, cutoff_date):
    if published_at is None:
        return False
    try:
        if isinstance(published_at, (int, float)):
            published_timestamp = pd.Timestamp(published_at, unit="s")
        else:
            published_timestamp = pd.Timestamp(published_at)
        if published_timestamp.tzinfo is not None:
            published_timestamp = published_timestamp.tz_localize(None)

        return published_timestamp <= pd.Timestamp(cutoff_date) + pd.Timedelta(
            days=1
        ) - pd.Timedelta(seconds=1)
    except Exception:
        return False


def _extract_news_articles(ticker_obj, max_articles, provider, cutoff_date):
    # yahoo news feed is live so historical cutoffs will usually return nothing
    raw_news = ticker_obj.news or []

    articles = []
    for item in raw_news:
        if len(articles) >= max_articles:
            break

        content = item.get("content") or {}
        published_at = content.get("pubDate") or item.get("providerPublishTime")
        if not _published_on_or_before(published_at, cutoff_date):
            continue
        title = content.get("title") or item.get("title") or ""
        summary_source = content.get("summary") or title
        articles.append(
            NewsArticle(
                title=title,
                published_at=published_at,
                source=content.get("provider") or item.get("publisher"),
                url=(
                    content.get("canonicalUrl", {}).get("url")
                    if isinstance(content.get("canonicalUrl"), dict)
                    else content.get("clickThroughUrl", {}).get("url")
                ),
                summary=_safe_summary(summary_source),
                sentiment=(
                    analyzer.polarity_scores(title)["compound"] if title else None
                ),
                provider=provider,
            )
        )
    return articles


_GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
_GDELT_MIN_INTERVAL_S = float(os.environ.get("GDELT_MIN_INTERVAL_S", "6.0"))
_GDELT_MAX_ATTEMPTS = int(os.environ.get("GDELT_MAX_ATTEMPTS", "10"))
_GDELT_MAX_BACKOFF_S = float(os.environ.get("GDELT_MAX_BACKOFF_S", "900"))
_gdelt_lock = threading.Lock()
_gdelt_last_call = 0.0


def _gdelt_parse_seendate(value):
    if not value:
        return None
    try:
        return pd.Timestamp(str(value).replace("Z", "")).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, TypeError):
        return None


def _gdelt_request(params, max_attempts=None):
    global _gdelt_last_call
    max_attempts = max_attempts or _GDELT_MAX_ATTEMPTS

    def _backoff(attempt, retry_after=None):
        try:
            hinted = float(retry_after) if retry_after else 0.0
        except ValueError:
            hinted = 0.0
        return max(hinted, min(30.0 * (2**attempt), _GDELT_MAX_BACKOFF_S))

    for attempt in range(max_attempts):
        with _gdelt_lock:
            wait = _GDELT_MIN_INTERVAL_S - (time.monotonic() - _gdelt_last_call)
            if wait > 0:
                time.sleep(wait)

            _gdelt_last_call = time.monotonic()
        try:
            response = requests.get(_GDELT_DOC_API, params=params, timeout=60)
        except requests.RequestException as exc:
            if attempt == max_attempts - 1:
                raise
            backoff = _backoff(attempt)
            logger.warning(
                "GDELT network error (%s); backing off %.0fs (attempt %d/%d)",
                exc,
                backoff,
                attempt + 1,
                max_attempts,
            )
            time.sleep(backoff)
            continue

        if response.status_code == 429 or response.status_code >= 500:
            if attempt == max_attempts - 1:
                response.raise_for_status()
            backoff = _backoff(attempt, response.headers.get("Retry-After"))
            logger.warning(
                "GDELT %s; backing off %.0fs (attempt %d/%d)",
                response.status_code,
                backoff,
                attempt + 1,
                max_attempts,
            )

            time.sleep(backoff)

            continue
        response.raise_for_status()
        try:
            return response.json().get("articles", [])
        except ValueError:
            if attempt == max_attempts - 1:
                raise RuntimeError(f"GDELT non-JSON response: {response.text[:200]}")
            backoff = _backoff(attempt)
            logger.warning(
                "GDELT non-JSON answer; backing off %.0fs (attempt %d/%d)",
                backoff,
                attempt + 1,
                max_attempts,
            )
            time.sleep(backoff)
    return []


def _gdelt_fetch_month(company_name, month):
    month_start, month_end = month_bounds(month)
    raw = _gdelt_request(
        {
            "query": f'"{company_name}" sourcelang:eng',
            "mode": "artlist",
            "format": "json",
            "maxrecords": "250",
            "startdatetime": month_start.replace("-", "") + "000000",
            "enddatetime": month_end.replace("-", "") + "235959",
            "sort": "datedesc",
        }
    )
    articles = []
    for item in raw:
        title = (item.get("title") or "").strip()
        published_at = _gdelt_parse_seendate(item.get("seendate"))
        if title and published_at:
            articles.append(
                {
                    "title": title,
                    "published_at": published_at,
                    "url": item.get("url"),
                    "source": item.get("domain"),
                }
            )
    return articles


def _fetch_news_gdelt(
    ticker, company_name, cutoff_date, lookback_days, max_articles, store=None
):
    store = store or NewsStore()
    start = (pd.Timestamp(cutoff_date) - pd.Timedelta(days=lookback_days)).strftime(
        "%Y-%m-%d"
    )

    store.ensure_window(
        ticker,
        start,
        cutoff_date,
        fetch_month=lambda month: _gdelt_fetch_month(company_name, month),
    )
    # Start with an empty list.
    articles = []

    seen_titles = set()
    for item in store.query(ticker, start, cutoff_date):
        if len(articles) >= max_articles:
            break
        title = str(item["title"]).strip()
        published_at = str(item["published_at"])

        if not title or not _published_on_or_before(published_at, cutoff_date):
            continue
        key = title.lower()
        if key in seen_titles:
            continue
        seen_titles.add(key)
        articles.append(
            NewsArticle(
                title=title,
                published_at=published_at,
                source=item.get("source"),
                url=item.get("url"),
                summary=title,
                sentiment=analyzer.polarity_scores(title)["compound"],
                provider="gdelt",
            )
        )
    return articles


def _press_release_bodies(filing):
    bodies = []
    try:
        for attachment in filing.attachments:
            doc_type = str(getattr(attachment, "document_type", "") or "")
            if not doc_type.upper().startswith("EX-99"):
                continue
            text = attachment.text()
            if text and text.strip():
                bodies.append(re.sub(r"\s+", " ", text).strip())
    except Exception as exc:
        logger.warning(
            "Failed to read 8-K exhibits for %s: %s",
            getattr(filing, "accession_no", "?"),
            exc,
        )
    return bodies


def _articles_from_8k_filings(filings, cutoff_date, lookback_days, max_articles):
    earliest = (pd.Timestamp(cutoff_date) - pd.Timedelta(days=lookback_days)).strftime(
        "%Y-%m-%d"
    )

    articles = []
    for filing in filings:
        if len(articles) >= max_articles:
            break
        filing_date = str(getattr(filing, "filing_date", "") or "")
        if not filing_date or filing_date > cutoff_date:
            continue
        if filing_date < earliest:
            break
        for body in _press_release_bodies(filing):
            if len(articles) >= max_articles:
                break
            cleaned = re.sub(r"^exhibit\s+99[.\d]*\s*", "", body, flags=re.IGNORECASE)

            title = cleaned[:180].strip()
            summary = _safe_summary(cleaned)
            articles.append(
                NewsArticle(
                    title=title,
                    published_at=filing_date,
                    source="SEC EDGAR 8-K",
                    url=getattr(filing, "filing_url", None) or None,
                    summary=summary,
                    sentiment=analyzer.polarity_scores(summary or title)["compound"],
                    provider="edgar_8k",
                )
            )
    return articles


def _fetch_news_edgar_8k(company, cutoff_date, lookback_days, max_articles):
    if company is None:
        return []
    try:
        filings = company.get_filings(form="8-K")
    except Exception as exc:
        logger.warning("Failed to list 8-K filings: %s", exc)
        return []
    return _articles_from_8k_filings(filings, cutoff_date, lookback_days, max_articles)


def _chunk_text(text, form_type, chunk_chars):
    cleaned = re.sub(r"\s+", " ", text).strip()

    if not cleaned:
        return []
    chunks = []
    for chunk_number, start in enumerate(range(0, len(cleaned), chunk_chars), start=1):
        chunk = cleaned[start : start + chunk_chars]
        chunks.append(
            FilingChunk(
                form_type=form_type,
                chunk_id=f"{form_type.lower()}-{chunk_number}",
                text=chunk,
                source_label=form_type,
            )
        )
    return chunks


def _extract_filing_text(filing):
    for attr in ("text", "markdown", "html"):
        candidate = getattr(filing, attr, None)
        if callable(candidate):
            try:
                result = candidate()
                if result:
                    return str(result)
            except Exception:
                continue
        elif candidate:
            return str(candidate)
    return str(filing)


def _edgar_company(ticker):
    email = os.environ.get("EMAIL")
    if not email:
        logger.warning("EMAIL not set, skipping SEC EDGAR access for %s", ticker)
        return None
    try:
        set_identity(email)
        company = Company(ticker)

        return company if company.is_company else None
    except Exception as exc:
        logger.warning("Failed to resolve EDGAR company for %s: %s", ticker, exc)
        return None


def _fetch_filings(company, ticker, chunk_chars, cutoff_date):
    if company is None:
        return [], []
    chunks = []
    selected = []
    for form_type in ("10-K", "10-Q"):
        try:
            filing = None
            for candidate in company.get_filings(form=form_type):
                filing_date = getattr(candidate, "filing_date", None)
                if filing_date is not None and str(filing_date) <= cutoff_date:
                    filing = candidate
                    break
            if filing is None:
                continue
            logger.info(
                "%s %s as of %s: using filing dated %s",
                ticker,
                form_type,
                cutoff_date,
                getattr(filing, "filing_date", "?"),
            )
            chunks.extend(
                _chunk_text(_extract_filing_text(filing), form_type, chunk_chars)
            )
            selected.append(
                {
                    "form_type": form_type,
                    "filing_date": str(getattr(filing, "filing_date", "")),
                    "accession_no": str(getattr(filing, "accession_no", "")),
                }
            )
        except Exception as exc:
            logger.warning("Failed to fetch %s for %s: %s", form_type, ticker, exc)
    return chunks, selected


def _fetch_point_in_time_news(ticker, company, config, ticker_obj):
    cutoff = config.information_cutoff_date
    lookback = config.news_lookback_days
    limit = config.max_news_articles
    if config.news_source == "yahoo":
        return (
            _extract_news_articles(
                ticker_obj, limit, provider="yahoo", cutoff_date=cutoff
            ),
            "yahoo",
        )
    if config.news_source == "edgar_8k":
        return _fetch_news_edgar_8k(company, cutoff, lookback, limit), "edgar_8k"
    # Start with an empty list.
    articles = []

    try:
        company_name = getattr(company, "name", None) or ticker
        articles = _fetch_news_gdelt(ticker, company_name, cutoff, lookback, limit)
    except Exception as exc:
        logger.warning("GDELT news fetch failed for %s: %s", ticker, exc)
    if articles:
        return articles, "gdelt"
    fallback = _fetch_news_edgar_8k(company, cutoff, lookback, limit)
    if fallback:
        logger.info(
            "%s: GDELT empty, using %d EDGAR 8-K disclosures", ticker, len(fallback)
        )
    return fallback, "edgar_8k_fallback"


def _fetch_stock_snapshot(ticker, config):
    ticker_obj = yf.Ticker(_yf_symbol(ticker))
    pre_start = (
        pd.Timestamp(config.information_cutoff_date)
        - pd.DateOffset(months=config.valuation_window_months)
    ).strftime("%Y-%m-%d")
    eval_end = (
        pd.Timestamp(config.portfolio_end_date) + pd.DateOffset(days=1)
    ).strftime("%Y-%m-%d")
    pre_history = ticker_obj.history(
        start=pre_start,
        end=config.information_cutoff_date,
        interval="1d",
        auto_adjust=False,
    )

    eval_history = ticker_obj.history(
        start=config.portfolio_start_date,
        end=eval_end,
        interval="1d",
        auto_adjust=False,
    )
    company = _edgar_company(ticker)

    news, news_provider = _fetch_point_in_time_news(ticker, company, config, ticker_obj)
    filings, filings_selected = _fetch_filings(
        company,
        ticker,
        chunk_chars=config.filing_chunk_chars,
        cutoff_date=config.information_cutoff_date,
    )
    return StockSnapshot(
        ticker=ticker,
        news_source=config.news_source,
        pre_window_prices=_history_to_price_bars(pre_history),
        evaluation_prices=_history_to_price_bars(eval_history),
        news=news,
        filings=filings,
        analyst_price_targets={},
        metadata={
            "news_provider_used": news_provider,
            "news_window": {
                "earliest": (
                    pd.Timestamp(config.information_cutoff_date)
                    - pd.Timedelta(days=config.news_lookback_days)
                ).strftime("%Y-%m-%d"),
                "cutoff": config.information_cutoff_date,
            },
            "news_cutoff_filtered": True,
            "analyst_targets_excluded": (
                "no as-of price-target history available without a licensed feed"
            ),
            "filings_available": bool(filings),
            "filings_as_of": config.information_cutoff_date,
            "filings_selected": filings_selected,
            "pre_window_rows": len(pre_history),
            "evaluation_rows": len(eval_history),
        },
    )


def compute_manifest_hash(manifest):
    payload = to_primitive(manifest)
    payload["manifest_hash"] = None

    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def freeze_experiment_dataset(config):
    stocks = {
        ticker: _fetch_stock_snapshot(ticker, config) for ticker in config.universe
    }

    unusable = [
        ticker
        for ticker, snapshot in stocks.items()
        if len(snapshot.evaluation_prices) < 2 or len(snapshot.pre_window_prices) < 2
    ]
    if unusable:
        raise MissingPriceDataError(
            unusable,
            f"window {config.portfolio_start_date}..{config.portfolio_end_date}, "
            f"cutoff {config.information_cutoff_date}",
        )
    annual_rf = config.risk_free_rate_annual
    max_eval_len = max(
        (len(snapshot.evaluation_prices) for snapshot in stocks.values()), default=0
    )
    daily_rf = annual_rf / 252
    manifest = ExperimentManifest(
        manifest_version="1.0",
        generated_at=utc_now(),
        information_cutoff_date=config.information_cutoff_date,
        portfolio_start_date=config.portfolio_start_date,
        portfolio_end_date=config.portfolio_end_date,
        universe=config.universe,
        benchmark_universe=config.benchmark_universe or list(config.universe),
        risk_free_rate_series=[daily_rf] * max_eval_len,
        stocks=stocks,
        metadata={
            "news_source": config.news_source,
            "tooling_mode": config.tooling_mode,
            "summarizer_model": os.environ.get("SUMMARIZER_MODEL")
            or os.environ.get("MODEL"),
            "summarizer_provider": (
                os.environ.get("SUMMARIZER_PROVIDER") or os.environ.get("PROVIDER")
            ),
            "data_substitutions": {
                "news": (
                    "Point-in-time substitution for the paper's Bloomberg feed: "
                    "GDELT archive headlines (falling back to SEC EDGAR 8-K "
                    "press-release bodies) restricted to published_at within "
                    "[cutoff - news_lookback_days, information_cutoff_date]. "
                    "Per-stock provider recorded as metadata.news_provider_used. "
                    "Analyst price targets remain excluded (no as-of history)."
                ),
                "filings_rag": (
                    "Keyword overlap over chunked text used in place of embedded vector "
                    "store (paper uses GPT-4o embeddings)."
                ),
                "summarizer": (
                    "LLM reflection-summarization (Together-hosted) used in place of "
                    "paper's GPT-4o summarizer."
                ),
                "primary_llm": (
                    "Together-hosted open model used in place of the paper's GPT-4o."
                ),
            },
        },
    )
    manifest.manifest_hash = compute_manifest_hash(manifest)
    return manifest


def save_manifest(manifest, path):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(to_primitive(manifest), indent=2), encoding="utf-8")
    return output


def load_manifest(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    stocks = {}
    for ticker, snapshot in payload["stocks"].items():
        stocks[ticker] = StockSnapshot(
            ticker=snapshot["ticker"],
            news_source=snapshot["news_source"],
            pre_window_prices=[
                PriceBar(**bar) for bar in snapshot["pre_window_prices"]
            ],
            evaluation_prices=[
                PriceBar(**bar) for bar in snapshot["evaluation_prices"]
            ],
            news=[NewsArticle(**article) for article in snapshot.get("news", [])],
            filings=[FilingChunk(**chunk) for chunk in snapshot.get("filings", [])],
            analyst_price_targets=snapshot.get("analyst_price_targets", {}),
            metadata=snapshot.get("metadata", {}),
        )
    manifest = ExperimentManifest(
        manifest_version=payload["manifest_version"],
        generated_at=payload["generated_at"],
        information_cutoff_date=payload["information_cutoff_date"],
        portfolio_start_date=payload["portfolio_start_date"],
        portfolio_end_date=payload["portfolio_end_date"],
        universe=payload["universe"],
        benchmark_universe=payload["benchmark_universe"],
        risk_free_rate_series=payload["risk_free_rate_series"],
        stocks=stocks,
        metadata=payload.get("metadata", {}),
        manifest_hash=payload.get("manifest_hash"),
    )

    computed = compute_manifest_hash(manifest)
    if manifest.manifest_hash != computed:
        logger.warning(
            "Manifest hash mismatch for %s: stored=%s computed=%s",
            path,
            manifest.manifest_hash,
            computed,
        )
    return manifest
