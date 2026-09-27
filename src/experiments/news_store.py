import os
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from src.util.log_config import setup_logging

logger = setup_logging("experiments.news_store")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    ticker TEXT NOT NULL,
    published_at TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT,
    source TEXT,
    provider TEXT NOT NULL,
    PRIMARY KEY (ticker, published_at, title)
);
CREATE TABLE IF NOT EXISTS coverage (
    ticker TEXT NOT NULL,
    month TEXT NOT NULL,
    provider TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (ticker, month, provider)
);
CREATE INDEX IF NOT EXISTS idx_articles_window ON articles (ticker, published_at);
"""


def default_store_path():
    env = os.environ.get("NEWS_STORE_PATH")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "experiments" / "news_store.sqlite3"


def months_for_window(start, end):
    """List the months between the start and end dates."""
    start_date, end_date = date.fromisoformat(start[:10]), date.fromisoformat(end[:10])
    if start_date > end_date:
        raise ValueError(f"start {start} after end {end}")
    months = []
    cursor = date(start_date.year, start_date.month, 1)
    while cursor <= end_date:
        months.append(cursor.strftime("%Y-%m"))
        cursor = date(cursor.year + (cursor.month == 12), cursor.month % 12 + 1, 1)

    return months


def month_bounds(month):
    """Get the first and last date of a month."""
    first = date.fromisoformat(f"{month}-01")

    next_first = date(first.year + (first.month == 12), first.month % 12 + 1, 1)
    return first.isoformat(), (next_first - timedelta(days=1)).isoformat()


class NewsStore:
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else default_store_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self):
        conn = sqlite3.connect(self.path)

        conn.executescript(_SCHEMA)
        return conn

    def has_month(self, ticker, month, provider="gdelt"):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM coverage WHERE ticker=? AND month=? AND provider=?",
                (ticker, month, provider),
            ).fetchone()
        return row is not None

    def ingest_month(
        self, ticker, month, articles, provider="gdelt", mark_covered=True
    ):
        """Save one month of articles and skip duplicates."""
        from .schemas import utc_now

        with self._connect() as conn:
            before = conn.total_changes
            conn.executemany(
                "INSERT OR IGNORE INTO articles "
                "(ticker, published_at, title, url, source, provider) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        ticker,
                        str(item["published_at"]),
                        str(item["title"]),
                        item.get("url"),
                        item.get("source"),
                        provider,
                    )
                    for item in articles
                    if item.get("title") and item.get("published_at")
                ],
            )
            inserted = conn.total_changes - before
            if mark_covered:
                conn.execute(
                    "INSERT OR REPLACE INTO coverage "
                    "(ticker, month, provider, fetched_at) VALUES (?, ?, ?, ?)",
                    (ticker, month, provider, utc_now()),
                )
        return inserted

    def query(self, ticker, start, end, limit=None):
        """Get articles between two dates, newest first."""
        sql = (
            "SELECT published_at, title, url, source, provider FROM articles "
            "WHERE ticker=? "
            "AND substr(published_at, 1, 10) >= ? "
            "AND substr(published_at, 1, 10) <= ? "
            "ORDER BY published_at DESC"
        )
        params = [ticker, start[:10], end[:10]]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            {
                "published_at": database_row[0],
                "title": database_row[1],
                "url": database_row[2],
                "source": database_row[3],
                "provider": database_row[4],
            }
            for database_row in rows
        ]

    def ensure_window(self, ticker, start, end, fetch_month, today=None):
        """Fetch articles for months that are still missing."""
        today = today or date.today().isoformat()
        for month in months_for_window(start, end):
            if self.has_month(ticker, month):
                continue
            _, month_end = month_bounds(month)
            articles = fetch_month(month)
            complete = month_end < today
            inserted = self.ingest_month(ticker, month, articles, mark_covered=complete)
            logger.info(
                "News store: %s %s -> %d new articles%s",
                ticker,
                month,
                inserted,
                "" if complete else " (month incomplete, not marked covered)",
            )
