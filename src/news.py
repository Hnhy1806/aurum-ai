"""
News collection for AurumAI.

Sources:
- Finnhub market news (main live source): free key, real-time financial headlines.
  No keyword search, so we filter locally with the Phase 6 classifier patterns.
- Federal Reserve RSS feeds: official monetary-policy press releases and speeches.
- GDELT DOC 2.0 API (optional, for slow history backfill only): free, no key,
  ~3-month searchable window, but heavily rate-limited in practice.

Pattern (same as market data): fetch -> standardize -> validate.
"""
from __future__ import annotations

import hashlib
import html
import logging
import os
import random
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
import requests

from src.classify import match_categories

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv not installed; rely on real environment variables
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_MIN_INTERVAL = 10.0  # seconds after the previous response; GDELT asks for >= 5s, but enforces it loosely
GDELT_BACKOFF_BASE = 15.0  # first wait after a 429; doubles each retry (15, 30, 60)
GDELT_MAX_RECORDS = 250    # hard cap per request
REQUEST_TIMEOUT = 30
USER_AGENT = "AurumAI/0.1 (personal research project)"

FINNHUB_URL = "https://finnhub.io/api/v1/news"
FINNHUB_CATEGORIES = ["general", "forex"]

EXCLUDE_PATTERN = re.compile(r"gold IRA|IRA rollover", re.IGNORECASE)  # promotional spam

# Search themes. The key is stored in the `query` column. It records WHY we
# fetched an article, not what the article is about (classification is Phase 6).
# GDELT rules: OR groups can't be nested; very short words (e.g. "Fed", "CPI")
# are rejected, so use longer phrases.
GDELT_QUERIES: dict[str, str] = {
    "gold":         '("gold price" OR "gold prices" OR bullion) -IRA',
    "central_bank": '("central bank gold" OR "gold reserves" OR "gold purchases")',
    "fed":          '("Federal Reserve" OR FOMC OR Powell)',
    "inflation":    '("consumer price index" OR inflation OR "PCE price")',
    "labor":        '("nonfarm payrolls" OR "jobs report" OR "unemployment rate")',
    "usd":          '("dollar index" OR "US dollar" OR greenback)',
    "yields":       '("Treasury yields" OR "bond yields" OR "10-year yield")',
    "oil":          '("oil prices" OR "crude oil" OR OPEC)',
    "geopolitics":  '(geopolitical OR "military strike" OR sanctions)',
    "macro_risk":   '("recession fears" OR "financial crisis" OR "market selloff")',
}

FED_FEEDS: dict[str, str] = {
    "fed_monetary_policy": "https://www.federalreserve.gov/feeds/press_monetary.xml",
    "fed_speeches":        "https://www.federalreserve.gov/feeds/speeches.xml",
}

NEWS_COLUMNS = [
    "article_id", "published_at", "fetched_at", "provider", "source",
    "headline", "description", "url", "query",
]

FUTURE_TOLERANCE = timedelta(minutes=5)  # small clock-skew allowance
MIN_HEADLINE_KEY_LEN = 25                # don't merge short generic titles
SYNDICATION_WINDOW = timedelta(hours=48) # same title within 48h = same story


# ---------------------------------------------------------------------------
# IDs and URL normalization
# ---------------------------------------------------------------------------
def normalize_url(url: str) -> str:
    """Lowercase the host, drop fragments and utm_* tracking params, drop the trailing slash."""
    parts = urlsplit(url.strip())
    query = [(k, v) for k, v in parse_qsl(parts.query) if not k.lower().startswith("utm_")]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, urlencode(query), ""))


def make_article_id(url: str) -> str:
    """Stable ID: the same URL always gives the same ID, so re-fetching is idempotent."""
    return hashlib.sha256(normalize_url(url).encode("utf-8")).hexdigest()[:16]


def _headline_key(headline: str) -> str:
    """Normalized headline used to spot the same wire story on different outlets."""
    key = re.sub(r"[^a-z0-9 ]", "", headline.lower())
    return re.sub(r"\s+", " ", key).strip()


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
_last_gdelt_call = 0.0


def _throttle_gdelt() -> None:
    """Wait until GDELT_MIN_INTERVAL has passed since the previous GDELT response."""
    wait = GDELT_MIN_INTERVAL - (time.monotonic() - _last_gdelt_call)
    if wait > 0:
        time.sleep(wait)


def _get(url: str, params: dict | None = None, max_retries: int = 3,
         is_gdelt: bool = False, headers: dict | None = None) -> requests.Response:
    """GET with a User-Agent, a timeout, and exponential backoff (with jitter) on 429/5xx."""
    global _last_gdelt_call
    all_headers = {"User-Agent": USER_AGENT, **(headers or {})}
    for attempt in range(max_retries + 1):
        if is_gdelt:
            _throttle_gdelt()
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT,
                                headers=all_headers)
        finally:
            if is_gdelt:
                # Count the gap from when the response ends: GDELT is often slow to answer.
                _last_gdelt_call = time.monotonic()

        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt == max_retries:
                break
            backoff = GDELT_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 3)
            logger.warning("HTTP %s from %s (attempt %d/%d), retrying in %.0fs",
                           resp.status_code, url, attempt + 1, max_retries + 1, backoff)
            time.sleep(backoff)
            continue
        resp.raise_for_status()
        return resp
    resp.raise_for_status()  # out of retries: raise so the caller can log and skip
    return resp


# ---------------------------------------------------------------------------
# Fetchers (return raw-ish frames with the standard column names)
# ---------------------------------------------------------------------------
def fetch_finnhub(category: str = "general", api_key: str | None = None,
                  min_id: int | None = None) -> pd.DataFrame:
    """
    Fetch the latest Finnhub market news for one category ("general", "forex", ...).

    Returns ALL articles (unfiltered); filter_by_theme() keeps the relevant ones.
    min_id: only return articles with a Finnhub id above this (for incremental runs).
    """
    api_key = api_key or os.getenv("FINNHUB_API_KEY")
    if not api_key:
        raise ValueError("FINNHUB_API_KEY is not set. Add it to your .env file.")

    params: dict = {"category": category}
    if min_id is not None:
        params["minId"] = min_id
    # Key goes in a header, not the URL, so it never appears in logs or tracebacks.
    resp = _get(FINNHUB_URL, params=params, headers={"X-Finnhub-Token": api_key})

    items = resp.json()
    if not isinstance(items, list):  # an error comes back as {"error": "..."}
        logger.warning("Finnhub '%s' unexpected response: %s", category, str(items)[:200])
        return pd.DataFrame(columns=NEWS_COLUMNS)
    if not items:
        return pd.DataFrame(columns=NEWS_COLUMNS)

    raw = pd.DataFrame(items)
    published = pd.to_datetime(raw["datetime"], unit="s", utc=True, errors="coerce")
    # How far back this batch reaches. If runs are further apart than this, news is missed.
    span_h = (published.max() - published.min()).total_seconds() / 3600
    logger.info("Finnhub '%s': %d articles covering the last %.1f hours", category, len(raw), span_h)
    return pd.DataFrame({
        # `datetime` = publication time as a UNIX timestamp (seconds, UTC)
        "published_at": published,
        "provider": "finnhub",
        "source": raw.get("source"),
        "headline": raw.get("headline"),
        "description": raw.get("summary"),
        "url": raw.get("url"),
        "query": f"finnhub_{category}",
    })


def filter_by_theme(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep articles that match at least one classifier category (src/classify.py),
    set `query` to the matched categories in lowercase (e.g. "fed,yields"),
    and drop promotional spam (EXCLUDE_PATTERN).
    """
    if df.empty:
        return df
    headlines = df["headline"].fillna("").astype(str)
    descriptions = df["description"].fillna("").astype(str)

    themes = pd.Series(
        [",".join(sorted(c.lower() for c in match_categories(h, d))) for h, d in zip(headlines, descriptions)],
        index=df.index,
    )
    spam = (headlines + " " + descriptions).str.contains(EXCLUDE_PATTERN)
    keep = (themes != "") & ~spam

    out = df[keep].copy()
    out["query"] = themes[keep]
    logger.info("filter_by_theme: kept %d of %d articles", len(out), len(df))
    return out


def fetch_gdelt(query_name: str, query: str, *, timespan: str = "1d",
                start: datetime | None = None, end: datetime | None = None,
                max_records: int = GDELT_MAX_RECORDS) -> pd.DataFrame:
    """
    Fetch English-language articles matching a GDELT query.

    Use `timespan` (e.g. "1d", "12h") for live collection, or start/end (UTC)
    for backfilling. Both must fall within GDELT's ~3-month window.
    """
    params = {
        "query": f"{query} sourcelang:english",
        "mode": "artlist",
        "format": "json",
        "maxrecords": max_records,
        "sort": "datedesc",
    }
    if start is not None:
        params["startdatetime"] = start.astimezone(timezone.utc).strftime("%Y%m%d%H%M%S")
        if end is not None:
            params["enddatetime"] = end.astimezone(timezone.utc).strftime("%Y%m%d%H%M%S")
    else:
        params["timespan"] = timespan

    resp = _get(GDELT_URL, params=params, is_gdelt=True)

    # GDELT reports query errors as plain text with HTTP 200, not as JSON.
    try:
        data = resp.json()
    except ValueError:
        logger.warning("GDELT non-JSON response for '%s': %s", query_name, resp.text[:200])
        return pd.DataFrame(columns=NEWS_COLUMNS)

    articles = data.get("articles", [])
    if len(articles) >= max_records:
        logger.warning("GDELT '%s' hit the %d-record cap; results are truncated. "
                       "Use a shorter window.", query_name, max_records)
    if not articles:
        return pd.DataFrame(columns=NEWS_COLUMNS)

    raw = pd.DataFrame(articles)
    return pd.DataFrame({
        # seendate = when GDELT first saw the article (UTC), e.g. 20261118T143000Z
        "published_at": pd.to_datetime(raw["seendate"], format="%Y%m%dT%H%M%SZ",
                                       utc=True, errors="coerce"),
        "provider": "gdelt",
        "source": raw.get("domain"),
        "headline": raw.get("title"),
        "description": None,  # the GDELT article list has no summary
        "url": raw.get("url"),
        "query": query_name,
    })


def fetch_fed_rss(feed_name: str, url: str) -> pd.DataFrame:
    """Parse a Federal Reserve RSS 2.0 feed (official releases and speeches)."""
    resp = _get(url)
    root = ET.fromstring(resp.content)

    rows = []
    for item in root.iter("item"):
        pub = item.findtext("pubDate")
        rows.append({
            "published_at": parsedate_to_datetime(pub) if pub else None,
            "provider": "fed_rss",
            "source": "federalreserve.gov",
            "headline": item.findtext("title"),
            "description": item.findtext("description"),
            "url": item.findtext("link"),
            "query": feed_name,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    return df


def clean_text(text: str | None) -> str | None:
    """Strip HTML tags and entities, collapse whitespace. Some feeds send HTML summaries."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    text = re.sub(r"<[^>]+>", " ", str(text))   # drop tags like <p>, <li>, <a href=...>
    text = html.unescape(text)                  # &amp; -> &, &#39; -> '
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


# ---------------------------------------------------------------------------
# Standardize
# ---------------------------------------------------------------------------
def standardize_news(df: pd.DataFrame, fetched_at: datetime | None = None) -> pd.DataFrame:
    """Clean, assign IDs, and deduplicate. Returns columns in NEWS_COLUMNS order."""
    fetched_at = fetched_at or datetime.now(timezone.utc)
    if df.empty:
        return pd.DataFrame(columns=NEWS_COLUMNS)

    out = df.copy()
    n_start = len(out)

    # Text cleanup
    out["headline"] = out["headline"].map(clean_text).astype("string")
    out["url"] = out["url"].astype("string").str.strip()
    out["description"] = out["description"].map(clean_text).astype("string")

    # Drop rows we can't use
    bad = out["headline"].isna() | (out["headline"] == "") | out["url"].isna() | (out["url"] == "")
    out["published_at"] = pd.to_datetime(out["published_at"], utc=True, errors="coerce")
    bad |= out["published_at"].isna()
    if bad.any():
        logger.info("Dropping %d rows with missing headline/url/timestamp", int(bad.sum()))
    out = out[~bad]

    out["article_id"] = out["url"].map(make_article_id)
    ts = pd.Timestamp(fetched_at)
    out["fetched_at"] = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    out = out.sort_values("published_at", kind="stable")

    # Dedup 1: same URL found by several queries -> one row, queries merged ("fed,inflation")
    queries = out.groupby("article_id")["query"].agg(
        lambda s: ",".join(sorted({q for val in s for q in str(val).split(",")})))
    out = out.drop_duplicates("article_id", keep="first")
    out["query"] = out["article_id"].map(queries)
    n_after_url = len(out)

    # Dedup 2: same wire story on different outlets -> keep the earliest copy.
    # Only within SYNDICATION_WINDOW: identical titles weeks apart are different events
    # (e.g. the Fed reuses "Federal Reserve issues FOMC statement" every meeting).
    key = out["headline"].map(_headline_key)
    gap = out.groupby(key)["published_at"].diff()        # time since previous same-title row
    is_dup = gap.notna() & (gap <= SYNDICATION_WINDOW) & (key.str.len() >= MIN_HEADLINE_KEY_LEN)
    out = out[~is_dup]

    logger.info("standardize_news: %d raw -> %d after URL dedup -> %d after headline dedup",
                n_start, n_after_url, len(out))
    return out[NEWS_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
def validate_news(df: pd.DataFrame) -> dict[str, int]:
    """Return a count per issue type (all zeros = clean). Raises on structural problems."""
    missing = set(NEWS_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"News frame is missing columns: {sorted(missing)}")
    if df.empty:
        return {}

    if df["published_at"].dt.tz is None:
        raise ValueError("published_at must be timezone-aware UTC")

    issues = {
        "duplicate_article_id": int(df["article_id"].duplicated().sum()),
        "missing_published_at": int(df["published_at"].isna().sum()),
        "empty_headline": int((df["headline"].fillna("").str.strip() == "").sum()),
        "future_published_at": int((df["published_at"] > df["fetched_at"] + FUTURE_TOLERANCE).sum()),
    }
    for name, count in issues.items():
        if count:
            logger.warning("validate_news: %s = %d", name, count)
    return issues


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def collect_news(include_gdelt: bool = False, gdelt_timespan: str = "1d") -> pd.DataFrame:
    """
    Fetch all sources, standardize, and validate. One failing source doesn't stop the run.

    Finnhub + Fed RSS always run. GDELT is opt-in because of its strict rate limiting.
    """
    fetched_at = datetime.now(timezone.utc)
    results: dict[str, pd.DataFrame | None] = {}

    for cat in FINNHUB_CATEGORIES:
        name = f"finnhub_{cat}"
        try:
            results[name] = filter_by_theme(fetch_finnhub(cat))
        except (requests.RequestException, ValueError) as e:
            logger.error("Finnhub '%s' failed: %s", cat, e)
            results[name] = None

    for name, url in FED_FEEDS.items():
        try:
            results[name] = fetch_fed_rss(name, url)
        except (requests.RequestException, ET.ParseError) as e:
            logger.error("Fed feed '%s' failed: %s", name, e)
            results[name] = None

    if include_gdelt:
        for name, q in GDELT_QUERIES.items():
            try:
                results[f"gdelt_{name}"] = fetch_gdelt(name, q, timespan=gdelt_timespan)
            except requests.RequestException as e:
                logger.error("GDELT '%s' failed: %s", name, e)
                results[f"gdelt_{name}"] = None

    failed = [n for n, f in results.items() if f is None]
    frames = [f for f in results.values() if f is not None and not f.empty]
    logger.info("collect_news: %d of %d sources returned articles; failed: %s",
                len(frames), len(results), failed or "none")
    if not frames:
        logger.warning("No news collected from any source")
        return pd.DataFrame(columns=NEWS_COLUMNS)

    news = standardize_news(pd.concat(frames, ignore_index=True), fetched_at=fetched_at)
    validate_news(news)
    return news
