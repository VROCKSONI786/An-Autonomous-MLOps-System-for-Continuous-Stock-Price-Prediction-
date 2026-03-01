"""
sentiment_analyzer.py
─────────────────────
Uses the NEW google-genai SDK exclusively (pip install -U google-genai).
Do NOT use google-generativeai — that package is deprecated and
will no longer receive updates or bug fixes.

Migration reference: https://ai.google.dev/gemini-api/docs/migrate

Key improvements over old code:
  - Pydantic schema → response.parsed (no manual JSON parsing, no fence stripping)
  - client.models.generate_content() — new centralized client pattern
  - Proper 429 handling: reads retry_delay seconds from error, waits, retries
  - Enforces 5s between requests (free tier: 15 RPM limit)
  - gemini-1.5-flash first: 1500 req/day free (gemini-2.0-flash = 0 in many regions)
"""

import os
import re
import time
import json
import random
import logging
import pandas as pd
from html.parser import HTMLParser
from typing import Optional
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── HTML cleaner ──────────────────────────────────────────────────────────────

class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts = []

    def handle_data(self, data):
        self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts).strip()


def _strip_html(raw: str) -> str:
    """Remove all HTML tags and collapse whitespace."""
    if not raw or not isinstance(raw, str):
        return ""
    stripper = _HTMLStripper()
    try:
        stripper.feed(raw)
        text = stripper.get_text()
    except Exception:
        text = re.sub(r"<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", text).strip()


def _parse_retry_seconds(error_str: str) -> int:
    """Extract retry_delay.seconds from a 429 error message body."""
    match = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", error_str)
    return int(match.group(1)) + 5 if match else 65


# ── Pydantic schema for structured output ─────────────────────────────────────
# The new SDK's response.parsed automatically deserialises JSON into this class.
# No manual json.loads(), no ```json fence stripping needed.

class SentimentResult(BaseModel):
    sentiment: str          # "positive" | "negative" | "neutral"
    confidence: float       # 0.0 – 1.0
    impact: str             # "high" | "medium" | "low"
    sector_impact: str      # e.g. "Banking", "IT", "general market"
    key_themes: list[str]   # ["earnings", "buyback", ...]
    reasoning: str          # one sentence

    @field_validator("sentiment")
    @classmethod
    def validate_sentiment(cls, v):
        v = v.lower()
        return v if v in ("positive", "negative", "neutral") else "neutral"

    @field_validator("impact")
    @classmethod
    def validate_impact(cls, v):
        v = v.lower()
        return v if v in ("high", "medium", "low") else "low"

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v):
        return max(0.0, min(1.0, float(v)))

    def to_dict(self) -> dict:
        return self.model_dump()


_NEUTRAL_RESULT = SentimentResult(
    sentiment="neutral",
    confidence=0.0,
    impact="low",
    sector_impact="unknown",
    key_themes=[],
    reasoning="analysis unavailable",
)

# Free-tier quota (requests/day, RPM):
#   gemini-1.5-flash      → 1500 req/day, 15 RPM   ← BEST free option, START HERE
#   gemini-1.5-flash-8b   → 1500 req/day, 15 RPM
#   gemini-2.0-flash-lite → varies by region (often 0 free)
#   gemini-2.0-flash      → 0 free in most regions (DO NOT USE)
_MODEL_PRIORITY = [
    "gemini-1.5-flash",      # Primary: guaranteed 1500 req/day free
    "gemini-1.5-flash-8b",   # Secondary: also 1500 req/day free
    # Commented out 2.0 models—they have 0 free quota in most regions and cause 429 errors
    # "gemini-2.0-flash-lite",
    # "gemini-2.0-flash",
]


# ── SentimentAnalyzer ─────────────────────────────────────────────────────────

class SentimentAnalyzer:
    """
    Analyze financial news sentiment using the new Google GenAI SDK.

    Rate-limit policy (free tier)
    ─────────────────────────────
    • min_delay_seconds=5.0  →  12 req/min  (safely under 15 RPM limit)
    • On 429: reads retry_delay from error body, waits that + jitter, retries
    • max_retries=2: after 2 retries gives up and returns neutral result
    • max_articles=15: conservative daily limit (1500 ÷ 100 runs ÷ safety margin)
    """

    def __init__(self, min_delay_seconds: float = 5.0, max_retries: int = 2):
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "No API key found. Set GEMINI_API_KEY in your .env file."
            )

        self.min_delay   = min_delay_seconds
        self.max_retries = max_retries

        # ── New SDK client (the ONLY correct way as of 2025) ─────────────────
        # Reference: https://ai.google.dev/gemini-api/docs/migrate#api-access
        try:
            from google import genai
            self._client     = genai.Client(api_key=api_key)
            self._model_name = self._pick_model()
            logger.info(f"google-genai SDK ready | model: {self._model_name}")
        except ImportError:
            raise ImportError(
                "google-genai package not installed.\n"
                "Run: pip install -U google-genai\n"
                "Do NOT use google-generativeai — it is deprecated."
            )

    # ── Model selection ───────────────────────────────────────────────────────

    def _pick_model(self) -> str:
        """Return the first available model from priority list (with models/ prefix)."""
        try:
            available = {m.name for m in self._client.models.list()}
            for candidate in _MODEL_PRIORITY:
                # Check both with and without prefix; return with prefix
                if candidate in available or f"models/{candidate}" in available:
                    model_with_prefix = f"models/{candidate}" if not candidate.startswith("models/") else candidate
                    logger.info(f"Selected model: {model_with_prefix}")
                    return model_with_prefix
        except Exception as e:
            logger.warning(f"Model listing failed: {e}. Defaulting to models/gemini-1.5-flash")
        return "models/gemini-1.5-flash"

    # ── Core generate with structured output + retry ──────────────────────────

    def _generate_structured(self, prompt: str) -> Optional[SentimentResult]:
        """
        Call the API requesting JSON parsed directly into SentimentResult.
        Uses response_mime_type + response_schema → response.parsed.
        No manual JSON parsing needed — the SDK handles it.
        Reference: https://ai.google.dev/gemini-api/docs/migrate#json-response
        """
        from google.genai import types

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SentimentResult,
            temperature=0.1,        # low temp → consistent structured output
            max_output_tokens=300,
        )

        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=prompt,
                    config=config,
                )
                # response.parsed is auto-deserialised SentimentResult instance
                if response.parsed is not None:
                    return response.parsed

                # Fallback: manual parse if .parsed is None for some reason
                raw = response.text or ""
                raw = re.sub(r"^```(?:json)?", "", raw.strip(), flags=re.IGNORECASE).strip()
                raw = re.sub(r"```$", "", raw).strip()
                return SentimentResult.model_validate_json(raw)

            except Exception as e:
                err = str(e)
                # Only treat as rate limit if it's actually a 429/rate issue, NOT 404
                is_rate_limit = any(
                    kw in err.lower()
                    for kw in ("429", "quota", "rate", "exceeded", "resource_exhausted")
                ) and "404" not in err
                
                # Log 404 as a configuration problem
                if "404" in err:
                    logger.error(f"Model endpoint 404 Not Found (configuration issue): {err[:200]}")
                    return None

                # If rate-limited, try to switch to another available model first
                if is_rate_limit and attempt < self.max_retries:
                    try:
                        available = {m.name for m in self._client.models.list()}
                    except Exception:
                        available = set()

                    switched = False
                    for cand in _MODEL_PRIORITY:
                        cand_with_prefix = f"models/{cand}" if not cand.startswith("models/") else cand
                        if cand_with_prefix == self._model_name:
                            continue
                        if cand in available or f"models/{cand}" in available:
                            logger.warning(f"Model {self._model_name} rate-limited; switching to {cand_with_prefix} and retrying.")
                            self._model_name = cand_with_prefix
                            switched = True
                            break

                    if switched:
                        # immediately retry with the new model
                        continue

                    # no alternate model available — fall back to waiting
                    wait = _parse_retry_seconds(err) + random.uniform(2, 8)
                    logger.warning(
                        f"  429 rate limit (attempt {attempt+1}/{self.max_retries+1}). "
                        f"No alternate model found; waiting {wait:.0f}s…"
                    )
                    time.sleep(wait)
                    continue

                # Log and give up after retries exhausted or non-rate-limit error
                if is_rate_limit:
                    logger.error(
                        f"Rate limit persists after {self.max_retries} retries. Returning neutral."
                    )
                else:
                    logger.error(f"API error: {err[:200]}")
                return None

    # ── Single article ────────────────────────────────────────────────────────

    def analyze_article_sentiment(
        self,
        title: str,
        summary: str,
        symbol: Optional[str] = None,
    ) -> dict:
        """Analyze one article. Always returns a dict (never raises)."""
        title   = _strip_html(str(title))   if pd.notna(title)   else ""
        summary = _strip_html(str(summary)) if pd.notna(summary) else ""
        sym_str = str(symbol) if symbol and pd.notna(symbol) else ""

        if not title and not summary:
            return {**_NEUTRAL_RESULT.to_dict(), "reasoning": "empty article"}

        prompt = (
            "You are a financial analyst specializing in Indian equities (NSE/BSE). "
            "Analyze the sentiment of the following news article.\n\n"
            f"Title: {title}\n"
            f"Summary: {summary}\n"
            + (f"Stock symbol: {sym_str}\n" if sym_str else "")
            + "\nClassify sentiment as positive, negative, or neutral. "
            "Be specific about which sector is impacted."
        )

        result = self._generate_structured(prompt)
        if result is None:
            return _NEUTRAL_RESULT.to_dict()
        return result.to_dict()

    # ── Bulk analysis ─────────────────────────────────────────────────────────

    def analyze_bulk_sentiment(
        self,
        news_df: pd.DataFrame,
        max_articles: int = 15,
    ) -> pd.DataFrame:
        """
        Analyze up to max_articles rows with enforced per-request delay.
        Skips rows that already have a valid sentiment value.

        Free tier budget: gemini-1.5-flash allows 1500 req/day.
        At max_articles=15 per run, that allows 100 pipeline runs/day.
        """
        df      = news_df.head(max_articles).copy()
        results = []
        total   = len(df)

        for i, (idx, row) in enumerate(df.iterrows()):
            # Skip already-analyzed rows
            existing = row.get("sentiment", None)
            if isinstance(existing, str) and existing in ("positive", "negative", "neutral"):
                logger.info(f"[{i+1}/{total}] Already analyzed — skipping.")
                results.append(row.to_dict())
                continue

            title_preview = str(row.get("title", ""))[:60]
            logger.info(f"[{i+1}/{total}] {title_preview}…")

            t_start = time.monotonic()

            sentiment = self.analyze_article_sentiment(
                title=row.get("title", ""),
                summary=row.get("summary", ""),
                symbol=row.get("symbol", None),
            )
            results.append({**row.to_dict(), **sentiment})

            # Enforce per-request delay to stay within free RPM limit
            elapsed   = time.monotonic() - t_start
            remaining = self.min_delay - elapsed
            if remaining > 0 and i < total - 1:
                logger.debug(f"  Rate-limit sleep {remaining:.1f}s")
                time.sleep(remaining)

        return pd.DataFrame(results)

    # ── Market summary ────────────────────────────────────────────────────────

    def get_market_sentiment_summary(self, sentiment_df: pd.DataFrame) -> str:
        """
        Generate a plain-text market outlook from aggregated sentiment.
        Falls back to a rule-based summary if API call fails.
        """
        if sentiment_df.empty or "sentiment" not in sentiment_df.columns:
            return "Insufficient data for market summary."

        counts   = sentiment_df["sentiment"].value_counts().to_dict()
        avg_conf = round(sentiment_df["confidence"].mean(), 2) if "confidence" in sentiment_df.columns else "N/A"
        top_secs = (
            sentiment_df["sector_impact"].value_counts().head(5).to_dict()
            if "sector_impact" in sentiment_df.columns else {}
        )

        # Rule-based fallback (used if API fails)
        pos = counts.get("positive", 0)
        neg = counts.get("negative", 0)
        neu = counts.get("neutral", 0)
        total = pos + neg + neu or 1
        dominant = max(counts, key=counts.get) if counts else "neutral"
        fallback = (
            f"Market sentiment is predominantly {dominant} "
            f"({pos} positive, {neg} negative, {neu} neutral out of {total} articles). "
            f"Average confidence: {avg_conf}. "
            f"Top impacted sectors: {', '.join(list(top_secs.keys())[:3]) or 'N/A'}."
        )

        prompt = (
            "Based on this Indian equity market sentiment data, write a concise "
            "3-sentence market outlook for retail investors. Be factual and specific.\n\n"
            f"Sentiment counts: {counts}\n"
            f"Average analyst confidence: {avg_conf}\n"
            f"Most impacted sectors: {top_secs}\n"
        )

        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Market summary API error: {e}")
            return fallback


# ── CLI smoke-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing SentimentAnalyzer with new google-genai SDK…\n")

    analyzer = SentimentAnalyzer(min_delay_seconds=5.0)

    test_articles = [
        {
            "title":   "Buy HDFC Bank; target of Rs 1,850: ICICI Securities",
            "summary": "ICICI Securities is bullish on HDFC Bank and has recommended "
                       "a buy rating with a target price of Rs 1,850 dated April 21, 2024.",
            "symbol":  "HDFCBANK.NS",
        },
        {
            "title":   "Infosys Q4 results: Net profit falls 12% YoY",
            "summary": "Infosys reported a net profit of Rs 6,128 crore for Q4 FY24, "
                       "down 12% year-on-year, missing analyst estimates.",
            "symbol":  "INFY.NS",
        },
    ]

    for art in test_articles:
        print(f"Article: {art['title']}")
        result = analyzer.analyze_article_sentiment(**art)
        print(json.dumps(result, indent=2))
        print()