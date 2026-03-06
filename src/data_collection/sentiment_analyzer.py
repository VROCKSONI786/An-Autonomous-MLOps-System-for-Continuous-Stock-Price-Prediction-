import os
import re
import time
import logging
import random
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
    if not raw or not isinstance(raw, str):
        return ""
    stripper = _HTMLStripper()
    try:
        stripper.feed(raw)
        text = stripper.get_text()
    except Exception:
        text = re.sub(r"<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", text).strip()


# ── Pydantic schema ───────────────────────────────────────────────────────────

class SentimentResult(BaseModel):
    sentiment:     str        # "positive" | "negative" | "neutral"
    confidence:    float      # 0.0 – 1.0
    impact:        str        # "high" | "medium" | "low"
    sector_impact: str        # e.g. "Banking", "IT", "general market"
    key_themes:    list[str]  # ["earnings", "buyback", ...]
    reasoning:     str        # one-sentence explanation

    @field_validator("sentiment")
    @classmethod
    def validate_sentiment(cls, v):
        v = str(v).lower().strip()
        return v if v in ("positive", "negative", "neutral") else "neutral"

    @field_validator("impact")
    @classmethod
    def validate_impact(cls, v):
        v = str(v).lower().strip()
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

# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a financial analyst specializing in Indian equities (NSE/BSE). "
    "Analyze the sentiment of the financial news article provided. "
    "Return ONLY a JSON object matching the required schema — no preamble, no markdown."
)

_HUMAN_TEMPLATE = (
    "Article Title: {title}\n"
    "Article Summary: {summary}\n"
    "{symbol_line}"
    "\nClassify the sentiment as positive, negative, or neutral. "
    "Be specific about which sector is impacted and provide a short reasoning."
)


# ── LLM Factory ──────────────────────────────────────────────────────────────

def _build_llm():
    """
    Try each provider in priority order based on available API keys.
    Returns a LangChain BaseChatModel instance.
    """

    # ── 1. Groq (Free tier — llama-3.1-8b-instant, 6000 tokens/min) ──────────────
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                api_key=groq_key,
                model="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=512,
            )
            logger.info("LLM Provider: Groq — llama-3.1-8b-instant (free tier)")
            return llm
        except ImportError:
            logger.warning("langchain_groq not installed → pip install langchain-groq")
        except Exception as e:
            logger.warning(f"Groq init failed: {e}")

    # ── 3. Google Gemini via LangChain wrapper ────────────────────────────────
    # Probes model names in order; uses the first one that responds successfully.
    # gemini-1.5-flash returns 404 in some regions — fallback list handles this.
    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if google_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import HumanMessage as _HM

            _GOOGLE_MODELS = [
                "gemini-2.0-flash",
                "gemini-2.0-flash-lite",
                "gemini-1.5-flash-latest",
                "gemini-1.5-flash",
                "gemini-1.5-pro",
            ]

            for _model in _GOOGLE_MODELS:
                try:
                    _llm = ChatGoogleGenerativeAI(
                        google_api_key=google_key,
                        model=_model,
                        temperature=0.1,
                        max_output_tokens=512,
                        convert_system_message_to_human=True,
                    )
                    _llm.invoke([_HM(content="Hi")])   # quick probe
                    logger.info(f"LLM Provider: Google Gemini — {_model} (LangChain wrapper)")
                    return _llm
                except Exception as _probe_err:
                    _msg = str(_probe_err)
                    if "404" in _msg or "not found" in _msg.lower() or "invalid" in _msg.lower():
                        logger.warning(f"Gemini model '{_model}' not available (404) — trying next…")
                        continue
                    logger.warning(f"Gemini '{_model}' non-404 error: {_msg[:120]}")
                    break   # auth/quota error — stop trying Gemini

            logger.warning("No Google Gemini model available in this region/account.")
        except ImportError:
            logger.warning("langchain_google_genai not installed → pip install langchain-google-genai")
        except Exception as e:
            logger.warning(f"Google Gemini init failed: {e}")

    raise ValueError(
        "\n\nNo LLM provider configured. Add ONE of these to your .env file:\n\n"
        "  GROQ_API_KEY=...       # FREE — https://console.groq.com\n"
        "  OPENAI_API_KEY=...     # Paid — https://platform.openai.com\n"
        "  GOOGLE_API_KEY=...     # Free tier — https://aistudio.google.com\n"
        "  ANTHROPIC_API_KEY=...  # Paid — https://console.anthropic.com\n"
    )


# ── SentimentAnalyzer ─────────────────────────────────────────────────────────

class SentimentAnalyzer:
    """
    Analyze financial news sentiment using LangChain structured output.

    The .with_structured_output(SentimentResult) binding enforces the Pydantic
    schema at the LLM provider level — no manual JSON parsing required.
    Provider selection is automatic based on available API keys in .env.
    """

    def __init__(self, min_delay_seconds: float = 2.0, max_retries: int = 3):
        self.min_delay   = min_delay_seconds
        self.max_retries = max_retries

        self._base_llm = _build_llm()
        # .with_structured_output → returns SentimentResult instances directly
        self._chain = self._base_llm.with_structured_output(SentimentResult)
        logger.info("SentimentAnalyzer ready (LangChain structured output chain).")

    # ── Single article ────────────────────────────────────────────────────────

    def analyze_article_sentiment(
        self,
        title:   str,
        summary: str,
        symbol:  Optional[str] = None,
    ) -> dict:
        """Analyze one article. Always returns a dict (never raises)."""
        title   = _strip_html(str(title))   if pd.notna(title)   else ""
        summary = _strip_html(str(summary)) if pd.notna(summary) else ""

        if not title and not summary:
            return {**_NEUTRAL_RESULT.to_dict(), "reasoning": "empty article"}

        sym_line = f"Stock Symbol: {symbol}\n" if symbol and pd.notna(symbol) else ""

        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=_HUMAN_TEMPLATE.format(
                title=title,
                summary=summary[:800],   # cap length to save tokens
                symbol_line=sym_line,
            )),
        ]

        for attempt in range(self.max_retries + 1):
            try:
                result: SentimentResult = self._chain.invoke(messages)
                return result.to_dict()

            except Exception as e:
                err = str(e)

                # 404 = wrong model name / endpoint — no point retrying
                if "404" in err or "not found" in err.lower():
                    logger.error("Model endpoint 404 — check model name/region. Returning neutral.")
                    return _NEUTRAL_RESULT.to_dict()

                is_rate_limit = any(
                    k in err.lower()
                    for k in ("429", "rate", "quota", "exceeded", "resource_exhausted")
                )

                if is_rate_limit and attempt < self.max_retries:
                    wait = 60 + random.uniform(5, 15)
                    logger.warning(
                        f"Rate limit (attempt {attempt+1}/{self.max_retries+1}). "
                        f"Waiting {wait:.0f}s…"
                    )
                    time.sleep(wait)
                    continue

                logger.error(f"Sentiment error (attempt {attempt+1}): {err[:200]}")
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)   # exponential back-off
                    continue
                break

        return _NEUTRAL_RESULT.to_dict()

    # ── Bulk analysis ─────────────────────────────────────────────────────────

    def analyze_bulk_sentiment(
        self,
        news_df:      pd.DataFrame,
        max_articles: int = 15,
    ) -> pd.DataFrame:
        """
        Analyze up to max_articles rows.
        Skips rows that already have a valid sentiment value.
        Enforces min_delay_seconds between requests to respect rate limits.
        """
        df      = news_df.head(max_articles).copy()
        results = []
        total   = len(df)

        for i, (idx, row) in enumerate(df.iterrows()):
            existing = row.get("sentiment", None)
            if isinstance(existing, str) and existing in ("positive", "negative", "neutral"):
                logger.info(f"[{i+1}/{total}] Already analyzed — skipping.")
                results.append(row.to_dict())
                continue

            title_preview = str(row.get("title", ""))[:60]
            logger.info(f"[{i+1}/{total}] Analyzing: {title_preview}…")

            t_start   = time.monotonic()
            sentiment = self.analyze_article_sentiment(
                title=row.get("title",   ""),
                summary=row.get("summary", ""),
                symbol=row.get("symbol",  None),
            )
            results.append({**row.to_dict(), **sentiment})

            elapsed   = time.monotonic() - t_start
            remaining = self.min_delay - elapsed
            if remaining > 0 and i < total - 1:
                time.sleep(remaining)

        return pd.DataFrame(results)

    # ── Market summary ────────────────────────────────────────────────────────

    def get_market_sentiment_summary(self, sentiment_df: pd.DataFrame) -> str:
        """
        Generate a plain-text market outlook from aggregated sentiment.
        Falls back to rule-based summary if the API call fails.
        """
        if sentiment_df.empty or "sentiment" not in sentiment_df.columns:
            return "Insufficient data for market summary."

        counts   = sentiment_df["sentiment"].value_counts().to_dict()
        avg_conf = (
            round(sentiment_df["confidence"].mean(), 2)
            if "confidence" in sentiment_df.columns else "N/A"
        )
        top_secs = (
            sentiment_df["sector_impact"].value_counts().head(5).to_dict()
            if "sector_impact" in sentiment_df.columns else {}
        )

        pos = counts.get("positive", 0)
        neg = counts.get("negative", 0)
        neu = counts.get("neutral",  0)
        total    = pos + neg + neu or 1
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
            from langchain_core.messages import HumanMessage
            response = self._base_llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            logger.error(f"Market summary API error: {e}")
            return fallback


# ── CLI smoke-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    print("Testing SentimentAnalyzer with LangChain…\n")

    analyzer = SentimentAnalyzer(min_delay_seconds=1.0)

    test_articles = [
        {
            "title":   "Buy HDFC Bank; target of Rs 1,850: ICICI Securities",
            "summary": "ICICI Securities is bullish on HDFC Bank and has recommended "
                       "a buy rating with a target price of Rs 1,850.",
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