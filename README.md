# 📡 RaniStocks — Social Sentiment Radar (Improved)

Real-time ticker intelligence scraped from Reddit, Stocktwits, Finviz & Yahoo Finance  
with VADER NLP sentiment scoring.

> **This is an improved fork of [Eliahur7/stocks_sentiment](https://github.com/Eliahur7/stocks_sentiment).**  
> See the [Changes](#changes) section for a full diff summary.

---

## What it does

On every manual scan the app:

1. Scrapes 4 sources for the most-mentioned stock tickers
2. Scores sentiment on every post/headline using VADER NLP (bullish / bearish / neutral)
3. Enriches with price data from Yahoo Finance
4. Displays a ranked leaderboard with mention bars, sentiment badges, source breakdown, and sample posts

## Sources

| Source | Method | Coverage |
|--------|--------|----------|
| Reddit | Public `.json` API (no auth needed) | Hot posts from r/wallstreetbets, r/stocks, r/investing, r/options, and more |
| Stocktwits | Public REST API | Trending symbols + message stream |
| Finviz | finvizfinance library | News & blog headlines |
| Yahoo Finance | Public HTML endpoint | Trending tickers page |

No API keys required.

---

## Changes

### 1. Security fix — XSS via `unsafe_allow_html`

**Original problem:** Raw Reddit/Stocktwits post text was injected directly into HTML
strings rendered with `unsafe_allow_html=True`. A post containing `<script>` tags or
HTML markup could break the UI or inject arbitrary markup into the page.

**Fix:** All user-generated text (post previews, ticker symbols, sentiment labels) is now
passed through Python's built-in `html.escape()` before being inserted into any HTML string.

```python
# Before — unsafe
preview = txt[:300].replace("\n", " ").strip()

# After — safe
preview = html.escape(txt[:300].replace("\n", " ").strip())
```

---

### 2. Ticker accuracy — real tickers incorrectly blacklisted

**Original problem:** The `TICKER_BLACKLIST` set contained several actively-traded tickers,
causing them to silently vanish from all scan results regardless of mention volume:

| Ticker | Company | Why it was wrongly blacklisted |
|--------|---------|-------------------------------|
| `AI` | C3.ai | Matched the word "AI" |
| `APP` | AppLovin | Matched the word "APP" |
| `RUN` | Sunrun | Common English word |
| `NOW` | ServiceNow | Common English word |
| `BUY` | — | Useful trading signal word |

**Fix — two-part:**

1. Removed incorrectly blacklisted tickers from the set.
2. Changed `extract_tickers()` to use a **two-tier extraction approach**:
   - `$TICKER` dollar-sign mentions → **high confidence**, bypass blacklist entirely
   - Plain ALL-CAPS words → lower confidence, blacklist still applied

```python
# High-confidence: $TICKER — skip blacklist (preserves $AI, $APP, $NOW, $NET, $RUN)
dollar_tickers = set(re.findall(r'\$([A-Z]{1,5})\b', text_upper))

# Lower-confidence: plain CAPS — apply blacklist
caps_words = set(re.findall(r'\b([A-Z]{2,5})\b', text_upper))
caps_filtered = {t for t in caps_words if t not in TICKER_BLACKLIST}

return list(dollar_tickers | caps_filtered)
```

---

### 3. Error visibility — silent source failures

**Original problem:** All four data-source fetchers used bare `except Exception: pass`
or `except Exception: continue`. Network errors, HTTP 429 rate-limit responses, and API
changes would silently produce empty results with no feedback to the user — the scan would
just return fewer tickers with no indication why.

**Fix:** All exception handlers now call `st.warning(f"Source X failed: {e}")` so
degraded results are visible rather than hidden.

```python
# Before
except Exception:
    pass

# After
except Exception as e:
    st.warning(f"Stocktwits trending symbols failed: {e}")
```

---

### 4. yfinance column compatibility

**Original problem:** `data["Close"]` raises a `KeyError` in some yfinance versions
where the column is named differently (behaviour changed in 0.2.x depending on
`auto_adjust` setting).

**Fix:** Safe column resolution with fallback:

```python
close = (
    data["Close"] if "Close" in data.columns
    else data.get("Adj Close", pd.DataFrame())
)
```

---

## Local development

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push this folder to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io/) → New app
3. Select repo → main file: `app.py` → Deploy

Live in ~60 seconds.

---

## Sentiment scoring

Each post title + body is scored with VADER (Valence Aware Dictionary and sEntiment Reasoner),
a rule-based NLP model purpose-built for short social media text.

| Score | Label |
|-------|-------|
| ≥ 0.15 | Bullish |
| 0.05 – 0.15 | Slightly Bullish |
| −0.05 – 0.05 | Neutral |
| −0.15 – −0.05 | Slightly Bearish |
| ≤ −0.15 | Bearish |

Individual post scores are averaged per ticker to produce the final compound score.

---

## Notes

- No API keys required — all sources use public endpoints
- Not financial advice — this is a sentiment monitoring tool, not a trading signal
- Reddit rate limits — the app includes `time.sleep(0.3)` between subreddit requests
- Price data is delayed 15–20 min via Yahoo Finance
- Each scan takes 15–30 seconds depending on source selection and post count

---

Built by RaniStocks · Educational use only, not a financal advice or a recommendation! 
