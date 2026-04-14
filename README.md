# 📡 RaniStocks — Social Sentiment Radar

> Real-time ticker intelligence scraped from Reddit, Stocktwits, Finviz & Yahoo Finance with VADER NLP sentiment scoring.

## What it does

On every manual scan, the app:
1. **Scrapes 4 sources** for the most-mentioned stock tickers
2. **Scores sentiment** on every post/headline using VADER NLP (bullish / bearish / neutral)
3. **Enriches with price data** from Yahoo Finance
4. **Displays a ranked leaderboard** with mention bars, sentiment badges, source breakdown, and sample posts

---

## Sources

| Source | Method | What's scraped |
|---|---|---|
| **Reddit** | Public `.json` API (no auth needed) | Hot posts from r/wallstreetbets, r/stocks, r/investing, r/options, and more |
| **Stocktwits** | Public REST API | Trending symbols + message stream |
| **Finviz** | `finvizfinance` library | News & blog headlines |
| **Yahoo Finance** | Public HTML endpoint | Trending tickers page |

No API keys required for any source.

---

## How to deploy

### Step 1 — Create a GitHub repo
1. Go to [github.com](https://github.com) → **New repository**
2. Name it `ranistocks-sentiment-radar`
3. Set to **Public** → **Create**

### Step 2 — Upload files
Upload both to the repo root:
- `app.py`
- `requirements.txt`

### Step 3 — Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. **New app** → select repo → main file: `app.py`
3. Click **Deploy**

Live in ~60 seconds.

---

## Local development

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## How to add more subreddits

Edit the `SUBREDDITS` list near the top of `app.py`:

```python
SUBREDDITS = [
    "wallstreetbets", "stocks", "investing", "options",
    "StockMarket", "SecurityAnalysis", "ValueInvesting", "Daytrading",
    # add more here ↓
    "pennystocks", "RobinHoodPennyStocks", "Superstonk",
]
```

---

## How the sentiment scoring works

Each post title + body is scored with **VADER** (Valence Aware Dictionary and sEntiment Reasoner), a rule-based NLP model purpose-built for short social media text.

| Compound score | Label |
|---|---|
| ≥ 0.15 | Bullish |
| 0.05 – 0.15 | Slightly Bullish |
| -0.05 – 0.05 | Neutral |
| -0.15 – -0.05 | Slightly Bearish |
| ≤ -0.15 | Bearish |

Individual post scores are averaged per ticker to produce the final compound score.

---

## Important notes

- **No API keys required** — all sources use public endpoints
- **Not financial advice** — this is a sentiment monitoring tool, not a trading signal
- **Reddit rate limits** — the app includes `time.sleep(0.3)` between subreddit requests to stay within Reddit's public API limits
- **Price data** is delayed 15–20 min via Yahoo Finance
- Each scan takes **15–30 seconds** depending on source selection and post count

---

*Built by RaniStocks · Educational use only*
