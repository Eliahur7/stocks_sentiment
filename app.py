"""
RaniStocks · Social Sentiment Radar
Scans Reddit, Stocktwits, and Finviz news to surface the most-discussed tickers
with bullish/bearish sentiment scoring.
"""

import streamlit as st
import requests
import re
import time
import yfinance as yf
import pandas as pd
from collections import Counter, defaultdict
from datetime import datetime
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ── Download VADER lexicon once ───────────────────────────────────────────────
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RaniStocks · Sentiment Radar",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
.main { background: #09090b; }
.block-container { padding: 1.5rem 2rem 3rem; max-width: 1400px; }

section[data-testid="stSidebar"] {
    background: #09090b;
    border-right: 1px solid #1c1c1e;
}
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

.stButton > button {
    background: #3b82f6;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 14px;
    padding: 0.6rem 1.5rem;
    width: 100%;
    letter-spacing: 0.03em;
    transition: background 0.15s;
}
.stButton > button:hover { background: #2563eb; }

div[data-testid="metric-container"] {
    background: #18181b;
    border: 1px solid #27272a;
    border-radius: 10px;
    padding: 0.85rem 1rem;
}
div[data-testid="metric-container"] label { color: #71717a; font-size: 11px; }
div[data-testid="metric-container"] div[data-testid="metric-value"] {
    color: #fafafa; font-size: 1.3rem; font-weight: 600;
}

.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #27272a;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #71717a;
    border: none;
    padding: 0.5rem 1.2rem;
    font-size: 13px;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    color: #fafafa;
    border-bottom: 2px solid #3b82f6;
}
hr { border-color: #27272a; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────

# Subreddits to scan
SUBREDDITS = [
    "wallstreetbets", "stocks", "investing", "options",
    "StockMarket", "SecurityAnalysis", "ValueInvesting", "Daytrading",
]

# Known noise words that look like tickers but aren't
TICKER_BLACKLIST = {
    "A", "I", "AM", "AN", "AT", "BE", "BY", "DO", "ET", "GO", "HE", "IF",
    "IN", "IS", "IT", "ME", "MY", "NO", "OF", "ON", "OR", "SO", "TO", "UP",
    "US", "WE", "AI", "AR", "AS", "AH", "OK", "PM", "EV", "PR", "HQ",
    "CEO", "CFO", "CTO", "IPO", "ETF", "ATH", "ATL", "IMO", "TBH", "FYI",
    "EOD", "EOW", "EPS", "PE", "DD", "IV", "OG", "RH", "SEC", "FDA", "IRS",
    "GDP", "CPI", "FED", "GDP", "YTD", "YOY", "QOQ", "TTM", "LOL", "WTF",
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HER",
    "WAS", "ONE", "OUR", "OUT", "DAY", "NEW", "MAY", "WAY", "WHO", "OIL",
    "GAS", "RUN", "BUY", "NOW", "TOP", "LOW", "HIT", "SET", "BIG", "OFF",
    "GET", "PUT", "CUT", "DIP", "MAX", "APP", "API", "GDP", "USD", "CAD",
    "EUR", "GBP", "JPY", "BTC", "ETH", "NFT", "DAO", "DeFi", "YOLO",
    "HODL", "FOMO", "BTFD", "SPAC", "ICO", "SaaS", "EBITDA",
}

HEADERS = {
    "User-Agent": "RaniStocks SentimentRadar/1.0 (educational project)"
}

# ── Sentiment engine ──────────────────────────────────────────────────────────

@st.cache_resource
def get_sia():
    return SentimentIntensityAnalyzer()

def score_sentiment(texts: list[str]) -> dict:
    """
    Returns dict with keys: compound, bullish_pct, bearish_pct, neutral_pct, label, color
    compound is mean VADER compound score across all texts (-1 to +1)
    """
    sia = get_sia()
    scores = []
    bull = bear = neut = 0
    for t in texts:
        if not t or not t.strip():
            continue
        s = sia.polarity_scores(t)["compound"]
        scores.append(s)
        if s >= 0.05:
            bull += 1
        elif s <= -0.05:
            bear += 1
        else:
            neut += 1

    if not scores:
        return {"compound": 0, "bullish_pct": 0, "bearish_pct": 0, "neutral_pct": 100,
                "label": "Neutral", "color": "#71717a"}

    compound = sum(scores) / len(scores)
    total = bull + bear + neut or 1
    bp = round(bull / total * 100)
    brp = round(bear / total * 100)
    np_ = round(neut / total * 100)

    if compound >= 0.15:
        label, color = "Bullish", "#22c55e"
    elif compound >= 0.05:
        label, color = "Slightly Bullish", "#86efac"
    elif compound <= -0.15:
        label, color = "Bearish", "#ef4444"
    elif compound <= -0.05:
        label, color = "Slightly Bearish", "#fca5a5"
    else:
        label, color = "Neutral", "#71717a"

    return {"compound": round(compound, 3), "bullish_pct": bp,
            "bearish_pct": brp, "neutral_pct": np_,
            "label": label, "color": color}


# ── Ticker extraction ─────────────────────────────────────────────────────────

def extract_tickers(text: str) -> list[str]:
    """Extract $TICKER mentions and ALL-CAPS words 2-5 chars from text."""
    if not text:
        return []
    # $TICKER style (highest confidence)
    dollar_tickers = re.findall(r'\$([A-Z]{1,5})\b', text.upper())
    # Also catch plain CAPS words 2–5 chars (lower confidence — filtered by blacklist)
    caps_words = re.findall(r'\b([A-Z]{2,5})\b', text.upper())
    combined = set(dollar_tickers) | set(caps_words)
    return [t for t in combined if t not in TICKER_BLACKLIST and len(t) >= 2]


# ── Data sources ──────────────────────────────────────────────────────────────

def fetch_reddit(subreddits: list[str], post_limit: int = 25) -> tuple[Counter, dict]:
    """
    Pull hot posts from each subreddit via the public .json endpoint.
    Returns (ticker_counter, ticker_texts_dict)
    """
    counter = Counter()
    texts = defaultdict(list)

    for sub in subreddits:
        try:
            url = f"https://www.reddit.com/r/{sub}/hot.json?limit={post_limit}"
            resp = requests.get(url, headers=HEADERS, timeout=8)
            if resp.status_code != 200:
                continue
            data = resp.json()
            posts = data.get("data", {}).get("children", [])
            for post in posts:
                p = post.get("data", {})
                title = p.get("title", "")
                body = p.get("selftext", "")
                full_text = f"{title} {body}"
                tickers = extract_tickers(full_text)
                for t in tickers:
                    counter[t] += 1
                    texts[t].append(full_text)
            time.sleep(0.3)  # be polite to Reddit
        except Exception:
            continue

    return counter, dict(texts)


def fetch_stocktwits() -> tuple[Counter, dict]:
    """
    Pull trending symbols from Stocktwits public API.
    Returns (ticker_counter, ticker_texts_dict)
    """
    counter = Counter()
    texts = defaultdict(list)
    try:
        url = "https://api.stocktwits.com/api/2/trending/symbols.json"
        resp = requests.get(url, headers=HEADERS, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            symbols = data.get("symbols", [])
            for s in symbols:
                ticker = s.get("symbol", "")
                watchlist = s.get("watchlist_count", 1)
                if ticker and ticker not in TICKER_BLACKLIST:
                    # Weight by watchlist count (normalize to mention-like score)
                    weight = max(1, watchlist // 5000)
                    counter[ticker] += weight
                    texts[ticker].append(s.get("title", ticker))
    except Exception:
        pass

    # Also try to get messages for top trending (sentiment)
    try:
        url2 = "https://api.stocktwits.com/api/2/streams/trending.json"
        resp2 = requests.get(url2, headers=HEADERS, timeout=8)
        if resp2.status_code == 200:
            messages = resp2.json().get("messages", [])
            for msg in messages:
                body = msg.get("body", "")
                symbol = msg.get("symbols", [{}])
                sentiment_raw = msg.get("entities", {}).get("sentiment", {})
                if symbol:
                    t = symbol[0].get("symbol", "")
                    if t:
                        counter[t] += 1
                        texts[t].append(body)
    except Exception:
        pass

    return counter, dict(texts)


def fetch_finviz_news() -> tuple[Counter, dict]:
    """
    Pull news from Finviz and extract tickers from headlines.
    finvizfinance library returns news_df and blogs_df.
    """
    counter = Counter()
    texts = defaultdict(list)
    try:
        from finvizfinance.news import News
        fnews = News()
        all_news = fnews.get_news()

        for key in ("news", "blogs"):
            df = all_news.get(key)
            if df is None or df.empty:
                continue
            for _, row in df.iterrows():
                headline = str(row.get("Title", ""))
                link = str(row.get("Link", ""))
                tickers = extract_tickers(headline)
                # Also try to get ticker from link URL (finviz links often have t=TICKER)
                url_match = re.search(r'[?&]t=([A-Z]{1,5})', link)
                if url_match:
                    tickers.append(url_match.group(1))
                for t in set(tickers):
                    if t not in TICKER_BLACKLIST:
                        counter[t] += 1
                        texts[t].append(headline)
    except Exception:
        pass

    return counter, dict(texts)


def fetch_yahoo_trending() -> tuple[Counter, dict]:
    """
    Scrape Yahoo Finance trending tickers (public endpoint).
    """
    counter = Counter()
    texts = defaultdict(list)
    try:
        url = "https://finance.yahoo.com/trending-tickers/"
        resp = requests.get(url, headers={
            **HEADERS,
            "Accept": "text/html,application/xhtml+xml",
        }, timeout=8)
        if resp.status_code == 200:
            # Extract ticker symbols from the page
            tickers_found = re.findall(r'"symbol":"([A-Z]{1,5})"', resp.text)
            for t in tickers_found:
                if t not in TICKER_BLACKLIST:
                    counter[t] += 2  # trending = higher weight
                    texts[t].append(f"{t} is trending on Yahoo Finance")
    except Exception:
        pass
    return counter, dict(texts)


# ── Price enrichment ──────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_price_data(tickers: tuple) -> dict:
    """Fetch current price and change for a list of tickers. Cached 5 min."""
    result = {}
    if not tickers:
        return result
    try:
        # batch download
        data = yf.download(list(tickers), period="2d", progress=False, auto_adjust=True)
        close = data["Close"] if "Close" in data else pd.DataFrame()
        if close.empty:
            return result
        for t in tickers:
            try:
                series = close[t].dropna() if t in close.columns else pd.Series()
                if len(series) >= 2:
                    prev = float(series.iloc[-2])
                    curr = float(series.iloc[-1])
                    chg = (curr - prev) / prev * 100
                    result[t] = {"price": curr, "chg": chg}
                elif len(series) == 1:
                    result[t] = {"price": float(series.iloc[-1]), "chg": 0.0}
            except Exception:
                continue
    except Exception:
        pass
    return result


# ── Main scan function ────────────────────────────────────────────────────────

def run_scan(selected_subs: list[str], post_limit: int, top_n: int,
             use_reddit: bool, use_stocktwits: bool,
             use_finviz: bool, use_yahoo: bool) -> pd.DataFrame:
    """
    Aggregate mentions from all sources, score sentiment, enrich with price.
    Returns sorted DataFrame.
    """
    total_counter = Counter()
    all_texts = defaultdict(list)
    source_hits = defaultdict(lambda: defaultdict(int))

    progress = st.progress(0, text="Starting scan...")

    step = 0
    total_steps = sum([use_reddit, use_stocktwits, use_finviz, use_yahoo])
    if total_steps == 0:
        st.warning("Select at least one data source.")
        return pd.DataFrame()

    # ── Reddit ──
    if use_reddit:
        progress.progress(step / total_steps, text=f"📡 Scanning {len(selected_subs)} subreddits...")
        rc, rt = fetch_reddit(selected_subs, post_limit)
        for t, c in rc.items():
            total_counter[t] += c
            all_texts[t].extend(rt.get(t, []))
            source_hits[t]["reddit"] = c
        step += 1
        progress.progress(step / total_steps, text="Reddit ✓")

    # ── Stocktwits ──
    if use_stocktwits:
        progress.progress(step / total_steps, text="📡 Scanning Stocktwits...")
        sc, st_ = fetch_stocktwits()
        for t, c in sc.items():
            total_counter[t] += c
            all_texts[t].extend(st_.get(t, []))
            source_hits[t]["stocktwits"] = c
        step += 1
        progress.progress(step / total_steps, text="Stocktwits ✓")

    # ── Finviz ──
    if use_finviz:
        progress.progress(step / total_steps, text="📡 Scanning Finviz news...")
        fc, ft = fetch_finviz_news()
        for t, c in fc.items():
            total_counter[t] += c
            all_texts[t].extend(ft.get(t, []))
            source_hits[t]["finviz"] = c
        step += 1
        progress.progress(step / total_steps, text="Finviz ✓")

    # ── Yahoo Finance ──
    if use_yahoo:
        progress.progress(step / total_steps, text="📡 Scanning Yahoo Finance trending...")
        yc, yt = fetch_yahoo_trending()
        for t, c in yc.items():
            total_counter[t] += c
            all_texts[t].extend(yt.get(t, []))
            source_hits[t]["yahoo"] = c
        step += 1
        progress.progress(step / total_steps, text="Yahoo ✓")

    progress.progress(1.0, text="Scoring sentiment...")

    # ── Top N tickers ──
    top_tickers = [t for t, _ in total_counter.most_common(top_n * 3)]

    # ── Sentiment ──
    rows = []
    for ticker in top_tickers[:top_n * 2]:
        texts_for_ticker = all_texts.get(ticker, [])
        sent = score_sentiment(texts_for_ticker)
        rows.append({
            "ticker": ticker,
            "mentions": total_counter[ticker],
            "reddit": source_hits[ticker].get("reddit", 0),
            "stocktwits": source_hits[ticker].get("stocktwits", 0),
            "finviz": source_hits[ticker].get("finviz", 0),
            "yahoo": source_hits[ticker].get("yahoo", 0),
            "sentiment_label": sent["label"],
            "sentiment_color": sent["color"],
            "compound": sent["compound"],
            "bullish_pct": sent["bullish_pct"],
            "bearish_pct": sent["bearish_pct"],
            "neutral_pct": sent["neutral_pct"],
            "sample_texts": texts_for_ticker[:5],
        })

    if not rows:
        progress.empty()
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("mentions", ascending=False).reset_index(drop=True)
    df = df.head(top_n)

    # ── Price data ──
    progress.progress(0.95, text="Fetching prices...")
    prices = get_price_data(tuple(df["ticker"].tolist()))
    df["price"] = df["ticker"].map(lambda t: prices.get(t, {}).get("price"))
    df["chg"] = df["ticker"].map(lambda t: prices.get(t, {}).get("chg"))

    progress.empty()
    return df


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='display:flex;align-items:center;gap:10px;margin-bottom:1.5rem;'>
        <div style='width:34px;height:34px;background:#3b82f6;border-radius:8px;
                    display:flex;align-items:center;justify-content:center;
                    font-size:18px;'>📡</div>
        <div>
            <div style='font-size:15px;font-weight:600;color:#fafafa;'>Sentiment Radar</div>
            <div style='font-size:10px;color:#52525b;'>by RaniStocks</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="font-size:11px;font-weight:600;color:#52525b;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:6px;">Data Sources</p>', unsafe_allow_html=True)
    use_reddit     = st.checkbox("Reddit (WSB, stocks, investing…)", value=True)
    use_stocktwits = st.checkbox("Stocktwits trending", value=True)
    use_finviz     = st.checkbox("Finviz news headlines", value=True)
    use_yahoo      = st.checkbox("Yahoo Finance trending", value=True)

    if use_reddit:
        st.markdown('<p style="font-size:11px;font-weight:600;color:#52525b;text-transform:uppercase;letter-spacing:0.07em;margin:12px 0 6px;">Reddit Subreddits</p>', unsafe_allow_html=True)
        selected_subs = st.multiselect(
            "Subreddits",
            SUBREDDITS,
            default=["wallstreetbets", "stocks", "investing", "options"],
            label_visibility="collapsed",
        )
        post_limit = st.slider("Posts per subreddit", 10, 100, 25, 5)
    else:
        selected_subs = []
        post_limit = 25

    st.markdown("---")
    st.markdown('<p style="font-size:11px;font-weight:600;color:#52525b;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:6px;">Results</p>', unsafe_allow_html=True)
    top_n = st.slider("Top tickers to show", 10, 50, 20, 5)

    st.markdown("---")
    scan_btn = st.button("🔍  Run Scan", use_container_width=True)

    st.markdown("""
    <div style='margin-top:1rem;font-size:10px;color:#3f3f46;text-align:center;'>
        Sources: Reddit · Stocktwits · Finviz · Yahoo Finance<br>
        Sentiment: VADER NLP<br>
        Prices: Yahoo Finance (delayed)
    </div>
    """, unsafe_allow_html=True)


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:1.5rem;'>
    <h1 style='font-size:1.8rem;font-weight:700;color:#fafafa;margin:0 0 4px;'>
        📡 Social Sentiment Radar
    </h1>
    <p style='font-size:14px;color:#71717a;margin:0;'>
        Real-time ticker intelligence scraped from Reddit, Stocktwits, Finviz & Yahoo Finance
        · Sentiment scored with VADER NLP
    </p>
</div>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
if "scan_time" not in st.session_state:
    st.session_state.scan_time = None

if scan_btn:
    with st.spinner(""):
        st.session_state.df = run_scan(
            selected_subs=selected_subs,
            post_limit=post_limit,
            top_n=top_n,
            use_reddit=use_reddit,
            use_stocktwits=use_stocktwits,
            use_finviz=use_finviz,
            use_yahoo=use_yahoo,
        )
        st.session_state.scan_time = datetime.now().strftime("%b %d, %Y · %I:%M %p")

df = st.session_state.df

# ── Empty state ───────────────────────────────────────────────────────────────
if df is None or df.empty:
    st.markdown("""
    <div style='text-align:center;padding:4rem 2rem;'>
        <div style='font-size:3rem;margin-bottom:1rem;'>📡</div>
        <div style='font-size:1.1rem;font-weight:500;color:#fafafa;margin-bottom:8px;'>
            Ready to scan
        </div>
        <div style='font-size:13px;color:#71717a;max-width:400px;margin:0 auto;'>
            Configure your sources in the sidebar, then hit <strong style='color:#3b82f6;'>Run Scan</strong>.
            Results are sourced live — each scan takes 15–30 seconds.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Summary bar ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='font-size:12px;color:#52525b;margin-bottom:1rem;'>
    Last scan: <span style='color:#a1a1aa;'>{st.session_state.scan_time}</span> ·
    {len(df)} tickers found
</div>
""", unsafe_allow_html=True)

total_mentions  = int(df["mentions"].sum())
bull_count      = len(df[df["sentiment_label"].str.contains("Bullish")])
bear_count      = len(df[df["sentiment_label"].str.contains("Bearish")])
neut_count      = len(df) - bull_count - bear_count
top_ticker      = df.iloc[0]["ticker"] if not df.empty else "—"
top_mentions    = int(df.iloc[0]["mentions"]) if not df.empty else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Mentions", f"{total_mentions:,}")
c2.metric("Tickers Found", len(df))
c3.metric("Bullish", f"{bull_count} tickers", f"{round(bull_count/len(df)*100)}%")
c4.metric("Bearish", f"{bear_count} tickers", f"-{round(bear_count/len(df)*100)}%")
c5.metric("Most Buzzed", top_ticker, f"{top_mentions} mentions")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_leaderboard, tab_sentiment, tab_sources, tab_raw = st.tabs([
    "🏆  Leaderboard", "🧠  Sentiment Breakdown", "📊  Source Breakdown", "🗂  Raw Data"
])


# ══════════════════════════════════════
# TAB 1 · Leaderboard
# ══════════════════════════════════════
with tab_leaderboard:
    st.markdown('<p style="font-size:13px;color:#71717a;margin-bottom:1rem;">Ranked by total cross-platform mentions. Sentiment labeled by VADER NLP.</p>', unsafe_allow_html=True)

    for i, row in df.iterrows():
        rank  = i + 1
        t     = row["ticker"]
        price = row.get("price")
        chg   = row.get("chg")

        price_str = f"${price:,.2f}" if price and price > 0 else "—"
        chg_color = "#22c55e" if chg and chg >= 0 else "#ef4444"
        chg_str   = f"{chg:+.2f}%" if chg is not None else "—"

        # Mention bar width (relative to max)
        bar_w = int(row["mentions"] / df["mentions"].max() * 100)

        # Source dots
        src_dots = ""
        if row["reddit"] > 0:
            src_dots += '<span style="background:#ff4500;color:white;border-radius:3px;padding:1px 6px;font-size:10px;margin-right:3px;">Reddit</span>'
        if row["stocktwits"] > 0:
            src_dots += '<span style="background:#1d9bf0;color:white;border-radius:3px;padding:1px 6px;font-size:10px;margin-right:3px;">Stocktwits</span>'
        if row["finviz"] > 0:
            src_dots += '<span style="background:#6366f1;color:white;border-radius:3px;padding:1px 6px;font-size:10px;margin-right:3px;">Finviz</span>'
        if row["yahoo"] > 0:
            src_dots += '<span style="background:#7e22ce;color:white;border-radius:3px;padding:1px 6px;font-size:10px;margin-right:3px;">Yahoo</span>'

        rank_color = {1: "#f59e0b", 2: "#94a3b8", 3: "#cd7f32"}.get(rank, "#3f3f46")

        st.markdown(f"""
        <div style='background:#18181b;border:1px solid #27272a;border-radius:10px;
                    padding:14px 18px;margin-bottom:8px;
                    border-left:3px solid {row["sentiment_color"]};'>
            <div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;'>
                <div style='display:flex;align-items:center;gap:14px;'>
                    <span style='font-size:1.1rem;font-weight:700;color:{rank_color};
                                 min-width:26px;'>#{rank}</span>
                    <div>
                        <span style='font-size:1.2rem;font-weight:700;color:#fafafa;
                                     font-family:monospace;letter-spacing:-0.5px;'>${t}</span>
                        <div style='margin-top:3px;'>{src_dots}</div>
                    </div>
                </div>
                <div style='display:flex;align-items:center;gap:20px;flex-wrap:wrap;'>
                    <div style='text-align:right;'>
                        <div style='font-size:1rem;font-weight:600;color:#fafafa;
                                    font-family:monospace;'>{price_str}</div>
                        <div style='font-size:12px;color:{chg_color};font-weight:600;'>{chg_str}</div>
                    </div>
                    <div style='text-align:center;'>
                        <div style='font-size:1rem;font-weight:600;color:#fafafa;'>{row["mentions"]}</div>
                        <div style='font-size:10px;color:#71717a;'>mentions</div>
                    </div>
                    <div style='background:{row["sentiment_color"]}22;border:1px solid {row["sentiment_color"]}55;
                                border-radius:6px;padding:4px 12px;text-align:center;'>
                        <div style='font-size:11px;font-weight:700;color:{row["sentiment_color"]};'>
                            {row["sentiment_label"].upper()}
                        </div>
                        <div style='font-size:10px;color:{row["sentiment_color"]}aa;'>
                            {row["compound"]:+.3f}
                        </div>
                    </div>
                </div>
            </div>
            <div style='margin-top:10px;background:#27272a;border-radius:4px;height:4px;'>
                <div style='height:4px;width:{bar_w}%;background:{row["sentiment_color"]};
                            border-radius:4px;opacity:0.7;'></div>
            </div>
            <div style='display:flex;gap:12px;margin-top:6px;'>
                <span style='font-size:10px;color:#52525b;'>
                    🐂 {row["bullish_pct"]}% Bullish &nbsp;|&nbsp;
                    🐻 {row["bearish_pct"]}% Bearish &nbsp;|&nbsp;
                    ⚖️ {row["neutral_pct"]}% Neutral
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════
# TAB 2 · Sentiment Breakdown
# ══════════════════════════════════════
with tab_sentiment:
    import plotly.express as px
    import plotly.graph_objects as go

    st.markdown("### Sentiment distribution")

    # Sentiment scatter: compound score vs mentions
    fig_scatter = go.Figure()
    colors_map = df["sentiment_color"].tolist()

    fig_scatter.add_trace(go.Scatter(
        x=df["compound"],
        y=df["mentions"],
        mode="markers+text",
        text=df["ticker"],
        textposition="top center",
        textfont=dict(color="#a1a1aa", size=10),
        marker=dict(
            color=df["compound"],
            colorscale=[[0, "#ef4444"], [0.5, "#71717a"], [1, "#22c55e"]],
            size=df["mentions"].apply(lambda m: max(8, min(40, m * 2))),
            opacity=0.85,
            colorbar=dict(
                title="Compound",
                tickfont=dict(color="#71717a"),
                titlefont=dict(color="#71717a"),
            ),
        ),
        hovertemplate="<b>$%{text}</b><br>Compound: %{x:.3f}<br>Mentions: %{y}<extra></extra>",
    ))

    fig_scatter.add_vline(x=0.05,  line_dash="dot", line_color="#22c55e", line_width=1)
    fig_scatter.add_vline(x=-0.05, line_dash="dot", line_color="#ef4444", line_width=1)
    fig_scatter.add_vrect(x0=0.05, x1=1,   fillcolor="#22c55e", opacity=0.04, line_width=0)
    fig_scatter.add_vrect(x0=-1,   x1=-0.05, fillcolor="#ef4444", opacity=0.04, line_width=0)

    fig_scatter.update_layout(
        template="plotly_dark",
        paper_bgcolor="#09090b", plot_bgcolor="#09090b",
        height=420,
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(title="Sentiment Score (compound)", showgrid=True, gridcolor="#18181b",
                   tickfont=dict(color="#71717a")),
        yaxis=dict(title="Mentions", showgrid=True, gridcolor="#18181b",
                   tickfont=dict(color="#71717a")),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("### Bullish vs Bearish breakdown — top 15")

    top15 = df.head(15).copy()
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        name="Bullish %",
        x=top15["ticker"],
        y=top15["bullish_pct"],
        marker_color="#22c55e",
        opacity=0.85,
    ))
    fig_bar.add_trace(go.Bar(
        name="Neutral %",
        x=top15["ticker"],
        y=top15["neutral_pct"],
        marker_color="#52525b",
        opacity=0.75,
    ))
    fig_bar.add_trace(go.Bar(
        name="Bearish %",
        x=top15["ticker"],
        y=top15["bearish_pct"],
        marker_color="#ef4444",
        opacity=0.85,
    ))

    fig_bar.update_layout(
        barmode="stack",
        template="plotly_dark",
        paper_bgcolor="#09090b", plot_bgcolor="#09090b",
        height=320,
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", x=0, y=1.1, font=dict(color="#71717a", size=11)),
        yaxis=dict(ticksuffix="%", showgrid=True, gridcolor="#18181b",
                   tickfont=dict(color="#71717a")),
        xaxis=dict(showgrid=False, tickfont=dict(color="#a1a1aa", size=11)),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Sample posts for selected ticker
    st.markdown("### Sample posts — click a ticker to read what's being said")
    selected_ticker = st.selectbox(
        "Ticker",
        df["ticker"].tolist(),
        label_visibility="collapsed",
    )
    ticker_row = df[df["ticker"] == selected_ticker].iloc[0]
    sample = ticker_row["sample_texts"]

    if sample:
        for i, txt in enumerate(sample[:5]):
            preview = txt[:300].replace("\n", " ").strip()
            if len(txt) > 300:
                preview += "…"
            st.markdown(f"""
            <div style='background:#18181b;border:1px solid #27272a;border-radius:8px;
                        padding:12px 14px;margin-bottom:6px;font-size:12px;
                        color:#d4d4d8;line-height:1.6;'>
                <span style='color:#52525b;font-size:10px;'>Post {i+1}</span><br>{preview}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No sample text available for this ticker.")


# ══════════════════════════════════════
# TAB 3 · Source Breakdown
# ══════════════════════════════════════
with tab_sources:
    import plotly.graph_objects as go

    st.markdown("### Mentions by source — top 20 tickers")

    top20 = df.head(20).copy()
    fig_src = go.Figure()

    src_cfg = [
        ("reddit",     "#ff4500", "Reddit"),
        ("stocktwits", "#1d9bf0", "Stocktwits"),
        ("finviz",     "#6366f1", "Finviz"),
        ("yahoo",      "#7e22ce", "Yahoo Finance"),
    ]
    for col, color, label in src_cfg:
        fig_src.add_trace(go.Bar(
            name=label,
            x=top20["ticker"],
            y=top20[col],
            marker_color=color,
            opacity=0.85,
        ))

    fig_src.update_layout(
        barmode="stack",
        template="plotly_dark",
        paper_bgcolor="#09090b", plot_bgcolor="#09090b",
        height=380,
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", x=0, y=1.1, font=dict(color="#71717a", size=11)),
        yaxis=dict(showgrid=True, gridcolor="#18181b", tickfont=dict(color="#71717a")),
        xaxis=dict(showgrid=False, tickfont=dict(color="#a1a1aa", size=11)),
    )
    st.plotly_chart(fig_src, use_container_width=True)

    # Source summary cards
    st.markdown("### Source contribution")
    s1, s2, s3, s4 = st.columns(4)

    reddit_total = int(df["reddit"].sum())
    st_total     = int(df["stocktwits"].sum())
    fv_total     = int(df["finviz"].sum())
    yh_total     = int(df["yahoo"].sum())
    grand        = reddit_total + st_total + fv_total + yh_total or 1

    for col, label, total, color in [
        (s1, "Reddit",       reddit_total, "#ff4500"),
        (s2, "Stocktwits",   st_total,     "#1d9bf0"),
        (s3, "Finviz",       fv_total,     "#6366f1"),
        (s4, "Yahoo Finance",yh_total,     "#7e22ce"),
    ]:
        pct = round(total / grand * 100)
        col.markdown(f"""
        <div style='background:#18181b;border:1px solid #27272a;border-radius:10px;
                    padding:14px;text-align:center;border-top:3px solid {color};'>
            <div style='font-size:1.4rem;font-weight:700;color:#fafafa;'>{total:,}</div>
            <div style='font-size:11px;color:#71717a;margin-top:2px;'>{label}</div>
            <div style='font-size:12px;color:{color};font-weight:600;margin-top:4px;'>
                {pct}% of signal
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════
# TAB 4 · Raw Data
# ══════════════════════════════════════
with tab_raw:
    st.markdown("### Export-ready data")
    display_df = df[[
        "ticker", "mentions", "reddit", "stocktwits", "finviz", "yahoo",
        "sentiment_label", "compound", "bullish_pct", "bearish_pct",
        "price", "chg"
    ]].copy()
    display_df.columns = [
        "Ticker", "Total Mentions", "Reddit", "Stocktwits", "Finviz", "Yahoo",
        "Sentiment", "Compound Score", "Bullish %", "Bearish %",
        "Price ($)", "Change (%)"
    ]

    st.dataframe(
        display_df.style.format({
            "Compound Score": "{:+.3f}",
            "Change (%)": lambda v: f"{v:+.2f}%" if v == v else "—",
            "Price ($)": lambda v: f"${v:,.2f}" if v == v else "—",
        }),
        use_container_width=True,
        height=500,
    )

    csv = display_df.to_csv(index=False)
    st.download_button(
        "⬇️  Download CSV",
        data=csv,
        file_name=f"sentiment_radar_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<p style='font-size:11px;color:#27272a;text-align:center;margin-top:3rem;'>
    RaniStocks Sentiment Radar · Educational use only · Not financial advice ·
    Reddit data via public JSON API · Prices delayed via Yahoo Finance
</p>
""", unsafe_allow_html=True)
