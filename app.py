import datetime
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf
import requests
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# ดาวน์โหลด VADER lexicon สำหรับ NLTK
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

st.set_page_config(page_title="Stock Dashboard", layout="wide")
st.title("Stock Dashboard")
st.markdown(
    "เปรียบเทียบราคาหุ้นและจัดการ watchlist ของคุณ โดยดึงข้อมูลจาก `SET` สำหรับหุ้นไทย และ `yfinance` สำหรับหุ้นต่างประเทศ"
)

GROUPS = {
    "Biotech": [
        "CRSP",
        "TEM",
        "JNJ",
        "REGN",
        "CMPS",
        "ATAI",
        "NTLA",
        "PFE",
        "BEAM",
        "UTHR",
        "MDGL",
        "LLY",
    ],
    "Tech-semicon": [
        "NVDA",
        "TSMC",
        "ASML",
        "MSFT",
        "META",
        "AAPL",
        "AMZN",
        "GOOG",
        "TSLA",
    ],
    "Mining": ["FCX", "AG", "RIO"],
    "SET (ตลาดหุ้นไทย)": [
        "PTT.BK",
        "AOT.BK",
        "KBANK.BK",
        "SCB.BK",
        "SCC.BK",
        "CPALL.BK",
        "ADVANC.BK",
        "BANPU.BK",
        "CPN.BK",
        "BAM.BK",
    ],
}

THAI_TICKERS_AUTO = {
    "PTT", "AOT", "KBANK", "SCB", "SCC", "CPALL", "ADVANC", "BANPU", "CPN", "BAM",
    "KTB", "BBL", "TMB", "TRUE", "JAS", "HMPRO", "BEM", "COM7", "BDMS",
    "CENTEL", "CRC", "EA", "BH", "MINT", "SCBX", "LH", "BGRIM", "GULF",
}

SET_BASE_URL = "https://www.set.or.th"
DEFAULT_GROUPS = set(GROUPS.keys())
GROUPS_FILE = Path(__file__).resolve().parent / "groups.json"


def normalize_ticker_input(text):
    """แปลง input เป็นรายการ ticker ที่เป็นตัวพิมพ์ใหญ่"""
    if not text:
        return []
    tickers = [t.strip().upper() for t in text.replace(";", ",").replace("\n", ",").split(",")]
    normalized = []
    for ticker in tickers:
        if not ticker:
            continue
        if "." not in ticker and ticker in THAI_TICKERS_AUTO:
            normalized.append(f"{ticker}.BK")
        else:
            normalized.append(ticker)
    return normalized


def load_groups_from_disk():
    if not GROUPS_FILE.exists():
        save_groups_to_disk(GROUPS)
        return {name: list(items) for name, items in GROUPS.items()}

    try:
        with GROUPS_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Invalid group file format")
        return {str(name): [str(ticker).upper() for ticker in tickers] for name, tickers in data.items()}
    except Exception:
        save_groups_to_disk(GROUPS)
        return {name: list(items) for name, items in GROUPS.items()}


def save_groups_to_disk(groups):
    safe_groups = {str(name): [str(ticker).upper() for ticker in tickers] for name, tickers in groups.items()}
    GROUPS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with GROUPS_FILE.open("w", encoding="utf-8") as f:
        json.dump(safe_groups, f, indent=2, ensure_ascii=False)


def get_group_store():
    if "groups" not in st.session_state:
        st.session_state["groups"] = load_groups_from_disk()
    return st.session_state["groups"]


# ============= Technical Indicators Functions =============
def calculate_rsi(series, period=14):
    """
    คำนวณ RSI (Relative Strength Index) โดยใช้ pandas
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


@st.cache_data(ttl=3600)
def calculate_technical_indicators(df_price):
    """
    Calculate EMA 20, EMA 50, and RSI for technical analysis
    """
    df = df_price.copy()
    
    # ได้รับชื่อคอลัมน์ (ควรเป็นหนึ่งคอลัมน์)
    price_column = df.columns[0]
    price_series = df[price_column]
    
    # EMA 20 และ EMA 50 ใช้ pandas .ewm()
    # EMA ใช้ span parameter: span = 2 / (N + 1) เป็นแบบ exponential weighted mean
    df['EMA_20'] = price_series.ewm(span=20, adjust=False).mean()
    df['EMA_50'] = price_series.ewm(span=50, adjust=False).mean()
    
    # RSI (Relative Strength Index) - custom calculation
    df['RSI'] = calculate_rsi(price_series, period=14)
    
    return df


def get_trading_signal(df):
    """
    ให้สัญญาณซื้อขายจากเส้น EMA และ RSI
    """
    if len(df) < 2:
        return "ไม่มีข้อมูลเพียงพอ"
    
    latest = df.iloc[-1]
    price = latest.iloc[0] if isinstance(latest, pd.Series) else latest
    ema_20 = latest['EMA_20']
    ema_50 = latest['EMA_50']
    rsi = latest['RSI']
    
    signal = []
    
    # EMA Signal
    if ema_20 > ema_50:
        signal.append("🟢 EMA 20 > EMA 50 (Uptrend)")
    else:
        signal.append("🔴 EMA 20 < EMA 50 (Downtrend)")
    
    # RSI Signal
    if rsi < 30:
        signal.append("🟢 RSI < 30 (Oversold - Buy Signal)")
    elif rsi > 70:
        signal.append("🔴 RSI > 70 (Overbought - Sell Signal)")
    else:
        signal.append(f"⚪ RSI = {rsi:.2f} (Neutral)")
    
    return " | ".join(signal)


# ============= Sentiment Analysis Functions =============
@st.cache_resource
def load_sentiment_model():
    """
    โหลด VADER Sentiment Analyzer (ไม่ต้องดาวน์โหลดโมเดลเพิ่มเติม)
    """
    return SentimentIntensityAnalyzer()


DEFAULT_YAHOO_FINANCE_API_KEY = "Sh2JBbbtk2HCBGenvZt8JPpFJMoHB73u"


def get_yahoo_finance_api_key():
    return st.secrets.get("YAHOO_FINANCE_API_KEY", DEFAULT_YAHOO_FINANCE_API_KEY)


def fetch_news(ticker, api_key=None):
    """
    ดึงข่าวล่าสุดจาก Yahoo Finance ผ่าน RapidAPI
    """
    if api_key is None:
        api_key = get_yahoo_finance_api_key()
    
    if not api_key:
        return []

    url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/stock/v2/get-news"
    params = {
        "category": ticker,
        "region": "US"
    }
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": "apidojo-yahoo-finance-v1.p.rapidapi.com"
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=7)
        if response.status_code == 200:
            data = response.json()
            result = data.get("items") or data.get("data") or []
            articles = []
            for item in result:
                if isinstance(item, dict):
                    title = item.get("title") or item.get("providerPublishTime") or ""
                    link = item.get("link") or item.get("url") or ""
                    source = item.get("publisher") or item.get("source") or "Yahoo Finance"
                    summary = item.get("summary") or item.get("description") or ""
                    articles.append({
                        "title": title,
                        "source": {"name": source},
                        "description": summary,
                        "url": link,
                    })
            return articles[:5]
        return []
    except Exception as e:
        st.error(f"ข้อผิดพลาดในการดึงข่าว: {e}")
        return []


def analyze_sentiment_vader(text, sentiment_model):
    """
    วิเคราะห์ Sentiment ของข้อความโดยใช้ VADER
    Return: {label: "POSITIVE"|"NEGATIVE"|"NEUTRAL", score: float}
    """
    scores = sentiment_model.polarity_scores(text)
    compound = scores['compound']
    
    # แปลง compound score (-1 to 1) เป็น sentiment label
    if compound >= 0.05:
        label = "POSITIVE"
    elif compound <= -0.05:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"
    
    # Convert to 0-1 range for consistency
    score = abs(compound) / 2 + 0.5
    
    return {"label": label, "score": score, "compound": compound}


# ============= Financial Statement Functions =============
@st.cache_data(ttl=3600)
def load_financial_statement(ticker_symbol):
    """
    ดึงข้อมูลงบการเงินและ ratios จาก yfinance
    """
    ticker_obj = yf.Ticker(ticker_symbol)
    try:
        info = ticker_obj.info or {}
        financials = ticker_obj.financials
        balance_sheet = ticker_obj.balance_sheet
        cashflow = ticker_obj.cashflow
        quarterly_financials = ticker_obj.quarterly_financials
        quarterly_balance_sheet = ticker_obj.quarterly_balance_sheet
        quarterly_cashflow = ticker_obj.quarterly_cashflow
        return (
            info,
            financials,
            balance_sheet,
            cashflow,
            quarterly_financials,
            quarterly_balance_sheet,
            quarterly_cashflow,
        )
    except Exception:
        empty = pd.DataFrame()
        return {}, empty, empty, empty, empty, empty, empty


@st.cache_resource
def get_set_session():
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "th-TH,th;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": f"{SET_BASE_URL}/",
        }
    )
    session.get(SET_BASE_URL, timeout=10)
    return session


def is_thai_ticker(ticker):
    if not isinstance(ticker, str):
        return False
    ticker_upper = ticker.upper()
    return ticker_upper.endswith(".BK") or ticker_upper in THAI_TICKERS_AUTO


def strip_set_suffix(ticker):
    return ticker.upper().replace(".BK", "")


@st.cache_data(ttl=3600)
def fetch_set_stock_chart(symbol, period="1Y"):
    try:
        session = get_set_session()
        response = session.get(
            f"{SET_BASE_URL}/api/set/stock/{symbol}/chart-quotation",
            params={"period": period},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except Exception:
        return {}


def parse_set_chart_to_df(chart_data, ticker):
    quotations = chart_data.get("quotations") or []
    if not quotations:
        return pd.DataFrame()
    df = pd.DataFrame(quotations)
    if df.empty or "price" not in df.columns:
        return pd.DataFrame()

    date_key = "localDatetime" if "localDatetime" in df.columns else "datetime"
    df.index = pd.to_datetime(df[date_key], errors="coerce")
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df = df.sort_index()
    df = df[["price"]].rename(columns={"price": ticker})
    return df


def choose_set_period(start, end):
    days = (end - start).days if isinstance(start, datetime.date) and isinstance(end, datetime.date) else 365
    if days <= 1:
        return "1D"
    if days <= 31:
        return "1M"
    return "1Y"


@st.cache_data(ttl=3600)
def load_set_price_history(ticker, start, end, price_type):
    symbol = strip_set_suffix(ticker)
    period = choose_set_period(start, end)
    chart_data = fetch_set_stock_chart(symbol, period)
    df = parse_set_chart_to_df(chart_data, ticker)
    if df.empty and period != "1Y":
        chart_data = fetch_set_stock_chart(symbol, "1Y")
        df = parse_set_chart_to_df(chart_data, ticker)

    if df.empty:
        return pd.DataFrame()

    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]

    if price_type not in ["Close", "Adj Close"]:
        df[ticker] = df[ticker]
    return df


@st.cache_data(ttl=3600)
def load_yfinance_prices(tickers, start, end, price_type):
    raw = yf.download(
        tickers,
        start=start,
        end=end + datetime.timedelta(days=1),
        progress=False,
        group_by="ticker",
        auto_adjust=False,
        threads=True,
    )

    if raw.empty:
        return pd.DataFrame()

    if len(tickers) == 1:
        ticker = tickers[0]
        if isinstance(raw.columns, pd.MultiIndex):
            try:
                data = raw[ticker][price_type].to_frame()
            except KeyError:
                data = raw[[price_type]].copy()
        else:
            data = raw[[price_type]].copy()
        data.columns = pd.MultiIndex.from_product([[ticker], data.columns])
        return data

    return pd.DataFrame({ticker: raw[ticker][price_type] for ticker in tickers})


@st.cache_data(ttl=3600)
def load_prices(tickers, start, end, price_type):
    if not tickers:
        return pd.DataFrame()

    thai_tickers = [ticker for ticker in tickers if is_thai_ticker(ticker)]
    other_tickers = [ticker for ticker in tickers if ticker not in thai_tickers]
    frames = []

    if other_tickers:
        yfinance_frame = load_yfinance_prices(other_tickers, start, end, price_type)
        if not yfinance_frame.empty:
            if isinstance(yfinance_frame.columns, pd.MultiIndex):
                yfinance_frame.columns = [col[0] if isinstance(col, tuple) else col for col in yfinance_frame.columns]
            frames.append(yfinance_frame)

    for ticker in thai_tickers:
        set_frame = load_set_price_history(ticker, start, end, price_type)
        if not set_frame.empty:
            frames.append(set_frame)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, axis=1).sort_index()
    return combined


def get_info_value(info, keys):
    if not isinstance(info, dict):
        return None
    for key in keys:
        if key in info:
            return info.get(key)
    return None


def get_row_as_series(df, keys):
    if df is None or df.empty:
        return pd.Series(dtype=object)
    for key in keys:
        if key in df.index:
            return df.loc[key]
    return pd.Series(dtype=object)


def get_series_value(series, keys):
    if series is None or series.empty:
        return None
    for key in keys:
        if key in series.index:
            return series.loc[key]
    return None


def format_value(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    if isinstance(value, (int, float)):
        if abs(value) >= 1_000_000_000:
            short = value / 1_000_000_000
            return f"{int(short)}B" if short.is_integer() else f"{short:.1f}B"
        if abs(value) >= 1_000_000:
            short = value / 1_000_000
            return f"{int(short)}M" if short.is_integer() else f"{short:.1f}M"
        if abs(value) >= 1_000:
            short = value / 1_000
            return f"{int(short)}K" if short.is_integer() else f"{short:.1f}K"
        return f"{value:.2f}"
    return str(value)


def format_ratio(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    if isinstance(value, (int, float)):
        return f"{value*100:.2f}%" if abs(value) <= 10 else f"{value:.2f}"
    return str(value)


def abbreviate_dataframe(df):
    if df is None or df.empty:
        return df

    def abbreviate_value(val):
        if isinstance(val, (int, float)) and not pd.isna(val):
            return format_value(val)
        return val

    return df.astype(object).apply(lambda col: col.map(abbreviate_value))


def safe_div(numerator, denominator):
    try:
        if denominator in (0, None) or pd.isna(denominator):
            return None
        return numerator / denominator
    except Exception:
        return None


def calculate_additional_ratios(info, financials, balance_sheet, cashflow=None):
    net_income = get_info_value(info, ["netIncomeToCommon", "netIncome"])
    total_assets = get_info_value(info, ["totalAssets"])
    total_equity = get_info_value(info, ["totalStockholderEquity", "totalStockholdersEquity", "shareholderEquity"])
    current_liabilities = get_info_value(info, ["currentLiabilities", "totalCurrentLiabilities"])
    revenue = get_info_value(info, ["totalRevenue", "revenue"])
    ebitda = get_info_value(info, ["ebitda", "ebitdaRaw"])
    if ebitda is None and not financials.empty:
        ebitda_series = get_row_as_series(financials, ["Ebitda", "EBITDA"])
        if not ebitda_series.empty:
            ebitda = ebitda_series.iloc[0]

    ebit = get_info_value(info, ["ebit", "operatingIncome", "ebitda"])
    if ebit is None and not financials.empty:
        ebit_series = get_row_as_series(financials, ["Ebit", "EBIT", "Operating Income"])
        if not ebit_series.empty:
            ebit = ebit_series.iloc[0]

    invested_capital = None
    if total_assets is not None and current_liabilities is not None:
        invested_capital = total_assets - current_liabilities

    free_cash_flow = get_info_value(info, ["freeCashflow"])
    if free_cash_flow is None and cashflow is not None and not cashflow.empty:
        fcf_series = get_row_as_series(cashflow, ["Free Cash Flow", "Free Cash Flow (FCF)"])
        if not fcf_series.empty:
            free_cash_flow = fcf_series.iloc[0]

    market_cap = get_info_value(info, ["marketCap"])

    ratios = {
        "ROE": safe_div(net_income, total_equity),
        "ROA": safe_div(net_income, total_assets),
        "ROIC": safe_div(ebit, invested_capital),
        "Gross Margin": get_info_value(info, ["grossMargins"]),
        "Operating Margin": get_info_value(info, ["operatingMargins"]),
        "Net Margin": get_info_value(info, ["profitMargins"]),
        "EBITDA Margin": safe_div(ebitda, revenue),
        "EBITDA": ebitda,
        "Free Cash Flow Yield": safe_div(free_cash_flow, market_cap),
        "Interest Coverage": get_info_value(info, ["interestCoverage"]),
    }
    return ratios


def build_yoy_summary(financials, balance_sheet, cashflow):
    if financials is None or financials.empty:
        return pd.DataFrame()

    metrics = {
        "Total Revenue": (financials, ["Total Revenue", "Revenue"]),
        "Gross Profit": (financials, ["Gross Profit"]),
        "Operating Income": (financials, ["Operating Income", "Income Before Tax"]),
        "Net Income": (financials, ["Net Income", "Net Income Applicable To Common Shares"]),
        "Total Assets": (balance_sheet, ["Total Assets"]),
        "Total Liabilities": (balance_sheet, ["Total Liab", "Total Liabilities"]),
        "Shareholders' Equity": (balance_sheet, ["Total Stockholder Equity", "Total Stockholders' Equity", "Stockholders Equity"]),
        "Operating Cash Flow": (cashflow, ["Total Cash From Operating Activities", "Operating Cash Flow"]),
        "Free Cash Flow": (cashflow, ["Free Cash Flow", "Free Cash Flow (FCF)"])
    }

    rows = {}
    for label, (df, keys) in metrics.items():
        rows[label] = get_row_as_series(df, keys)

    if not rows:
        return pd.DataFrame()

    summary_df = pd.DataFrame(rows).transpose()
    summary_df.columns = [str(col) for col in summary_df.columns]
    return summary_df


def build_trend_chart(financials, cashflow):
    chart_data = {}
    if financials is not None and not financials.empty:
        revenue = get_row_as_series(financials, ["Total Revenue", "Revenue"])
        net_income = get_row_as_series(financials, ["Net Income", "Net Income Applicable To Common Shares"])
        if not revenue.empty:
            chart_data["Revenue"] = revenue
        if not net_income.empty:
            chart_data["Net Income"] = net_income

    if cashflow is not None and not cashflow.empty:
        op_cash = get_row_as_series(cashflow, ["Total Cash From Operating Activities", "Operating Cash Flow"])
        if not op_cash.empty:
            chart_data["Operating Cash Flow"] = op_cash

    if chart_data:
        df_chart = pd.DataFrame(chart_data)
        return df_chart
    return pd.DataFrame()


group_store = get_group_store()

with st.sidebar:
    st.header("Watchlist / กลุ่มหุ้น")
    st.markdown("กำหนดกลุ่มหุ้นและเพิ่มหุ้นใหม่ได้เอง เพื่อให้เป็น dashboard watchlist ส่วนตัว")

    group_options = list(group_store.keys())
    if group_options:
        selected_group = st.selectbox("เลือกกลุ่มหุ้น", group_options)
    else:
        selected_group = None

    with st.expander("จัดการกลุ่มหุ้น"):
        new_group_name = st.text_input("ชื่อกลุ่มใหม่", value="", key="new_group_name")
        new_group_tickers = st.text_input(
            "Ticker ของหุ้นในกลุ่มใหม่ (คั่นด้วย ,)",
            value="",
            key="new_group_tickers",
        )

        if st.button("เพิ่มกลุ่มใหม่"):
            group_name = new_group_name.strip()
            tickers = normalize_ticker_input(new_group_tickers)
            if not group_name:
                st.error("โปรดระบุชื่อกลุ่มก่อน")
            elif group_name in group_store:
                st.warning("ชื่อกลุ่มนี้มีอยู่แล้ว กรุณาใช้ชื่ออื่น")
            else:
                group_store[group_name] = tickers
                save_groups_to_disk(group_store)
                st.success(f"เพิ่มกลุ่ม '{group_name}' เรียบร้อยแล้ว")
                st.rerun()

        if st.button("ลบกลุ่มนี้"):
            del group_store[selected_group]
            save_groups_to_disk(group_store)
            st.success(f"ลบกลุ่ม '{selected_group}' เรียบร้อยแล้ว")
            st.rerun()

    st.markdown("---")

    if selected_group is None:
        st.warning("ยังไม่มีกลุ่มหุ้นในระบบ โปรดสร้างกลุ่มใหม่ก่อน")
        selected_tickers = []
    else:
        selected_tickers = st.multiselect(
            "เลือกหุ้น",
            group_store[selected_group],
            default=group_store[selected_group][: min(5, len(group_store[selected_group]))],
        )

        if selected_tickers and any(is_thai_ticker(ticker) for ticker in selected_tickers):
            st.info(
                "หุ้นไทยจาก SET จะใช้ราคาปิดเป็นข้อมูลหลัก หากเลือกประเภทราคาที่ไม่ใช่ Close หรือ Adj Close ระบบจะยังคงแสดงราคา SET ล่าสุด"
            )

        with st.expander("เพิ่ม/ลบหุ้นในกลุ่มนี้"):
            add_ticker = st.text_input("เพิ่ม ticker ใหม่", value="", key="add_ticker")
            if st.button("เพิ่มหุ้นในกลุ่มนี้"):
                normalized_tickers = normalize_ticker_input(add_ticker)
                if not normalized_tickers:
                    st.error("โปรดระบุ Ticker ก่อน")
                else:
                    ticker = normalized_tickers[0]
                    if ticker in group_store[selected_group]:
                        st.warning(f"{ticker} อยู่ในกลุ่มนี้แล้ว")
                    else:
                        group_store[selected_group].append(ticker)
                        save_groups_to_disk(group_store)
                        st.success(f"เพิ่ม {ticker} ในกลุ่ม '{selected_group}' เรียบร้อยแล้ว")
                        st.rerun()

            remove_tickers = st.multiselect(
                "เลือกหุ้นเพื่อลบจากกลุ่ม",
                group_store[selected_group],
                key="remove_tickers",
            )
            if st.button("ลบหุ้นจากกลุ่มนี้"):
                if remove_tickers:
                    for ticker in remove_tickers:
                        if ticker in group_store[selected_group]:
                            group_store[selected_group].remove(ticker)
                    save_groups_to_disk(group_store)
                    st.success(f"ลบหุ้นที่เลือกออกจากกลุ่ม '{selected_group}' เรียบร้อยแล้ว")
                    st.rerun()
                else:
                    st.warning("โปรดเลือกหุ้นอย่างน้อยหนึ่งตัวเพื่อจะลบ")

    st.markdown("---")
    start_date = st.date_input(
        "วันเริ่มต้น",
        value=datetime.date.today() - datetime.timedelta(days=365),
        max_value=datetime.date.today(),
    )
    end_date = st.date_input(
        "วันสิ้นสุด",
        value=datetime.date.today(),
        max_value=datetime.date.today(),
    )
    price_type = st.selectbox(
        "ประเภทราคาที่แสดง",
        ["Adj Close", "Close", "Open", "High", "Low"],
        index=0,
    )

    st.markdown("---")
    st.subheader("ตัวเลือกเพิ่มเติม")
    show_technical = st.checkbox("แสดง Technical Indicators", value=True)
    show_financials = st.checkbox("แสดง Financial Statement", value=True)
    show_sentiment = st.checkbox("แสดง Sentiment Analysis", value=True)

if not selected_tickers:
    st.warning("โปรดเลือกหุ้นอย่างน้อยหนึ่งตัวเพื่อแสดงกราฟ")
    st.stop()

if start_date > end_date:
    st.error("วันที่เริ่มต้นต้องอยู่ก่อนวันที่สิ้นสุด")
    st.stop()

@st.cache_data(ttl=3600)
def load_prices(tickers, start, end, price_type):
    raw = yf.download(
        tickers,
        start=start,
        end=end + datetime.timedelta(days=1),
        progress=False,
        group_by="ticker",
        auto_adjust=False,
        threads=True,
    )

    if raw.empty:
        return pd.DataFrame()

    if len(tickers) == 1:
        ticker = tickers[0]
        if isinstance(raw.columns, pd.MultiIndex):
            try:
                data = raw[ticker][price_type].to_frame()
            except KeyError:
                data = raw[[price_type]].copy()
        else:
            data = raw[[price_type]].copy()

        data.columns = pd.MultiIndex.from_product([[ticker], data.columns])
        return data

    return raw

price_data = load_prices(selected_tickers, start_date, end_date, price_type)

if price_data.empty:
    st.warning("ไม่พบข้อมูลหุ้นในช่วงวันที่ที่เลือก")
    st.stop()

if len(selected_tickers) == 1:
    selected = selected_tickers[0]
    df = price_data[selected][price_type].to_frame(name=selected)
else:
    df = pd.DataFrame(
        {ticker: price_data[ticker][price_type] for ticker in selected_tickers}
    )

st.subheader(f"กราฟราคา: {selected_group}")
line_chart = st.line_chart(df)

st.markdown("---")

latest = df.iloc[-1]
first = df.iloc[0]
change_pct = (latest - first) / first * 100
summary = pd.DataFrame(
    {
        "Latest": latest,
        "Change %": change_pct,
    }
)
summary["Latest"] = summary["Latest"].map("{:.2f}".format)
summary["Change %"] = summary["Change %"].map("{:+.2f}%".format)

st.subheader("สรุปราคาและการเปลี่ยนแปลง")
st.table(summary)

st.markdown("---")

# ============= Technical Indicators Section =============
if show_technical and len(selected_tickers) == 1:
    st.subheader("📊 Technical Indicators (ตัวชี้วัดทางเทคนิค)")
    
    ticker = selected_tickers[0]
    df_indicator = price_data[ticker]["Adj Close" if "Adj Close" in price_data[ticker].columns else "Close"].to_frame(name=ticker)
    df_with_indicators = calculate_technical_indicators(df_indicator)
    
    # แสดงสัญญาณซื้อขาย
    signal = get_trading_signal(df_with_indicators)
    st.info(f"🎯 **สัญญาณซื้อขาย**: {signal}")
    
    # สร้าง chart ที่แสดง EMA
    chart_data = df_with_indicators[[ticker, 'EMA_20', 'EMA_50']].copy()
    st.line_chart(chart_data)
    
    # แสดง RSI
    col1, col2, col3 = st.columns(3)
    latest_data = df_with_indicators.iloc[-1]
    with col1:
        st.metric("ราคาปัจจุบัน", f"${latest_data[ticker]:.2f}")
    with col2:
        st.metric("RSI (14)", f"{latest_data['RSI']:.2f}")
    with col3:
        st.metric("EMA 20 vs 50", f"{(latest_data['EMA_20'] - latest_data['EMA_50']):.2f}")
    
    with st.expander("ดูข้อมูล Technical Indicators"):
        st.dataframe(df_with_indicators.tail(20))

elif show_technical and len(selected_tickers) > 1:
    st.info("💡 Technical Indicators จะแสดงเมื่อเลือกหุ้น **เพียงหนึ่งตัว** เท่านั้น")

st.markdown("---")

# ============= Financial Statement Section =============
if show_financials:
    st.subheader("📑 Financial Statement")
    
    if len(selected_tickers) == 1:
        ticker = selected_tickers[0]
        (
            info,
            financials,
            balance_sheet,
            cashflow,
            quarterly_financials,
            quarterly_balance_sheet,
            quarterly_cashflow,
        ) = load_financial_statement(ticker)
        
        if not info:
            st.error("ไม่สามารถดึงข้อมูลงบการเงินได้")
        else:
            latest_cashflow = cashflow.iloc[:, 0] if not cashflow.empty else pd.Series(dtype='float64')
            latest_balance = balance_sheet.iloc[:, 0] if not balance_sheet.empty else pd.Series(dtype='float64')
            
            ratios = calculate_additional_ratios(info, financials, balance_sheet, cashflow)
            metrics = {
                "P/E": info.get("trailingPE"),
                "Forward P/E": info.get("forwardPE"),
                "P/BV": info.get("priceToBook"),
                "P/S": info.get("priceToSalesTrailing12Months"),
                "D/E": info.get("debtToEquity"),
                "Current Ratio": info.get("currentRatio"),
                "Quick Ratio": info.get("quickRatio"),
                "Market Cap": info.get("marketCap"),
                "Revenue (TTM)": info.get("totalRevenue"),
                "Net Income (TTM)": info.get("netIncomeToCommon"),
                "EPS (TTM)": info.get("trailingEps"),
                "Operating Cash Flow": get_series_value(latest_cashflow, [
                    "Total Cash From Operating Activities",
                    "Operating Cash Flow",
                ]),
                "Free Cash Flow": info.get("freeCashflow"),
                "Cash & Short Term Investments": info.get("totalCash"),
                "Total Debt": info.get("totalDebt"),
                "Shareholders' Equity": get_series_value(latest_balance, [
                    "Total Stockholder Equity",
                    "Total Stockholders' Equity",
                    "Stockholders Equity",
                ]),
            }

            kpi_cards = [
                ("P/E", metrics.get("P/E")),
                ("ROIC", ratios.get("ROIC")),
                ("EBITDA Margin", ratios.get("EBITDA Margin")),
                ("FCF Yield", ratios.get("Free Cash Flow Yield")),
                ("Current Ratio", metrics.get("Current Ratio")),
                ("D/E", metrics.get("D/E")),
            ]
            for row_start in range(0, len(kpi_cards), 3):
                cols = st.columns(3)
                for col, (label, value) in zip(cols, kpi_cards[row_start:row_start+3]):
                    if label in ["ROIC", "EBITDA Margin", "FCF Yield"]:
                        display_value = format_ratio(value)
                    else:
                        display_value = format_value(value)
                    col.metric(label, display_value)
            
            metrics_df = pd.DataFrame([
                {
                    "Metric": name,
                    "Value": format_ratio(value) if name in [
                        "ROE",
                        "ROA",
                        "ROIC",
                        "Gross Margin",
                        "Operating Margin",
                        "Net Margin",
                        "EBITDA Margin",
                        "Free Cash Flow Yield",
                        "Interest Coverage",
                    ] else format_value(value),
                }
                for name, value in {**metrics, **ratios}.items()
            ])
            st.table(metrics_df)
            
            yoy_summary = build_yoy_summary(financials, balance_sheet, cashflow)
            if not yoy_summary.empty:
                with st.expander("สรุปงบปีต่อปี / ไตรมาส"):
                    st.dataframe(abbreviate_dataframe(yoy_summary.fillna("-").astype(object)))
            
            trend_chart = build_trend_chart(financials, cashflow)
            if not trend_chart.empty:
                with st.expander("กราฟ Revenue / Net Income / Operating Cash Flow"):
                    fig = px.line(
                        trend_chart.reset_index(),
                        x=trend_chart.index.name or trend_chart.reset_index().columns[0],
                        y=trend_chart.columns,
                        markers=True,
                    )
                    fig.update_yaxes(tickformat="~s")
                    st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("งบกำไรขาดทุน (Financials)"):
                st.dataframe(abbreviate_dataframe(financials.fillna("-").astype(object)))
            
            with st.expander("งบดุล (Balance Sheet)"):
                st.dataframe(abbreviate_dataframe(balance_sheet.fillna("-").astype(object)))
            
            with st.expander("กระแสเงินสด (Cash Flow)"):
                st.dataframe(abbreviate_dataframe(cashflow.fillna("-").astype(object)))
    else:
        summary_rows = []
        for ticker in selected_tickers:
            info, _, _, _, _, _, _ = load_financial_statement(ticker)
            ratios = calculate_additional_ratios(info, pd.DataFrame(), pd.DataFrame())
            summary_rows.append(
                {
                    "Ticker": ticker,
                    "P/E": format_value(info.get("trailingPE")),
                    "P/BV": format_value(info.get("priceToBook")),
                    "P/S": format_value(info.get("priceToSalesTrailing12Months")),
                    "D/E": format_value(info.get("debtToEquity")),
                    "Current Ratio": format_value(info.get("currentRatio")),
                    "ROE": format_ratio(ratios.get("ROE")),
                    "ROA": format_ratio(ratios.get("ROA")),
                    "ROIC": format_ratio(ratios.get("ROIC")),
                    "FCF Yield": format_ratio(ratios.get("Free Cash Flow Yield")),
                }
            )
        
        st.dataframe(pd.DataFrame(summary_rows))

st.markdown("---")

# ============= Sentiment Analysis Section =============
if show_sentiment and len(selected_tickers) == 1:
    st.subheader("📰 Sentiment Analysis (วิเคราะห์ความรู้สึกข่าว)")
    
    ticker = selected_tickers[0]
    
    st.write(f"กำลังดึงข่าวล่าสุดสำหรับ **{ticker}**...")
    
    # ตรวจสอบว่าผู้ใช้ได้ตั้งค่า API Key หรือไม่
    api_key_placeholder = st.secrets.get("NEWS_API_KEY", "")
    
    if not api_key_placeholder:
        st.warning(
            "⚠️ **ไม่พบ News API Key**  \n"
            "โปรดสร้างไฟล์ `.streamlit/secrets.toml` และเพิ่ม:  \n"
            "```\nNEWS_API_KEY = \"your_api_key_here\"\n```\n"
            "ดึง API Key จาก: https://newsapi.org"
        )
    else:
        articles = fetch_news(ticker, api_key_placeholder)
        
        if articles:
            try:
                sentiment_model = load_sentiment_model()
                
                # สร้างรายการข่าว
                sentiments_data = []
                for article in articles:
                    title = article.get("title", "")
                    description = article.get("description", "") or ""
                    text = f"{title}. {description}"
                    
                    sentiment_result = analyze_sentiment_vader(text, sentiment_model)
                    sentiments_data.append({
                        "Title": title,
                        "Source": article.get("source", {}).get("name", "Unknown"),
                        "Sentiment": sentiment_result["label"],
                        "Score": sentiment_result["score"],
                        "URL": article.get("url", "")
                    })
                
                # คำนวณ Sentiment ทั้งหมด
                positive = sum(1 for s in sentiments_data if s["Sentiment"] == "POSITIVE")
                negative = sum(1 for s in sentiments_data if s["Sentiment"] == "NEGATIVE")
                neutral = sum(1 for s in sentiments_data if s["Sentiment"] not in ["POSITIVE", "NEGATIVE"])
                
                # แสดงสรุป
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🟢 Positive", positive)
                with col2:
                    st.metric("🔴 Negative", negative)
                with col3:
                    st.metric("⚪ Neutral", neutral)
                
                # Sentiment Score ทั้งหมด
                avg_positive_score = sum(
                    s["Score"] for s in sentiments_data if s["Sentiment"] == "POSITIVE"
                ) / max(positive, 1)
                
                if positive > negative:
                    st.success(f"✅ **โดยรวม: Positive** (Average Score: {avg_positive_score:.2f})")
                elif negative > positive:
                    st.error(f"❌ **โดยรวม: Negative**")
                else:
                    st.info("⚪ **โดยรวม: Neutral**")
                
                # แสดงรายละเอียดข่าว
                with st.expander("ดูรายละเอียดข่าวทั้งหมด"):
                    for idx, item in enumerate(sentiments_data, 1):
                        sentiment_emoji = "🟢" if item["Sentiment"] == "POSITIVE" else "🔴" if item["Sentiment"] == "NEGATIVE" else "⚪"
                        st.write(
                            f"{sentiment_emoji} **{idx}. {item['Title']}**  \n"
                            f"📰 {item['Source']} | Score: {item['Score']:.2f}  \n"
                            f"[อ่านข่าวเต็ม]({item['URL']})"
                        )
            
            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาดในการวิเคราะห์ Sentiment: {e}")
        else:
            st.info("ℹ️ ไม่พบข่าวสำหรับหุ้น นี้")

elif show_sentiment and len(selected_tickers) > 1:
    st.info("💡 Sentiment Analysis จะแสดงเมื่อเลือกหุ้น **เพียงหนึ่งตัว** เท่านั้น")

st.markdown("---")

with st.expander("ดูข้อมูลราคาย้อนหลัง"):
    st.dataframe(df)

st.caption("สังเกต: ข้อมูลราคาถูกดึงจาก yfinance ซึ่งอาจใช้เวลารอเมื่อต้องโหลดหลายตัวพร้อมกัน")
