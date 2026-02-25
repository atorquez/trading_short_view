import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# VERSION 1.5

import streamlit as st

# MUST come before any Streamlit layout or widgets
st.set_page_config(layout="wide")

# Custom dark theme styling
st.markdown("""
<style>

    /* Global background */
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background-color: #0d1117 !important;
        color: #e6e6e6 !important;
    }

    /* Main content area */
    [data-testid="stAppViewContainer"] > .main, .block-container {
        background-color: #0d1117 !important;
        color: #e6e6e6 !important;
    }

    /* Labels */
    label, .stTextInput label {
        font-weight: 600 !important;
        color: #e6e6e6 !important;
    }

    /* Table header */
    thead tr th {
        font-size: 18px !important;
        font-weight: 800 !important;
        background-color: #161b22 !important;
        color: #f1c40f !important;
        border-bottom: 2px solid #f1c40f !important;
    }

    /* Table body */
    tbody tr td {
        font-size: 16px !important;
        color: #e6e6e6 !important;
    }

    tbody tr:hover {
        background-color: #1f2937 !important;
    }

    /* Buttons */
    .stButton>button {
        background-color: #f1c40f !important;
        color: black !important;
        font-weight: 700 !important;
        border-radius: 6px !important;
        height: 3em !important;
        width: 10em !important;
    }

    .stButton>button:hover {
        background-color: #d4ac0d !important;
        color: white !important;
    }

</style>
""", unsafe_allow_html=True)

st.write("Trade-short-term")
st.title("📈 Stock Decisions")


# ===============================
# CORE CONFIG DEFAULTS
# ===============================

DEFAULT_TICKER = "ZTS"
DEFAULT_START = "2015-01-01"
DEFAULT_INITIAL_CAPITAL = 100000
DEFAULT_RISK_PER_TRADE = 0.01
DEFAULT_HOLD_DAYS = 5

MA_FAST = 5
MA_MED = 20
MA_SLOW = 50
HH_LOOKBACK = 5
LL_LOOKBACK = 3


# ===============================
# DATA & INDICATORS
# ===============================

def download_data(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end)

    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns]
    df.columns = [c.split("_")[0] for c in df.columns]

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['MA_FAST'] = df['Close'].rolling(MA_FAST).mean()
    df['MA_MED'] = df['Close'].rolling(MA_MED).mean()
    df['MA_SLOW'] = df['Close'].rolling(MA_SLOW).mean()

    df['MA_SLOW_slope'] = df['MA_SLOW'] > df['MA_SLOW'].shift(1)

    df['HH'] = df['High'].rolling(HH_LOOKBACK).max().shift(1)
    df['LL'] = df['Low'].rolling(LL_LOOKBACK).min().shift(1)

    return df


def add_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['trend_filter'] = (df['Close'] > df['MA_SLOW']) & (df['MA_SLOW_slope'])

    df['entry_signal'] = (
        (df['Close'] > df['HH']) &
        (df['Close'] > df['MA_MED']) &
        (df['MA_MED'] > df['MA_MED'].shift(1)) &
        df['trend_filter']
    )

    return df


# ===============================
# BACKTEST ENGINE
# ===============================

def run_backtest(df: pd.DataFrame,
                 initial_capital: float,
                 risk_per_trade: float,
                 hold_days: int):

    df = df.copy()

    capital = initial_capital
    equity_curve = []
    position = 0
    entry_price = 0.0
    entry_index = None
    holding_counter = 0

    trades = []

    start_i = max(MA_SLOW, HH_LOOKBACK + 1, LL_LOOKBACK + 1)

    for i in range(start_i, len(df)):
        row = df.iloc[i]
        idx = df.index[i]

        if position > 0:
            equity = capital + position * row['Close']
        else:
            equity = capital
        equity_curve.append({'Date': idx, 'Equity': equity})

        # EXIT LOGIC
        if position > 0:
            holding_counter += 1

            stop_price = df['LL'].iloc[i]
            if np.isnan(stop_price):
                stop_price = entry_price * 0.97

            exit_flag = False
            exit_reason = None

            if row['Close'] < stop_price:
                exit_flag = True
                exit_reason = "stop"

            if not exit_flag and holding_counter >= hold_days:
                exit_flag = True
                exit_reason = "time"

            if exit_flag:
                exit_price = row['Close']
                pnl = (exit_price - entry_price) * position
                capital += pnl

                trades.append({
                    'entry_date': entry_index,
                    'exit_date': idx,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position': position,
                    'pnl': pnl,
                    'return_pct': pnl / (entry_price * position),
                    'holding_period': holding_counter,
                    'exit_reason': exit_reason
                })

                position = 0
                entry_price = 0.0
                entry_index = None
                holding_counter = 0
                continue

        # ENTRY LOGIC
        if position == 0 and row['entry_signal']:
            stop_level = df['LL'].iloc[i]
            if np.isnan(stop_level) or stop_level >= row['Close']:
                continue

            risk_amount = capital * risk_per_trade
            risk_per_share = max(row['Close'] - stop_level, row['Close'] * 0.005)
            shares = int(risk_amount / risk_per_share)
            if shares <= 0:
                continue

            position = shares
            entry_price = row['Close']
            entry_index = idx
            holding_counter = 0

    equity_df = pd.DataFrame(equity_curve).set_index('Date')
    trades_df = pd.DataFrame(trades)

    return equity_df, trades_df, capital, position, entry_price, holding_counter


# ===============================
# PERFORMANCE METRICS
# ===============================

def compute_performance(equity_df, trades_df, initial_capital):
    perf = {}

    if equity_df.empty:
        return {"error": "No equity data"}

    equity = equity_df['Equity']
    returns = equity.pct_change().dropna()

    total_return = equity.iloc[-1] / initial_capital - 1
    perf['final_equity'] = float(equity.iloc[-1])
    perf['total_return_pct'] = float(total_return * 100)

    days = (equity.index[-1] - equity.index[0]).days
    years = days / 252 if days > 0 else 0
    if years > 0:
        cagr = (equity.iloc[-1] / initial_capital) ** (1 / years) - 1
    else:
        cagr = np.nan
    perf['CAGR'] = float(cagr * 100) if not np.isnan(cagr) else None

    if not returns.empty:
        vol = returns.std() * np.sqrt(252)
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else np.nan
    else:
        vol = np.nan
        sharpe = np.nan
    perf['volatility_pct'] = float(vol * 100) if not np.isnan(vol) else None
    perf['sharpe'] = float(sharpe) if not np.isnan(sharpe) else None

    roll_max = equity.cummax()
    dd = equity / roll_max - 1
    perf['max_drawdown_pct'] = float(dd.min() * 100)

    if not trades_df.empty:
        pnl = trades_df['pnl']
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]

        perf['num_trades'] = int(len(trades_df))
        perf['win_rate_pct'] = float((len(wins) / len(trades_df)) * 100)
        perf['avg_win'] = float(wins.mean()) if len(wins) > 0 else None
        perf['avg_loss'] = float(losses.mean()) if len(losses) > 0 else None

        if len(wins) > 0 and len(losses) > 0:
            profit_factor = wins.sum() / abs(losses.sum())
        else:
            profit_factor = np.nan
        perf['profit_factor'] = float(profit_factor) if not np.isnan(profit_factor) else None

        perf['expectancy'] = float(pnl.mean())
        perf['avg_holding_period'] = float(trades_df['holding_period'].mean())
    else:
        perf['num_trades'] = 0

    return perf


# ===============================
# DASHBOARD COMPONENTS
# ===============================

def recent_signal_snapshot(df: pd.DataFrame, days: int = 7):
    cols = ['Close', 'HH', 'LL', 'MA_MED', 'MA_SLOW', 'trend_filter', 'entry_signal']
    available = [c for c in cols if c in df.columns]
    return df[available].tail(days)


def todays_recommendation(df, position, entry_price, hold_counter, hold_days):
    if df.empty:
        return "NO DATA"

    last = df.iloc[-1]

    if position == 0:
        return "BUY" if last['entry_signal'] else "NO SIGNAL"

    stop_price = last['LL']
    if np.isnan(stop_price):
        stop_price = entry_price * 0.97

    if last['Close'] < stop_price:
        return "SELL (STOP LOSS)"

    if hold_counter >= hold_days:
        return "SELL (TIME EXIT)"

    return "HOLD"


def compute_trend_strength(df):
    if len(df) < 2:
        return 0

    last = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0

    if last['Close'] > last['MA_SLOW']:
        score += 30
    if last['Close'] > last['MA_MED']:
        score += 20
    if last['MA_SLOW'] > prev['MA_SLOW']:
        score += 25
    if last['MA_MED'] > prev['MA_MED']:
        score += 15
    if last['Close'] > last['HH'] * 0.98:
        score += 10

    return min(score, 100)


def compute_risk_meter(df):
    if df.empty:
        return {"stop_distance_pct": None, "volatility_pct": None, "risk_level": "N/A"}

    last = df.iloc[-1]

    stop = last['LL']
    if np.isnan(stop):
        stop = last['Close'] * 0.97

    stop_distance = (last['Close'] - stop) / last['Close']
    vol = (df['High'] - df['Low']).rolling(14).mean().iloc[-1] / last['Close']

    return {
        "stop_distance_pct": round(stop_distance * 100, 2),
        "volatility_pct": round(vol * 100, 2),
        "risk_level": (
            "LOW" if stop_distance < 0.03 else
            "MEDIUM" if stop_distance < 0.06 else
            "HIGH"
        )
    }


def generate_signal_chart(df, last_n=180):
    if df.empty:
        return go.Figure()

    d = df.tail(last_n)

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=d.index,
        open=d['Open'],
        high=d['High'],
        low=d['Low'],
        close=d['Close'],
        name='Price'
    ))

    fig.add_trace(go.Scatter(x=d.index, y=d['MA_MED'], mode='lines',
                             name='MA20', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=d.index, y=d['MA_SLOW'], mode='lines',
                             name='MA50', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=d.index, y=d['HH'], mode='lines',
                             name='HH5', line=dict(color='green', dash='dot')))
    fig.add_trace(go.Scatter(x=d.index, y=d['LL'], mode='lines',
                             name='LL3', line=dict(color='red', dash='dot')))

    fig.update_layout(
        title=f"{ticker} – Price & Signals",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        height=600
    )

    return fig


# ===============================
# STREAMLIT APP
# ===============================

st.set_page_config(page_title="Short-Term Trade Engine", layout="wide")

st.title("Swing Breakout Backtest & Signal Dashboard")

with st.sidebar:
    st.header("Parameters")

    ticker = st.text_input("Ticker", value=DEFAULT_TICKER)
    start_date = st.text_input("Start Date", value=DEFAULT_START)
    end_date = st.text_input("End Date (optional)", value="")

    initial_capital = st.number_input("Initial Capital", value=DEFAULT_INITIAL_CAPITAL, step=1000)
    risk_per_trade = st.number_input("Risk per Trade (fraction)", value=DEFAULT_RISK_PER_TRADE, step=0.005, format="%.3f")
    hold_days = st.number_input("Hold Days", value=DEFAULT_HOLD_DAYS, step=1)

    run_button = st.button("Run Backtest")

if run_button:
    end = end_date if end_date.strip() != "" else None

    df = download_data(ticker, start_date, end)

    if df.empty:
        st.error("No data downloaded. Check ticker or date range.")
    else:
        df = add_indicators(df)
        df = add_signals(df)

        equity_df, trades_df, final_capital, position, entry_price, holding_counter = run_backtest(
            df, initial_capital, risk_per_trade, hold_days
        )

        perf = compute_performance(equity_df, trades_df, initial_capital)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Performance")
            for k, v in perf.items():
                st.write(f"**{k}**: {v}")

        with col2:
            st.subheader("Today")
            rec = todays_recommendation(df, position, entry_price, holding_counter, hold_days)
            st.write(f"**Recommendation:** {rec}")

            trend_score = compute_trend_strength(df)
            st.write(f"**Trend Strength:** {trend_score}/100")

        with col3:
            st.subheader("Risk")
            risk = compute_risk_meter(df)
            st.write(risk)

        st.subheader("Recent Signal Snapshot")
        st.dataframe(recent_signal_snapshot(df, days=10))

        st.subheader("Equity Curve")
        if not equity_df.empty:
            st.line_chart(equity_df['Equity'])

        st.subheader("Price & Signals")
        fig = generate_signal_chart(df)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Trades")
        if not trades_df.empty:
            st.dataframe(trades_df)
        else:
            st.write("No trades generated for this configuration.")
else:
    st.info("Set parameters in the sidebar and click **Run Backtest**.")

#DEFINITONS
st.markdown("---")
st.subheader("📘 Key Indicator Definitions")

st.markdown("""
| **Indicator** | **Meaning** |
|---------------|-------------|
| **HH5** | Highest High of the last 5 days. Used as the breakout trigger. |
| **LL3** | Lowest Low of the last 3 days. Used as the stop-loss level. |
| **MA20** | 20-day moving average. Measures short-term trend direction. |
| **MA50** | 50-day moving average. Measures medium-term trend strength. |
| **Trend Strength** | Composite score (0–100) based on price vs. MA20/MA50 and slope direction. |
| **Entry Signal** | True when price breaks above HH5 *and* all trend filters are satisfied. |
| **Stop Distance %** | Distance between current price and LL3 stop level, expressed as a percentage. |
| **Volatility %** | Short-term price volatility used for risk assessment. |
| **Risk Level** | LOW / MEDIUM / HIGH based on stop distance and volatility. |
| **Exit Reason** | Why the trade closed: `time` (5-day rule) or `stop` (LL3 break). |
| **Today's Recommendation** | The system’s action for today: **BUY**, **SELL**, or **NO SIGNAL**. |
| **Why Today’s Recommendation?** | **BUY** = breakout above HH5 + trend filters valid. **SELL** = active trade hit stop or time exit. **NO SIGNAL** = breakout conditions not met today. |
""")