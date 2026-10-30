"""
Nifty 50 + Bank Nifty synchronized EMA-cross / Marubozu signal tool.

Install:  pip install streamlit yfinance plotly pandas streamlit-autorefresh
Run:      streamlit run nifty_signal_tool.py

Signal (per index, per closed candle):
  BUY  = EMA fast crosses above EMA slow AND bullish marubozu (body >= X% of range)
  SELL = EMA fast crosses below EMA slow AND bearish marubozu
Final signal fires only when BOTH indices show the same direction on the
same candle (or within the "sync window" you set, default 0 = same candle).
"""
import smtplib
from email.message import EmailMessage

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

SYMBOLS = {"Nifty 50": "^NSEI", "Bank Nifty": "^NSEBANK"}
PERIOD_FOR = {"1m": "5d", "5m": "5d", "15m": "30d", "30m": "30d", "1h": "60d"}

st.set_page_config(page_title="Nifty / Bank Nifty Signals", layout="wide")

# ---------------- Sidebar settings ----------------
with st.sidebar:
    st.header("Settings")
    tf = st.selectbox("Time frame", list(PERIOD_FOR), index=1)
    fast = st.number_input("Fast EMA", 2, 200, 9)
    slow = st.number_input("Slow EMA", 3, 400, 15)
    body_pct = st.slider("Min candle body % of range", 50, 100, 90)
    sync_window = st.number_input("Sync window (candles)", 0, 5, 0,
                                  help="0 = both indices must signal on the same candle")
    refresh = st.number_input("Refresh every (sec)", 10, 300, 30)
    st.subheader("Alerts")
    use_sound = st.checkbox("Sound", True)
    use_popup = st.checkbox("Pop-up toast", True)
    use_email = st.checkbox("Email", False)
    if use_email:
        smtp_host = st.text_input("SMTP host", "smtp.gmail.com")
        smtp_port = st.number_input("SMTP port", 1, 65535, 465)
        mail_user = st.text_input("Sender email")
        mail_pass = st.text_input("App password", type="password")
        mail_to = st.text_input("Send to")

st_autorefresh(interval=int(refresh) * 1000, key="auto")


# ---------------- Data ----------------
@st.cache_data(ttl=20, show_spinner=False)
def load(symbol: str, interval: str) -> pd.DataFrame:
    df = yf.download(symbol, period=PERIOD_FOR[interval], interval=interval,
                     progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError("empty data returned")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    # Quick sanity checks for data discrepancies
    bad = (df["High"] < df[["Open", "Close"]].max(axis=1)) | (df["Low"] > df[["Open", "Close"]].min(axis=1))
    return df[~bad]


def analyse(df: pd.DataFrame, fast: int, slow: int, body_pct: int) -> pd.DataFrame:
    d = df.copy()
    d["ema_f"] = d["Close"].ewm(span=fast, adjust=False).mean()
    d["ema_s"] = d["Close"].ewm(span=slow, adjust=False).mean()
    above = d["ema_f"] > d["ema_s"]
    d["cross_up"] = above & ~above.shift(1, fill_value=above.iloc[0])
    d["cross_dn"] = ~above & above.shift(1, fill_value=above.iloc[0])
    rng = (d["High"] - d["Low"]).replace(0, pd.NA)
    body = (d["Close"] - d["Open"]).abs()
    d["marubozu"] = (body / rng * 100) >= body_pct
    d["bull"] = d["marubozu"] & (d["Close"] > d["Open"])
    d["bear"] = d["marubozu"] & (d["Close"] < d["Open"])
    # Signal on the cross candle itself being a marubozu
    d["buy"] = d["cross_up"] & d["bull"]
    d["sell"] = d["cross_dn"] & d["bear"]
    return d


def combine(a: pd.DataFrame, b: pd.DataFrame, window: int) -> pd.DataFrame:
    idx = a.index.intersection(b.index)
    out = pd.DataFrame(index=idx)
    for side in ("buy", "sell"):
        ra = a[side].reindex(idx).fillna(False).rolling(window + 1, min_periods=1).max().astype(bool)
        rb = b[side].reindex(idx).fillna(False).rolling(window + 1, min_periods=1).max().astype(bool)
        # fire on the candle where the second index confirms
        out[side] = ra & rb & (a[side].reindex(idx).fillna(False) | b[side].reindex(idx).fillna(False))
    return out


# ---------------- Alerts ----------------
def send_email(subject: str, body: str):
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, mail_user, mail_to
    msg.set_content(body)
    with smtplib.SMTP_SSL(smtp_host, int(smtp_port)) as s:
        s.login(mail_user, mail_pass)
        s.send_message(msg)


def beep():
    st.markdown(
        """<audio autoplay><source src="data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=" type="audio/wav"></audio>
        <script>
        const c=new (window.AudioContext||window.webkitAudioContext)();
        const o=c.createOscillator();o.connect(c.destination);o.frequency.value=880;
        o.start();setTimeout(()=>o.stop(),600);
        </script>""", unsafe_allow_html=True)


# ---------------- Main ----------------
st.title("Nifty 50 + Bank Nifty — Synchronized Signal Monitor")

data, errors = {}, []
for name, sym in SYMBOLS.items():
    try:
        data[name] = analyse(load(sym, tf), int(fast), int(slow), body_pct)
    except Exception as e:  # connection issues, empty data, etc.
        errors.append(f"{name}: {e}")

if errors:
    st.error("Data problem — signals paused. " + " | ".join(errors))
    st.stop()

n, b = data["Nifty 50"], data["Bank Nifty"]
# Use only fully closed candles (drop the still-forming last bar)
n, b = n.iloc[:-1], b.iloc[:-1]
if abs((n.index[-1] - b.index[-1]).total_seconds()) > 60:
    st.warning("Index feeds are out of sync by more than a minute; waiting for both to update.")

sig = combine(n, b, int(sync_window))

# Charts side by side
fig = make_subplots(rows=1, cols=2, subplot_titles=list(SYMBOLS))
for col, (name, d) in enumerate(zip(SYMBOLS, (n, b)), start=1):
    fig.add_trace(go.Candlestick(x=d.index, open=d.Open, high=d.High, low=d.Low,
                                 close=d.Close, name=name, showlegend=False), 1, col)
    fig.add_trace(go.Scatter(x=d.index, y=d.ema_f, name=f"EMA {fast}",
                             line=dict(color="orange", width=1.3), showlegend=col == 1), 1, col)
    fig.add_trace(go.Scatter(x=d.index, y=d.ema_s, name=f"EMA {slow}",
                             line=dict(color="royalblue", width=1.3), showlegend=col == 1), 1, col)
    for side, symbol, color, y in (("buy", "triangle-up", "lime", "Low"),
                                   ("sell", "triangle-down", "red", "High")):
        t = sig.index[sig[side]]
        if len(t):
            off = 0.9985 if side == "buy" else 1.0015
            fig.add_trace(go.Scatter(x=t, y=d.loc[t, y] * off, mode="markers",
                                     marker=dict(symbol=symbol, size=16, color=color),
                                     name=side.upper(), showlegend=col == 1), 1, col)
fig.update_layout(height=650, xaxis_rangeslider_visible=False, xaxis2_rangeslider_visible=False,
                  margin=dict(l=10, r=10, t=40, b=10))
# hide non-trading gaps
fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[15.5, 9.25], pattern="hour")])
st.plotly_chart(fig, use_container_width=True)

# Latest signal + alerting (only once per candle)
last_ts = sig.index[-1]
state = st.session_state.setdefault("alerted", set())
for side, label in (("buy", "BUY"), ("sell", "SELL")):
    if sig[side].iloc[-1]:
        st.success(f"{label} signal on both indices — candle {last_ts:%Y-%m-%d %H:%M}")
        key = (side, last_ts)
        if key not in state:
            state.add(key)
            msg = f"{label}: EMA{fast}/{slow} cross + marubozu on Nifty & Bank Nifty ({tf}) at {last_ts:%H:%M}"
            if use_popup:
                st.toast(msg, icon="🚨")
            if use_sound:
                beep()
            if use_email:
                try:
                    send_email(f"{label} signal", msg)
                except Exception as e:
                    st.warning(f"Email failed: {e}")
        break
else:
    st.info("No synchronized signal on the latest closed candle.")

with st.expander("Recent synchronized signals"):
    hist = sig[sig.buy | sig.sell].tail(20)
    st.dataframe(hist.assign(direction=hist.buy.map({True: "BUY", False: "SELL"}))[["direction"]])

st.caption("Free Yahoo data is delayed/unofficial. For live trading use a broker feed "
           "(Zerodha Kite, Upstox, Angel One, Dhan) by replacing load(). Educational tool, not financial advice.")
