import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io

# Google Drive
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="Trading Performance Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'About': "Professional Trading Performance Dashboard"}
)

# =========================================================
# Google Drive helpers
# =========================================================
def get_gdrive_service():
    if "gcp_service_account" not in st.secrets:
        st.error("GCP service account credentials not found in Streamlit Secrets.")
        st.info("Add your service account JSON to .streamlit/secrets.toml under [gcp_service_account].")
        return None
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=['https://www.googleapis.com/auth/drive.readonly']
    )
    return build('drive', 'v3', credentials=creds)

def download_file_from_drive(file_id, service):
    try:
        request = service.files().get_media(fileId=file_id)
        file_bytes = io.BytesIO()
        downloader = MediaIoBaseDownload(file_bytes, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        file_bytes.seek(0)
        return file_bytes
    except Exception as e:
        st.error(f"Error downloading file with ID '{file_id}'. Details: {e}")
        return None

def read_gdrive_csv(file_bytes):
    if file_bytes is None:
        return None
    try:
        return pd.read_csv(file_bytes)
    except Exception:
        file_bytes.seek(0)
        return pd.read_csv(file_bytes, encoding="latin1")

def read_gdrive_parquet(file_bytes):
    if file_bytes is None:
        return None
    return pd.read_parquet(file_bytes)

# =========================================================
# Helpers
# =========================================================
LOCAL_TZ = 'America/Los_Angeles'

def to_local_naive(ts):
    s = pd.to_datetime(ts, errors='coerce')
    try:
        if s.dt.tz is not None:
            s = s.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    except Exception:
        try:
            s = s.dt.tz_localize('UTC').dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
        except Exception:
            s = s.dt.tz_localize(None)
    return s

def lower_strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.lower().strip() for c in out.columns]
    return out

# =========================================================
# Data loader (with probability merge)
# =========================================================
@st.cache_data(ttl=600)
def load_data():
    service = get_gdrive_service()
    if service is None:
        return None, None, None

    TRADES_FILE_ID = "PUT_TRADES_FILE_ID_HERE"
    OHLC_FILE_ID   = "PUT_OHLC_FILE_ID_HERE"
    PROB_FILE_ID   = "PUT_PROB_LOG_FILE_ID_HERE"

    trades = read_gdrive_csv(download_file_from_drive(TRADES_FILE_ID, service))
    ohlc   = read_gdrive_parquet(download_file_from_drive(OHLC_FILE_ID, service))
    probs  = read_gdrive_csv(download_file_from_drive(PROB_FILE_ID, service))

    if trades is not None:
        trades = lower_strip_cols(trades)
        if 'product_id' in trades.columns and 'asset' not in trades.columns:
            trades = trades.rename(columns={'product_id':'asset'})
        trades = trades.rename(columns={'side':'action','size':'quantity'})
        trades['timestamp'] = to_local_naive(trades['timestamp'])

    if ohlc is not None:
        ohlc = lower_strip_cols(ohlc)
        if 'product_id' in ohlc.columns and 'asset' not in ohlc.columns:
            ohlc = ohlc.rename(columns={'product_id':'asset'})
        ohlc['timestamp'] = to_local_naive(ohlc['timestamp'])

    if probs is not None and not probs.empty and ohlc is not None and not ohlc.empty:
        probs = lower_strip_cols(probs)
        if 'product_id' in probs.columns and 'asset' not in probs.columns:
            probs = probs.rename(columns={'product_id':'asset'})
        rename_map = {}
        for c in probs.columns:
            cl = c.lower()
            if cl.startswith("p_up"): rename_map[c] = 'p_up'
            if cl.startswith("p_down"): rename_map[c] = 'p_down'
        probs = probs.rename(columns=rename_map)
        probs['timestamp'] = to_local_naive(probs['timestamp'])
        ohlc = pd.merge_asof(
            ohlc.sort_values(['asset','timestamp']),
            probs.sort_values(['asset','timestamp']),
            on='timestamp',
            by='asset',
            direction='nearest',
            tolerance=pd.Timedelta('1min')
        )
    else:
        if ohlc is not None:
            ohlc['p_up'] = np.nan
            ohlc['p_down'] = np.nan

    return trades, ohlc, probs

trades_df, ohlc_df, probs_df = load_data()

# =========================================================
# Chart
# =========================================================
if ohlc_df is not None and not ohlc_df.empty:
    selected_asset = st.selectbox("Select Asset", ohlc_df['asset'].unique())

    filtered = ohlc_df[ohlc_df['asset'] == selected_asset].copy()
    filtered = filtered.sort_values("timestamp")

    def fmt(v): return f"{float(v):.4f}" if pd.notna(v) else "—"
    hovertext = [
        "📅 " + row['timestamp'].strftime("%Y-%m-%d %H:%M:%S") +
        f"<br>Open: ${fmt(row['open'])}" +
        f"<br>High: ${fmt(row['high'])}" +
        f"<br>Low: ${fmt(row['low'])}" +
        f"<br>Close: ${fmt(row['close'])}" +
        f"<br>P-Up: {fmt(row.get('p_up'))}" +
        f"<br>P-Down: {fmt(row.get('p_down'))}"
        for _, row in filtered.iterrows()
    ]

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=filtered["timestamp"],
        open=filtered["open"], high=filtered["high"],
        low=filtered["low"], close=filtered["close"],
        name="Price",
        text=hovertext,
        hoverinfo="text"
    ))

    # Example buy/sell markers if trades exist
    if trades_df is not None and not trades_df.empty:
        asset_trades = trades_df[trades_df['asset'] == selected_asset]
        buys = asset_trades[asset_trades['action'] == 'buy']
        sells = asset_trades[asset_trades['action'] == 'sell']

        fig.add_trace(go.Scatter(
            x=buys['timestamp'], y=buys['price'],
            mode='markers', name='Buy',
            marker=dict(color='#10b981', symbol='triangle-up', size=14, line=dict(width=2, color='#059669')),
            hovertemplate='<b>BUY</b><br>Price: $%{y:.2f}<br>%{x}<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=sells['timestamp'], y=sells['price'],
            mode='markers', name='Sell',
            marker=dict(color='#ef4444', symbol='triangle-down', size=14, line=dict(width=2, color='#dc2626')),
            hovertemplate='<b>SELL</b><br>Price: $%{y:.2f}<br>%{x}<extra></extra>'
        ))

    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("No OHLC data available. Check your file IDs.")
