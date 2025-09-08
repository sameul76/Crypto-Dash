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
# P&L calc (kept minimal for this view)
# =========================================================
def calculate_pnl_and_metrics(trades_df):
    if trades_df is None or trades_df.empty:
        return {}, pd.DataFrame(), {}
    pnl_per_asset, positions = {}, {}
    df = trades_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['pnl'] = 0.0
    df['cumulative_pnl'] = 0.0
    total_cum = 0.0
    win = loss = 0
    gp = gl = 0.0
    peak = mdd = 0.0

    for i, row in df.iterrows():
        asset = row['asset']; action = row['action']
        price = float(row['price']); qty = float(row['quantity'])
        if asset not in positions:
            positions[asset] = {'quantity': 0.0, 'cost': 0.0}
            pnl_per_asset[asset] = 0.0
        cur = 0.0
        if action == 'buy':
            positions[asset]['cost'] += qty * price
            positions[asset]['quantity'] += qty
        elif action == 'sell' and positions[asset]['quantity'] > 0:
            avg = positions[asset]['cost'] / positions[asset]['quantity']
            trade_qty = min(qty, positions[asset]['quantity'])
            realized = (price - avg) * trade_qty
            pnl_per_asset[asset] += realized
            total_cum += realized
            cur = realized
            if realized > 0: win += 1; gp += realized
            else: loss += 1; gl += abs(realized)
            positions[asset]['cost'] -= avg * trade_qty
            positions[asset]['quantity'] -= trade_qty

        df.loc[i, 'pnl'] = cur
        df.loc[i, 'cumulative_pnl'] = total_cum
        peak = max(peak, total_cum)
        mdd = max(mdd, peak - total_cum)

    closed = win + loss
    stats = {
        'win_rate': (win / closed * 100) if closed > 0 else 0,
        'profit_factor': (gp / gl) if gl > 0 else float('inf'),
        'total_trades': closed,
        'avg_win': (gp / win) if win > 0 else 0,
        'avg_loss': (gl / loss) if loss > 0 else 0,
        'max_drawdown': mdd
    }
    df['asset_cumulative_pnl'] = df.groupby('asset')['pnl'].cumsum()
    return pnl_per_asset, df, stats

# =========================================================
# Header
# =========================================================
st.markdown("""
<div style="text-align:center; padding:2rem 0;">
  <h1 style="margin:0;">Trading Performance Analytics</h1>
  <p style="margin:.25rem 0; opacity:.7;">Hover to see P-Up / P-Down. Like a heartbeat monitor, but for your portfolio.</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# Data loading (Drive) with probability merge
# =========================================================
@st.cache_data(ttl=600)
def load_data():
    service = get_gdrive_service()
    if service is None:
        return None, None, None, None

    # ✅ Your actual file IDs from the links you provided
    TRADES_FILE_ID = "1zSdFcG4Xlh_iSa180V6LRSeEucAokXYk"  # trade data CSV
    OHLC_FILE_ID   = "1kxOB0PCGaT7N0Ljv4-EUDqYCnnTyd0SK"  # OHLCV parquet
    PROB_FILE_ID   = "1YKuVKIQ5Esgo8G44ODfbnc6_UsSY4CFX"  # probability log CSV

    trades = read_gdrive_csv(download_file_from_drive(TRADES_FILE_ID, service))
    ohlc   = read_gdrive_parquet(download_file_from_drive(OHLC_FILE_ID, service))
    probs  = read_gdrive_csv(download_file_from_drive(PROB_FILE_ID, service))

    # -------- Trades normalize
    if trades is not None and not trades.empty:
        trades = lower_strip_cols(trades)
        if 'product_id' in trades.columns and 'asset' not in trades.columns:
            trades = trades.rename(columns={'product_id':'asset'})
        trades = trades.rename(columns={'side':'action','size':'quantity'})
        trades['timestamp'] = to_local_naive(trades['timestamp'])
        if 'action' in trades.columns:
            trades['action'] = trades['action'].str.lower()
        req_tr = ['timestamp','asset','action','price','quantity']
        if not all(c in trades.columns for c in req_tr):
            st.error(f"Trades file missing required columns: {req_tr}")
            trades = None

    # -------- OHLC normalize
    if ohlc is not None and not ohlc.empty:
        ohlc = lower_strip_cols(ohlc)
        if 'product_id' in ohlc.columns and 'asset' not in ohlc.columns:
            ohlc = ohlc.rename(columns={'product_id':'asset'})
        req_ohlc = ['timestamp','asset','open','high','low','close']
        if not all(c in ohlc.columns for c in req_ohlc):
            st.error(f"OHLC file missing required columns: {req_ohlc}")
            ohlc = None
        else:
            ohlc['timestamp'] = to_local_naive(ohlc['timestamp'])

    # -------- Probability log: normalize + merge_asof (±1 min) per asset
    if probs is not None and not probs.empty and ohlc is not None and not ohlc.empty:
        probs = lower_strip_cols(probs)
        if 'product_id' in probs.columns and 'asset' not in probs.columns:
            probs = probs.rename(columns={'product_id':'asset'})
        # standardize probability headers to p_up / p_down
        rename_map = {}
        for c in list(probs.columns):
            cl = c.lower()
            if cl in {'p_up','p-up','pup','prob_up','p_up_prob','puprob'}:      rename_map[c] = 'p_up'
            if cl in {'p_down','p-down','pdown','prob_down','p_down_prob','pdownprob'}: rename_map[c] = 'p_down'
        probs = probs.rename(columns=rename_map)

        keep = [c for c in ['timestamp','asset','p_up','p_down'] if c in probs.columns]
        probs = probs[keep].copy()
        probs['timestamp'] = to_local_naive(probs['timestamp'])
        for pc in ['p_up','p_down']:
            if pc in probs.columns:
                probs[pc] = pd.to_numeric(probs[pc], errors='coerce')

        ohlc = ohlc.sort_values(['asset','timestamp'])
        probs = probs.sort_values(['asset','timestamp'])
        ohlc = pd.merge_asof(
            ohlc,
            probs,
            on='timestamp',
            by='asset',
            direction='nearest',
            tolerance=pd.Timedelta('1min')
        )
    else:
        if ohlc is not None:
            if 'p_up' not in ohlc.columns:   ohlc['p_up'] = np.nan
            if 'p_down' not in ohlc.columns: ohlc['p_down'] = np.nan

    # P&L calc (optional, but handy)
    if trades is not None and not trades.empty:
        pnl_summary, trades_with_pnl, stats = calculate_pnl_and_metrics(trades)
    else:
        pnl_summary, trades_with_pnl, stats = {}, pd.DataFrame(), {}

    return trades_with_pnl, pnl_summary, ohlc, stats

trades_df, pnl_summary, ohlc_df, summary_stats = load_data()

# =========================================================
# Main charts
# =========================================================
if ohlc_df is None or ohlc_df.empty:
    st.error("No OHLC data available. Check your file access and columns.")
else:
    # Asset picker
    assets = sorted(ohlc_df['asset'].dropna().unique())
    selected_asset = st.selectbox("Select Asset", assets, index=0)

    # Time range
    colA, colB = st.columns([3,1])
    with colA:
        pass
    with colB:
        time_range = st.selectbox('Time Range', options=['30 days','7 days','1 day','All'], index=0)

    # Filter OHLC for asset + time
    asset_ohlc = ohlc_df[ohlc_df['asset'] == selected_asset].copy()
    asset_ohlc['timestamp'] = pd.to_datetime(asset_ohlc['timestamp'])
    if asset_ohlc.empty:
        st.warning("No rows for selected asset.")
    else:
        end_date = asset_ohlc['timestamp'].max()
        if time_range == '1 day':
            start_date = end_date - timedelta(days=1)
        elif time_range == '7 days':
            start_date = end_date - timedelta(days=7)
        elif time_range == '30 days':
            start_date = end_date - timedelta(days=30)
        else:
            start_date = asset_ohlc['timestamp'].min()

        ohlc = asset_ohlc[(asset_ohlc['timestamp'] >= start_date) & (asset_ohlc['timestamp'] <= end_date)].copy()
        ohlc = ohlc.sort_values('timestamp').reset_index(drop=True)

        # Build hover text (like your working script)
        def fmt(v): 
            return f"{float(v):.4f}" if v is not None and pd.notna(v) else "—"

        hovertext = [
            "📅 " + row['timestamp'].strftime("%Y-%m-%d %H:%M:%S") +
            f"<br>Open: ${fmt(row['open'])}" +
            f"<br>High: ${fmt(row['high'])}" +
            f"<br>Low: ${fmt(row['low'])}" +
            f"<br>Close: ${fmt(row['close'])}" +
            f"<br>P-Up: {fmt(row.get('p_up'))}" +
            f"<br>P-Down: {fmt(row.get('p_down'))}"
            for _, row in ohlc.iterrows()
        ]

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=ohlc["timestamp"],
            open=ohlc["open"], high=ohlc["high"],
            low=ohlc["low"],  close=ohlc["close"],
            name="Price",
            text=hovertext,
            hoverinfo="text"
        ))

        # Overlay trades (optional)
        if trades_df is not None and not trades_df.empty:
            asset_trades = trades_df[trades_df['asset'] == selected_asset]
            if not asset_trades.empty:
                buys  = asset_trades[asset_trades['action'] == 'buy']
                sells = asset_trades[asset_trades['action'] == 'sell']
                if not buys.empty:
                    fig.add_trace(go.Scatter(
                        x=buys['timestamp'], y=buys['price'],
                        mode='markers', name='Buy',
                        marker=dict(color='#10b981', symbol='triangle-up', size=14, line=dict(width=2, color='#059669')),
                        hovertemplate='<b>BUY</b><br>Price: $%{y:.2f}<br>%{x}<extra></extra>'
                    ))
                if not sells.empty:
                    fig.add_trace(go.Scatter(
                        x=sells['timestamp'], y=sells['price'],
                        mode='markers', name='Sell',
                        marker=dict(color='#ef4444', symbol='triangle-down', size=14, line=dict(width=2, color='#dc2626')),
                        hovertemplate='<b>SELL</b><br>Price: $%{y:.2f}<br>%{x}<extra></extra>'
                    ))

        fig.update_layout(
            template='plotly_white',
            xaxis_rangeslider_visible=True,
            hovermode='x unified',
            yaxis_title="Price (USD)",
            title=f"{selected_asset} — Price & Probabilities"
        )
        st.plotly_chart(fig, use_container_width=True)

# (Optional) quick P&L timeline
if trades_df is not None and not trades_df.empty:
    st.markdown("### Portfolio P&L (Quick View)")
    buys = trades_df[trades_df['action'] == 'buy']
    sells = trades_df[trades_df['action'] == 'sell']
    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Scatter(x=trades_df['timestamp'], y=trades_df['cumulative_pnl'],
                                 mode='lines', name='Cumulative P&L'))
    fig_pnl.add_trace(go.Scatter(x=buys['timestamp'], y=buys['cumulative_pnl'],
                                 mode='markers', name='Buys'))
    fig_pnl.add_trace(go.Scatter(x=sells['timestamp'], y=sells['cumulative_pnl'],
                                 mode='markers', name='Sells'))
    fig_pnl.update_layout(template='plotly_white', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig_pnl, use_container_width=True)
