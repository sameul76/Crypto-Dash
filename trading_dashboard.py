import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io

# Imports for Google Drive
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# --- Configuration ---
st.set_page_config(
    page_title="Trading Bot P&L Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Google Drive Data Loading ---

def get_gdrive_service():
    """Initializes the Google Drive API service using Streamlit's secrets."""
    if "gcp_service_account" not in st.secrets:
        st.error("GCP service account credentials not found in Streamlit Secrets.")
        st.info("Please follow the setup guide to add your credentials to your app's settings.")
        return None
    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=['https://www.googleapis.com/auth/drive.readonly'])
    service = build('drive', 'v3', credentials=creds)
    return service

def download_file_from_drive(file_id, service):
    """Downloads a file from Google Drive into an in-memory byte buffer."""
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
        st.error(f"Error downloading file with ID '{file_id}' from Google Drive.")
        st.warning(f"Details: {e}")
        st.info("Please ensure the file ID is correct and you have shared the file with the service account email.")
        return None

def read_gdrive_csv(file_bytes):
    """Reads CSV data from a byte buffer."""
    if file_bytes is None:
        return None
    parser_kwargs = {'engine': 'python', 'on_bad_lines': 'skip'}
    try:
        return pd.read_csv(file_bytes, encoding='utf-8', **parser_kwargs)
    except UnicodeDecodeError:
        file_bytes.seek(0)
        return pd.read_csv(file_bytes, encoding='latin1', **parser_kwargs)

def read_gdrive_parquet(file_bytes):
    """Reads Parquet data from a byte buffer."""
    if file_bytes is None:
        return None
    return pd.read_parquet(file_bytes)

# --- P&L Calculation ---

def calculate_pnl(trades_df):
    """Calculates per-asset P&L and a running total P&L."""
    if trades_df is None or trades_df.empty:
        return {}, pd.DataFrame()

    pnl_per_asset = {}
    positions = {}
    
    df = trades_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    df['pnl'] = 0.0
    df['cumulative_pnl'] = 0.0
    total_cumulative_pnl = 0

    for index, row in df.iterrows():
        asset = row['asset']
        action = row['action']
        price = row['price']
        quantity = row['quantity']

        if asset not in positions:
            positions[asset] = {'quantity': 0, 'cost': 0}
            pnl_per_asset[asset] = 0

        current_pnl = 0
        if action == 'buy':
            positions[asset]['cost'] += quantity * price
            positions[asset]['quantity'] += quantity
        elif action == 'sell':
            if positions[asset]['quantity'] > 0:
                avg_cost_per_unit = positions[asset]['cost'] / positions[asset]['quantity']
                realized_pnl = (price - avg_cost_per_unit) * min(quantity, positions[asset]['quantity'])
                pnl_per_asset[asset] += realized_pnl
                total_cumulative_pnl += realized_pnl
                current_pnl = realized_pnl
                positions[asset]['cost'] -= avg_cost_per_unit * min(quantity, positions[asset]['quantity'])
                positions[asset]['quantity'] -= quantity

        df.loc[index, 'pnl'] = current_pnl
        df.loc[index, 'cumulative_pnl'] = total_cumulative_pnl

    return pnl_per_asset, df

# --- Main App ---
st.title("📈 Trading Bot Performance Dashboard")

@st.cache_data(ttl=600)
def load_data():
    gdrive_service = get_gdrive_service()
    if gdrive_service is None:
        return None, None, None

    TRADES_FILE_ID = "1zSdFcG4Xlh_iSa180V6LRSeEucAokXYk"
    OHLC_FILE_ID = "1kxOB0PCGaT7N0Ljv4-EUDqYCnnTyd0SK"
    
    trades_bytes = download_file_from_drive(TRADES_FILE_ID, gdrive_service)
    ohlc_bytes = download_file_from_drive(OHLC_FILE_ID, gdrive_service)

    trades = read_gdrive_csv(trades_bytes)
    ohlc_data = read_gdrive_parquet(ohlc_bytes)
    
    if trades is not None:
        # --- FIX for column mapping ---
        trades = trades.rename(columns={
            'product_id': 'asset',
            'side': 'action',
            'size': 'quantity'
        })
        trades.columns = [col.lower().strip() for col in trades.columns]
        if 'action' in trades.columns:
            trades['action'] = trades['action'].str.lower()
        
        required_trade_cols = ['timestamp', 'asset', 'action', 'price', 'quantity']
        if not all(col in trades.columns for col in required_trade_cols):
            st.error(f"Trades file is missing required columns after mapping. Expected: {required_trade_cols}, Found: {list(trades.columns)}")
            return None, None, ohlc_data

    if ohlc_data is not None:
        ohlc_data.columns = [col.lower().strip() for col in ohlc_data.columns]
        # **NEW**: Parquet often uses 'product_id' as well, so we map it here too.
        if 'product_id' in ohlc_data.columns:
             ohlc_data = ohlc_data.rename(columns={'product_id': 'asset'})
        required_ohlc_cols = ['timestamp', 'asset', 'open', 'high', 'low', 'close']
        if not all(col in ohlc_data.columns for col in required_ohlc_cols):
            st.error(f"OHLC file is missing required columns after mapping. Expected: {required_ohlc_cols}, Found: {list(ohlc_data.columns)}")
            return trades, None, None
            
    if trades is not None:
        pnl_summary, trades_with_pnl = calculate_pnl(trades)
        return trades_with_pnl, pnl_summary, ohlc_data
    return pd.DataFrame(), {}, pd.DataFrame()

trades_df, pnl_summary, ohlc_df = load_data()

st.sidebar.header("Controls")
st.sidebar.info("This dashboard visualizes trading performance using data from your Google Drive.")
if st.sidebar.button('🔄 Refresh Data'):
    st.cache_data.clear()
    st.rerun()

if trades_df is not None and not trades_df.empty:
    st.subheader("Total P&L per Asset")
    if pnl_summary:
        cols = st.columns(min(len(pnl_summary), 5))
        asset_names = sorted(pnl_summary.keys())
        for i, asset in enumerate(asset_names):
            pnl_value = pnl_summary.get(asset, 0)
            with cols[i % 5]:
                st.metric(label=asset, value=f"${pnl_value:,.2f}")
    st.markdown("---")

    st.subheader("Overall P&L Timeline")
    buys = trades_df[trades_df['action'] == 'buy']
    sells = trades_df[trades_df['action'] == 'sell']
    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Scatter(x=trades_df['timestamp'], y=trades_df['cumulative_pnl'], mode='lines', name='Cumulative P&L', line=dict(color='royalblue', width=2)))
    fig_pnl.add_trace(go.Scatter(x=buys['timestamp'], y=buys['cumulative_pnl'], mode='markers', name='Buy', marker=dict(color='green', symbol='triangle-up', size=10)))
    fig_pnl.add_trace(go.Scatter(x=sells['timestamp'], y=sells['cumulative_pnl'], mode='markers', name='Sell', marker=dict(color='red', symbol='triangle-down', size=10)))
    fig_pnl.update_layout(title='Cumulative P&L with Trade Markers', xaxis_title='Date', yaxis_title='P&L (USD)', template='plotly_white', xaxis=dict(rangeslider=dict(visible=True), type="date"))
    st.plotly_chart(fig_pnl, use_container_width=True)
    st.markdown("---")

    st.subheader("Asset-Specific Analysis")
    if ohlc_df is not None and 'asset' in ohlc_df.columns:
        available_assets_ohlc = sorted(ohlc_df['asset'].unique())
        selected_asset = st.selectbox('Select Asset', options=available_assets_ohlc, index=0)

        if selected_asset:
            asset_trades = trades_df[trades_df['asset'] == selected_asset]
            asset_ohlc = ohlc_df[ohlc_df['asset'] == selected_asset].copy()
            
            asset_ohlc['timestamp'] = pd.to_datetime(asset_ohlc['timestamp'])
            asset_trades['timestamp'] = pd.to_datetime(asset_trades['timestamp'])

            fig_asset = go.Figure()
            fig_asset.add_trace(go.Candlestick(x=asset_ohlc['timestamp'], open=asset_ohlc['open'], high=asset_ohlc['high'], low=asset_ohlc['low'], close=asset_ohlc['close'], name='Candles'))
            
            asset_buys = asset_trades[asset_trades['action'] == 'buy']
            asset_sells = asset_trades[asset_trades['action'] == 'sell']
            fig_asset.add_trace(go.Scatter(x=asset_buys['timestamp'], y=asset_buys['price'], mode='markers', name='Buy', marker=dict(color='rgba(0, 255, 0, 0.9)', symbol='triangle-up', size=12, line=dict(width=2, color='DarkGreen'))))
            fig_asset.add_trace(go.Scatter(x=asset_sells['timestamp'], y=asset_sells['price'], mode='markers', name='Sell', marker=dict(color='rgba(255, 0, 0, 0.9)', symbol='triangle-down', size=12, line=dict(width=2, color='DarkRed'))))

            fig_asset.update_layout(title=f'Price History and Trades for {selected_asset}', xaxis_title='Date', yaxis_title='Price (USD)', template='plotly_white', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig_asset, use_container_width=True)
    elif ohlc_df is not None:
        st.warning("Could not process OHLC data. Ensure the Google Drive file is correct and contains an 'asset' column.")

