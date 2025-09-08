import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# --- P&L and Metrics Calculation ---

def calculate_pnl_and_metrics(trades_df):
    """Calculates per-asset P&L, a running total P&L, and performance metrics."""
    if trades_df is None or trades_df.empty:
        return {}, pd.DataFrame(), {}

    pnl_per_asset = {}
    positions = {}
    
    df = trades_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    df['pnl'] = 0.0
    df['cumulative_pnl'] = 0.0
    total_cumulative_pnl = 0
    
    # Metrics variables
    winning_trades = 0
    losing_trades = 0
    gross_profit = 0
    gross_loss = 0
    peak_pnl = 0
    max_drawdown = 0

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
                
                # Update metrics
                if realized_pnl > 0:
                    winning_trades += 1
                    gross_profit += realized_pnl
                else:
                    losing_trades += 1
                    gross_loss += abs(realized_pnl)

                positions[asset]['cost'] -= avg_cost_per_unit * min(quantity, positions[asset]['quantity'])
                positions[asset]['quantity'] -= quantity

        df.loc[index, 'pnl'] = current_pnl
        df.loc[index, 'cumulative_pnl'] = total_cumulative_pnl
        
        # Calculate drawdown
        peak_pnl = max(peak_pnl, total_cumulative_pnl)
        drawdown = peak_pnl - total_cumulative_pnl
        max_drawdown = max(max_drawdown, drawdown)

    # Finalize metrics
    total_closed_trades = winning_trades + losing_trades
    summary_stats = {
        'win_rate': (winning_trades / total_closed_trades * 100) if total_closed_trades > 0 else 0,
        'profit_factor': (gross_profit / gross_loss) if gross_loss > 0 else float('inf'),
        'total_trades': total_closed_trades,
        'avg_win': (gross_profit / winning_trades) if winning_trades > 0 else 0,
        'avg_loss': (gross_loss / losing_trades) if losing_trades > 0 else 0,
        'max_drawdown': max_drawdown
    }
    
    df['asset_cumulative_pnl'] = df.groupby('asset')['pnl'].cumsum()

    return pnl_per_asset, df, summary_stats

# --- Main App ---
st.title("📈 Trading Bot Performance Dashboard")

@st.cache_data(ttl=600)
def load_data():
    gdrive_service = get_gdrive_service()
    if gdrive_service is None:
        return None, None, None, None

    TRADES_FILE_ID = "1zSdFcG4Xlh_iSa180V6LRSeEucAokXYk"
    OHLC_FILE_ID = "1kxOB0PCGaT7N0Ljv4-EUDqYCnnTyd0SK"
    
    trades_bytes = download_file_from_drive(TRADES_FILE_ID, gdrive_service)
    ohlc_bytes = download_file_from_drive(OHLC_FILE_ID, gdrive_service)

    trades = read_gdrive_csv(trades_bytes)
    ohlc_data = read_gdrive_parquet(ohlc_bytes)
    
    if trades is not None:
        trades = trades.rename(columns={'product_id': 'asset', 'side': 'action', 'size': 'quantity'})
        trades.columns = [col.lower().strip() for col in trades.columns]
        if 'action' in trades.columns:
            trades['action'] = trades['action'].str.lower()
        
        required_trade_cols = ['timestamp', 'asset', 'action', 'price', 'quantity']
        if not all(col in trades.columns for col in required_trade_cols):
            st.error(f"Trades file is missing required columns: {required_trade_cols}")
            return None, None, ohlc_data, None

    if ohlc_data is not None:
        ohlc_data.columns = [col.lower().strip() for col in ohlc_data.columns]
        if 'product_id' in ohlc_data.columns:
             ohlc_data = ohlc_data.rename(columns={'product_id': 'asset'})
        required_ohlc_cols = ['timestamp', 'asset', 'open', 'high', 'low', 'close']
        if not all(col in ohlc_data.columns for col in required_ohlc_cols):
            st.error(f"OHLC file is missing required columns: {required_ohlc_cols}")
            return trades, None, None, None
            
    if trades is not None:
        pnl_summary, trades_with_pnl, summary_stats = calculate_pnl_and_metrics(trades)
        return trades_with_pnl, pnl_summary, ohlc_data, summary_stats
    return pd.DataFrame(), {}, pd.DataFrame(), {}

trades_df, pnl_summary, ohlc_df, summary_stats = load_data()

# --- Sidebar ---
st.sidebar.header("Controls")
st.sidebar.info("Dashboard visualizes trading performance from Google Drive.")
if st.sidebar.button('🔄 Refresh Data'):
    st.cache_data.clear()
    st.rerun()

if pnl_summary:
    st.sidebar.header("P&L Summary")
    total_pnl = sum(pnl_summary.values())
    color = "green" if total_pnl >= 0 else "red"
    st.sidebar.markdown(f'**Overall P&L:** <span style="color:{color};">${total_pnl:,.2f}</span>', unsafe_allow_html=True)
    st.sidebar.markdown("---")
    for asset, pnl in sorted(pnl_summary.items()):
        color = "green" if pnl >= 0 else "red"
        st.sidebar.markdown(f'**{asset}:** <span style="color:{color};">${pnl:,.2f}</span>', unsafe_allow_html=True)

if summary_stats:
    st.sidebar.header("Performance Metrics")
    st.sidebar.metric("Total Closed Trades", f"{summary_stats['total_trades']:,}")
    st.sidebar.metric("Win Rate", f"{summary_stats['win_rate']:.2f}%")

# --- Main Content ---
if trades_df is not None and not trades_df.empty:
    
    if summary_stats:
        st.subheader("Key Performance Indicators")
        kpi_cols = st.columns(2)
        kpi_cols[0].metric("Win Rate", f"{summary_stats['win_rate']:.2f}%")
        kpi_cols[1].metric("Total Closed Trades", f"{summary_stats['total_trades']:,}")
        st.markdown("---")
    
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

            if asset_ohlc.empty and asset_trades.empty:
                st.warning(f"No OHLC or trade data available for {selected_asset}.")
            else:
                asset_ohlc['timestamp'] = pd.to_datetime(asset_ohlc['timestamp'])
                asset_trades['timestamp'] = pd.to_datetime(asset_trades['timestamp'])

                end_date = asset_ohlc['timestamp'].max() if not asset_ohlc.empty else datetime.now()
                start_date = end_date - timedelta(days=30)
                if not asset_ohlc.empty and start_date < asset_ohlc['timestamp'].min():
                    start_date = asset_ohlc['timestamp'].min()
                
                fig_asset = make_subplots(specs=[[[{"secondary_y": True}]]])
                
                if not asset_ohlc.empty:
                    fig_asset.add_trace(go.Candlestick(x=asset_ohlc['timestamp'], open=asset_ohlc['open'], high=asset_ohlc['high'], low=asset_ohlc['low'], close=asset_ohlc['close'], name='Candles'), secondary_y=False)
                
                fig_asset.add_trace(go.Scatter(x=asset_trades['timestamp'], y=asset_trades['asset_cumulative_pnl'], mode='lines', name='Asset P&L', line=dict(color='orange')), secondary_y=True)

                asset_buys = asset_trades[asset_trades['action'] == 'buy']
                asset_sells = asset_trades[asset_trades['action'] == 'sell']
                
                fig_asset.add_trace(go.Scatter(x=asset_buys['timestamp'], y=asset_buys['price'], mode='markers', name='Buy', marker=dict(color='rgba(0, 255, 0, 0.9)', symbol='triangle-up', size=16, line=dict(width=2, color='DarkGreen'))), secondary_y=False)
                fig_asset.add_trace(go.Scatter(x=asset_sells['timestamp'], y=asset_sells['price'], mode='markers', name='Sell', marker=dict(color='rgba(255, 0, 0, 0.9)', symbol='triangle-down', size=16, line=dict(width=2, color='DarkRed'))), secondary_y=False)

                for _, trade in asset_buys.iterrows():
                    fig_asset.add_shape(type="line", x0=trade['timestamp'], y0=0, x1=trade['timestamp'], y1=1, yref='paper', line=dict(color="rgba(0, 255, 0, 0.5)", width=1, dash="dash"))

                for _, trade in asset_sells.iterrows():
                    fig_asset.add_shape(type="line", x0=trade['timestamp'], y0=0, x1=trade['timestamp'], y1=1, yref='paper', line=dict(color="rgba(255, 0, 0, 0.5)", width=1, dash="dash"))

                fig_asset.update_layout(title=f'Price History and Trades for {selected_asset}', template='plotly_white', xaxis_rangeslider_visible=True, xaxis_range=[start_date, end_date])
                fig_asset.update_yaxes(title_text="Price (USD)", secondary_y=False)
                fig_asset.update_yaxes(title_text="P&L (USD)", secondary_y=True, showgrid=False)

                st.plotly_chart(fig_asset, use_container_width=True)
                st.info("💡 **Tip:** To scroll through history, drag the middle of the range slider at the bottom of the chart.")

    elif ohlc_df is not None:
        st.warning("Could not process OHLC data. Ensure the Google Drive file is correct and contains an 'asset' column.")

