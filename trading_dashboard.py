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

# Function to authorize and build the Google Drive service
def get_gdrive_service():
    """Initializes the Google Drive API service using Streamlit's secrets."""
    # The structure of the secrets should match the JSON key file.
    # We use st.secrets for secure credential management.
    scopes = ['https://www.googleapis.com/auth/drive.readonly']
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=scopes
    )
    service = build('drive', 'v3', credentials=creds)
    return service

# Function to download and read a file from Google Drive into a pandas DataFrame
def read_gdrive_file(file_id, service):
    """Downloads a file from Google Drive and returns it as a pandas DataFrame."""
    try:
        request = service.files().get_media(fileId=file_id)
        # Use io.BytesIO to handle the downloaded byte stream
        file_bytes = io.BytesIO()
        downloader = MediaIoBaseDownload(file_bytes, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        
        # Once downloaded, reset the stream's position and read with pandas
        file_bytes.seek(0)
        # Assuming the file is a CSV. Change accordingly for other formats (e.g., pd.read_excel).
        df = pd.read_csv(file_bytes)
        return df
    except Exception as e:
        st.error(f"Error reading file with ID '{file_id}' from Google Drive: {e}")
        return None

# --- P&L Calculation ---

def calculate_pnl(trades_df):
    """Calculates per-asset P&L and a running total P&L."""
    if trades_df is None or trades_df.empty:
        return {}, pd.DataFrame()

    pnl_per_asset = {}
    positions = {}
    
    # Ensure 'timestamp' is a datetime object for sorting
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

# --- Data Loading and Caching ---
# Use caching to avoid re-downloading data on every interaction.
@st.cache_data(ttl=600) # Cache for 10 minutes
def load_data():
    # !!! IMPORTANT !!!
    # REPLACE THESE WITH YOUR ACTUAL GOOGLE DRIVE FILE IDs
    # You can get the File ID from the shareable link.
    # e.g., in "drive.google.com/file/d/THIS_IS_THE_ID/view", the ID is "THIS_IS_THE_ID"
    TRADES_FILE_ID = "YOUR_TRADES_FILE_ID_HERE"
    OHLC_FILE_ID = "YOUR_OHLC_DATA_FILE_ID_HERE"
    
    # Initialize the Google Drive service
    gdrive_service = get_gdrive_service()
    
    # Load the data files
    trades = read_gdrive_file(TRADES_FILE_ID, gdrive_service)
    ohlc_data = read_gdrive_file(OHLC_FILE_ID, gdrive_service)
    
    if trades is not None:
        pnl_summary, trades_with_pnl = calculate_pnl(trades)
        return trades_with_pnl, pnl_summary, ohlc_data
    return pd.DataFrame(), {}, pd.DataFrame()


trades_df, pnl_summary, ohlc_df = load_data()

# --- Sidebar ---
st.sidebar.header("Controls")
st.sidebar.info("This dashboard visualizes trading performance using data from your Google Drive.")
if st.sidebar.button('🔄 Refresh Data'):
    st.cache_data.clear()
    st.rerun()

# --- Main App Logic (Proceed only if data is loaded) ---
if trades_df is not None and not trades_df.empty:
    # --- P&L Summary Section ---
    st.subheader("Total P&L per Asset")
    cols = st.columns(len(pnl_summary))
    asset_names = sorted(pnl_summary.keys())
    for i, asset in enumerate(asset_names):
        pnl_value = pnl_summary[asset]
        with cols[i]:
            st.metric(label=asset, value=f"${pnl_value:,.2f}")
    st.markdown("---")

    # --- Chart 1: Overall P&L Over Time ---
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

    # --- Chart 2: Asset-Specific Candlestick Chart ---
    st.subheader("Asset-Specific Analysis")
    # Assuming your OHLC data has an 'asset' column to filter by.
    # If not, you might need a separate file for each asset's OHLC data.
    if ohlc_df is not None and 'asset' in ohlc_df.columns:
        available_assets_ohlc = ohlc_df['asset'].unique()
        selected_asset = st.selectbox('Select Asset', options=available_assets_ohlc, index=0)

        if selected_asset:
            asset_trades = trades_df[trades_df['asset'] == selected_asset]
            asset_ohlc = ohlc_df[ohlc_df['asset'] == selected_asset]
            # Ensure timestamp columns are datetime objects for plotting
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
    else:
        st.warning("Could not load or process OHLC data. Ensure the Google Drive file is correct and contains an 'asset' column.")
else:
    st.header("Could not load trading data from Google Drive.")
    st.warning("Please check the following:")
    st.markdown("""
    1.  Have you replaced `"YOUR_TRADES_FILE_ID_HERE"` in the script with your actual file ID?
    2.  Have you configured your Streamlit Secrets correctly?
    3.  Did you share the Google Drive file with your service account's email?
    """)

