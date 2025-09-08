import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import io

# Imports for Google Drive
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# --- Configuration ---
st.set_page_config(
    page_title="🚀 Trading Bot Performance",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Advanced Trading Bot Performance Dashboard"
    }
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .profit-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    
    .loss-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    
    .neutral-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #3498db 100%);
    }
    
    .stSelectbox > div > div {
        background-color: rgba(255,255,255,0.1);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- Google Drive Data Loading (same as before) ---
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

# --- P&L and Metrics Calculation (same as before) ---
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
# Cool animated header
st.markdown("""
<div class="main-header">
    <h1>🚀 Elite Trading Bot Dashboard</h1>
    <p style="font-size: 1.2em; opacity: 0.9;">Real-time Performance Analytics & Intelligence</p>
</div>
""", unsafe_allow_html=True)

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

# --- Enhanced Sidebar ---
with st.sidebar:
    st.markdown("## 🎛️ Control Center")
    
    # Auto-refresh toggle
    auto_refresh = st.toggle("🔄 Auto-refresh (5min)", value=False)
    if auto_refresh:
        st.rerun()
    
    if st.button('🔄 Manual Refresh', type="primary"):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    
    if pnl_summary:
        st.markdown("## 💰 Portfolio Overview")
        total_pnl = sum(pnl_summary.values())
        
        # Dynamic styling based on P&L
        if total_pnl > 0:
            pnl_color = "#00ff88"
            pnl_emoji = "📈"
        elif total_pnl < 0:
            pnl_color = "#ff4757"
            pnl_emoji = "📉"
        else:
            pnl_color = "#ffa502"
            pnl_emoji = "➖"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {pnl_color}20, {pnl_color}10); 
                    padding: 1rem; border-radius: 10px; text-align: center; margin-bottom: 1rem;">
            <h3 style="margin: 0; color: {pnl_color};">{pnl_emoji} ${total_pnl:,.2f}</h3>
            <p style="margin: 0; opacity: 0.8;">Total Portfolio P&L</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Asset Breakdown")
        for asset, pnl in sorted(pnl_summary.items(), key=lambda x: x[1], reverse=True):
            color = "#00ff88" if pnl >= 0 else "#ff4757"
            st.markdown(f"""
            <div style="background: {color}20; padding: 0.5rem; border-radius: 8px; margin: 0.3rem 0;">
                <strong>{asset}:</strong> <span style="color: {color};">${pnl:,.2f}</span>
            </div>
            """, unsafe_allow_html=True)
    
    if summary_stats:
        st.markdown("---")
        st.markdown("## 📈 Performance Metrics")
        
        # Win rate with progress bar
        win_rate = summary_stats['win_rate']
        st.markdown(f"**Win Rate:** {win_rate:.1f}%")
        st.progress(win_rate / 100)
        
        st.metric("🎯 Total Trades", f"{summary_stats['total_trades']:,}")
        st.metric("💸 Max Drawdown", f"${summary_stats['max_drawdown']:,.2f}")
        
        if summary_stats['profit_factor'] != float('inf'):
            st.metric("⚖️ Profit Factor", f"{summary_stats['profit_factor']:.2f}")

# --- Main Content ---
if trades_df is not None and not trades_df.empty:
    
    # Enhanced P&L metrics cards
    st.markdown("## 💎 Asset Performance Overview")
    if pnl_summary:
        # Create dynamic columns based on number of assets
        num_assets = len(pnl_summary)
        cols_per_row = min(num_assets, 4)
        
        for i in range(0, num_assets, cols_per_row):
            cols = st.columns(cols_per_row)
            asset_slice = list(pnl_summary.items())[i:i+cols_per_row]
            
            for j, (asset, pnl_value) in enumerate(asset_slice):
                with cols[j]:
                    # Determine card style based on P&L
                    if pnl_value > 0:
                        card_class = "profit-card"
                        emoji = "🟢"
                    elif pnl_value < 0:
                        card_class = "loss-card"
                        emoji = "🔴"
                    else:
                        card_class = "neutral-card"
                        emoji = "🟡"
                    
                    st.markdown(f"""
                    <div class="metric-card {card_class}">
                        <h3 style="margin: 0;">{emoji} {asset}</h3>
                        <h2 style="margin: 0.5rem 0;">${pnl_value:,.2f}</h2>
                        <p style="margin: 0; opacity: 0.9;">Total P&L</p>
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown("---")

    # Enhanced P&L Timeline with better styling
    st.markdown("## 🌊 P&L Flow Visualization")
    
    buys = trades_df[trades_df['action'] == 'buy']
    sells = trades_df[trades_df['action'] == 'sell']
    
    fig_pnl = go.Figure()
    
    # Main P&L line with gradient fill
    fig_pnl.add_trace(go.Scatter(
        x=trades_df['timestamp'], 
        y=trades_df['cumulative_pnl'], 
        mode='lines',
        name='Cumulative P&L',
        line=dict(color='#00d4ff', width=3),
        fill='tonexty',
        fillcolor='rgba(0, 212, 255, 0.1)'
    ))
    
    # Buy markers with custom styling
    fig_pnl.add_trace(go.Scatter(
        x=buys['timestamp'], 
        y=buys['cumulative_pnl'], 
        mode='markers',
        name='Buy Orders',
        marker=dict(
            color='#00ff88',
            symbol='triangle-up',
            size=12,
            line=dict(width=2, color='#00cc66')
        ),
        hovertemplate='<b>BUY</b><br>Date: %{x}<br>P&L: $%{y:,.2f}<extra></extra>'
    ))
    
    # Sell markers with custom styling
    fig_pnl.add_trace(go.Scatter(
        x=sells['timestamp'], 
        y=sells['cumulative_pnl'], 
        mode='markers',
        name='Sell Orders',
        marker=dict(
            color='#ff4757',
            symbol='triangle-down',
            size=12,
            line=dict(width=2, color='#cc3644')
        ),
        hovertemplate='<b>SELL</b><br>Date: %{x}<br>P&L: $%{y:,.2f}<extra></extra>'
    ))
    
    fig_pnl.update_layout(
        title=dict(
            text='💰 Cumulative P&L Journey',
            font=dict(size=24, color='#2c3e50')
        ),
        xaxis_title='Timeline',
        yaxis_title='P&L (USD)',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="date",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.3)',
            zerolinewidth=2
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig_pnl, use_container_width=True)
    st.markdown("---")

    # Enhanced Asset Analysis
    st.markdown("## 🔍 Deep Asset Analysis")
    
    if ohlc_df is not None and 'asset' in ohlc_df.columns:
        available_assets_ohlc = sorted(ohlc_df['asset'].unique())
        
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_asset = st.selectbox(
                '🎯 Select Asset for Analysis', 
                options=available_assets_ohlc, 
                index=0
            )
        with col2:
            time_range = st.selectbox(
                '📅 Time Range',
                options=['30 days', '7 days', '1 day', 'All'],
                index=0
            )

        if selected_asset:
            asset_trades = trades_df[trades_df['asset'] == selected_asset]
            asset_ohlc = ohlc_df[ohlc_df['asset'] == selected_asset].copy()

            if asset_ohlc.empty and asset_trades.empty:
                st.warning(f"No data available for {selected_asset}.")
            else:
                asset_ohlc['timestamp'] = pd.to_datetime(asset_ohlc['timestamp'])
                asset_trades['timestamp'] = pd.to_datetime(asset_trades['timestamp'])

                # Calculate time range
                end_date = asset_ohlc['timestamp'].max() if not asset_ohlc.empty else datetime.now()
                if time_range == '1 day':
                    start_date = end_date - timedelta(days=1)
                elif time_range == '7 days':
                    start_date = end_date - timedelta(days=7)
                elif time_range == '30 days':
                    start_date = end_date - timedelta(days=30)
                else:  # All
                    start_date = asset_ohlc['timestamp'].min() if not asset_ohlc.empty else end_date - timedelta(days=30)
                
                # Fixed subplot creation
                fig_asset = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Filter data for time range
                mask_ohlc = (asset_ohlc['timestamp'] >= start_date) & (asset_ohlc['timestamp'] <= end_date)
                mask_trades = (asset_trades['timestamp'] >= start_date) & (asset_trades['timestamp'] <= end_date)
                
                filtered_ohlc = asset_ohlc[mask_ohlc]
                filtered_trades = asset_trades[mask_trades]
                
                if not filtered_ohlc.empty:
                    fig_asset.add_trace(go.Candlestick(
                        x=filtered_ohlc['timestamp'],
                        open=filtered_ohlc['open'],
                        high=filtered_ohlc['high'],
                        low=filtered_ohlc['low'],
                        close=filtered_ohlc['close'],
                        name='Price',
                        increasing_line_color='#00ff88',
                        decreasing_line_color='#ff4757'
                    ), secondary_y=False)
                
                # P&L line
                if not filtered_trades.empty:
                    fig_asset.add_trace(go.Scatter(
                        x=filtered_trades['timestamp'],
                        y=filtered_trades['asset_cumulative_pnl'],
                        mode='lines',
                        name='Asset P&L',
                        line=dict(color='#ffa502', width=3)
                    ), secondary_y=True)

                    # Trade markers
                    asset_buys = filtered_trades[filtered_trades['action'] == 'buy']
                    asset_sells = filtered_trades[filtered_trades['action'] == 'sell']
                    
                    if not asset_buys.empty:
                        fig_asset.add_trace(go.Scatter(
                            x=asset_buys['timestamp'],
                            y=asset_buys['price'],
                            mode='markers',
                            name='Buy',
                            marker=dict(
                                color='#00ff88',
                                symbol='triangle-up',
                                size=16,
                                line=dict(width=2, color='#00cc66')
                            ),
                            hovertemplate='<b>BUY</b><br>Price: $%{y:,.2f}<br>%{x}<extra></extra>'
                        ), secondary_y=False)
                    
                    if not asset_sells.empty:
                        fig_asset.add_trace(go.Scatter(
                            x=asset_sells['timestamp'],
                            y=asset_sells['price'],
                            mode='markers',
                            name='Sell',
                            marker=dict(
                                color='#ff4757',
                                symbol='triangle-down',
                                size=16,
                                line=dict(width=2, color='#cc3644')
                            ),
                            hovertemplate='<b>SELL</b><br>Price: $%{y:,.2f}<br>%{x}<extra></extra>'
                        ), secondary_y=False)

                fig_asset.update_layout(
                    title=dict(
                        text=f'🚀 {selected_asset} - Advanced Trading Analysis',
                        font=dict(size=20, color='#2c3e50')
                    ),
                    template='plotly_white',
                    xaxis_rangeslider_visible=True,
                    hovermode='x unified',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                fig_asset.update_yaxes(title_text="💵 Price (USD)", secondary_y=False)
                fig_asset.update_yaxes(title_text="📈 P&L (USD)", secondary_y=True, showgrid=False)

                st.plotly_chart(fig_asset, use_container_width=True)
                
                # Asset statistics
                if not asset_trades.empty:
                    asset_pnl = asset_trades['pnl'].sum()
                    num_trades = len(asset_trades)
                    avg_trade = asset_pnl / num_trades if num_trades > 0 else 0
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("💰 Asset P&L", f"${asset_pnl:,.2f}")
                    col2.metric("🔢 Total Trades", num_trades)
                    col3.metric("📊 Avg per Trade", f"${avg_trade:,.2f}")

    elif ohlc_df is not None:
        st.warning("Could not process OHLC data. Ensure the file contains an 'asset' column.")

else:
    st.error("No trading data available. Please check your data connection.")
    st.info("🔧 Make sure your Google Drive files are properly configured and accessible.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; opacity: 0.6; padding: 2rem;">
    <p>🚀 Elite Trading Dashboard • Real-time Analytics • Built with ❤️</p>
</div>
""", unsafe_allow_html=True)
