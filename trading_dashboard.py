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
    page_title="Trading Performance Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Professional Trading Performance Dashboard"
    }
)

# Custom CSS for professional modern styling with better contrast
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main-header {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        border-radius: 20px;
        margin-bottom: 3rem;
        box-shadow: 0 20px 60px rgba(37, 99, 235, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .main-header h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        margin: 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-family: 'Inter', sans-serif;
        font-weight: 300;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    .professional-section {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(229, 231, 235, 0.8);
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .section-title {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1.5rem;
        color: #1f2937;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
    
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid rgba(209, 213, 219, 0.8);
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #2563eb;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.2);
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    .css-1d391kg { background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%); }
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #f1f5f9 0%, #e2e8f0 100%);
        color: #1f2937;
    }
    .sidebar-metric {
        background: rgba(255, 255, 255, 0.9);
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border: 1px solid rgba(229, 231, 235, 0.8);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        color: #1f2937;
    }
    .sidebar-metric:hover { background: rgba(255, 255, 255, 1); transform: translateY(-2px); }
    .css-1v3fvcr h2 { color: #1f2937 !important; font-weight: 600; }
    .css-1v3fvcr h3 { color: #374151 !important; font-weight: 500; }
    [data-testid="stSidebar"] .markdown-text-container { color: #1f2937 !important; }
    .stProgress > div > div > div { background-color: #e5e7eb; }
</style>
""", unsafe_allow_html=True)

# --- Google Drive Data Loading ---
def get_gdrive_service():
    """Initializes the Google Drive API service using Streamlit's secrets."""
    if "gcp_service_account" not in st.secrets:
        st.error("GCP service account credentials not found in Streamlit Secrets.")
        st.info("Please add your service account JSON to .streamlit/secrets.toml as gcp_service_account.")
        return None
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=['https://www.googleapis.com/auth/drive.readonly']
    )
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
        st.info("Ensure the file ID is correct and the service account has access.")
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
        price = float(row['price'])
        quantity = float(row['quantity'])

        if asset not in positions:
            positions[asset] = {'quantity': 0.0, 'cost': 0.0}
            pnl_per_asset[asset] = 0.0

        current_pnl = 0.0
        if action == 'buy':
            positions[asset]['cost'] += quantity * price
            positions[asset]['quantity'] += quantity
        elif action == 'sell':
            if positions[asset]['quantity'] > 0:
                avg_cost_per_unit = positions[asset]['cost'] / positions[asset]['quantity']
                trade_qty = min(quantity, positions[asset]['quantity'])
                realized_pnl = (price - avg_cost_per_unit) * trade_qty
                pnl_per_asset[asset] += realized_pnl
                total_cumulative_pnl += realized_pnl
                current_pnl = realized_pnl

                if realized_pnl > 0:
                    winning_trades += 1
                    gross_profit += realized_pnl
                else:
                    losing_trades += 1
                    gross_loss += abs(realized_pnl)

                positions[asset]['cost'] -= avg_cost_per_unit * trade_qty
                positions[asset]['quantity'] -= trade_qty

        df.loc[index, 'pnl'] = current_pnl
        df.loc[index, 'cumulative_pnl'] = total_cumulative_pnl
        
        peak_pnl = max(peak_pnl, total_cumulative_pnl)
        drawdown = peak_pnl - total_cumulative_pnl
        max_drawdown = max(max_drawdown, drawdown)

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
st.markdown("""
<div class="main-header">
    <h1>Trading Performance Analytics</h1>
    <p style="font-size: 1.1em;">Professional Portfolio Management Dashboard</p>
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

# --- Sidebar ---
with st.sidebar:
    st.markdown("## Portfolio Control")
    auto_refresh = st.toggle("Auto-refresh (5min)", value=False)
    if auto_refresh:
        st.rerun()
    if st.button('Refresh Data', type="primary"):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    if pnl_summary:
        st.markdown("## Portfolio Summary")
        total_pnl = sum(pnl_summary.values())
        if total_pnl > 0:
            pnl_color, pnl_bg, pnl_emoji = "#10b981", "rgba(16,185,129,0.1)", "↗"
        elif total_pnl < 0:
            pnl_color, pnl_bg, pnl_emoji = "#ef4444", "rgba(239,68,68,0.1)", "↘"
        else:
            pnl_color, pnl_bg, pnl_emoji = "#6b7280", "rgba(107,114,128,0.1)", "→"
        st.markdown(f"""
        <div class="sidebar-metric" style="background: {pnl_bg}; border-color: {pnl_color};">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <span style="font-size: 0.9rem; opacity: 0.8;">Total P&L</span>
                <span style="font-size: 1.2rem;">{pnl_emoji}</span>
            </div>
            <div style="font-size: 1.4rem; font-weight: 600; color: {pnl_color}; margin-top: 0.3rem;">
                ${total_pnl:,.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("### Asset Allocation")
        for asset, pnl in sorted(pnl_summary.items(), key=lambda x: x[1], reverse=True):
            color = "#10b981" if pnl >= 0 else "#ef4444"
            bg_color = "rgba(16, 185, 129, 0.1)" if pnl >= 0 else "rgba(239, 68, 68, 0.1)"
            st.markdown(f"""
            <div class="sidebar-metric" style="background: {bg_color}; border-color: {color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 500; font-size: 0.9rem;">{asset}</span>
                    <span style="color: {color}; font-weight: 600;">${pnl:,.2f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    if summary_stats:
        st.markdown("---")
        st.markdown("## Key Metrics")
        win_rate = summary_stats['win_rate']
        st.markdown(f"""
        <div class="sidebar-metric">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.9rem;">Win Rate</span>
                <span style="font-weight: 600; color: #1f2937;">{win_rate:.1f}%</span>
            </div>
            <div style="background: rgba(229, 231, 235, 0.8); border-radius: 8px; height: 6px; margin-top: 0.5rem;">
                <div style="background: #10b981; height: 6px; border-radius: 8px; width: {win_rate}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sidebar-metric">
            <div style="display: flex; justify-content: space-between;">
                <span style="font-size: 0.9rem;">Total Trades</span>
                <span style="font-weight: 600; color: #1f2937;">{summary_stats['total_trades']:,}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sidebar-metric">
            <div style="display: flex; justify-content: space-between;">
                <span style="font-size: 0.9rem;">Max Drawdown</span>
                <span style="font-weight: 600; color: #ef4444;">${summary_stats['max_drawdown']:,.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if summary_stats['profit_factor'] != float('inf'):
            pf_color = "#10b981" if summary_stats['profit_factor'] > 1 else "#ef4444"
            st.markdown(f"""
            <div class="sidebar-metric">
                <div style="display: flex; justify-content: space-between;">
                    <span style="font-size: 0.9rem;">Profit Factor</span>
                    <span style="font-weight: 600; color: {pf_color};">{summary_stats['profit_factor']:.2f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# --- Main Content ---
if trades_df is not None and not trades_df.empty:
    # P&L Timeline
    st.markdown('<div class="professional-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Portfolio Performance Timeline</h2>', unsafe_allow_html=True)
    buys = trades_df[trades_df['action'] == 'buy']
    sells = trades_df[trades_df['action'] == 'sell']
    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Scatter(
        x=trades_df['timestamp'], 
        y=trades_df['cumulative_pnl'], 
        mode='lines',
        name='Cumulative P&L',
        line=dict(color='#1e3c72', width=3),
        fill='tonexty',
        fillcolor='rgba(30, 60, 114, 0.1)'
    ))
    fig_pnl.add_trace(go.Scatter(
        x=buys['timestamp'], 
        y=buys['cumulative_pnl'], 
        mode='markers',
        name='Buy Orders',
        marker=dict(color='#10b981', symbol='triangle-up', size=10, line=dict(width=2, color='#059669')),
        hovertemplate='<b>BUY</b><br>Date: %{x}<br>P&L: $%{y:,.2f}<extra></extra>'
    ))
    fig_pnl.add_trace(go.Scatter(
        x=sells['timestamp'], 
        y=sells['cumulative_pnl'], 
        mode='markers',
        name='Sell Orders',
        marker=dict(color='#ef4444', symbol='triangle-down', size=10, line=dict(width=2, color='#dc2626')),
        hovertemplate='<b>SELL</b><br>Date: %{x}<br>P&L: $%{y:,.2f}<extra></extra>'
    ))
    fig_pnl.update_layout(
        title=dict(text='Cumulative P&L Performance', font=dict(size=20, color='#1e3c72', family='Inter')),
        xaxis_title='Timeline',
        yaxis_title='P&L (USD)',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(family='Inter')),
        xaxis=dict(rangeslider=dict(visible=True), type="date", showgrid=True, gridcolor='rgba(30, 60, 114, 0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(30, 60, 114, 0.1)', zeroline=True, zerolinecolor='rgba(30, 60, 114, 0.3)', zerolinewidth=2),
        plot_bgcolor='rgba(248, 250, 252, 0.8)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter')
    )
    st.plotly_chart(fig_pnl, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Asset Analysis
    st.markdown('<div class="professional-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Asset Analysis</h2>', unsafe_allow_html=True)
    if ohlc_df is not None and 'asset' in ohlc_df.columns:
        available_assets = ohlc_df['asset'].unique()
        cvx_assets = [asset for asset in available_assets if 'CVX' in asset.upper()]
        asset_options = []
        if cvx_assets:
            base_cvx_asset = cvx_assets[0]
            cvx_data = ohlc_df[ohlc_df['asset'] == base_cvx_asset].copy()
            cvx_data['timestamp'] = pd.to_datetime(cvx_data['timestamp'])
            cvx_data = cvx_data.sort_values('timestamp').set_index('timestamp')
            cvx_1min = cvx_data.copy()
            cvx_1min['asset'] = 'CVX_1min'
            cvx_5min = cvx_data.resample('5min').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'asset': 'first'
            }).dropna()
            cvx_5min['asset'] = 'CVX_5min'
            cvx_1min = cvx_1min.reset_index()
            cvx_5min = cvx_5min.reset_index()
            ohlc_df = pd.concat([ohlc_df, cvx_1min, cvx_5min], ignore_index=True)
            asset_options.append(('1 min CVX', 'CVX_1min'))
            asset_options.append(('5 min CVX', 'CVX_5min'))
        for asset in sorted(available_assets):
            asset_options.append((asset, asset))
        col1, col2 = st.columns([3, 1])
        with col1:
            display_options = [option[0] for option in asset_options]
            selected_display = st.selectbox('Select Asset', options=display_options, index=0)
            selected_asset = next(option[1] for option in asset_options if option[0] == selected_display)
        with col2:
            time_range = st.selectbox('Time Range', options=['30 days', '7 days', '1 day', 'All'], index=0)

        if selected_asset:
            asset_trades = trades_df[trades_df['asset'] == selected_asset]
            asset_ohlc = ohlc_df[ohlc_df['asset'] == selected_asset].copy()
            if asset_ohlc.empty and asset_trades.empty:
                st.warning(f"No data available for {selected_asset}.")
            else:
                asset_ohlc['timestamp'] = pd.to_datetime(asset_ohlc['timestamp'])
                asset_trades['timestamp'] = pd.to_datetime(asset_trades['timestamp'])
                end_date = asset_ohlc['timestamp'].max() if not asset_ohlc.empty else datetime.now()
                if time_range == '1 day':
                    start_date = end_date - timedelta(days=1)
                elif time_range == '7 days':
                    start_date = end_date - timedelta(days=7)
                elif time_range == '30 days':
                    start_date = end_date - timedelta(days=30)
                else:
                    start_date = asset_ohlc['timestamp'].min() if not asset_ohlc.empty else end_date - timedelta(days=30)
                fig_asset = make_subplots(specs=[[{"secondary_y": True}]])
                mask_ohlc = (asset_ohlc['timestamp'] >= start_date) & (asset_ohlc['timestamp'] <= end_date)
                mask_trades = (asset_trades['timestamp'] >= start_date) & (asset_trades['timestamp'] <= end_date)
                filtered_ohlc = asset_ohlc[mask_ohlc].copy()
                filtered_trades = asset_trades[mask_trades].copy()

                if not filtered_ohlc.empty:
                    # ---------- FIX: Hovertemplate + customdata + valid kwargs ----------
                    def find_col(candidates, cols):
                        for c in candidates:
                            if c in cols:
                                return c
                        return None

                    cols_lower = set(filtered_ohlc.columns.str.lower())
                    colmap = {c.lower(): c for c in filtered_ohlc.columns}

                    p_up_key   = find_col(["p_up", "p-up", "pup", "puprob", "p_up_prob", "puprobability"], cols_lower)
                    p_down_key = find_col(["p_down", "p-down", "pdown", "pdownprob", "p_down_prob", "pdownprobability"], cols_lower)

                    def fmt_series(key):
                        if key is None:
                            return ["—"] * len(filtered_ohlc)
                        s = filtered_ohlc[colmap[key]]
                        return [f"{float(v):.4f}" if pd.notna(v) else "—" for v in s]

                    pu_str = fmt_series(p_up_key)
                    pd_str = fmt_series(p_down_key)
                    customdata = np.column_stack([pu_str, pd_str])

                    # Ensure numeric dtypes for O/H/L/C
                    for c in ["open", "high", "low", "close"]:
                        filtered_ohlc[c] = pd.to_numeric(filtered_ohlc[c], errors="coerce")

                    fig_asset.add_trace(
                        go.Candlestick(
                            x=pd.to_datetime(filtered_ohlc['timestamp']),
                            open=filtered_ohlc['open'].astype(float),
                            high=filtered_ohlc['high'].astype(float),
                            low=filtered_ohlc['low'].astype(float),
                            close=filtered_ohlc['close'].astype(float),
                            name='Price',
                            increasing=dict(line=dict(color='#10b981'), fillcolor='rgba(16,185,129,0.3)'),
                            decreasing=dict(line=dict(color='#ef4444'), fillcolor='rgba(239,68,68,0.3)'),
                            customdata=customdata,
                            hovertemplate=(
                                "📅 %{x|%Y-%m-%d %H:%M:%S}<br>"
                                "Open: $%{open:.4f}<br>"
                                "High: $%{high:.4f}<br>"
                                "Low: $%{low:.4f}<br>"
                                "Close: $%{close:.4f}<br>"
                                "P-Up: %{customdata[0]}<br>"
                                "P-Down: %{customdata[1]}<extra></extra>"
                            ),
                        ),
                        secondary_y=False
                    )

                # P&L line + trade markers
                if not filtered_trades.empty:
                    fig_asset.add_trace(go.Scatter(
                        x=filtered_trades['timestamp'],
                        y=filtered_trades['asset_cumulative_pnl'],
                        mode='lines',
                        name='Asset P&L',
                        line=dict(color='#1e3c72', width=3)
                    ), secondary_y=True)

                    buys_a = filtered_trades[filtered_trades['action'] == 'buy']
                    sells_a = filtered_trades[filtered_trades['action'] == 'sell']
                    if not buys_a.empty:
                        fig_asset.add_trace(go.Scatter(
                            x=buys_a['timestamp'],
                            y=buys_a['price'],
                            mode='markers',
                            name='Buy',
                            marker=dict(color='#10b981', symbol='triangle-up', size=14, line=dict(width=2, color='#059669')),
                            hovertemplate='<b>BUY</b><br>Price: $%{y:,.2f}<br>%{x}<extra></extra>'
                        ), secondary_y=False)
                    if not sells_a.empty:
                        fig_asset.add_trace(go.Scatter(
                            x=sells_a['timestamp'],
                            y=sells_a['price'],
                            mode='markers',
                            name='Sell',
                            marker=dict(color='#ef4444', symbol='triangle-down', size=14, line=dict(width=2, color='#dc2626')),
                            hovertemplate='<b>SELL</b><br>Price: $%{y:,.2f}<br>%{x}<extra></extra>'
                        ), secondary_y=False)

                fig_asset.update_layout(
                    title=dict(text=f'{selected_asset} - Price & Performance Analysis',
                               font=dict(size=18, color='#1e3c72', family='Inter')),
                    template='plotly_white',
                    xaxis_rangeslider_visible=True,
                    hovermode='x unified',
                    plot_bgcolor='rgba(248, 250, 252, 0.8)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Inter'),
                    xaxis=dict(showgrid=True, gridcolor='rgba(30, 60, 114, 0.1)'),
                    yaxis=dict(showgrid=True, gridcolor='rgba(30, 60, 114, 0.1)')
                )
                fig_asset.update_yaxes(title_text="Price (USD)", secondary_y=False, title_font=dict(family='Inter'))
                fig_asset.update_yaxes(title_text="P&L (USD)", secondary_y=True, showgrid=False, title_font=dict(family='Inter'))
                st.plotly_chart(fig_asset, use_container_width=True)

                # Asset stats
                if not asset_trades.empty:
                    asset_pnl = asset_trades['pnl'].sum()
                    num_trades = len(asset_trades)
                    avg_trade = asset_pnl / num_trades if num_trades > 0 else 0
                    col1, col2, col3 = st.columns(3)
                    pnl_color = "#10b981" if asset_pnl >= 0 else "#ef4444"
                    col1.markdown(f"""
                    <div style="background: rgba(30, 60, 114, 0.1); padding: 1rem; border-radius: 12px; text-align: center;">
                        <div style="font-size: 0.9rem; color: #6b7280; margin-bottom: 0.5rem;">Asset P&L</div>
                        <div style="font-size: 1.5rem; font-weight: 600; color: {pnl_color};">${asset_pnl:,.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    col2.markdown(f"""
                    <div style="background: rgba(30, 60, 114, 0.1); padding: 1rem; border-radius: 12px; text-align: center;">
                        <div style="font-size: 0.9rem; color: #6b7280; margin-bottom: 0.5rem;">Total Trades</div>
                        <div style="font-size: 1.5rem; font-weight: 600; color: #1e3c72;">{num_trades}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    avg_color = "#10b981" if avg_trade >= 0 else "#ef4444"
                    col3.markdown(f"""
                    <div style="background: rgba(30, 60, 114, 0.1); padding: 1rem; border-radius: 12px; text-align: center;">
                        <div style="font-size: 0.9rem; color: #6b7280; margin-bottom: 0.5rem;">Avg per Trade</div>
                        <div style="font-size: 1.5rem; font-weight: 600; color: {avg_color};">${avg_trade:,.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
    elif ohlc_df is not None:
        st.warning("Could not process OHLC data. Ensure the file contains an 'asset' column.")
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.error("No trading data available. Please check your data connection.")
    st.info("Make sure your Google Drive files are properly configured and accessible.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; opacity: 0.6; padding: 2rem; font-family: 'Inter', sans-serif;">
    <p style="margin: 0; color: #6b7280;">Professional Trading Analytics Dashboard</p>
    <p style="margin: 0; font-size: 0.8rem; color: #9ca3af;">Real-time Portfolio Performance & Risk Management</p>
</div>
""", unsafe_allow_html=True)
