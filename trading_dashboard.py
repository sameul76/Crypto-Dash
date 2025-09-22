import io
import re
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from datetime import datetime, timedelta
from pytz import timezone

# Note: Reading Parquet files with pandas requires the 'pyarrow' library.
# pip install streamlit pandas plotly pyarrow requests pytz
import pyarrow  # noqa: F401 (ensures pyarrow engine is available)

# =========================
# App Configuration
# =========================
st.set_page_config(page_title="Crypto Trading Strategy", layout="wide")

# =========================
# Auto-Refresh Setup
# =========================
REFRESH_INTERVAL = 300  # 5 minutes in seconds

# Initialize session state for auto-refresh
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()
if 'auto_refresh_enabled' not in st.session_state:
    st.session_state.auto_refresh_enabled = True

LOCAL_TZ = "America/Los_Angeles"
DEFAULT_ASSET = "GIGA-USD"

# =========================
# Google Drive Links
# =========================
# Trade log file (trades_log.parquet) - UPDATED
TRADES_LINK = "https://drive.google.com/file/d/1hyM37eafLgvMo8RJDtw9GSEdZ-LQ05ks/view?usp=sharing"
# Features/market data file (trading_data_complete.parquet)
MARKET_LINK = "https://drive.google.com/file/d/1h9CIU4ro6JPpoBXYeH_oZD7x7af-teB1/view?usp=sharing"

# =========================
# Helpers — data processing
# =========================
def lower_strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.lower().strip() for c in out.columns]
    return out

def unify_symbol(val: str) -> str:
    if not isinstance(val, str):
        return val
    s = val.strip().upper().replace("_", "-")
    if "GIGA" in s:
        return "GIGA-USD"
    return s

def normalize_prob_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    p_up_variations = {"p_up", "p-up", "pup", "pup prob", "p up"}
    p_down_variations = {"p_down", "p-down", "pdown", "pdown prob", "p down"}
    for col in df.columns:
        col_lower = col.lower().replace("_", " ").replace("-", " ")
        if col_lower in p_up_variations:
            rename_map[col] = "p_up"
        elif col_lower in p_down_variations:
            rename_map[col] = "p_down"
    if rename_map:
        df = df.rename(columns=rename_map)
    if "p_up" not in df.columns: df["p_up"] = np.nan
    if "p_down" not in df.columns: df["p_down"] = np.nan
    return df

def calculate_pnl_and_metrics(trades_df: pd.DataFrame):
    if trades_df is None or trades_df.empty:
        return {}, pd.DataFrame(), {}
    pnl_per_asset, positions = {}, {}
    df = trades_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["pnl"], df["cumulative_pnl"] = 0.0, 0.0
    total, win, loss, gp, gl, peak, mdd = 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0
    
    for i, row in df.iterrows():
        asset = row.get("asset", "")
        action = str(row.get("unified_action", "")).lower().strip()
        
        try:
            price = float(row.get("price", 0))
            qty = float(row.get("quantity", 0))
        except (ValueError, TypeError):
            price, qty = 0.0, 0.0
            
        if asset not in positions:
            positions[asset] = {"quantity": 0.0, "cost": 0.0}
            pnl_per_asset[asset] = 0.0
            
        cur_pnl = 0.0
        
        if action in ["buy", "open"]:
            positions[asset]["cost"] += qty * price
            positions[asset]["quantity"] += qty
        elif action in ["sell", "close"]:
            if positions[asset]["quantity"] > 0:
                avg_cost = positions[asset]["cost"] / max(positions[asset]["quantity"], 1e-12)
                trade_qty = min(qty, positions[asset]["quantity"])
                realized_pnl = (price - avg_cost) * trade_qty
                pnl_per_asset[asset] += realized_pnl
                total += realized_pnl
                cur_pnl = realized_pnl
                if realized_pnl > 0:
                    win += 1
                    gp += realized_pnl
                else:
                    loss += 1
                    gl += abs(realized_pnl)
                positions[asset]["cost"] -= avg_cost * trade_qty
                positions[asset]["quantity"] -= trade_qty
                
        df.loc[i, "pnl"] = cur_pnl
        df.loc[i, "cumulative_pnl"] = total
        peak = max(peak, total)
        mdd = max(mdd, peak - total)
        
    closed_trades = win + loss
    stats = {
        "win_rate": (win / closed_trades * 100) if closed_trades else 0,
        "profit_factor": (gp / gl) if gl > 0 else float("inf"),
        "total_trades": closed_trades,
        "avg_win": (gp / win) if win else 0,
        "avg_loss": (gl / loss) if loss else 0,
        "max_drawdown": mdd,
    }
    df["asset_cumulative_pnl"] = df.groupby("asset")["pnl"].cumsum()
    return pnl_per_asset, df, stats

def calculate_open_positions(trades_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df is None or trades_df.empty or market_df is None or market_df.empty:
        return pd.DataFrame()
    positions = {}
    for _, row in trades_df.sort_values("timestamp").iterrows():
        asset = row.get("asset", "")
        action = str(row.get("unified_action", "")).lower().strip()
        try:
            qty = float(row.get("quantity", 0.0))
            price = float(row.get("price", 0.0))
        except (ValueError, TypeError):
            qty, price = 0.0, 0.0
            
        if asset not in positions:
            positions[asset] = {"quantity": 0.0, "cost": 0.0}
            
        if action in ["buy", "open"]:
            positions[asset]["cost"] += qty * price
            positions[asset]["quantity"] += qty
        elif action in ["sell", "close"]:
            if positions[asset]["quantity"] > 0:
                avg_cost = positions[asset]["cost"] / max(positions[asset]["quantity"], 1e-12)
                sell_qty = min(qty, positions[asset]["quantity"])
                positions[asset]["cost"] -= avg_cost * sell_qty
                positions[asset]["quantity"] -= sell_qty
                
    open_positions = []
    for asset, data in positions.items():
        if data["quantity"] > 1e-9:
            latest_market = market_df[market_df['asset'] == asset]
            if not latest_market.empty:
                latest_price = latest_market.loc[latest_market['timestamp'].idxmax()]['close']
                avg_entry = data["cost"] / data["quantity"]
                current_value = latest_price * data["quantity"]
                unrealized_pnl = current_value - data["cost"]
                open_positions.append({
                    "Asset": asset, "Quantity": data["quantity"], "Avg. Entry Price": avg_entry,
                    "Current Price": latest_price, "Current Value ($)": current_value,
                    "Unrealized P&L ($)": unrealized_pnl,
                })
    return pd.DataFrame(open_positions)

def match_trades_fifo(trades_df: pd.DataFrame):
    if trades_df is None or trades_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    matched_trades = []
    open_positions = []
    
    for asset, group in trades_df.groupby('asset'):
        buys = [row.to_dict() for _, row in group[group['unified_action'].isin(['buy', 'open'])].sort_values('timestamp').iterrows()]
        sells = [row.to_dict() for _, row in group[group['unified_action'].isin(['sell', 'close'])].sort_values('timestamp').iterrows()]

        for sell in sells:
            sell_qty_remaining = sell.get('quantity', 0)
            
            while sell_qty_remaining > 1e-9 and buys:
                buy = buys[0]
                buy_qty_remaining = buy.get('quantity', 0)
                trade_qty = min(sell_qty_remaining, buy_qty_remaining)

                if trade_qty > 0:
                    pnl = (sell['price'] - buy['price']) * trade_qty
                    hold_time = sell['timestamp'] - buy['timestamp']
                    
                    matched_trades.append({
                        'Asset': asset, 'Quantity': trade_qty,
                        'Buy Time': buy['timestamp'], 'Buy Price': buy['price'],
                        'Sell Time': sell['timestamp'], 'Sell Price': sell['price'],
                        'Hold Time': hold_time, 'P&L ($)': pnl,
                        'Reason Buy': buy.get('reason'), 'Reason Sell': sell.get('reason')
                    })

                    sell_qty_remaining -= trade_qty
                    buys[0]['quantity'] -= trade_qty

                    if buys[0]['quantity'] < 1e-9:
                        buys.pop(0)
        
        open_positions.extend(buys)

    matched_df = pd.DataFrame(matched_trades) if matched_trades else pd.DataFrame()
    open_df = pd.DataFrame(open_positions) if open_positions else pd.DataFrame()

    if not matched_df.empty:
        matched_df['P&L %'] = (matched_df['P&L ($)'] / (matched_df['Buy Price'] * matched_df['Quantity'])) * 100
        matched_df = matched_df.sort_values("Sell Time", ascending=False)
        
    if not open_df.empty:
        open_df = open_df.sort_values("timestamp", ascending=False)

    return matched_df, open_df

def extract_drive_id(url_or_id: str) -> str:
    if not url_or_id: return ""
    s = url_or_id.strip()
    m = re.search(r"/d/([A-Za-z0-9_-]{20,})", s)
    if m: return m.group(1)
    m = re.search(r"[?&]id=([A-Za-z0-9_-]{20,})", s)
    if m: return m.group(1)
    return s

def download_drive_file_bytes(url_or_id: str) -> bytes | None:
    fid = extract_drive_id(url_or_id)
    if not fid: return None
    try:
        base = "https://drive.google.com/uc?export=download"
        with requests.Session() as s:
            r1 = s.get(base, params={"id": fid}, stream=True, timeout=60)
            if "text/html" in (r1.headers.get("Content-Type") or ""):
                m = re.search(r"confirm=([0-9A-Za-z_-]+)", r1.text)
                if m:
                    params = {"id": fid, "confirm": m.group(1)}
                    r2 = s.get(base, params=params, stream=True, timeout=60)
                    r2.raise_for_status()
                    return r2.content
            r1.raise_for_status()
            return r1.content
    except Exception as e:
        st.error(f"Network error downloading file ID {fid}: {e}")
        return None

def _read_parquet_bytes(b: bytes, label: str) -> pd.DataFrame:
    if not b:
        st.warning(f"{label}: no bytes downloaded")
        return pd.DataFrame()
    if b[:4] != b"PAR1":
        try:
            df_csv = pd.read_csv(io.BytesIO(b))
            st.warning(f"{label}: bytes are not Parquet; parsed as CSV instead.")
            return df_csv
        except Exception:
            st.error(f"{label}: not a Parquet file (magic mismatch) and CSV fallback failed.")
            return pd.DataFrame()
    try:
        return pd.read_parquet(io.BytesIO(b))
    except Exception as e:
        st.error(f"{label}: failed to read Parquet: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def load_data(trades_link: str, market_link: str):
    trades = pd.DataFrame()
    market = pd.DataFrame()

    raw_trades = download_drive_file_bytes(trades_link)
    if raw_trades:
        trades = _read_parquet_bytes(raw_trades, "Trades")

    raw_market = download_drive_file_bytes(market_link)
    if raw_market:
        market = _read_parquet_bytes(raw_market, "Market")
    
    if not trades.empty:
        trades = lower_strip_cols(trades)
        
        column_mapping = {}
        if "value" in trades.columns: column_mapping["value"] = "usd_value"
        
        if "side" in trades.columns and "action" in trades.columns:
            column_mapping["side"] = "trade_direction" 
        elif "side" in trades.columns and "action" not in trades.columns:
            column_mapping["side"] = "action"
        
        trades = trades.rename(columns=column_mapping)
        
        if "action" in trades.columns:
            trades["unified_action"] = trades["action"].str.upper().map({
                "OPEN": "buy",
                "CLOSE": "sell"
            }).fillna(trades["action"].str.lower())
        elif "trade_direction" in trades.columns:
            trades["unified_action"] = trades["trade_direction"]
        elif "side" in trades.columns:
            trades["unified_action"] = trades["side"]
        else:
            trades["unified_action"] = "unknown"
        
        if "asset" in trades.columns: 
            trades["asset"] = trades["asset"].apply(unify_symbol)
        
        if 'timestamp' in trades.columns: 
            trades['timestamp'] = pd.to_datetime(trades['timestamp'], errors='coerce')
            if trades['timestamp'].dt.tz is None:
                trades['timestamp'] = trades['timestamp'].dt.tz_localize('UTC')
            else:
                trades['timestamp'] = trades['timestamp'].dt.tz_convert('UTC')

        for col in ["quantity", "price", "usd_value", "p_up", "p_down", "pnl", "pnl_pct"]:
            if col in trades.columns: 
                trades[col] = pd.to_numeric(trades[col], errors="coerce")
    
    if not market.empty:
        market = lower_strip_cols(market)
        market = market.rename(columns={"product_id": "asset"})
        
        if 'timestamp' in market.columns: 
            market['timestamp'] = pd.to_datetime(market['timestamp'], errors='coerce')
            if market['timestamp'].dt.tz is None:
                market['timestamp'] = market['timestamp'].dt.tz_localize('UTC')
            else:
                market['timestamp'] = market['timestamp'].dt.tz_convert('UTC')

        if 'asset' in market.columns: market["asset"] = market["asset"].apply(unify_symbol)
        market = normalize_prob_columns(market)
        for col in ["open", "high", "low", "close", "p_up", "p_down"]:
            if col in market.columns: market[col] = pd.to_numeric(market[col], errors="coerce")
    
    pnl_summary, trades_with_pnl, stats = calculate_pnl_and_metrics(trades) if not trades.empty else ({}, pd.DataFrame(), {})
    return trades_with_pnl, pnl_summary, stats, market

# =========================
# Auto-Refresh Logic & Main App Setup
# =========================
check_auto_refresh()
st.markdown("## Crypto Trading Strategy")
st.caption("ML Signals with Price-Based Exit Logic")
trades_df, pnl_summary, summary_stats, market_df = load_data(TRADES_LINK, MARKET_LINK)
with st.expander("🔎 Debug — data status"):
    st.write({
        "trades_shape": tuple(trades_df.shape) if isinstance(trades_df, pd.DataFrame) else None,
        "market_shape": tuple(market_df.shape) if isinstance(market_df, pd.DataFrame) else None,
        "market_columns": list(market_df.columns)[:12] if not market_df.empty else [],
        "assets_count": int(market_df["asset"].nunique()) if not market_df.empty and "asset" in market_df.columns else 0,
        "assets_sample": sorted(market_df["asset"].dropna().unique())[:10] if not market_df.empty and "asset" in market_df.columns else [],
    })

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>Crypto Strategy</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 2])
    with col1:
        auto_refresh = st.toggle("🔄 Auto-Refresh (5min)", value=st.session_state.auto_refresh_enabled)
        st.session_state.auto_refresh_enabled = auto_refresh
    with col2:
        col2a, col2b = st.columns(2)
        with col2a:
            if st.button("🔄"):
                st.cache_data.clear()
                st.session_state.last_refresh = time.time()
                st.rerun()
        with col2b:
            if st.button("🗑️"):
                st.cache_data.clear()
                st.session_state.clear()
                st.rerun()
    
    if st.session_state.auto_refresh_enabled:
        time_until_refresh = REFRESH_INTERVAL - (time.time() - st.session_state.last_refresh)
        if time_until_refresh > 0:
            minutes, seconds = divmod(int(time_until_refresh), 60)
            st.markdown(f"<p style='text-align: center; font-size: 0.8em; color: #4CAF50;'>⏱️ Next refresh: {minutes:02d}:{seconds:02d}</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='text-align: center; font-size: 0.8em; color: #4CAF50;'>🔄 Refreshing...</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p style='text-align: center; font-size: 0.8em; color: #888;'>Auto-refresh disabled</p>", unsafe_allow_html=True)
    
    st.markdown("---")

    # Date Range and Freshness Display
    # ... (This logic remains the same)

# =========================
# Main content tabs
# =========================
tab1, tab2, tab3 = st.tabs(["📈 Price & Trades", "💰 P&L Analysis", "📜 Trade History"])

with tab1:
    assets = sorted(market_df["asset"].dropna().unique()) if (not market_df.empty and "asset" in market_df.columns) else []
    if assets:
        default_index = assets.index(DEFAULT_ASSET) if DEFAULT_ASSET in assets else 0
        selected_asset = st.selectbox("Select Asset to View", assets, index=default_index, key="asset_select")
        range_choice = st.selectbox("Select Date Range", ["4 hours", "12 hours", "1 day", "3 days", "7 days", "30 days", "All"], index=0, key="range_select")
        asset_market_data = market_df[market_df['asset'] == selected_asset] if not market_df.empty else pd.DataFrame()
        if not asset_market_data.empty:
            last_price = asset_market_data.sort_values('timestamp').iloc[-1]['close']
            
            if last_price < 0.001: price_format = ",.8f"
            elif last_price < 1: price_format = ",.6f"
            else: price_format = ",.2f"
            st.metric(f"Last Price for {selected_asset}", f"${last_price:{price_format}}")

        with st.expander("🔍 Data Resolution Inspector"):
             if not asset_market_data.empty:
                df_sorted = asset_market_data.sort_values('timestamp')
                time_diffs = df_sorted['timestamp'].diff().dropna()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Total Data Points:** {len(df_sorted):,}")
                    min_ts_local = df_sorted['timestamp'].min().tz_convert(LOCAL_TZ)
                    max_ts_local = df_sorted['timestamp'].max().tz_convert(LOCAL_TZ)
                    st.write(f"**Date Range:** {min_ts_local.strftime('%Y-%m-%d %H:%M')} to {max_ts_local.strftime('%Y-%m-%d %H:%M')}")
                with col2:
                    if not time_diffs.empty:
                        mode_diff = time_diffs.mode()[0] if not time_diffs.mode().empty else time_diffs.median()
                        st.write(f"**Most Common Interval:** {mode_diff}")
                        st.write(f"**Min Interval:** {time_diffs.min()}")
                        st.write(f"**Max Interval:** {time_diffs.max()}")
                with col3:
                    recent_data = df_sorted.tail(10)
                    st.write("**Last 10 Timestamps:**")
                    for ts in recent_data['timestamp']:
                        ts_local = ts.tz_convert(LOCAL_TZ)
                        st.write(f"• {ts_local.strftime('%m/%d %H:%M:%S')}")

        st.markdown("---")
        df = asset_market_data.sort_values("timestamp")
        if not df.empty:
            end_date = df["timestamp"].max()
            if range_choice == "4 hours": start_date = end_date - timedelta(hours=4)
            elif range_choice == "12 hours": start_date = end_date - timedelta(hours=12)
            elif range_choice == "1 day": start_date = end_date - timedelta(days=1)
            elif range_choice == "3 days": start_date = end_date - timedelta(days=3)
            elif range_choice == "7 days": start_date = end_date - timedelta(days=7)
            elif range_choice == "30 days": start_date = end_date - timedelta(days=30)
            else: start_date = df["timestamp"].min()
            vis = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)].copy()
            if not vis.empty:
                
                vis['timestamp'] = vis['timestamp'].dt.tz_convert(LOCAL_TZ)
                
                start_date_local = start_date.tz_convert(LOCAL_TZ)
                end_date_local = end_date.tz_convert(LOCAL_TZ)
                st.info(f"Showing {len(vis):,} candles from {start_date_local.strftime('%Y-%m-%d %H:%M')} to {end_date_local.strftime('%Y-%m-%d %H:%M')}")
                
                fig = go.Figure()
                
                fig.add_trace(go.Candlestick(
                    x=vis["timestamp"], open=vis["open"], high=vis["high"], low=vis["low"], close=vis["close"], name=selected_asset, 
                    increasing_line_color='#26a69a', decreasing_line_color='#ef5350', 
                    increasing_fillcolor='rgba(38, 166, 154, 0.5)', decreasing_fillcolor='rgba(239, 83, 80, 0.5)', 
                    line=dict(width=1), hoverinfo='none' 
                ))
                
                price_format = ".8f" if vis['close'].iloc[-1] < 0.001 else ".6f" if vis['close'].iloc[-1] < 1 else ".4f"
                hover_template = (
                    '<b>Time: %{x|%Y-%m-%d %H:%M} ('+ LOCAL_TZ +')</b><br><br>' +
                    'Open: %{customdata[0]:' + price_format + '}<br>' +
                    'High: %{customdata[1]:' + price_format + '}<br>' +
                    'Low: %{customdata[2]:' + price_format + '}<br>' +
                    'Close: %{customdata[3]:' + price_format + '}<extra></extra>'
                )

                fig.add_trace(go.Scatter(
                    x=vis['timestamp'], y=vis['high'], mode='markers',
                    marker=dict(color='rgba(0,0,0,0)', size=0),
                    customdata=vis[['open', 'high', 'low', 'close']],
                    hovertemplate=hover_template, name='OHLC', showlegend=False
                ))

                if 'p_up' in vis.columns and 'p_down' in vis.columns:
                    prob_data = vis.dropna(subset=['p_up', 'p_down'])
                    if not prob_data.empty:
                        prob_data = prob_data.copy()
                        prob_data['confidence'] = abs(prob_data['p_up'] - prob_data['p_down'])
                        prob_data['signal_strength'] = prob_data['confidence'] * 100
                        colors = ['#ff6b6b' if p_down > p_up else '#51cf66' for p_up, p_down in zip(prob_data['p_up'], prob_data['p_down'])]
                        fig.add_trace(go.Scatter(x=prob_data["timestamp"], y=prob_data["close"], mode='markers', marker=dict(size=prob_data['signal_strength'] / 5 + 3, color=colors, opacity=0.7, line=dict(width=1, color='white')), name='ML Signals', customdata=list(zip(prob_data['p_up'], prob_data['p_down'], prob_data['confidence'])), hovertemplate='<b>ML Signal</b><br>Time: %{x|%Y-%m-%d %H:%M}<br>Price: $%{y:.6f}<br>P(Up): %{customdata[0]:.3f}<br>P(Down): %{customdata[1]:.3f}<br>Confidence: %{customdata[2]:.3f}<extra></extra>'))
                
                if not trades_df.empty:
                    asset_trades = trades_df[(trades_df["asset"] == selected_asset) & (trades_df["timestamp"] >= start_date) & (trades_df["timestamp"] <= end_date)].copy()
                    
                    if not asset_trades.empty:
                        asset_trades['timestamp'] = asset_trades['timestamp'].dt.tz_convert(LOCAL_TZ)
                    
                    if not asset_trades.empty:
                        buy_trades = asset_trades[asset_trades["unified_action"].str.lower().isin(["buy", "open"])]
                        if not buy_trades.empty:
                            fig.add_trace(go.Scatter(x=buy_trades["timestamp"], y=buy_trades["price"], mode="markers+text", name="BUY", marker=dict(symbol='triangle-up', size=16, color='#4caf50', line=dict(width=2, color='white')), text=['▲'] * len(buy_trades), textposition="top center", textfont=dict(size=12, color='#4caf50'), customdata=buy_trades.get('reason', ''), hovertemplate='<b>BUY ORDER</b><br>Time: %{x|%Y-%m-%d %H:%M}<br>Price: $%{y:.6f}<br>Reason: %{customdata}<extra></extra>'))
                        sell_trades = asset_trades[asset_trades["unified_action"].str.lower().isin(["sell", "close"])]
                        if not sell_trades.empty:
                            fig.add_trace(go.Scatter(x=sell_trades["timestamp"], y=sell_trades["price"], mode="markers+text", name="SELL", marker=dict(symbol='triangle-down', size=16, color='#f44336', line=dict(width=2, color='white')), text=['▼'] * len(sell_trades), textposition="bottom center", textfont=dict(size=12, color='#f44336'), customdata=sell_trades.get('reason', ''), hovertemplate='<b>SELL ORDER</b><br>Time: %{x|%Y-%m-%d %H:%M}<br>Price: $%{y:.6f}<br>Reason: %{customdata}<extra></extra>'))
                        
                        # --- MODIFICATION: Add trade markers at the bottom of the chart ---
                        y_position = vis['low'].min() - (vis['high'].max() - vis['low'].min()) * 0.02
                        if not buy_trades.empty:
                            fig.add_trace(go.Scatter(x=buy_trades["timestamp"], y=[y_position] * len(buy_trades),mode="markers", name="Buy Signal (Bottom)", marker=dict(symbol='triangle-up', size=8, color='#4caf50', line=dict(width=1, color='white')), showlegend=False, hovertemplate='<b>BUY</b><br>Time: %{x|%Y-%m-%d %H:%M}<extra></extra>'))
                        if not sell_trades.empty:
                             fig.add_trace(go.Scatter(x=sell_trades["timestamp"], y=[y_position] * len(sell_trades),mode="markers", name="Sell Signal (Bottom)", marker=dict(symbol='triangle-down', size=8, color='#f44336', line=dict(width=1, color='white')), showlegend=False, hovertemplate='<b>SELL</b><br>Time: %{x|%Y-%m-%d %H:%M}<extra></extra>'))

                        sorted_trades = asset_trades.sort_values("timestamp")
                        open_trades = []
                        for _, trade in sorted_trades.iterrows():
                            trade_action = str(trade.get('unified_action', '')).lower()
                            if trade_action in ['buy', 'open']: 
                                open_trades.append(trade)
                            elif trade_action in ['sell', 'close'] and open_trades:
                                buy_trade = open_trades.pop(0)
                                pnl = float(trade.get('price', 0)) - float(buy_trade.get('price', 0))
                                pnl_pct = (pnl / float(buy_trade.get('price', 1))) * 100
                                line_color = "#4caf50" if pnl >= 0 else "#f44336"
                                line_width = 4 if abs(pnl_pct) > 10 else 3 if abs(pnl_pct) > 5 else 2
                                fig.add_trace(go.Scatter(x=[buy_trade['timestamp'], trade['timestamp']], y=[buy_trade['price'], trade['price']], mode='lines', line=dict(color=line_color, width=line_width, dash='solid'), opacity=0.8, showlegend=False, hovertemplate=f'<b>Trade P&L</b><br>P&L: ${pnl:.6f} ({pnl_pct:+.2f}%)<br>Hold Time: {trade["timestamp"] - buy_trade["timestamp"]}<extra></extra>', name='Trade P&L'))
                
                if range_choice in ["4 hours", "12 hours"]: 
                    tick_format = '%H:%M'
                else: 
                    tick_format = '%m/%d %H:%M'
                
                price_range = vis['high'].max() - vis['low'].min()
                y_padding = price_range * 0.05
                fig.update_layout(
                    title=f"{selected_asset} — Minute-Level Price & ML Signals ({range_choice})", 
                    template="plotly_white", 
                    xaxis_rangeslider_visible=False, 
                    xaxis=dict(
                        title=f"Time ({LOCAL_TZ})", 
                        type='date', 
                        tickformat=tick_format, 
                        showgrid=True, 
                        gridcolor='rgba(128,128,128,0.1)',
                        tickangle=45
                    ), 
                    yaxis=dict(
                        title="Price (USD)", 
                        tickformat='.8f' if vis['close'].iloc[-1] < 0.001 else '.6f' if vis['close'].iloc[-1] < 1 else '.4f', 
                        showgrid=True, 
                        gridcolor='rgba(128,128,128,0.1)', 
                        range=[vis['low'].min() - y_padding, vis['high'].max() + y_padding]
                    ), 
                    hovermode="x unified", 
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5), 
                    height=750, 
                    margin=dict(l=60, r=20, t=80, b=80), 
                    plot_bgcolor='rgba(250,250,250,0.8)'
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath'], 'scrollZoom': True})
                
                # ... (rest of tab1 code remains the same)

with tab2:
    # (Code remains the same)

with tab3:
    # (Code remains the same)

# =========================
# Trigger auto-refresh check at the end 
# =========================
if st.session_state.auto_refresh_enabled:
    time.sleep(5)
    st.rerun()
