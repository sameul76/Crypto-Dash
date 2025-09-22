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

def to_local_naive(series: pd.Series) -> pd.Series:
    """
    Fixed timezone handling - doesn't force UTC when timestamps are already local
    """
    s = series.copy()
    
    # Convert to datetime without forcing timezone assumptions
    if pd.api.types.is_numeric_dtype(s):
        # Only if truly numeric (unix timestamps), assume UTC
        s = pd.to_datetime(s, unit="s", errors="coerce", utc=True)
        s = s.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    else:
        # If already datetime or string, don't assume timezone
        s = pd.to_datetime(s, errors="coerce")
        
        # Only convert timezone if source has timezone info
        if s.dt.tz is not None:
            s = s.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
        # If timezone-naive, assume already in correct local time (don't convert)
    
    return s

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
        # Use unified_action column
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
        
        # Handle different action formats from your bot
        if action in ["buy", "open"]:  # Your bot uses "buy" in side column
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
        # Use unified_action column
        action = str(row.get("unified_action", "")).lower().strip()
        try:
            qty = float(row.get("quantity", 0.0))
            price = float(row.get("price", 0.0))
        except (ValueError, TypeError):
            qty, price = 0.0, 0.0
            
        if asset not in positions:
            positions[asset] = {"quantity": 0.0, "cost": 0.0}
            
        # Handle different action formats from your bot
        if action in ["buy", "open"]:  # Your bot uses both formats
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

def get_trade_display_info(action: str):
    if action == "buy": return "BUY", "green", "triangle-up"
    elif action == "sell": return "SELL", "red", "triangle-down"
    else: return "TRADE", "blue", "circle"

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
    raw_trades = download_drive_file_bytes(trades_link)
    trades = _read_parquet_bytes(raw_trades, "Trades")
    raw_market = download_drive_file_bytes(market_link)
    market = _read_parquet_bytes(raw_market, "Market")
    
    if not trades.empty:
        trades = lower_strip_cols(trades)
        
        # Handle your specific trade file structure
        column_mapping = {}
        if "value" in trades.columns: column_mapping["value"] = "usd_value"
        
        # Since your file has both 'action' and 'side', use 'side' for trading direction
        # and keep 'action' for position type (OPEN/CLOSE)
        if "side" in trades.columns and "action" in trades.columns:
            # Rename side to trade_direction to avoid conflict
            column_mapping["side"] = "trade_direction" 
        elif "side" in trades.columns and "action" not in trades.columns:
            column_mapping["side"] = "action"
        
        trades = trades.rename(columns=column_mapping)
        
        # Create a unified action column for buy/sell logic
        if "trade_direction" in trades.columns:
            trades["unified_action"] = trades["trade_direction"]
        elif "action" in trades.columns:
            trades["unified_action"] = trades["action"]
        else:
            trades["unified_action"] = "unknown"
        
        if "asset" in trades.columns: 
            trades["asset"] = trades["asset"].apply(unify_symbol)
        
        # Fix timestamp parsing for your ISO format - NO TIMEZONE CONVERSION
        if 'timestamp' in trades.columns: 
            # Your timestamps are already in local time, just parse them as-is
            trades['timestamp'] = pd.to_datetime(trades['timestamp'], errors='coerce')
            # DO NOT convert timezone - your timestamps are already correct local time
            # Remove any timezone info if pandas added it
            if trades['timestamp'].dt.tz is not None:
                trades['timestamp'] = trades['timestamp'].dt.tz_localize(None)
        
        # Convert numeric columns
        for col in ["quantity", "price", "usd_value", "p_up", "p_down", "pnl", "pnl_pct"]:
            if col in trades.columns: 
                trades[col] = pd.to_numeric(trades[col], errors="coerce")
    
    if not market.empty:
    market = lower_strip_cols(market)
    market = market.rename(columns={"product_id": "asset"})
    
    # FIXED: Treat market timestamps the same as trade timestamps
    if 'timestamp' in market.columns: 
        market['timestamp'] = pd.to_datetime(market['timestamp'], errors='coerce')
        # Remove any timezone info if pandas added it
        if market['timestamp'].dt.tz is not None:
            market['timestamp'] = market['timestamp'].dt.tz_localize(None)
    
    if 'asset' in market.columns: market["asset"] = market["asset"].apply(unify_symbol)
    market = normalize_prob_columns(market)
    for col in ["open", "high", "low", "close", "p_up", "p_down"]:
        if col in market.columns: market[col] = pd.to_numeric(market[col], errors="coerce")
    
    pnl_summary, trades_with_pnl, stats = calculate_pnl_and_metrics(trades) if not trades.empty else ({}, pd.DataFrame(), {})
    return trades_with_pnl, pnl_summary, stats, market

# =========================
# Auto-Refresh Logic
# =========================
def check_auto_refresh():
    current_time = time.time()
    time_since_refresh = current_time - st.session_state.last_refresh
    
    if st.session_state.auto_refresh_enabled and time_since_refresh >= REFRESH_INTERVAL:
        st.session_state.last_refresh = current_time
        st.cache_data.clear()
        st.rerun()
    
    return time_since_refresh

# Check auto-refresh at the start
time_since_refresh = check_auto_refresh()

# =========================
# Main App
# =========================
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
# SIDEBAR WITH AUTO-REFRESH
# =========================
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>Crypto Strategy</h1>", unsafe_allow_html=True)
    
    # --- AUTO-REFRESH CONTROLS ---
    col1, col2 = st.columns([3, 1])
    with col1:
        auto_refresh = st.toggle("🔄 Auto-Refresh (5min)", value=st.session_state.auto_refresh_enabled)
        st.session_state.auto_refresh_enabled = auto_refresh
    with col2:
        if st.button("🔄"):
            st.cache_data.clear()
            st.session_state.last_refresh = time.time()
            st.rerun()
    
    # Show countdown if auto-refresh is enabled
    if st.session_state.auto_refresh_enabled:
        time_until_refresh = REFRESH_INTERVAL - time_since_refresh
        if time_until_refresh > 0:
            minutes = int(time_until_refresh // 60)
            seconds = int(time_until_refresh % 60)
            st.markdown(f"<p style='text-align: center; font-size: 0.8em; color: #4CAF50;'>⏱️ Next refresh: {minutes:02d}:{seconds:02d}</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='text-align: center; font-size: 0.8em; color: #4CAF50;'>🔄 Refreshing...</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p style='text-align: center; font-size: 0.8em; color: #888;'>Auto-refresh disabled</p>", unsafe_allow_html=True)
    
    st.markdown("---")

    # --- IMPROVED DATE DISPLAY: Show market data when trade data is old ---
    date_source = "No Data"
    min_date = max_date = None
    
    # Check trade data first
    if not trades_df.empty and 'timestamp' in trades_df.columns:
        trade_min = trades_df['timestamp'].min()
        trade_max = trades_df['timestamp'].max()
        if pd.notna(trade_min) and pd.notna(trade_max):
            # Only use trade dates if they span more than 1 day AND are recent
            days_span = (trade_max - trade_min).days
            days_old = (datetime.now(timezone(LOCAL_TZ)).replace(tzinfo=None) - trade_max).days
            
            if days_span > 0 and days_old <= 3:  # Recent trade data
                min_date, max_date = trade_min, trade_max
                date_source = "Trade Data"
    
    # Fall back to market data if trade data is limited/old
    if min_date is None and not market_df.empty and 'timestamp' in market_df.columns:
        market_min = market_df['timestamp'].min()
        market_max = market_df['timestamp'].max()
        if pd.notna(market_min) and pd.notna(market_max):
            min_date, max_date = market_min, market_max
            date_source = "Market Data"
    
    # Display the date range
    if min_date and max_date:
        st.markdown(f"<p style='text-align: center;'><strong>{min_date.strftime('%m/%d/%y')} - {max_date.strftime('%m/%d/%y')}</strong></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; font-size: 0.8em; color: grey;'>Source: {date_source}</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='text-align: center; color: orange;'>⚠️ No Date Range Available</p>", unsafe_allow_html=True)

    # Last updated timestamp
    now_local = datetime.now(timezone(LOCAL_TZ))
    st.markdown(f"<p style='text-align: center; font-size: 0.9em; color: grey;'>Last updated: {now_local.strftime('%Y-%m-%d %H:%M')}</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Data status indicators
    col1, col2 = st.columns(2)
    with col1:
        trade_status = "✅" if not trades_df.empty else "⚠️"
        trade_count = len(trades_df) if not trades_df.empty else 0
        st.markdown(f"{trade_status} **Trades:** {trade_count}")
    with col2:
        market_status = "✅" if not market_df.empty else "❌"
        market_count = len(market_df) if not market_df.empty else 0
        st.markdown(f"{market_status} **Market:** {market_count:,}")
    
    # Show asset count
    if not market_df.empty and "asset" in market_df.columns:
        asset_count = market_df["asset"].nunique()
        st.markdown(f"📊 **Assets:** {asset_count}")
    
    st.markdown("---")

    st.markdown("## 📊 Strategy Stats")
    if summary_stats:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Closed Trades", f"{summary_stats.get('total_trades', 0):,}")
            st.metric("Win Rate", f"{summary_stats.get('win_rate', 0):.1f}%")
        with col2:
            pf = summary_stats.get('profit_factor', 0)
            st.metric("Profit Factor", "∞" if np.isinf(pf) else f"{pf:.2f}")
            st.metric("Avg Win ($)", f"${summary_stats.get('avg_win', 0):.2f}")

    st.markdown("---")

    st.markdown("## 💵 Realized P&L")
    if pnl_summary:
        total_pnl = sum(pnl for pnl in pnl_summary.values() if pd.notna(pnl))
        st.metric("Overall P&L", f"${total_pnl:,.2f}")
        st.markdown("**By Asset**")
        for asset, pnl in sorted(pnl_summary.items(), key=lambda kv: kv[1], reverse=True):
            color = "#10b981" if pnl >= 0 else "#ef4444"
            st.markdown(f"<div style='display:flex;justify-content:space-between'><span>{asset}</span><span style='color:{color};font-weight:600'>${pnl:,.2f}</span></div>", unsafe_allow_html=True)
    else:
        st.info("No realized P&L yet.")

    st.markdown("---")
    st.markdown("## 📈 Current Holdings & Watchlist")
    open_positions_df = pd.DataFrame()
    if not market_df.empty and not trades_df.empty:
        open_positions_df = calculate_open_positions(trades_df, market_df)
    if not open_positions_df.empty:
        st.markdown("**Open Positions**")
        for _, pos in open_positions_df.iterrows():
            pnl = pos["Unrealized P&L ($)"]
            color = "#16a34a" if pnl >= 0 else "#ef4444"
            asset_name = pos["Asset"]
            current_price = pos["Current Price"]
            price_format = ".6f" if current_price < 1 else ".2f"
            st.markdown(f"""<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: -10px;"><p style="color:{color}; font-weight:bold; margin: 0;">{asset_name}</p><p style="color:black; font-weight:bold; margin: 0;">${current_price:{price_format}}</p></div>""", unsafe_allow_html=True)
            st.caption(f"Qty: {pos['Quantity']:.4f} | Entry: ${pos['Avg. Entry Price']:.6f} | P&L: ${pnl:.2f}")
        st.markdown("---")

    if not market_df.empty and "asset" in market_df.columns:
        st.markdown("**Watchlist**")
        all_assets = sorted(market_df["asset"].dropna().unique())
        held_assets = set(open_positions_df['Asset']) if not open_positions_df.empty else set()
        for asset in all_assets:
            if asset not in held_assets:
                latest_market_data = market_df[market_df['asset'] == asset]
                if not latest_market_data.empty:
                    last_price = latest_market_data.sort_values('timestamp').iloc[-1]['close']
                    price_format = ".6f" if last_price < 1 else ".2f"
                    price_str = f"${last_price:{price_format}}"
                else:
                    price_str = "N/A"
                st.markdown(f"""<div style="display: flex; justify-content: space-between; align-items: center;"><p style="color:grey; margin: 0;">{asset}</p><p style="color:grey; font-weight:bold; margin: 0;">{price_str}</p></div>""", unsafe_allow_html=True)

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
            st.metric(f"Last Price for {selected_asset}", f"${last_price:,.6f}" if last_price < 1 else f"${last_price:,.2f}")
        with st.expander("🔍 Data Resolution Inspector"):
            if not asset_market_data.empty:
                df_sorted = asset_market_data.sort_values('timestamp')
                time_diffs = df_sorted['timestamp'].diff().dropna()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Total Data Points:** {len(df_sorted):,}")
                    st.write(f"**Date Range:** {df_sorted['timestamp'].min().strftime('%Y-%m-%d %H:%M')} to {df_sorted['timestamp'].max().strftime('%Y-%m-%d %H:%M')}")
                with col2:
                    if len(time_diffs) > 0:
                        mode_diff = time_diffs.mode()[0] if len(time_diffs.mode()) > 0 else time_diffs.median()
                        st.write(f"**Most Common Interval:** {mode_diff}")
                        st.write(f"**Min Interval:** {time_diffs.min()}")
                        st.write(f"**Max Interval:** {time_diffs.max()}")
                with col3:
                    recent_data = df_sorted.tail(10)
                    st.write("**Last 10 Timestamps:**")
                    for ts in recent_data['timestamp']:
                        st.write(f"• {ts.strftime('%m/%d %H:%M:%S')}")

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
                st.info(f"Showing {len(vis):,} candles from {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}")
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=vis["timestamp"], open=vis["open"], high=vis["high"], low=vis["low"], close=vis["close"], name=selected_asset, increasing_line_color='#26a69a', decreasing_line_color='#ef5350', increasing_fillcolor='rgba(38, 166, 154, 0.5)', decreasing_fillcolor='rgba(239, 83, 80, 0.5)', line=dict(width=1)))
                if 'p_up' in vis.columns and 'p_down' in vis.columns:
                    prob_data = vis.dropna(subset=['p_up', 'p_down'])
                    if not prob_data.empty:
                        prob_data = prob_data.copy()
                        prob_data['confidence'] = abs(prob_data['p_up'] - prob_data['p_down'])
                        prob_data['signal_strength'] = prob_data['confidence'] * 100
                        colors = ['#ff6b6b' if p_down > p_up else '#51cf66' for p_up, p_down in zip(prob_data['p_up'], prob_data['p_down'])]
                        fig.add_trace(go.Scatter(x=prob_data["timestamp"], y=prob_data["close"], mode='markers', marker=dict(size=prob_data['signal_strength'] / 5 + 3, color=colors, opacity=0.7, line=dict(width=1, color='white')), name='ML Signals', customdata=list(zip(prob_data['p_up'], prob_data['p_down'], prob_data['confidence'])), hovertemplate='<b>ML Signal</b><br>Time: %{x}<br>Price: $%{y:.6f}<br>P(Up): %{customdata[0]:.3f}<br>P(Down): %{customdata[1]:.3f}<br>Confidence: %{customdata[2]:.3f}<extra></extra>'))
                if not trades_df.empty:
                    asset_trades = trades_df[(trades_df["asset"] == selected_asset) & (trades_df["timestamp"] >= start_date) & (trades_df["timestamp"] <= end_date)].copy()
                    if not asset_trades.empty:
                        # Use unified_action for consistent buy/sell detection
                        buy_trades = asset_trades[asset_trades["unified_action"].str.lower().isin(["buy", "open"])]
                        if not buy_trades.empty:
                            fig.add_trace(go.Scatter(x=buy_trades["timestamp"], y=buy_trades["price"], mode="markers+text", name="BUY", marker=dict(symbol='triangle-up', size=16, color='#4caf50', line=dict(width=2, color='white')), text=['▲'] * len(buy_trades), textposition="top center", textfont=dict(size=12, color='#4caf50'), customdata=buy_trades.get('reason', ''), hovertemplate='<b>BUY ORDER</b><br>Time: %{x}<br>Price: $%{y:.6f}<br>Reason: %{customdata}<extra></extra>'))
                        sell_trades = asset_trades[asset_trades["unified_action"].str.lower().isin(["sell", "close"])]
                        if not sell_trades.empty:
                            fig.add_trace(go.Scatter(x=sell_trades["timestamp"], y=sell_trades["price"], mode="markers+text", name="SELL", marker=dict(symbol='triangle-down', size=16, color='#f44336', line=dict(width=2, color='white')), text=['▼'] * len(sell_trades), textposition="bottom center", textfont=dict(size=12, color='#f44336'), customdata=sell_trades.get('reason', ''), hovertemplate='<b>SELL ORDER</b><br>Time: %{x}<br>Price: $%{y:.6f}<br>Reason: %{customdata}<extra></extra>'))
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
                if range_choice in ["4 hours", "12 hours"]: tick_format = '%H:%M'; dtick = 'M30'
                elif range_choice == "1 day": tick_format = '%H:%M'; dtick = 'M60'
                elif range_choice == "3 days": tick_format = '%m/%d %H:%M'; dtick = 'M360'
                else: tick_format = '%m/%d'; dtick = None
                price_range = vis['high'].max() - vis['low'].min()
                y_padding = price_range * 0.05
                fig.update_layout(title=f"{selected_asset} — Minute-Level Price & ML Signals ({range_choice})", template="plotly_white", xaxis_rangeslider_visible=False, xaxis=dict(title="Time", type='date', tickformat=tick_format, dtick=dtick, showgrid=True, gridcolor='rgba(128,128,128,0.1)', tickangle=45 if range_choice in ["4 hours", "12 hours", "1 day"] else 0), yaxis=dict(title="Price (USD)", tickformat='.6f' if vis['close'].iloc[-1] < 1 else '.4f', showgrid=True, gridcolor='rgba(128,128,128,0.1)', range=[vis['low'].min() - y_padding, vis['high'].max() + y_padding]), hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5), height=750, margin=dict(l=60, r=20, t=80, b=80), plot_bgcolor='rgba(250,250,250,0.8)')
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath'], 'scrollZoom': True})
                
                # Show trade statistics if trades exist for this asset
                asset_trades = trades_df[(trades_df["asset"] == selected_asset) & (trades_df["timestamp"] >= start_date) & (trades_df["timestamp"] <= end_date)].copy() if not trades_df.empty else pd.DataFrame()
                if not asset_trades.empty:
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1: st.metric("Total Trades", len(asset_trades))
                    with col2: 
                        buy_count = len(asset_trades[asset_trades["unified_action"].str.lower().isin(["buy", "open"])])
                        st.metric("Buy Orders", buy_count)
                    with col3: 
                        sell_count = len(asset_trades[asset_trades["unified_action"].str.lower().isin(["sell", "close"])])
                        st.metric("Sell Orders", sell_count)
                    with col4:
                        if 'pnl' in asset_trades.columns: 
                            period_pnl = asset_trades['pnl'].sum()
                            st.metric("Period P&L", f"${period_pnl:.6f}")
                    with col5:
                        if len(asset_trades) >= 2: 
                            time_span = asset_trades['timestamp'].max() - asset_trades['timestamp'].min()
                            st.metric("Trading Span", f"{time_span}")
                if 'p_up' in vis.columns and 'p_down' in vis.columns:
                    st.markdown("---")
                    st.markdown("### 🤖 ML Signal Analysis")
                    prob_data = vis.dropna(subset=['p_up', 'p_down'])
                    if not prob_data.empty:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1: bullish_signals = len(prob_data[prob_data['p_up'] > prob_data['p_down']]); st.metric("Bullish Signals", f"{bullish_signals} ({bullish_signals/len(prob_data)*100:.1f}%)")
                        with col2: bearish_signals = len(prob_data[prob_data['p_down'] > prob_data['p_up']]); st.metric("Bearish Signals", f"{bearish_signals} ({bearish_signals/len(prob_data)*100:.1f}%)")
                        with col3: avg_confidence = abs(prob_data['p_up'] - prob_data['p_down']).mean(); st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                        with col4: high_conf_signals = len(prob_data[abs(prob_data['p_up'] - prob_data['p_down']) > 0.1]); st.metric("High Confidence", f"{high_conf_signals} (>10%)")
            else:
                st.warning(f"No data for {selected_asset} in the selected date range of {range_choice}.")
        else:
            st.warning(f"No market data found for {selected_asset}.")
    else:
        st.warning("Market data not loaded or available.")

with tab2:
    if not trades_df.empty and "timestamp" in trades_df.columns and "cumulative_pnl" in trades_df.columns:
        st.markdown("### Strategy Performance Analysis")
        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Scatter(x=trades_df["timestamp"], y=trades_df["cumulative_pnl"], mode="lines", name="Cumulative P&L", line=dict(color="blue", width=2)))
        fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Break Even")
        fig_pnl.update_layout(title="Total Portfolio P&L", template="plotly_white", yaxis_title="P&L (USD)", xaxis_title="Date", hovermode="x unified")
        st.plotly_chart(fig_pnl, use_container_width=True)
    else:
        st.warning("No trade data loaded to analyze P&L.")

with tab3:
    if not trades_df.empty:
        st.markdown("### Complete Trade Log")
        
        # Get available columns from your trade file, avoiding duplicates
        available_cols = []
        priority_cols = ["timestamp", "asset", "unified_action", "action", "trade_direction", "quantity", "price", "usd_value", "reason", "p_up", "p_down", "pnl", "pnl_pct", "confidence", "take_profit", "stop_loss", "trading_mode"]
        
        # Add columns in priority order, avoiding duplicates
        for col in priority_cols:
            if col in trades_df.columns and col not in available_cols:
                available_cols.append(col)
        
        # Add any remaining columns
        for col in trades_df.columns:
            if col not in available_cols:
                available_cols.append(col)
        
        display_df = trades_df[available_cols].copy()
        
        # Handle timestamp
        if "timestamp" in display_df.columns:
            display_df["timestamp"] = pd.to_datetime(display_df["timestamp"], errors="coerce")
            display_df["Time"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
            display_df.drop(columns=["timestamp"], inplace=True)
        
        # Create a safe rename dictionary that only renames existing columns
        rename_dict = {}
        column_renames = {
            "asset": "Asset",
            "unified_action": "Direction", 
            "action": "Action",
            "trade_direction": "Side",
            "quantity": "Quantity",
            "price": "Price", 
            "usd_value": "USD Value",
            "reason": "Reason",
            "p_up": "P(Up)",
            "p_down": "P(Down)",
            "pnl": "P&L ($)",
            "pnl_pct": "P&L %",
            "confidence": "Confidence",
            "take_profit": "Take Profit",
            "stop_loss": "Stop Loss",
            "trading_mode": "Mode"
        }
        
        # Only add to rename_dict if column exists and target name doesn't already exist
        for old_name, new_name in column_renames.items():
            if old_name in display_df.columns and new_name not in display_df.columns:
                rename_dict[old_name] = new_name
        
        if rename_dict:
            display_df.rename(columns=rename_dict, inplace=True)
        
        if "Time" in display_df.columns:
            display_df = display_df.sort_values("Time", ascending=False)
            
        st.dataframe(display_df, use_container_width=True)
    else:
        st.warning("No trade history to display.")

# =========================
# Trigger auto-refresh check at the end 
# =========================
if st.session_state.auto_refresh_enabled:
    time.sleep(5)  # Wait 5 seconds before checking again
    st.rerun()

