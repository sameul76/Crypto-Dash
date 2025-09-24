import io
import re
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from datetime import datetime, timedelta
import pytz
from decimal import Decimal, getcontext

# Set precision for Decimal calculations
getcontext().prec = 30

# Note: Reading Parquet files with pandas requires the 'pyarrow' and 'pytz' libraries.
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

DEFAULT_ASSET = "GIGA-USD"

# =========================
# Google Drive Links
# =========================
# Trade log file (trades_log.parquet) - UPDATED
TRADES_LINK = "https://drive.google.com/file/d/1hyM37eafLgvMo8RJDtw9GSEdZ-LQ05ks/view?usp=sharing"
# Features/market data file exported by your OHLCV service (parquet or csv)
MARKET_LINK = "https://drive.google.com/file/d/1JaNhwQTcYOZ-tpP_ZwHXHtNzo-GpW-TO/view?usp=sharing"

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
    # Keep existing column names, but normalize common variants for p_up/p_down.
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
    # Ensure columns exist (will stay NaN if file doesn't include them)
    for c in ["p_up", "p_down", "p_hold", "confidence"]:
        if c not in df.columns:
            df[c] = np.nan
    return df

def _fallback_confidence(df_like: pd.DataFrame) -> pd.Series:
    """
    If 'confidence' is missing, produce a safe fallback:
    - If p_hold exists: 3-class margin = max(p) - second_best(p)
    - Else: abs(p_up - p_down)
    """
    if "p_hold" in df_like.columns and df_like["p_hold"].notna().any():
        probs = df_like[["p_down", "p_hold", "p_up"]].to_numpy(dtype=float)
        best = np.nanmax(probs, axis=1)
        second = np.partition(probs, -2, axis=1)[:, -2]
        conf = best - second
        conf = np.clip(conf, 0.0, 1.0)
        conf[~np.isfinite(conf)] = 0.0
        return pd.Series(conf, index=df_like.index)
    else:
        return (df_like["p_up"] - df_like["p_down"]).abs()

def format_timedelta_hours_minutes(td):
    """Formats a pandas Timedelta into a total hours and minutes string like 'HH:MM'."""
    if pd.isna(td):
        return "N/A"
    total_seconds = int(td.total_seconds())
    total_minutes, _ = divmod(total_seconds, 60)
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours:02d}:{minutes:02d}"

def calculate_pnl_and_metrics(trades_df: pd.DataFrame):
    if trades_df is None or trades_df.empty:
        return {}, pd.DataFrame(), {}
    pnl_per_asset, positions = {}, {}
    df = trades_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["pnl"], df["cumulative_pnl"] = Decimal(0), Decimal(0)
    total, win, loss, gp, gl, peak, mdd = Decimal(0), 0, 0, Decimal(0), Decimal(0), Decimal(0), Decimal(0)
    
    for i, row in df.iterrows():
        asset = row.get("asset", "")
        action = str(row.get("unified_action", "")).lower().strip()
        
        price = row.get("price", Decimal(0))
        qty = row.get("quantity", Decimal(0))
            
        if asset not in positions:
            positions[asset] = {"quantity": Decimal(0), "cost": Decimal(0)}
            pnl_per_asset[asset] = Decimal(0)
            
        cur_pnl = Decimal(0)
        
        if action in ["buy", "open"]:
            positions[asset]["cost"] += qty * price
            positions[asset]["quantity"] += qty
        elif action in ["sell", "close"]:
            if positions[asset]["quantity"] > 0:
                avg_cost = positions[asset]["cost"] / max(positions[asset]["quantity"], Decimal("1e-12"))
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
    # Fix for TypeError: Use transform to preserve index alignment with Decimal objects
    if not df.empty and 'pnl' in df.columns:
        df["asset_cumulative_pnl"] = df.groupby("asset")["pnl"].transform(lambda x: x.cumsum())
    
    return pnl_per_asset, df, stats

def calculate_open_positions(trades_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df is None or trades_df.empty or market_df is None or market_df.empty:
        return pd.DataFrame()
    positions = {}
    for _, row in trades_df.sort_values("timestamp").iterrows():
        asset = row.get("asset", "")
        action = str(row.get("unified_action", "")).lower().strip()
        
        qty = row.get("quantity", Decimal(0))
        price = row.get("price", Decimal(0))
            
        if asset not in positions:
            positions[asset] = {"quantity": Decimal(0), "cost": Decimal(0)}
            
        if action in ["buy", "open"]:
            positions[asset]["cost"] += qty * price
            positions[asset]["quantity"] += qty
        elif action in ["sell", "close"]:
            if positions[asset]["quantity"] > 0:
                avg_cost = positions[asset]["cost"] / max(positions[asset]["quantity"], Decimal("1e-12"))
                sell_qty = min(qty, positions[asset]["quantity"])
                positions[asset]["cost"] -= avg_cost * sell_qty
                positions[asset]["quantity"] -= sell_qty
                
    open_positions = []
    for asset, data in positions.items():
        if data["quantity"] > Decimal("1e-9"):
            latest_market = market_df[market_df['asset'] == asset]
            if not latest_market.empty:
                latest_price = Decimal(str(latest_market.loc[latest_market['timestamp'].idxmax()]['close']))
                avg_entry = data["cost"] / data["quantity"]
                current_value = latest_price * data["quantity"]
                unrealized_pnl = current_value - data["cost"]
                open_positions.append({
                    "Asset": asset, "Quantity": float(data["quantity"]), "Avg. Entry Price": float(avg_entry),
                    "Current Price": float(latest_price), "Current Value ($)": float(current_value),
                    "Unrealized P&L ($)": float(unrealized_pnl),
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
            sell_qty_remaining = sell.get('quantity', Decimal(0))
            
            while sell_qty_remaining > Decimal("1e-9") and buys:
                if buys[0]['timestamp'] >= sell['timestamp']:
                    break

                buy = buys[0]
                buy_qty_remaining = buy.get('quantity', Decimal(0))
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

                    if buys[0]['quantity'] < Decimal("1e-9"):
                        buys.pop(0)
            
        open_positions.extend(buys)

    matched_df = pd.DataFrame(matched_trades) if matched_trades else pd.DataFrame()
    open_df = pd.DataFrame(open_positions) if open_positions else pd.DataFrame()

    if not matched_df.empty:
        buy_cost = matched_df['Buy Price'] * matched_df['Quantity']
        is_buy_cost_zero = buy_cost < Decimal("1e-18")
        matched_df['P&L %'] = 100 * np.where(is_buy_cost_zero, 0, matched_df['P&L ($)'] / buy_cost)
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

@st.cache_data(ttl=60)
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
            trades["unified_action"] = trades["action"].str.upper().map({"OPEN": "buy", "CLOSE": "sell"}).fillna(trades["action"].str.lower())
        elif "trade_direction" in trades.columns: trades["unified_action"] = trades["trade_direction"]
        elif "side" in trades.columns: trades["unified_action"] = trades["side"]
        else: trades["unified_action"] = "unknown"
        
        if "asset" in trades.columns: trades["asset"] = trades["asset"].apply(unify_symbol)
        if 'timestamp' in trades.columns: trades['timestamp'] = pd.to_datetime(trades['timestamp'], errors='coerce')

        # Convert financial columns to Decimal for precision
        for col in ["quantity", "price", "usd_value", "pnl", "pnl_pct"]:
            if col in trades.columns:
                trades[col] = trades[col].apply(lambda x: Decimal(str(x)) if pd.notna(x) else Decimal(0))
    
    if not market.empty:
        market = lower_strip_cols(market)
        market = market.rename(columns={"product_id": "asset"})
        market = normalize_prob_columns(market)
        
        if 'timestamp' in market.columns: market['timestamp'] = pd.to_datetime(market['timestamp'], errors='coerce')
        if 'asset' in market.columns: market["asset"] = market["asset"].apply(unify_symbol)

        # Coerce numeric columns we depend on for plotting/metrics
        for col in ["open", "high", "low", "close", "p_up", "p_down", "p_hold", "confidence"]:
            if col in market.columns:
                market[col] = pd.to_numeric(market[col], errors="coerce")
    
    pnl_summary, trades_with_pnl, stats = calculate_pnl_and_metrics(trades) if not trades.empty else ({}, pd.DataFrame(), {})
    return trades_with_pnl, pnl_summary, stats, market

# =========================
# Auto-Refresh Logic
# =========================
def check_auto_refresh():
    current_time = time.time()
    time_since_refresh = current_time - st.session_state.get('last_refresh', 0)
    
    if st.session_state.get('auto_refresh_enabled', False) and time_since_refresh >= REFRESH_INTERVAL:
        st.session_state.last_refresh = current_time
        st.cache_data.clear()
        st.rerun()
    
    return time_since_refresh

time_since_refresh = check_auto_refresh()

# =========================
# Main App
# =========================
st.markdown("## Crypto Trading Strategy")
st.caption("ML Signals with Price-Based Exit Logic")

trades_df, pnl_summary, summary_stats, market_df = load_data(TRADES_LINK, MARKET_LINK)

# Add debug information to see actual timestamps
with st.expander("🔍 Data Freshness Debug", expanded=True):
    if not trades_df.empty and 'timestamp' in trades_df.columns:
        latest_trade = trades_df['timestamp'].max()
        st.write(f"**Latest Trade Timestamp:** {latest_trade}")
        st.write(f"**Total Trades:** {len(trades_df)}")
        
    if not market_df.empty and 'timestamp' in market_df.columns:
        latest_market = market_df['timestamp'].max()
        st.write(f"**Latest Market Timestamp:** {latest_market}")
        st.write(f"**Total Market Records:** {len(market_df)}")
    
    st.write(f"**Cache Age:** {time.time() - st.session_state.last_refresh:.1f} seconds")
    
    pst = pytz.timezone('America/Los_Angeles')
    now_utc = datetime.now(pytz.utc)
    now_pst = now_utc.astimezone(pst)
    st.write(f"**Current Time (PST):** {now_pst.strftime('%Y-%m-%d %H:%M:%S %Z')}")

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
        auto_refresh = st.toggle("🔄 Auto-Refresh (5min)", value=st.session_state.get('auto_refresh_enabled', True))
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
    
    if st.button("🔍 Force Fresh Load (No Cache)"):
        st.cache_data.clear()
        st.session_state.clear()
        st.rerun()
    
    if st.session_state.get('auto_refresh_enabled', True):
        time_until_refresh = REFRESH_INTERVAL - time_since_refresh
        if time_until_refresh > 0:
            minutes, seconds = divmod(int(time_until_refresh), 60)
            st.markdown(f"<p style='text-align: center; font-size: 0.8em; color: #4CAF50;'>⏱️ Next refresh: {minutes:02d}:{seconds:02d}</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='text-align: center; font-size: 0.8em; color: #4CAF50;'>🔄 Refreshing...</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p style='text-align: center; font-size: 0.8em; color: #888;'>Auto-refresh disabled</p>", unsafe_allow_html=True)
    
    st.markdown("---")

    date_source = "No Data"
    min_date = max_date = None
    
    if not trades_df.empty and 'timestamp' in trades_df.columns and trades_df['timestamp'].notna().any():
        trade_min = trades_df['timestamp'].min()
        trade_max = trades_df['timestamp'].max()
        if pd.notna(trade_min) and pd.notna(trade_max):
            days_span = (trade_max - trade_min).days
            now_utc = datetime.now(pytz.utc)
            trade_max_utc = trade_max.tz_localize('UTC') if trade_max.tzinfo is None else trade_max
            days_old = (now_utc - trade_max_utc).days
            if days_span >= 0 and days_old <= 3:
                min_date, max_date = trade_min, trade_max
                date_source = "Trade Data"
    
    if min_date is None and not market_df.empty and 'timestamp' in market_df.columns and market_df['timestamp'].notna().any():
        market_min = market_df['timestamp'].min()
        market_max = market_df['timestamp'].max()
        if pd.notna(market_min) and pd.notna(market_max):
            min_date, max_date = market_min, market_max
            date_source = "Market Data"
    
    if min_date and max_date:
        st.markdown(f"<p style='text-align: center;'><strong>{min_date.strftime('%m/%d/%y')} - {max_date.strftime('%m/%d/%y')}</strong></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; font-size: 0.8em; color: grey;'>Source: {date_source}</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='text-align: center; color: orange;'>⚠️ No Date Range Available</p>", unsafe_allow_html=True)

    latest_data_time = None
    data_source_for_time = "No Data"
    
    if not trades_df.empty and 'timestamp' in trades_df.columns and trades_df['timestamp'].notna().any():
        latest_trade = trades_df['timestamp'].max()
        if pd.notna(latest_trade):
            latest_data_time = latest_trade
            data_source_for_time = "Latest Trade"
    
    if not market_df.empty and 'timestamp' in market_df.columns and market_df['timestamp'].notna().any():
        latest_market = market_df['timestamp'].max()
        if pd.notna(latest_market):
            if latest_data_time is None or latest_market > latest_data_time:
                latest_data_time = latest_market
                data_source_for_time = "Latest Market Data"
    
    if latest_data_time:
        st.markdown(f"<p style='text-align: center; font-size: 0.9em; color: grey;'>{data_source_for_time}: {latest_data_time.strftime('%Y-%m-%d %H:%M')}</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p style='text-align: center; font-size: 0.9em; color: grey;'>No data timestamps available</p>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        trade_status = "✅" if not trades_df.empty else "⚠️"
        trade_count = len(trades_df) if not trades_df.empty else 0
        st.markdown(f"{trade_status} **Trades:** {trade_count}")
    with col2:
        market_status = "✅" if not market_df.empty else "❌"
        market_count = len(market_df) if not market_df.empty else 0
        st.markdown(f"{market_status} **Market:** {market_count:,}")
    
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
            st.metric("Profit Factor", "∞" if isinstance(pf, float) and np.isinf(pf) else f"{pf:.2f}")
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
            
            if current_price < 0.001: price_format = ".8f"
            elif current_price < 1: price_format = ".6f"
            else: price_format = ".2f"

            avg_entry_price = pos['Avg. Entry Price']
            if avg_entry_price < 0.001: entry_price_format = ".8f"
            elif avg_entry_price < 1: entry_price_format = ".6f"
            else: entry_price_format = ".2f"

            st.markdown(f"""<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: -10px;"><p style="color:{color}; font-weight:bold; margin: 0;">{asset_name}</p><p style="color:black; font-weight:bold; margin: 0;">${current_price:{price_format}}</p></div>""", unsafe_allow_html=True)
            st.caption(f"Qty: {pos['Quantity']:.4f} | Entry: ${avg_entry_price:{entry_price_format}} | P&L: ${pnl:.2f}")
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
                    
                    if last_price < 0.001: price_format = ".8f"
                    elif last_price < 1: price_format = ".6f"
                    else: price_format = ".2f"
                    
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
        selected_asset = st.selectbox("Select Asset to View", assets, index=default_index, key="asset_select_main")
        range_choice = st.selectbox("Select Date Range", ["4 hours", "12 hours", "1 day", "3 days", "7 days", "30 days", "All"], index=0, key="range_select_main")
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
                    min_ts = df_sorted['timestamp'].min()
                    max_ts = df_sorted['timestamp'].max()
                    st.write(f"**Date Range:** {min_ts.strftime('%Y-%m-%d %H:%M')} to {max_ts.strftime('%Y-%m-%d %H:%M')}")
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
                
                price_range = vis['high'].max() - vis['low'].min()
                y_padding_top = price_range * 0.05
                y_padding_bottom = price_range * 0.15
                
                y_min_range = vis['low'].min() - y_padding_bottom
                y_max_range = vis['high'].max() + y_padding_top
                
                marker_y_position = y_min_range + (price_range * 0.02)
                
                fig = go.Figure()
                
                fig.add_trace(go.Candlestick(
                    x=vis["timestamp"], open=vis["open"], high=vis["high"], low=vis["low"], close=vis["close"], name=selected_asset, 
                    increasing_line_color='#26a69a', decreasing_line_color='#ef5350', 
                    increasing_fillcolor='rgba(38, 166, 154, 0.5)', decreasing_fillcolor='rgba(239, 83, 80, 0.5)', 
                    line=dict(width=1)
                ))

                # ==== Signals: use confidence from file if present ====
                if {'p_up','p_down'}.issubset(vis.columns):
                    prob_data = vis.dropna(subset=['p_up', 'p_down']).copy()
                    if not prob_data.empty:
                        if 'confidence' in prob_data.columns and prob_data['confidence'].notna().any():
                            # use provided confidence
                            pass
                        else:
                            # fallback (prefer 3-class margin if p_hold available)
                            prob_data['confidence'] = _fallback_confidence(prob_data)

                        prob_data['signal_strength'] = prob_data['confidence'] * 100
                        colors = ['#ff6b6b' if p_down > p_up else '#51cf66'
                                  for p_up, p_down in zip(prob_data['p_up'], prob_data['p_down'])]

                        fig.add_trace(go.Scatter(
                            x=prob_data["timestamp"],
                            y=prob_data["close"],
                            mode='markers',
                            marker=dict(
                                size=prob_data['signal_strength'] / 5 + 3,
                                color=colors,
                                opacity=0.7,
                                line=dict(width=1, color='white')
                            ),
                            name='ML Signals',
                            customdata=list(zip(
                                prob_data['p_up'],
                                prob_data['p_down'],
                                prob_data.get('p_hold', pd.Series(index=prob_data.index, dtype=float)).fillna(np.nan),
                                prob_data['confidence']
                            )),
                            hovertemplate=(
                                "<b>ML Signal</b><br>"
                                "Time: %{x|%Y-%m-%d %H:%M}<br>"
                                "Price: $%{y:.6f}<br>"
                                "P(Up): %{customdata[0]:.3f}<br>"
                                "P(Down): %{customdata[1]:.3f}<br>"
                                "P(Hold): %{customdata[2]:.3f}<br>"
                                "Confidence: %{customdata[3]:.3f}"
                                "<extra></extra>"
                            )
                        ))

                # ==== Trade markers ====
                if not trades_df.empty:
                    asset_trades = trades_df[
                        (trades_df["asset"] == selected_asset) &
                        (trades_df["timestamp"] >= start_date) &
                        (trades_df["timestamp"] <= end_date)
                    ].copy()
                    
                    if not asset_trades.empty:
                        buy_trades = asset_trades[asset_trades["unified_action"].str.lower().isin(["buy", "open"])]
                        if not buy_trades.empty:
                            buy_prices = buy_trades["price"].apply(float)
                            buy_reasons = buy_trades.get('reason', pd.Series([''] * len(buy_trades))).fillna('')
                            fig.add_trace(go.Scatter(
                                x=buy_trades["timestamp"], 
                                y=[marker_y_position] * len(buy_trades), 
                                mode="markers",
                                name="BUY", 
                                marker=dict(symbol='triangle-up', size=14, color='#4caf50', line=dict(width=1, color='white')),
                                customdata=np.stack((buy_prices, buy_reasons), axis=-1),
                                hovertemplate='<b>BUY</b> @ $%{customdata[0]:.8f}<br>%{x|%H:%M:%S}<br>Reason: %{customdata[1]}<extra></extra>'
                            ))

                        sell_trades = asset_trades[asset_trades["unified_action"].str.lower().isin(["sell", "close"])]
                        if not sell_trades.empty:
                            sell_prices = sell_trades["price"].apply(float)
                            sell_reasons = sell_trades.get('reason', pd.Series([''] * len(sell_trades))).fillna('')
                            fig.add_trace(go.Scatter(
                                x=sell_trades["timestamp"], 
                                y=[marker_y_position] * len(sell_trades),
                                mode="markers",
                                name="SELL", 
                                marker=dict(symbol='triangle-down', size=14, color='#f44336', line=dict(width=1, color='white')),
                                customdata=np.stack((sell_prices, sell_reasons), axis=-1),
                                hovertemplate='<b>SELL</b> @ $%{customdata[0]:.8f}<br>%{x|%H:%M:%S}<br>Reason: %{customdata[1]}<extra></extra>'
                            ))
                            
                if range_choice in ["4 hours", "12 hours"]: 
                    tick_format = '%H:%M'
                else: 
                    tick_format = '%m/%d %H:%M'
                
                fig.update_layout(
                    title=f"{selected_asset} — Price & Trades ({range_choice})", 
                    template="plotly_white", 
                    xaxis_rangeslider_visible=False, 
                    xaxis=dict(
                        title="Time", type='date', tickformat=tick_format, 
                        showgrid=True, gridcolor='rgba(128,128,128,0.1)', tickangle=-45
                    ), 
                    yaxis=dict(
                        title="Price (USD)", 
                        tickformat='.8f' if vis['close'].iloc[-1] < 0.001 else '.6f' if vis['close'].iloc[-1] < 1 else '.4f', 
                        showgrid=True, gridcolor='rgba(128,128,128,0.1)', 
                        range=[y_min_range, y_max_range]
                    ), 
                    hovermode="x unified", 
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5), 
                    height=750, 
                    margin=dict(l=60, r=20, t=80, b=80), 
                    plot_bgcolor='rgba(250,250,250,0.8)'
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath'], 'scrollZoom': True})
                
                # Period trade stats
                asset_trades = trades_df[
                    (trades_df["asset"] == selected_asset) &
                    (trades_df["timestamp"] >= start_date) &
                    (trades_df["timestamp"] <= end_date)
                ].copy() if not trades_df.empty else pd.DataFrame()
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
                            st.metric("Trading Span", format_timedelta_hours_minutes(time_span))
                
                # ==== Signal analysis summary ====
                if {'p_up','p_down'}.issubset(vis.columns):
                    st.markdown("---")
                    st.markdown("### 🤖 ML Signal Analysis")
                    prob_data = vis.dropna(subset=['p_up', 'p_down']).copy()
                    if not prob_data.empty:
                        if 'confidence' not in prob_data.columns or prob_data['confidence'].isna().all():
                            prob_data['confidence'] = _fallback_confidence(prob_data)

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            bullish_signals = len(prob_data[prob_data['p_up'] > prob_data['p_down']])
                            st.metric("Bullish Signals", f"{bullish_signals} ({bullish_signals/len(prob_data)*100:.1f}%)")
                        with col2:
                            bearish_signals = len(prob_data[prob_data['p_down'] > prob_data['p_up']])
                            st.metric("Bearish Signals", f"{bearish_signals} ({bearish_signals/len(prob_data)*100:.1f}%)")
                        with col3:
                            avg_confidence = prob_data['confidence'].mean()
                            st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                        with col4:
                            high_conf_signals = len(prob_data[prob_data['confidence'] > 0.10])
                            st.metric("High Confidence", f"{high_conf_signals} (>10%)")

            else:
                st.warning(f"No data for {selected_asset} in the selected date range of {range_choice}.")
        else:
            st.warning(f"No market data found for {selected_asset}.")
    else:
        st.warning("Market data not loaded or available.")

with tab2:
    if not trades_df.empty and "timestamp" in trades_df.columns and "cumulative_pnl" in trades_df.columns:
        st.markdown("### Strategy Performance Analysis")
        # Convert Decimal to float for plotting
        plot_df = trades_df.copy()
        plot_df['cumulative_pnl'] = plot_df['cumulative_pnl'].apply(float)
        
        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Scatter(x=plot_df["timestamp"], y=plot_df["cumulative_pnl"], mode="lines", name="Cumulative P&L", line=dict(color="blue", width=2)))
        fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Break Even")
        fig_pnl.update_layout(title="Total Portfolio P&L", template="plotly_white", yaxis_title="P&L (USD)", xaxis_title="Date", hovermode="x unified")
        st.plotly_chart(fig_pnl, use_container_width=True)
    else:
        st.warning("No trade data loaded to analyze P&L.")

with tab3:
    st.markdown("### Trade History")
    if not trades_df.empty:
        matched_df, open_df = match_trades_fifo(trades_df)
        
        st.markdown("#### Completed Trades (FIFO)")
        if not matched_df.empty:
            display_matched = matched_df.copy()
            # Convert Decimal columns to float for Streamlit display formatting
            for col in ['Quantity', 'Buy Price', 'Sell Price', 'P&L ($)', 'P&L %']:
                if col in display_matched.columns:
                    display_matched[col] = display_matched[col].apply(float)

            display_matched['Buy Time'] = display_matched['Buy Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            display_matched['Sell Time'] = display_matched['Sell Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            display_matched['Hold Time'] = display_matched['Hold Time'].apply(format_timedelta_hours_minutes)

            st.dataframe(display_matched[['Asset', 'Quantity', 'Buy Time', 'Buy Price', 'Sell Time', 'Sell Price', 'Hold Time', 'P&L ($)', 'P&L %']],
                column_config={
                    "Asset": st.column_config.TextColumn(width="small"),
                    "Quantity": st.column_config.NumberColumn(format="%.4f", width="small"),
                    "Buy Price": st.column_config.NumberColumn(format="$%.8f"),
                    "Sell Price": st.column_config.NumberColumn(format="$%.8f"),
                    "P&L ($)": st.column_config.NumberColumn(format="$%.4f"),
                    "P&L %": st.column_config.NumberColumn(format="%.4f%%"),
                },
                use_container_width=True, hide_index=True
            )
        else:
            st.info("No completed (buy/sell) trades found.")

        st.markdown("---")
        st.markdown("#### Open Positions")
        if not open_df.empty:
            display_open = open_df.copy()
            # Convert Decimal to float for display
            for col in ['quantity', 'price']:
                if col in display_open.columns:
                    display_open[col] = display_open[col].apply(float)

            display_open['Time'] = display_open['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            display_open = display_open.rename(columns={'asset': 'Asset', 'quantity': 'Quantity', 'price': 'Price', 'reason': 'Reason'})
            
            st.dataframe(display_open[['Time', 'Asset', 'Quantity', 'Price', 'Reason']], 
                column_config={
                    "Asset": st.column_config.TextColumn(width="small"),
                    "Quantity": st.column_config.NumberColumn(format="%.4f", width="small"),
                    "Price": st.column_config.NumberColumn(format="$%.8f"),
                },
                use_container_width=True, hide_index=True
            )
        else:
            st.info("No open positions.")
            
    else:
        st.warning("No trade history to display.")

# Auto-refresh is handled by the check_auto_refresh() function at the top

