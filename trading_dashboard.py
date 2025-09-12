import io
import re
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pytz import timezone
# Note: Reading Parquet files with pandas requires the 'pyarrow' library.
# You may need to install it: pip install pyarrow
import pyarrow

# =========================
# App Configuration
# =========================
st.set_page_config(page_title="Giga Trading Strategy Analytics", layout="wide")
LOCAL_TZ = "America/Los_Angeles"
DEFAULT_ASSET = "GIGA-USD"

# =========================
# Google Drive Links - UPDATED
# =========================
# This is the trade log file (trade_history_master.parquet)
TRADES_LINK = "https://drive.google.com/file/d/1t60dS-c9R28evHCC-6AJ7sZZKQqPJA81/view?usp=sharing"
# This is the features/market data file (trading_data_complete.parquet)
MARKET_LINK = "https://drive.google.com/file/d/1BGV1Viib4nA3ge7xqCZ4v-oKDkwYmiVj/view?usp=sharing"

# =========================
# Helpers — data processing
# =========================
def lower_strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.lower().strip() for c in out.columns]
    return out

def to_local_naive(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        series = pd.to_datetime(series, unit='s', errors='coerce')
    else:
        series = pd.to_datetime(series, errors='coerce')

    if series.dt.tz is None:
        return series.dt.tz_localize('UTC').dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    else:
        return series.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)

def unify_symbol(val: str) -> str:
    if not isinstance(val, str):
        return val
    s = val.strip().upper().replace("_", "-")
    if "GIGA" in s:
        return "GIGA-USD"
    return s

def calculate_pnl_and_metrics(trades_df):
    if trades_df is None or trades_df.empty:
        return {}, pd.DataFrame(), {}

    pnl_per_asset, positions = {}, {}
    df = trades_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["pnl"], df["cumulative_pnl"] = 0.0, 0.0
    total, win, loss, gp, gl, peak, mdd = 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0

    for i, row in df.iterrows():
        asset = row["asset"]
        action = row["action"]
        price = float(row["price"])
        qty = float(row["quantity"])

        if asset not in positions:
            positions[asset] = {"quantity": 0.0, "cost": 0.0}
            pnl_per_asset[asset] = 0.0

        cur_pnl = 0.0

        if action == "buy":
            positions[asset]["cost"] += qty * price
            positions[asset]["quantity"] += qty
        elif action == "sell":
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

def calculate_open_positions(trades_df, market_df):
    if trades_df is None or trades_df.empty or market_df is None or market_df.empty:
        return pd.DataFrame()

    positions = {}
    for _, row in trades_df.sort_values("timestamp").iterrows():
        asset = row["asset"]
        action = row["action"]
        qty = float(row["quantity"])
        price = float(row["price"])

        if asset not in positions:
            positions[asset] = {"quantity": 0.0, "cost": 0.0}

        if action == "buy":
            positions[asset]["cost"] += qty * price
            positions[asset]["quantity"] += qty
        elif action == "sell":
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
                    "Asset": asset,
                    "Quantity": data["quantity"],
                    "Avg. Entry Price": avg_entry,
                    "Current Price": latest_price,
                    "Current Value ($)": current_value,
                    "Unrealized P&L ($)": unrealized_pnl,
                })
    return pd.DataFrame(open_positions)

def get_trade_display_info(action):
    if action == "buy":
        return "BUY", "green", "triangle-up"
    elif action == "sell":
        return "SELL", "red", "triangle-down"
    else:
        return "TRADE", "blue", "circle"

def extract_drive_id(url_or_id: str) -> str:
    if not url_or_id: return ""
    s = url_or_id.strip(); m = re.search(r"/d/([A-Za-z0-9_-]{20,})", s)
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

@st.cache_data(ttl=600)
def load_data(trades_link, market_link):
    raw_trades = download_drive_file_bytes(trades_link)
    trades = pd.read_parquet(io.BytesIO(raw_trades)) if raw_trades else pd.DataFrame()
    
    raw_market = download_drive_file_bytes(market_link)
    market = pd.read_parquet(io.BytesIO(raw_market)) if raw_market else pd.DataFrame()

    if not trades.empty:
        trades = lower_strip_cols(trades)
        
        # *** KEY MODIFICATION: Align column names from bot to app ***
        column_mapping = {}
        if "product_id" in trades.columns:
            column_mapping["product_id"] = "asset"
        if "side" in trades.columns:
            column_mapping["side"] = "action" # Rename 'side' to 'action'
        if "size" in trades.columns:
            column_mapping["size"] = "quantity" # Rename 'size' to 'quantity'
        
        trades = trades.rename(columns=column_mapping)
        
        if 'timestamp' in trades.columns:
            trades['timestamp'] = to_local_naive(trades['timestamp'])
        for col in ["quantity", "price", "usd_value"]:
            if col in trades.columns:
                trades[col] = pd.to_numeric(trades[col], errors="coerce")

    if not market.empty:
        market = lower_strip_cols(market)
        market = market.rename(columns={"product_id": "asset"})
        if 'timestamp' in market.columns:
            market['timestamp'] = to_local_naive(market['timestamp'])
        market["asset"] = market["asset"].apply(unify_symbol)
        for col in ["open", "high", "low", "close"]:
            if col in market.columns:
                market[col] = pd.to_numeric(market[col], errors="coerce")

    pnl_summary, trades_with_pnl, stats = calculate_pnl_and_metrics(trades) if not trades.empty else ({}, pd.DataFrame(), {})
    return trades_with_pnl, pnl_summary, stats, market

# =========================
# Main App
# =========================
st.markdown("## Giga Trading Strategy Dashboard")
st.caption("ML Signals with Price-Based Exit Logic")
trades_df, pnl_summary, summary_stats, market_df = load_data(TRADES_LINK, MARKET_LINK)

with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>Giga Strategy</h1>", unsafe_allow_html=True)
    
    if not trades_df.empty and 'timestamp' in trades_df.columns:
        min_date, max_date = trades_df['timestamp'].min(), trades_df['timestamp'].max()
        if pd.notna(min_date) and pd.notna(max_date):
            st.markdown(f"<p style='text-align: center;'><strong>{min_date.strftime('%m/%d/%y')} - {max_date.strftime('%m/%d/%y')}</strong></p>", unsafe_allow_html=True)

    now_local = datetime.now(timezone(LOCAL_TZ))
    st.markdown(f"<p style='text-align: center; font-size: 0.9em; color: grey;'>Last updated: {now_local.strftime('%Y-%m-%d %H:%M')}</p>", unsafe_allow_html=True)
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
    
    st.markdown("## 📈 Current Holdings")
    if not market_df.empty and not trades_df.empty:
        open_positions_df = calculate_open_positions(trades_df, market_df)
        if not open_positions_df.empty:
            for _, pos in open_positions_df.iterrows():
                pnl = pos["Unrealized P&L ($)"]
                color = "lightgreen" if pnl > 0 else "salmon"
                st.markdown(f'<p style="color:{color}; font-weight:bold; margin-bottom:0px;">{pos["Asset"]}</p>', unsafe_allow_html=True)
                st.caption(f"Qty: {pos['Quantity']:.4f} | Entry: ${pos['Avg. Entry Price']:.6f} | P&L: ${pnl:.2f}")
        else:
            st.info("No open positions.")
    else:
        st.info("No data to calculate open positions.")

# Main content tabs
tab1, tab2, tab3 = st.tabs(["📈 Price & Trades", "💰 P&L Analysis", "📜 Trade History"])

with tab1:
    assets = sorted(market_df["asset"].dropna().unique()) if not market_df.empty else []
    if assets:
        default_index = assets.index(DEFAULT_ASSET) if DEFAULT_ASSET in assets else 0
        selected_asset = st.selectbox("Select Asset", assets, index=default_index)
        range_choice = st.selectbox("Select Date Range", ["30 days", "7 days", "1 day", "All"], index=0)

        df = market_df[market_df["asset"] == selected_asset].sort_values("timestamp")
        if not df.empty:
            end_date = df["timestamp"].max()
            if range_choice == "1 day": start_date = end_date - timedelta(days=1)
            elif range_choice == "7 days": start_date = end_date - timedelta(days=7)
            elif range_choice == "30 days": start_date = end_date - timedelta(days=30)
            else: start_date = df["timestamp"].min()

            vis = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)].copy()
            
            fig = go.Figure(data=go.Candlestick(
                x=vis["timestamp"], open=vis["open"], high=vis["high"], low=vis["low"], close=vis["close"],
                name=selected_asset
            ))

            if not trades_df.empty:
                asset_trades = trades_df[(trades_df["asset"] == selected_asset) & (trades_df["timestamp"] >= start_date) & (trades_df["timestamp"] <= end_date)]
                
                for action in ["buy", "sell"]:
                    action_trades = asset_trades[asset_trades["action"] == action]
                    if not action_trades.empty:
                        display_name, color, symbol = get_trade_display_info(action)
                        fig.add_trace(go.Scatter(
                            x=action_trades["timestamp"], y=action_trades["price"], mode="markers", name=display_name,
                            marker=dict(symbol=symbol, size=10, color=color, line=dict(width=1, color='Black')),
                            hovertemplate=f"<b>{display_name}</b><br>Price: %{{y:.6f}}<br>Reason: %{{text}}<extra></extra>",
                            text=action_trades['reason']
                        ))

            fig.update_layout(
                template="plotly_white", xaxis_rangeslider_visible=False, hovermode="x unified",
                yaxis_title="Price (USD)", title=f"{selected_asset} — Price & Trade Activity",
                legend=dict(orientation="h", y=1.03, x=0.5, xanchor="center"),
                height=600, margin=dict(l=40, r=20, t=60, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Market data not loaded or available.")

with tab2:
    if not trades_df.empty:
        st.markdown("### Strategy Performance Analysis")
        
        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Scatter(
            x=trades_df["timestamp"], y=trades_df["cumulative_pnl"],
            mode="lines", name="Cumulative P&L", line=dict(color="blue", width=2)
        ))
        fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Break Even")
        
        fig_pnl.update_layout(
            title="Total Portfolio P&L", template="plotly_white", yaxis_title="P&L (USD)",
            xaxis_title="Date", hovermode="x unified"
        )
        st.plotly_chart(fig_pnl, use_container_width=True)
    else:
        st.warning("No trade data loaded to analyze P&L.")

with tab3:
    if not trades_df.empty:
        st.markdown("### Complete Trade Log")
        display_df = trades_df[["timestamp", "asset", "action", "quantity", "price", "usd_value", "reason"]].copy()
        display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        display_df.columns = ["Time", "Asset", "Action", "Quantity", "Price", "USD Value", "Reason"]
        st.dataframe(display_df.sort_values("Time", ascending=False), use_container_width=True)
    else:
        st.warning("No trade history to display.")
