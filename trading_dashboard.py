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
st.set_page_config(page_title="PAXG Transfer Strategy Analytics", layout="wide")
LOCAL_TZ = "America/Los_Angeles"
DEFAULT_ASSET = "GIGA-USD"

# =========================
# Google Drive Links - Updated for PAXG Strategy (Synced Files)
# =========================
TRADES_LINK = "https://drive.google.com/file/d/19ASI0V2NkaeIJT6UNZemRUgUWubBhdPT/view?usp=sharing"  # trade_history_master.parquet
MARKET_LINK = "https://drive.google.com/file/d/1uVpl125-qzH1Q9KrBEt-vFHTEphzptNQ/view?usp=sharing"   # features_latest.parquet

# =========================
# Helpers — data processing
# =========================
def lower_strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.lower().strip() for c in out.columns]
    return out

def to_local_naive(series: pd.Series) -> pd.Series:
    """
    Converts a pandas Series of timestamps to naive local time objects.
    Handles numeric (Unix), naive datetime, and aware datetime formats.
    """
    if pd.api.types.is_numeric_dtype(series):
        series = pd.to_datetime(series, unit='s', errors='coerce')
    else:
        series = pd.to_datetime(series, errors='coerce')

    if series.dt.tz is None:
        return series.dt.tz_localize('UTC').dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    else:
        return series.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)

def unify_symbol(val: str) -> str:
    """Unify GIGA-like ids to GIGA-USD; leave everything else untouched."""
    if not isinstance(val, str):
        return val
    s = val.strip()
    s_upper = s.upper().replace("_", "-")
    if "GIGA" in s_upper:
        return "GIGA-USD"
    return s

def normalize_prob_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Finds and standardizes column names for 'p_up' and 'p_down' probabilities.
    """
    rename_map = {}
    p_up_variations = {"p_up", "p-up", "pup", "prob_up", "p_up_prob", "puprob", "p up"}
    p_down_variations = {"p_down", "p-down", "pdown", "prob_down", "p_down_prob", "pdownprob", "p down"}

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

def calculate_pnl_and_metrics(trades_df):
    """
    UPDATED: Calculate P&L for PAXG transfer strategy
    Handles CRYPTO_BUY, CRYPTO_TO_PAXG_SELL, CRYPTO_TO_PAXG_BUY trade types
    """
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
        trade_type = row.get("trade_type", "SIMULATED")
        price = float(row["price"])
        qty = float(row["quantity"])
        
        if asset not in positions:
            positions[asset] = {"quantity": 0.0, "cost": 0.0}
            pnl_per_asset[asset] = 0.0
        
        cur_pnl = 0.0
        
        # Handle different trade types
        if action == "buy" or trade_type in ["CRYPTO_BUY", "CRYPTO_TO_PAXG_BUY"]:
            # Buy: Add to position
            positions[asset]["cost"] += qty * price
            positions[asset]["quantity"] += qty
            
        elif action == "sell" or trade_type in ["CRYPTO_TO_PAXG_SELL"]:
            # Sell: Realize P&L if we have position
            if positions[asset]["quantity"] > 0:
                avg_cost = positions[asset]["cost"] / max(positions[asset]["quantity"], 1e-12)
                trade_qty = min(qty, positions[asset]["quantity"])
                realized_pnl = (price - avg_cost) * trade_qty
                
                # Update asset P&L
                pnl_per_asset[asset] += realized_pnl
                total += realized_pnl
                cur_pnl = realized_pnl
                
                # Update trade statistics
                if realized_pnl > 0:
                    win += 1
                    gp += realized_pnl
                else:
                    loss += 1
                    gl += abs(realized_pnl)
                
                # Update position
                positions[asset]["cost"] -= avg_cost * trade_qty
                positions[asset]["quantity"] -= trade_qty
        
        # Record P&L for this row
        df.loc[i, "pnl"] = cur_pnl
        df.loc[i, "cumulative_pnl"] = total
        
        # Update drawdown tracking
        peak = max(peak, total)
        mdd = max(mdd, peak - total)

    closed_trades = win + loss
    stats = {
        "win_rate": (win / closed_trades * 100) if closed_trades else 0,
        "profit_factor": (gp / gl) if gl > 0 else float("inf"),
        "total_trades": closed_trades,
        "total_transfers": len(df[df.get("trade_type", "").str.contains("PAXG", na=False)]) // 2,  # Each transfer = 2 trades
        "avg_win": (gp / win) if win else 0,
        "avg_loss": (gl / loss) if loss else 0,
        "max_drawdown": mdd,
    }
    
    # Calculate per-asset cumulative P&L
    df["asset_cumulative_pnl"] = df.groupby("asset")["pnl"].cumsum()
    
    return pnl_per_asset, df, stats

def calculate_open_positions(trades_df, market_df):
    """
    UPDATED: Calculate open positions including PAXG holdings
    """
    if trades_df is None or trades_df.empty or market_df is None or market_df.empty: 
        return pd.DataFrame()
    
    positions = {}
    
    # Process all trades to build current positions
    for _, row in trades_df.sort_values("timestamp").iterrows():
        asset = row["asset"]
        action = row["action"]
        trade_type = row.get("trade_type", "SIMULATED")
        qty = float(row["quantity"])
        price = float(row["price"])
        
        if asset not in positions:
            positions[asset] = {"quantity": 0.0, "cost": 0.0}
        
        if action == "buy" or trade_type in ["CRYPTO_BUY", "CRYPTO_TO_PAXG_BUY"]:
            positions[asset]["cost"] += qty * price
            positions[asset]["quantity"] += qty
        elif action == "sell" or trade_type in ["CRYPTO_TO_PAXG_SELL"]:
            if positions[asset]["quantity"] > 0:
                avg_cost = positions[asset]["cost"] / max(positions[asset]["quantity"], 1e-12)
                sell_qty = min(qty, positions[asset]["quantity"])
                positions[asset]["cost"] -= avg_cost * sell_qty
                positions[asset]["quantity"] -= sell_qty
    
    # Build open positions dataframe
    open_positions = []
    for asset, data in positions.items():
        if data["quantity"] > 1e-9:  # Has meaningful position
            latest_market = market_df[market_df['asset'] == asset]
            if not latest_market.empty:
                latest_price = latest_market.loc[latest_market['timestamp'].idxmax()]['close']
                avg_entry = data["cost"] / data["quantity"]
                current_value = latest_price * data["quantity"]
                unrealized_pnl = current_value - data["cost"]
                
                # Determine position type
                position_type = "PAXG (Safe Haven)" if asset == "PAXG-USD" else "Crypto"
                
                open_positions.append({
                    "Asset": asset,
                    "Type": position_type,
                    "Quantity": data["quantity"],
                    "Avg. Entry Price": avg_entry,
                    "Current Price": latest_price,
                    "Current Value ($)": current_value,
                    "Unrealized P&L ($)": unrealized_pnl,
                })
    
    return pd.DataFrame(open_positions)

def get_trade_display_info(trade_type, action):
    """
    UPDATED: Get display information for different trade types
    """
    if trade_type == "CRYPTO_BUY":
        return "BUY", "green", "triangle-up"
    elif trade_type == "CRYPTO_TO_PAXG_SELL":
        return "→PAXG", "orange", "arrow-right"
    elif trade_type == "CRYPTO_TO_PAXG_BUY":
        return "PAXG", "gold", "circle"
    elif action == "buy":
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
    market = pd.read_csv(io.BytesIO(raw_market)) if raw_market else pd.DataFrame()

    if not trades.empty:
        trades = lower_strip_cols(trades)
        trades = trades.rename(columns={"product_id": "asset", "side": "action", "size": "quantity"})
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
        market = normalize_prob_columns(market)
        for col in ["open", "high", "low", "close"]:
            if col in market.columns: 
                market[col] = pd.to_numeric(market[col], errors="coerce")

    pnl_summary, trades_with_pnl, stats = calculate_pnl_and_metrics(trades) if not trades.empty else ({}, pd.DataFrame(), {})
    return trades_with_pnl, pnl_summary, stats, market

# =========================
# Main App
# =========================
st.markdown("## PAXG Transfer Strategy Dashboard")
st.caption("Risk-On/Risk-Off Strategy: Crypto ↔ PAXG (Gold) Transfers")
trades_df, pnl_summary, summary_stats, market_df = load_data(TRADES_LINK, MARKET_LINK)

with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>PAXG Strategy</h1>", unsafe_allow_html=True)
    
    # Date range display
    if not trades_df.empty and 'timestamp' in trades_df.columns:
        min_date, max_date = trades_df['timestamp'].min(), trades_df['timestamp'].max()
        if pd.notna(min_date) and pd.notna(max_date):
            st.markdown(f"<p style='text-align: center;'><strong>{min_date.strftime('%m/%d/%y')} - {max_date.strftime('%m/%d/%y')}</strong></p>", unsafe_allow_html=True)

    local_tz_obj = timezone(LOCAL_TZ)
    now_local = datetime.now(local_tz_obj)
    st.markdown(f"<p style='text-align: center; font-size: 0.9em; color: grey;'>Last updated: {now_local.strftime('%Y-%m-%d %H:%M')}</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Strategy Stats
    st.markdown("## 📊 Strategy Stats")
    if summary_stats:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Trades", f"{summary_stats.get('total_trades', 0):,}")
            st.metric("Win Rate", f"{summary_stats.get('win_rate', 0):.1f}%")
        with col2:
            st.metric("PAXG Transfers", f"{summary_stats.get('total_transfers', 0):,}")
            pf = summary_stats.get('profit_factor', 0)
            st.metric("Profit Factor", "∞" if np.isinf(pf) else f"{pf:.2f}")
    
    st.markdown("---")
    
    # Realized P&L
    st.markdown("## 💵 Realized P&L")
    if pnl_summary:
        total_pnl = sum(p for p in pnl_summary.values() if pd.notna(p))
        st.metric("Overall P&L", f"${total_pnl:,.2f}")
        st.markdown("**By Asset**")
        for asset, pnl in sorted(pnl_summary.items(), key=lambda kv: kv[1], reverse=True):
            color = "#10b981" if pnl >= 0 else "#ef4444"
            asset_display = f"🥇 {asset}" if asset == "PAXG-USD" else asset
            st.markdown(f"<div style='display:flex;justify-content:space-between'><span>{asset_display}</span><span style='color:{color};font-weight:600'>${pnl:,.2f}</span></div>", unsafe_allow_html=True)
    else: 
        st.info("No realized P&L yet.")
    
    st.markdown("---")
    
    # Current Positions
    st.markdown("## 📈 Current Holdings")
    if not trades_df.empty:
        last_trade_times = trades_df.groupby('asset')['timestamp'].max()
    else:
        last_trade_times = pd.Series()

    if not market_df.empty:
        open_positions_df = calculate_open_positions(trades_df, market_df)
        
        if not open_positions_df.empty:
            # Separate crypto and PAXG positions
            crypto_positions = open_positions_df[open_positions_df['Type'] == 'Crypto']
            paxg_positions = open_positions_df[open_positions_df['Type'] == 'PAXG (Safe Haven)']
            
            # Display PAXG positions first (safe haven)
            for _, pos in paxg_positions.iterrows():
                asset = pos['Asset']
                pnl = pos["Unrealized P&L ($)"]
                color = "gold"
                st.markdown(f'<p style="color:{color}; font-weight:bold; margin-bottom:0px;">🥇 {asset} (Safe Haven)</p>', unsafe_allow_html=True)
                st.caption(f"Value: ${pos['Current Value ($)']:.2f} | P&L: ${pnl:.2f}")
            
            # Display crypto positions
            for _, pos in crypto_positions.iterrows():
                asset = pos['Asset']
                pnl = pos["Unrealized P&L ($)"]
                color = "lightgreen" if pnl > 0 else "salmon"
                st.markdown(f'<p style="color:{color}; font-weight:bold; margin-bottom:0px;">{asset}</p>', unsafe_allow_html=True)
                st.caption(f"Qty: {pos['Quantity']:.4f} | Entry: ${pos['Avg. Entry Price']:.6f} | P&L: ${pnl:.2f}")
        
        # Show assets with no current positions
        if not market_df.empty:
            all_assets = sorted(list(set(market_df["asset"].dropna().unique()) | set(trades_df["asset"].dropna().unique() if not trades_df.empty else [])))
            held_assets = set(open_positions_df['Asset'].tolist() if not open_positions_df.empty else [])
            
            for asset in all_assets:
                if asset not in held_assets:
                    last_trade_time_str = ""
                    if asset in last_trade_times:
                        last_trade_time = last_trade_times[asset]
                        last_trade_time_str = f" ({last_trade_time.strftime('%m/%d')})"
                    
                    asset_display = f"🥇 {asset} (No Position){last_trade_time_str}" if asset == "PAXG-USD" else f"{asset} (No Position){last_trade_time_str}"
                    st.markdown(f'<p style="color:lightgray; margin-bottom:0px;">{asset_display}</p>', unsafe_allow_html=True)
    
    st.markdown("---")

    # Chart Controls
    st.markdown("## ⚙️ Chart Controls")
    selected_asset = None
    if not market_df.empty:
        assets = sorted(market_df["asset"].dropna().unique())
        default_index = assets.index(DEFAULT_ASSET) if DEFAULT_ASSET in assets else 0
        selected_asset = st.sidebar.selectbox("Select Asset", assets, index=default_index)
        range_choice = st.sidebar.selectbox("Select Date Range", ["30 days", "7 days", "1 day", "All"], index=0)

# Main content tabs
tab1, tab2, tab3 = st.tabs(["📈 Price & Transfers", "💰 P&L Analysis", "🔄 Transfer History"])

with tab1:
    if selected_asset and not market_df.empty:
        df = market_df[market_df["asset"] == selected_asset].sort_values("timestamp")
        if not df.empty:
            end_date = df["timestamp"].max()
            if range_choice == "1 day": start_date = end_date - timedelta(days=1)
            elif range_choice == "7 days": start_date = end_date - timedelta(days=7)
            elif range_choice == "30 days": start_date = end_date - timedelta(days=30)
            else: start_date = df["timestamp"].min()

            vis = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)].copy()
            if not vis.empty:
                rescale = (vis["high"].max() < 0.01)
                ylabel = "Price (µUSD)" if rescale else "Price (USD)"
                if rescale:
                    for col in ["open", "high", "low", "close"]: vis[col] *= 1e6
                
                def fmt(v, d=6): return f"{float(v):.{d}f}" if pd.notna(v) else "—"
                
                # Enhanced hover text with strategy context
                hovertext = []
                for _, r in vis.iterrows():
                    p_up, p_down = r.get('p_up'), r.get('p_down')
                    strategy_signal = "📈 Risk-On (Crypto)" if (pd.notna(p_up) and pd.notna(p_down) and p_up > p_down) else "🥇 Risk-Off (PAXG)" if (pd.notna(p_up) and pd.notna(p_down) and p_down > p_up) else "—"
                    
                    hover = f"📅 {r['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}<br>O: {fmt(r['open'])}<br>H: {fmt(r['high'])}<br>L: {fmt(r['low'])}<br>C: {fmt(r['close'])}<br><b>P-Up: {fmt(p_up, 4)}</b><br><b>P-Down: {fmt(p_down, 4)}</b><br><b>{strategy_signal}</b>"
                    hovertext.append(hover)
                
                fig = go.Figure(data=go.Candlestick(
                    x=vis["timestamp"], 
                    open=vis["open"], 
                    high=vis["high"], 
                    low=vis["low"], 
                    close=vis["close"], 
                    name=selected_asset, 
                    text=hovertext, 
                    hoverinfo="text"
                ))

                # Add trade markers with enhanced styling
                if not trades_df.empty:
                    asset_trades = trades_df[
                        (trades_df["asset"] == selected_asset) & 
                        (trades_df["timestamp"] >= start_date) & 
                        (trades_df["timestamp"] <= end_date)
                    ].copy()
                    
                    if not asset_trades.empty:
                        if rescale: 
                            asset_trades = asset_trades.copy()  # Avoid warning
                            asset_trades.loc[:, "price"] *= 1e6
                        
                        # Group by trade type
                        for trade_type in asset_trades.get("trade_type", ["SIMULATED"]).unique():
                            if pd.isna(trade_type):
                                continue
                            
                            type_trades = asset_trades[asset_trades.get("trade_type", "SIMULATED") == trade_type]
                            if not type_trades.empty:
                                display_name, color, symbol = get_trade_display_info(trade_type, type_trades.iloc[0]["action"])
                                
                                fig.add_trace(go.Scatter(
                                    x=type_trades["timestamp"], 
                                    y=type_trades["price"], 
                                    mode="markers", 
                                    name=display_name, 
                                    marker=dict(symbol=symbol, size=10, color=color),
                                    hovertemplate=f"<b>{display_name}</b><br>Price: %{{y}}<br>Time: %{{x}}<extra></extra>"
                                ))

                title_suffix = "🥇 (Gold-Backed Safe Haven)" if selected_asset == "PAXG-USD" else ""
                fig.update_layout(
                    template="plotly_white", 
                    xaxis_rangeslider_visible=False, 
                    hovermode="x unified", 
                    yaxis_title=ylabel, 
                    title=f"{selected_asset} {title_suffix} — Price & Transfer Activity", 
                    legend=dict(orientation="h", y=1.03, x=0.5, xanchor="center"), 
                    height=600, 
                    margin=dict(l=40, r=20, t=60, b=40)
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Market data not loaded or no asset selected.")

with tab2:
    if not trades_df.empty:
        st.markdown("### Strategy Performance Analysis")
        
        if summary_stats and summary_stats.get('total_trades', 0) > 0:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Win Rate", f"{summary_stats['win_rate']:.2f}%")
            pf = summary_stats['profit_factor']
            m2.metric("Profit Factor", "∞" if np.isinf(pf) else f"{pf:.2f}")
            m3.metric("Closed Trades", f"{summary_stats['total_trades']:,}")
            m4.metric("PAXG Transfers", f"{summary_stats['total_transfers']:,}")
        
        show_all_assets = st.checkbox("Show Portfolio P&L (all assets combined)", value=True)
        assets_for_plot = trades_df if show_all_assets else trades_df[trades_df['asset'] == (selected_asset or DEFAULT_ASSET)]
        pnl_col = "cumulative_pnl" if show_all_assets else "asset_cumulative_pnl"
        title = "Total Portfolio P&L (Risk-On/Risk-Off Strategy)" if show_all_assets else f"P&L for {selected_asset or DEFAULT_ASSET}"
        
        if not assets_for_plot.empty and assets_for_plot[pnl_col].sum() != 0:
            fig_pnl = go.Figure()
            
            # Main P&L line
            fig_pnl.add_trace(go.Scatter(
                x=assets_for_plot["timestamp"], 
                y=assets_for_plot[pnl_col], 
                mode="lines", 
                name="Cumulative P&L",
                line=dict(color="blue", width=2)
            ))
            
            # Add zero line
            fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Break Even")
            
            fig_pnl.update_layout(
                title=title, 
                template="plotly_white", 
                yaxis_title="P&L (USD)",
                xaxis_title="Date",
                hovermode="x unified"
            )
            st.plotly_chart(fig_pnl, use_container_width=True)
        else:
            st.warning("No P&L data to display for the selected scope.")
    else:
        st.warning("No trade data loaded.")

with tab3:
    if not trades_df.empty:
        st.markdown("### Transfer Activity Log")
        
        # Filter for transfer-related trades
        transfer_trades = trades_df[trades_df.get("trade_type", "").str.contains("PAXG", na=False)]
        
        if not transfer_trades.empty:
            # Display recent transfers
            recent_transfers = transfer_trades.tail(20).copy()
            recent_transfers = recent_transfers.sort_values("timestamp", ascending=False)
            
            # Format for display
            display_df = recent_transfers[["timestamp", "asset", "trade_type", "quantity", "price", "usd_value"]].copy()
            display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
            display_df.columns = ["Time", "Asset", "Transfer Type", "Quantity", "Price", "USD Value"]
            
            st.dataframe(display_df, use_container_width=True)
            
            # Transfer summary
            st.markdown("### Transfer Summary")
            crypto_to_paxg = len(transfer_trades[transfer_trades["trade_type"] == "CRYPTO_TO_PAXG_SELL"])
            paxg_buys = len(transfer_trades[transfer_trades["trade_type"] == "CRYPTO_TO_PAXG_BUY"])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Crypto Sales", crypto_to_paxg)
            with col2:
                st.metric("PAXG Purchases", paxg_buys)
            with col3:
                total_paxg_value = transfer_trades[transfer_trades["trade_type"] == "CRYPTO_TO_PAXG_BUY"]["usd_value"].sum()
                st.metric("Total PAXG Value", f"${total_paxg_value:,.2f}")
                
        else:
            st.info("No PAXG transfers recorded yet. Strategy will show transfers when ML signals trigger risk-off moves.")
    else:
        st.warning("No trade data loaded.")
