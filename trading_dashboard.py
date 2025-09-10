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
st.set_page_config(page_title="Trading Analytics", layout="wide")
LOCAL_TZ = "America/Los_Angeles"
DEFAULT_ASSET = "GIGA-USD"

# =========================
# Google Drive CSV Links
# =========================
TRADES_LINK = "https://drive.google.com/file/d/1En36aZ-mYP1qmmFR5LZwYxJHmygikRhb/view?usp=sharing"
MARKET_LINK = "https://drive.google.com/file/d/18SSSVO4U0jhCVL_SiZjQgd50Ei2dJVKK/view?usp=sharing"  # OHLCV + features

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
    if trades_df is None or trades_df.empty: return {}, pd.DataFrame(), {}
    pnl_per_asset, positions = {}, {}
    df = trades_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["pnl"], df["cumulative_pnl"] = 0.0, 0.0
    total, win, loss, gp, gl, peak, mdd = 0.0, 0, 0, 0.0, 0.0, 0.0, 0.0

    for i, row in df.iterrows():
        asset, action, price, qty = row["asset"], row["action"], float(row["price"]), float(row["quantity"])
        if asset not in positions:
            positions[asset] = {"quantity": 0.0, "cost": 0.0}
            pnl_per_asset[asset] = 0.0
        
        cur = 0.0
        if action == "buy":
            positions[asset]["cost"] += qty * price
            positions[asset]["quantity"] += qty
        elif action == "sell" and positions[asset]["quantity"] > 0:
            avg_cost = positions[asset]["cost"] / max(positions[asset]["quantity"], 1e-12)
            trade_qty = min(qty, positions[asset]["quantity"])
            realized = (price - avg_cost) * trade_qty
            pnl_per_asset[asset] += realized
            total += realized
            cur = realized
            if realized > 0: win += 1; gp += realized
            else: loss += 1; gl += abs(realized)
            positions[asset]["cost"] -= avg_cost * trade_qty
            positions[asset]["quantity"] -= trade_qty
        
        df.loc[i, "pnl"], df.loc[i, "cumulative_pnl"] = cur, total
        peak, mdd = max(peak, total), max(mdd, peak - total)

    closed = win + loss
    stats = {
        "win_rate": (win / closed * 100) if closed else 0,
        "profit_factor": (gp / gl) if gl > 0 else float("inf"),
        "total_trades": closed, "avg_win": (gp / win) if win else 0,
        "avg_loss": (gl / loss) if loss else 0, "max_drawdown": mdd,
    }
    df["asset_cumulative_pnl"] = df.groupby("asset")["pnl"].cumsum()
    return pnl_per_asset, df, stats

def calculate_open_positions(trades_df, market_df):
    if trades_df is None or trades_df.empty or market_df is None or market_df.empty: return pd.DataFrame()
    positions = {}
    for _, row in trades_df.sort_values("timestamp").iterrows():
        asset, action, qty, price = row["asset"], row["action"], float(row["quantity"]), float(row["price"])
        if asset not in positions: positions[asset] = {"quantity": 0.0, "cost": 0.0}
        if action == "buy":
            positions[asset]["cost"] += qty * price
            positions[asset]["quantity"] += qty
        elif action == "sell" and positions[asset]["quantity"] > 0:
            avg_cost = positions[asset]["cost"] / max(positions[asset]["quantity"], 1e-12)
            positions[asset]["cost"] -= qty * avg_cost
            positions[asset]["quantity"] -= qty
    
    open_positions = []
    for asset, data in positions.items():
        if data["quantity"] > 1e-9:
            latest_market = market_df[market_df['asset'] == asset]
            if not latest_market.empty:
                latest_price = latest_market.loc[latest_market['timestamp'].idxmax()]['close']
                avg_entry = data["cost"] / data["quantity"]
                unrealized_pnl = (latest_price * data["quantity"]) - data["cost"]
                open_positions.append({
                    "Asset": asset, "Quantity": data["quantity"], "Avg. Entry Price": avg_entry,
                    "Current Price": latest_price, "Unrealized P&L ($)": unrealized_pnl,
                })
    return pd.DataFrame(open_positions)

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
            if col in trades.columns: trades[col] = pd.to_numeric(trades[col], errors="coerce")
    
    if not market.empty:
        market = lower_strip_cols(market)
        market = market.rename(columns={"product_id": "asset"})
        if 'timestamp' in market.columns:
            market['timestamp'] = to_local_naive(market['timestamp'])
        market["asset"] = market["asset"].apply(unify_symbol)
        market = normalize_prob_columns(market)
        for col in ["open", "high", "low", "close"]:
            if col in market.columns: market[col] = pd.to_numeric(market[col], errors="coerce")

    pnl_summary, trades_with_pnl, stats = calculate_pnl_and_metrics(trades) if not trades.empty else ({}, pd.DataFrame(), {})
    return trades_with_pnl, pnl_summary, stats, market

st.markdown("## Trading Analytics Dashboard")
st.caption("View position status, P&L and chart controls in the sidebar.")
trades_df, pnl_summary, summary_stats, market_df = load_data(TRADES_LINK, MARKET_LINK)

with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>Trade Analytics</h1>", unsafe_allow_html=True)
    if not trades_df.empty and 'timestamp' in trades_df.columns:
        min_date, max_date = trades_df['timestamp'].min(), trades_df['timestamp'].max()
        if pd.notna(min_date) and pd.notna(max_date):
            st.markdown(f"<p style='text-align: center;'><strong>{min_date.strftime('%m/%d/%y')} - {max_date.strftime('%m/%d/%y')}</strong></p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='text-align: center; color: red;'><strong>Invalid Date Range</strong></p>", unsafe_allow_html=True)

    local_tz_obj = timezone(LOCAL_TZ)
    now_local = datetime.now(local_tz_obj)
    st.markdown(f"<p style='text-align: center; font-size: 0.9em; color: grey;'>Last updated: {now_local.strftime('%Y-%m-%d %H:%M')}</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("## 💵 Realized P&L")
    if pnl_summary:
        total_pnl = sum(p for p in pnl_summary.values() if pd.notna(p))
        st.metric("Overall P&L", f"${total_pnl:,.2f}")
        st.markdown("**By Asset**")
        for asset, pnl in sorted(pnl_summary.items(), key=lambda kv: kv[1], reverse=True):
            color = "#10b981" if pnl >= 0 else "#ef4444"
            st.markdown(f"<div style='display:flex;justify-content:space-between'><span>{asset}</span><span style='color:{color};font-weight:600'>${pnl:,.2f}</span></div>", unsafe_allow_html=True)
    else: st.info("No realized P&L yet.")
    st.markdown("---")
    
    # --- MODIFIED: Positions Status with Last Trade Time ---
    st.markdown("## 📊 Positions Status")
    if not trades_df.empty:
        last_trade_times = trades_df.groupby('asset')['timestamp'].max()
    else:
        last_trade_times = pd.Series()

    if not market_df.empty:
        open_positions_df = calculate_open_positions(trades_df, market_df)
        all_assets = sorted(list(set(market_df["asset"].dropna().unique()) | set(trades_df["asset"].dropna().unique())))
        lookup = open_positions_df.set_index("Asset") if not open_positions_df.empty else pd.DataFrame()
        
        for asset in all_assets:
            last_trade_time_str = ""
            if asset in last_trade_times:
                last_trade_time = last_trade_times[asset]
                last_trade_time_str = f" ({last_trade_time.strftime('%m/%d/%y %H:%M')})"
            
            asset_display_name = f"{asset}{last_trade_time_str}"
            
            if not lookup.empty and asset in lookup.index:
                pos = lookup.loc[asset]
                pnl, color = pos["Unrealized P&L ($)"], "lightgreen" if pos["Unrealized P&L ($)"] > 0 else "salmon"
                st.markdown(f'<p style="color:{color}; font-weight:bold; margin-bottom:0px;">{asset_display_name}</p>', unsafe_allow_html=True)
                st.caption(f"Qty: {pos['Quantity']:.4f} | Avg Entry: ${pos['Avg. Entry Price']:.6f} | Current: ${pos['Current Price']:.6f}")
            else:
                st.markdown(f'<p style="color:royalblue; margin-bottom:0px;">{asset_display_name}</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("## ⚙️ Chart Controls")
    selected_asset = None
    if not market_df.empty:
        assets = sorted(market_df["asset"].dropna().unique())
        default_index = assets.index(DEFAULT_ASSET) if DEFAULT_ASSET in assets else 0
        selected_asset = st.sidebar.selectbox("Select Asset", assets, index=default_index)
        range_choice = st.sidebar.selectbox("Select Date Range", ["30 days", "7 days", "1 day", "All"], index=0)

tab1, tab2 = st.tabs(["📈 Candlestick Analysis", "💰 P&L Analysis"])
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
                hovertext = [f"📅 {r['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}<br>O: {fmt(r['open'])}<br>H: {fmt(r['high'])}<br>L: {fmt(r['low'])}<br>C: {fmt(r['close'])}<br><b>P-Up: {fmt(r.get('p_up'), 4)}</b><br><b>P-Down: {fmt(r.get('p_down'), 4)}</b>" for _, r in vis.iterrows()]
                fig = go.Figure(data=go.Candlestick(x=vis["timestamp"], open=vis["open"], high=vis["high"], low=vis["low"], close=vis["close"], name=selected_asset, text=hovertext, hoverinfo="text"))

                if not trades_df.empty:
                    asset_trades = trades_df[(trades_df["asset"] == selected_asset) & (trades_df["timestamp"] >= start_date) & (trades_df["timestamp"] <= end_date)].copy()
                    if not asset_trades.empty:
                        if rescale: asset_trades.loc[:, "price"] *= 1e6
                        buys = asset_trades[asset_trades["action"] == "buy"]
                        sells = asset_trades[asset_trades["action"] == "sell"]
                        if not buys.empty: fig.add_trace(go.Scatter(x=buys["timestamp"], y=buys["price"], mode="markers", name="BUY", marker=dict(symbol="triangle-up", size=10, color="green")))
                        if not sells.empty: fig.add_trace(go.Scatter(x=sells["timestamp"], y=sells["price"], mode="markers", name="SELL", marker=dict(symbol="triangle-down", size=10, color="red")))

                fig.update_layout(template="plotly_white", xaxis_rangeslider_visible=False, hovermode="x unified", yaxis_title=ylabel, title=f"{selected_asset} — Price & Trades", legend=dict(orientation="h", y=1.03, x=0.5, xanchor="center"), height=600, margin=dict(l=40, r=20, t=60, b=40))
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Market data not loaded or no asset selected.")

with tab2:
    if not trades_df.empty:
        st.markdown("### Realized P&L (from closed trades)")
        if summary_stats and summary_stats.get('total_trades', 0) > 0:
            m1, m2, m3 = st.columns(3)
            m1.metric("Win Rate", f"{summary_stats['win_rate']:.2f}%")
            pf = summary_stats['profit_factor']
            m2.metric("Profit Factor", "∞" if np.isinf(pf) else f"{pf:.2f}")
            m3.metric("Total Closed Trades", f"{summary_stats['total_trades']:,}")
        
        show_all_assets = st.checkbox("Show P&L for all assets (Portfolio)", value=True)
        assets_for_plot = trades_df if show_all_assets else trades_df[trades_df['asset'] == (selected_asset or DEFAULT_ASSET)]
        pnl_col = "cumulative_pnl" if show_all_assets else "asset_cumulative_pnl"
        title = "Total Portfolio Cumulative P&L" if show_all_assets else f"Cumulative P&L for {selected_asset or DEFAULT_ASSET}"
        
        if not assets_for_plot.empty and assets_for_plot[pnl_col].sum() != 0:
            fig_pnl = go.Figure(data=go.Scatter(x=assets_for_plot["timestamp"], y=assets_for_plot[pnl_col], mode="lines", name="Cumulative P&L"))
            fig_pnl.update_layout(title=title, template="plotly_white", yaxis_title="P&L (USD)")
            st.plotly_chart(fig_pnl, use_container_width=True)
        else:
            st.warning("No P&L data to display for the selected scope.")
    else:
        st.warning("No trade data loaded.")
