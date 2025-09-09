import io
import re
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta

# =========================
# App Configuration
# =========================
st.set_page_config(page_title="Trading Analytics", layout="wide")
LOCAL_TZ = "America/Los_Angeles"
DEFAULT_ASSET = "GIGA-USD"

# =========================
# HARDCODED FILE LINKS
# =========================
TRADES_LINK = "https://drive.google.com/file/d/1zSdFcG4Xlh_iSa180V6LRSeEucAokXYk/view?usp=sharing"
MARKET_LINK = "https://drive.google.com/file/d/1tvY7CheH_p5f3uaE7VPUYS78hDHNmX_C/view?usp=sharing"

# =========================
# Helper Functions (Data Processing)
# =========================
def lower_strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.lower().strip() for c in out.columns]
    return out

def to_local_naive(ts):
    s = pd.to_datetime(ts, errors="coerce")
    try:
        if s.dt.tz is not None:
            s = s.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    except Exception:
        try:
            s = s.dt.tz_localize("UTC").dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
        except Exception:
            s = s.dt.tz_localize(None)
    return s

def unify_symbol(val: str) -> str:
    if not isinstance(val, str): return val
    s = val.strip()
    s_upper = s.upper().replace("_", "-")
    if "GIGA" in s_upper: return "GIGA-USD"
    return s

def normalize_prob_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for c in list(df.columns):
        cl = c.lower()
        if cl in {"p_up", "p-up", "pup", "prob_up", "p_up_prob", "puprob"}: rename_map[c] = "p_up"
        if cl in {"p_down", "p-down", "pdown", "prob_down", "p_down_prob", "pdownprob"}: rename_map[c] = "p_down"
    if rename_map: df = df.rename(columns=rename_map)
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
            avg = positions[asset]["cost"] / positions[asset]["quantity"]
            trade_qty = min(qty, positions[asset]["quantity"])
            realized = (price - avg) * trade_qty
            pnl_per_asset[asset] += realized
            total += realized
            cur = realized
            if realized > 0: win += 1; gp += realized
            else: loss += 1; gl += abs(realized)
            positions[asset]["cost"] -= avg * trade_qty
            positions[asset]["quantity"] -= trade_qty
        df.loc[i, "pnl"], df.loc[i, "cumulative_pnl"] = cur, total
        peak, mdd = max(peak, total), max(mdd, peak - total)

    closed = win + loss
    stats = {
        "win_rate": (win / closed * 100) if closed else 0,
        "profit_factor": (gp / gl) if gl > 0 else float("inf"),
        "total_trades": closed,
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
        asset, action, qty, price = row["asset"], row["action"], float(row["quantity"]), float(row["price"])
        if asset not in positions:
            positions[asset] = {"quantity": 0.0, "cost": 0.0}
        if action == "buy":
            positions[asset]["cost"] += qty * price
            positions[asset]["quantity"] += qty
        elif action == "sell" and positions[asset]["quantity"] > 0:
            avg_cost_per_unit = positions[asset]["cost"] / positions[asset]["quantity"]
            positions[asset]["cost"] -= qty * avg_cost_per_unit
            positions[asset]["quantity"] -= qty
    
    open_positions = []
    for asset, data in positions.items():
        if data["quantity"] > 1e-9:
            latest_market_data = market_df[market_df['asset'] == asset]
            if not latest_market_data.empty:
                latest_price = latest_market_data.loc[latest_market_data['timestamp'].idxmax()]['close']
                avg_entry_price = data["cost"] / data["quantity"] if data["quantity"] > 0 else 0
                current_value = data["quantity"] * latest_price
                unrealized_pnl = current_value - data["cost"]
                open_positions.append({
                    "Asset": asset, "Quantity": data["quantity"], "Avg. Entry Price": avg_entry_price,
                    "Current Price": latest_price, "Unrealized P&L ($)": unrealized_pnl,
                })
    return pd.DataFrame(open_positions)


# =========================
# Helper Functions (Data Loading)
# =========================
def extract_drive_id(url_or_id: str) -> str:
    if not url_or_id: return ""
    s = url_or_id.strip(); m = re.search(r"/d/([A-Za-z0-9_-]{20,})", s)
    if m: return m.group(1); m = re.search(r"[?&]id=([A-Za-z0-9_-]{20,})", s)
    if m: return m.group(1); return s

def download_drive_csv_bytes(url_or_id: str) -> bytes | None:
    fid = extract_drive_id(url_or_id)
    if not fid: return None
    try:
        base = "https://drive.google.com/uc?export=download"
        with requests.Session() as s:
            r1 = s.get(base, params={"id": fid}, stream=True)
            if "text/html" in (r1.headers.get("Content-Type") or ""):
                m = re.search(r"confirm=([0-9A-Za-z_-]+)", r1.text)
                if m:
                    params = {"id": fid, "confirm": m.group(1)}
                    r2 = s.get(base, params=params, stream=True)
                    return r2.content
            return r1.content
    except Exception as e:
        st.error(f"Network error downloading file ID {fid}: {e}")
        return None

def read_csv_best_effort(raw: bytes, label: str) -> pd.DataFrame | None:
    if not raw: st.warning(f"No data bytes received for {label} to parse."); return None
    if b"<html" in (raw[:512] or b"").lower():
        st.error(f"Google Drive returned an HTML page for {label}, not a file. Check sharing permissions.")
        return None
    try: return pd.read_csv(io.BytesIO(raw))
    except Exception: st.error(f"Failed to parse {label} as CSV after download."); return None


# =========================
# Main Data Loading Function (Cached)
# =========================
@st.cache_data(ttl=600)
def load_data(trades_link, market_link):
    raw_trades = download_drive_csv_bytes(trades_link)
    if raw_trades is None: st.error("Failed to download bytes for Trades CSV.")
    trades = read_csv_best_effort(raw_trades, "Trades CSV")
    
    raw_market = download_drive_csv_bytes(market_link)
    if raw_market is None: st.error("Failed to download bytes for Market CSV.")
    market = read_csv_best_effort(raw_market, "Market CSV")

    if trades is not None and not trades.empty:
        trades = lower_strip_cols(trades)
        if "product_id" in trades.columns: trades = trades.rename(columns={"product_id": "asset"})
        trades = trades.rename(columns={"side": "action", "size": "quantity"})
        trades["asset"] = trades["asset"].apply(unify_symbol)
        trades["timestamp"] = to_local_naive(trades["timestamp"])
        if "action" in trades.columns: trades["action"] = trades["action"].str.lower()
        for col in ["quantity", "price", "usd_value"]:
            if col in trades.columns: trades[col] = pd.to_numeric(trades[col], errors="coerce")

    if market is not None and not market.empty:
        market = lower_strip_cols(market)
        if "product_id" in market.columns: market = market.rename(columns={"product_id": "asset"})
        market["asset"] = market["asset"].apply(unify_symbol)
        market["timestamp"] = to_local_naive(market["timestamp"])
        market = normalize_prob_columns(market)
        for col in ["open", "high", "low", "close"]: market[col] = pd.to_numeric(market[col], errors="coerce")

    pnl_summary, trades_with_pnl, stats = calculate_pnl_and_metrics(trades) if trades is not None else ({}, pd.DataFrame(), {})
    return trades_with_pnl, pnl_summary, stats, market


# =========================
# Streamlit App UI
# =========================
st.markdown("## Trading Analytics Dashboard")
st.caption("View position status and chart controls in the sidebar.")

# --- Load data using the hardcoded links ---
trades_df, pnl_summary, summary_stats, market_df = load_data(TRADES_LINK, MARKET_LINK)

# --- Sidebar ---
st.sidebar.markdown("## 📊 Positions Status")
if market_df is not None and not market_df.empty:
    open_positions_df = calculate_open_positions(trades_df, market_df)
    
    all_assets = sorted(market_df["asset"].unique())
    open_positions_lookup = open_positions_df.set_index("Asset") if not open_positions_df.empty else pd.DataFrame()

    for asset in all_assets:
        if asset in open_positions_lookup.index:
            pos = open_positions_lookup.loc[asset]
            pnl = pos["Unrealized P&L ($)"]
            color = "lightgreen" if pnl > 0 else "salmon"
            
            st.sidebar.markdown(f'<p style="color:{color}; font-weight:bold; margin-bottom:0px;">{asset}</p>', unsafe_allow_html=True)
            st.sidebar.caption(
                f"Qty: {pos['Quantity']:.4f} | "
                f"Avg Entry: ${pos['Avg. Entry Price']:.6f} | "
                f"Current: ${pos['Current Price']:.6f}"
            )
        else:
            st.sidebar.markdown(f'<p style="color:royalblue; margin-bottom:0px;">{asset}</p>', unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("## ⚙️ Chart Controls")
selected_asset = None
if market_df is not None and not market_df.empty:
    assets = sorted(market_df["asset"].unique())
    default_index = assets.index(DEFAULT_ASSET) if DEFAULT_ASSET in assets else 0
    selected_asset = st.sidebar.selectbox("Select Asset", assets, index=default_index)
    range_choice = st.sidebar.selectbox("Select Date Range", ["30 days", "7 days", "1 day", "All"], index=0)

# --- Main Page Tabs ---
tab1, tab2 = st.tabs(["📈 Candlestick Analysis", "💰 P&L Analysis"])

with tab1:
    if selected_asset is None:
        st.warning("Market data not loaded. Cannot display chart.")
    else:
        df = market_df[market_df["asset"] == selected_asset].copy().sort_values("timestamp")
        end_date = df["timestamp"].max()
        if range_choice == "1 day": start_date = end_date - timedelta(days=1)
        elif range_choice == "7 days": start_date = end_date - timedelta(days=7)
        elif range_choice == "30 days": start_date = end_date - timedelta(days=30)
        else: start_date = df["timestamp"].min()
        vis = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]

        if vis.empty:
            st.warning("No market data in the selected date range for this asset.")
        else:
            def fmt(v, decimals=4):
                try: return f"{float(v):.{decimals}f}"
                except (ValueError, TypeError): return "—"

            hovertext = [
                f"📅 {row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
                + f"<br>Open: ${fmt(row['open'])}"
                + f"<br>High: ${fmt(row['high'])}"
                + f"<br>Low: ${fmt(row['low'])}"
                + f"<br>Close: ${fmt(row['close'])}"
                + f"<br><b>P-Up: {fmt(row.get('p_up'))}</b>"
                + f"<br><b>P-Down: {fmt(row.get('p_down'))}</b>"
                for _, row in vis.iterrows()
            ]
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=vis["timestamp"], open=vis["open"], high=vis["high"], low=vis["low"], close=vis["close"],
                name=selected_asset,
                text=hovertext,
                hoverinfo="text"
            ))
            
            if trades_df is not None:
                asset_trades = trades_df[(trades_df["asset"] == selected_asset) & (trades_df["timestamp"] >= start_date) & (trades_df["timestamp"] <= end_date)]
                if not asset_trades.empty:
                    buys = asset_trades[asset_trades["action"] == "buy"]
                    sells = asset_trades[asset_trades["action"] == "sell"]
                    if not buys.empty: fig.add_trace(go.Scatter(x=buys["timestamp"], y=buys["price"], mode="markers", name="BUY", marker=dict(symbol="triangle-up", size=10, color="green")))
                    if not sells.empty: fig.add_trace(go.Scatter(x=sells["timestamp"], y=sells["price"], mode="markers", name="SELL", marker=dict(symbol="triangle-down", size=10, color="red")))
            
            fig.update_layout(
                template="plotly_white", xaxis_rangeslider_visible=False, hovermode="x unified",
                yaxis_title="Price (USD)", yaxis=dict(tickformat='.10f'),
                title=f"{selected_asset} — Price & Trades",
                legend=dict(orientation="h", y=1.03, x=0.5, xanchor="center"), height=600,
            )
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    if trades_df is None or trades_df.empty:
        st.warning("No trade data loaded to perform P&L analysis.")
    else:
        st.markdown("### Realized P&L (from closed trades)")
        if summary_stats and summary_stats['total_trades'] > 0:
            m1, m2, m3 = st.columns(3)
            m1.metric("Win Rate", f"{summary_stats['win_rate']:.2f}%")
            m2.metric("Profit Factor", f"{summary_stats['profit_factor']:.2f}")
            m3.metric("Total Closed Trades", f"{summary_stats['total_trades']:,}")
        
        show_all_assets = st.checkbox("Show P&L for all assets (Portfolio)", value=True)
        if selected_asset:
            pnl_data = trades_df if show_all_assets else trades_df[trades_df['asset'] == selected_asset]
            pnl_col = "cumulative_pnl" if show_all_assets else "asset_cumulative_pnl"
            title = "Total Portfolio Cumulative P&L" if show_all_assets else f"Cumulative P&L for {selected_asset}"
            
            if pnl_data.empty or pnl_data[pnl_col].sum() == 0:
                st.warning(f"No P&L data to display.")
            else:
                fig_pnl = go.Figure()
                fig_pnl.add_trace(go.Scatter(x=pnl_data["timestamp"], y=pnl_data[pnl_col], mode="lines", name="Cumulative P&L"))
                fig_pnl.update_layout(title=title, template="plotly_white", yaxis_title="P&L (USD)")
                st.plotly_chart(fig_pnl, use_container_width=True)
