import io
import re
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
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
MARKET_LINK = "https://drive.google.com/file/d/18SSSVO4U0jhCVL_SiZjQgd50Ei2dJVKK/view?usp=sharing"  # OHLCV + probs

# =========================
# Helpers — data processing
# =========================
def lower_strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.lower().strip() for c in out.columns]
    return out

def to_local_naive(ts):
    # Try converting assuming it's already a datetime object
    s = pd.to_datetime(ts, errors="coerce")
    try:
        if s.dt.tz is not None:
            s = s.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    except Exception:
        # If that fails, it might be a Unix timestamp (numeric)
        try:
             s = pd.to_datetime(ts, unit='s', errors='coerce').dt.tz_localize('UTC').dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
        except Exception:
            try:
                s = s.dt.tz_localize("UTC").dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
            except Exception:
                s = s.dt.tz_localize(None)
    return s


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
    rename_map = {}
    for c in list(df.columns):
        cl = c.lower()
        if cl in {"p_up", "p-up", "pup", "prob_up", "p_up_prob", "puprob"}:
            rename_map[c] = "p_up"
        if cl in {"p_down", "p-down", "pdown", "prob_down", "p_down_prob", "pdownprob"}:
            rename_map[c] = "p_down"
    if rename_map:
        df = df.rename(columns=rename_map)
    if "p_up" not in df.columns:
        df["p_up"] = np.nan
    if "p_down" not in df.columns:
        df["p_down"] = np.nan
    return df

def calculate_pnl_and_metrics(trades_df):
    """Realized P&L on sells vs avg cost; returns (per_asset dict, enriched df, stats)."""
    if trades_df is None or trades_df.empty:
        return {}, pd.DataFrame(), {}
    pnl_per_asset, positions = {}, {}
    df = trades_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["pnl"], df["cumulative_pnl"] = 0.0, 0.0
    total = 0.0
    win = loss = 0
    gp = gl = 0.0
    peak = mdd = 0.0

    for i, row in df.iterrows():
        asset, action = row["asset"], row["action"]
        price, qty = float(row["price"]), float(row["quantity"])
        if asset not in positions:
            positions[asset] = {"quantity": 0.0, "cost": 0.0}
            pnl_per_asset[asset] = 0.0
        cur = 0.0
        if action == "buy":
            positions[asset]["cost"] += qty * price
            positions[asset]["quantity"] += qty
        elif action == "sell" and positions[asset]["quantity"] > 0:
            avg = positions[asset]["cost"] / max(positions[asset]["quantity"], 1e-12)
            trade_qty = min(qty, positions[asset]["quantity"])
            realized = (price - avg) * trade_qty
            pnl_per_asset[asset] += realized
            total += realized
            cur = realized
            if realized > 0:
                win += 1; gp += realized
            else:
                loss += 1; gl += abs(realized)
            positions[asset]["cost"] -= avg * trade_qty
            positions[asset]["quantity"] -= trade_qty
        df.loc[i, "pnl"] = cur
        df.loc[i, "cumulative_pnl"] = total
        peak = max(peak, total)
        mdd = max(mdd, peak - total)

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
    """Naive open-position tracker and unrealized P&L snapshot from latest market close."""
    if trades_df is None or trades_df.empty or market_df is None or market_df.empty:
        return pd.DataFrame()

    positions = {}
    for _, row in trades_df.sort_values("timestamp").iterrows():
        asset, action = row["asset"], row["action"]
        qty, price = float(row["quantity"]), float(row["price"])
        if asset not in positions:
            positions[asset] = {"quantity": 0.0, "cost": 0.0}
        if action == "buy":
            positions[asset]["cost"] += qty * price
            positions[asset]["quantity"] += qty
        elif action == "sell" and positions[asset]["quantity"] > 0:
            avg_cost_per_unit = positions[asset]["cost"] / max(positions[asset]["quantity"], 1e-12)
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
# Helpers — Drive download
# =========================
def extract_drive_id(url_or_id: str) -> str:
    """Pull Google Drive file id from full URL or return the id if it's already one."""
    if not url_or_id:
        return ""
    s = url_or_id.strip()
    m = re.search(r"/d/([A-Za-z0-9_-]{20,})", s)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([A-Za-z0-9_-]{20,})", s)
    if m:
        return m.group(1)
    return s

def download_drive_csv_bytes(url_or_id: str) -> bytes | None:
    fid = extract_drive_id(url_or_id)
    if not fid:
        return None
    try:
        base = "https://drive.google.com/uc?export=download"
        with requests.Session() as s:
            r1 = s.get(base, params={"id": fid}, stream=True, timeout=60)
            # Handle big-file confirm page
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

def read_csv_best_effort(raw: bytes, label: str) -> pd.DataFrame | None:
    if not raw:
        st.warning(f"No data bytes received for {label}.")
        return None
    # Bail if Drive returned HTML (permissions/big-file)
    if b"<html" in (raw[:512] or b"").lower():
        st.error(f"Google Drive returned HTML for {label} (check sharing).")
        return None
    for kwargs in [
        {"encoding": "utf-8"},
        {"encoding": "utf-8", "engine": "python", "sep": None},  # sniff delimiter
        {"encoding": "latin1", "engine": "python"},
        {"encoding": "latin1", "engine": "python", "on_bad_lines": "skip"},
    ]:
        try:
            return pd.read_csv(io.BytesIO(raw), **kwargs)
        except Exception:
            continue
    st.error(f"Failed to parse {label} as CSV.")
    return None

# =========================
# Data loading (cached)
# =========================
@st.cache_data(ttl=600)
def load_data(trades_link, market_link):
    # --- Load Trades (Parquet) ---
    raw_trades = download_drive_csv_bytes(trades_link)
    trades = None
    if raw_trades:
        if b"<html" in (raw_trades[:512] or b"").lower():
            st.error("Google Drive returned an HTML page for the Trades file. Please check sharing permissions.")
        else:
            try:
                trades = pd.read_parquet(io.BytesIO(raw_trades))
            except Exception as e:
                st.error(f"Failed to parse Trades data as Parquet: {e}")
    
    # --- Load Market Data (CSV) ---
    raw_market = download_drive_csv_bytes(market_link)
    market = read_csv_best_effort(raw_market, "Market CSV") if raw_market is not None else None

    # --- Process Trades Data ---
    if trades is not None and not trades.empty:
        trades = lower_strip_cols(trades)
        if "product_id" in trades.columns:
            trades = trades.rename(columns={"product_id": "asset"})
        trades = trades.rename(columns={"side": "action", "size": "quantity"})
        # IMPORTANT: Convert Unix timestamp (float) to datetime
        if 'timestamp' in trades.columns and pd.api.types.is_numeric_dtype(trades['timestamp']):
             trades['timestamp'] = pd.to_datetime(trades['timestamp'], unit='s', errors='coerce')
        trades["timestamp"] = to_local_naive(trades["timestamp"])
        if "action" in trades.columns:
            trades["action"] = trades["action"].str.lower()
        for col in ["quantity", "price", "usd_value"]:
            if col in trades.columns:
                trades[col] = pd.to_numeric(trades[col], errors="coerce")
    else:
        trades = pd.DataFrame()

    # --- Process Market Data ---
    if market is not None and not market.empty:
        market = lower_strip_cols(market)
        if "product_id" in market.columns:
            market = market.rename(columns={"product_id": "asset"})
        market["asset"] = market["asset"].apply(unify_symbol)
        market["timestamp"] = to_local_naive(market["timestamp"])
        market = normalize_prob_columns(market)
        for col in ["open", "high", "low", "close"]:
            market[col] = pd.to_numeric(market[col], errors="coerce")
    else:
        market = pd.DataFrame()

    pnl_summary, trades_with_pnl, stats = calculate_pnl_and_metrics(trades) if not trades.empty else ({}, pd.DataFrame(), {})
    return trades_with_pnl, pnl_summary, stats, market

# =========================
# UI
# =========================
st.markdown("## Trading Analytics Dashboard")
st.caption("View position status, P&L and chart controls in the sidebar.")

trades_df, pnl_summary, summary_stats, market_df = load_data(TRADES_LINK, MARKET_LINK)

# --- Sidebar: Realized P&L ---
with st.sidebar:
    st.markdown("## 💵 Realized P&L")
    if pnl_summary:
        total_pnl = sum(v for v in pnl_summary.values() if pd.notna(v))
        st.metric("Overall P&L", f"${total_pnl:,.2f}")
        st.markdown("**By Asset**")
        for asset, pnl in sorted(pnl_summary.items(), key=lambda kv: kv[1], reverse=True):
            color = "#10b981" if pnl >= 0 else "#ef4444"
            st.markdown(
                f"<div style='display:flex;justify-content:space-between'>"
                f"<span>{asset}</span>"
                f"<span style='color:{color};font-weight:600'>${pnl:,.2f}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
    else:
        st.info("No realized P&L yet (need closed SELLs).")

st.sidebar.markdown("---")
# --- Sidebar: Positions ---
st.sidebar.markdown("## 📊 Positions Status")
if not market_df.empty:
    open_positions_df = calculate_open_positions(trades_df, market_df)
    all_assets = sorted(market_df["asset"].dropna().unique())
    open_positions_lookup = open_positions_df.set_index("Asset") if not open_positions_df.empty else pd.DataFrame()
    for asset in all_assets:
        if not open_positions_lookup.empty and asset in open_positions_lookup.index:
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
if not market_df.empty:
    assets = sorted(market_df["asset"].dropna().unique())
    default_index = assets.index(DEFAULT_ASSET) if DEFAULT_ASSET in assets else 0
    selected_asset = st.sidebar.selectbox("Select Asset", assets, index=default_index)
    range_choice = st.sidebar.selectbox("Select Date Range", ["30 days", "7 days", "1 day", "All"], index=0)

# --- NEW: Sidebar Date Range Display ---
if not trades_df.empty and 'timestamp' in trades_df.columns:
    st.sidebar.markdown("---")
    st.sidebar.markdown("###  Trade File Date Range")
    min_date = trades_df['timestamp'].min()
    max_date = trades_df['timestamp'].max()
    if pd.notna(min_date) and pd.notna(max_date):
        st.sidebar.success(
            f"**Earliest Trade:**\n\n{min_date.strftime('%Y-%m-%d')}\n\n"
            f"**Latest Trade:**\n\n{max_date.strftime('%Y-%m-%d')}"
        )
    else:
        st.sidebar.warning("Could not determine the date range.")

# --- NEW: Sidebar Last Updated Time ---
st.sidebar.markdown("---")
st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# --- Tabs ---
tab1, tab2 = st.tabs(["📈 Candlestick Analysis", "💰 P&L Analysis"])

with tab1:
    if selected_asset is None or market_df.empty:
        st.warning("Market data not loaded. Cannot display chart.")
    else:
        df = market_df[market_df["asset"] == selected_asset].copy().sort_values("timestamp")
        if df.empty:
            st.warning("No market rows for selected asset.")
        else:
            end_date = df["timestamp"].max()
            if range_choice == "1 day":
                start_date = end_date - timedelta(days=1)
            elif range_choice == "7 days":
                start_date = end_date - timedelta(days=7)
            elif range_choice == "30 days":
                start_date = end_date - timedelta(days=30)
            else:
                start_date = df["timestamp"].min()

            vis = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)].copy()

            if vis.empty:
                st.warning("No market data in the selected date range for this asset.")
            else:
                # -------- Auto-rescale to µUSD for micro-priced assets --------
                y_min, y_max = vis["low"].min(), vis["high"].max()
                rescale = (pd.to_numeric(y_max, errors="coerce") < 0.01)
                if rescale:
                    for c in ["open", "high", "low", "close"]:
                        vis[c] = pd.to_numeric(vis[c], errors="coerce") * 1e6
                    ylabel = "Price (µUSD)"
                else:
                    ylabel = "Price (USD)"

                def fmt(v, decimals=6):
                    try:
                        return f"{float(v):.{decimals}f}"
                    except (ValueError, TypeError):
                        return "—"

                hovertext = [
                    f"📅 {row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"
                    + f"<br>Open: {fmt(row['open'])}"
                    + f"<br>High: {fmt(row['high'])}"
                    + f"<br>Low: {fmt(row['low'])}"
                    + f"<br>Close: {fmt(row['close'])}"
                    + f"<br><b>P-Up: {fmt(row.get('p_up'))}</b>"
                    + f"<br><b>P-Down: {fmt(row.get('p_down'))}</b>"
                    for _, row in vis.iterrows()
                ]

                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=vis["timestamp"], open=vis["open"], high=vis["high"], low=vis["low"], close=vis["close"],
                    name=selected_asset, text=hovertext, hoverinfo="text"
                ))

                # ---- Trades overlay (scale markers if needed) ----
                asset_trades = pd.DataFrame() # Initialize empty DataFrame
                if not trades_df.empty:
                    asset_trades = trades_df[
                        (trades_df["asset"] == selected_asset) &
                        (trades_df["timestamp"] >= start_date) &
                        (trades_df["timestamp"] <= end_date)
                    ].copy()
                    if not asset_trades.empty:
                        if rescale:
                            asset_trades["price"] = pd.to_numeric(asset_trades["price"], errors="coerce") * 1e6
                        buys = asset_trades[asset_trades["action"] == "buy"]
                        sells = asset_trades[asset_trades["action"] == "sell"]
                        if not buys.empty:
                            fig.add_trace(go.Scatter(
                                x=buys["timestamp"], y=buys["price"], mode="markers",
                                name="BUY", marker=dict(symbol="triangle-up", size=10, color="green")
                            ))
                        if not sells.empty:
                            fig.add_trace(go.Scatter(
                                x=sells["timestamp"], y=sells["price"], mode="markers",
                                name="SELL", marker=dict(symbol="triangle-down", size=10, color="red")
                            ))

                fig.update_layout(
                    template="plotly_white", xaxis_rangeslider_visible=False, hovermode="x unified",
                    yaxis_title=ylabel, title=f"{selected_asset} — Price & Trades",
                    legend=dict(orientation="h", y=1.03, x=0.5, xanchor="center"), height=600,
                    margin=dict(l=40, r=20, t=60, b=40),
                )
                st.plotly_chart(fig, use_container_width=True)

                # --- Debugging Expander ---
                with st.expander("Debug: View Filtered Trade Data for Chart"):
                    st.write(f"The table below shows the trade data being used to plot markers for **{selected_asset}** between **{start_date.strftime('%Y-%m-%d')}** and **{end_date.strftime('%Y-%m-%d')}**.")
                    if asset_trades.empty:
                        st.warning("No trades were found for the current asset and date range.")
                        st.write("Full (unfiltered) trade data loaded:")
                        st.dataframe(trades_df)
                    else:
                        st.dataframe(asset_trades)


with tab2:
    if trades_df is None or trades_df.empty:
        st.warning("No trade data loaded to perform P&L analysis.")
    else:
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
        
        if assets_for_plot.empty or assets_for_plot[pnl_col].sum() == 0:
            st.warning("No P&L data to display.")
        else:
            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Scatter(x=assets_for_plot["timestamp"], y=assets_for_plot[pnl_col], mode="lines", name="Cumulative P&L"))
            fig_pnl.update_layout(title=title, template="plotly_white", yaxis_title="P&L (USD)")
            st.plotly_chart(fig_pnl, use_container_width=True)
