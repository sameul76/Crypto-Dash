import io
import re
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from decimal import Decimal, getcontext

# ========= App setup =========
st.set_page_config(page_title="Crypto Trading Strategy", layout="wide")
getcontext().prec = 30  # precision for Decimal math

# ---- Google Drive links (update if files move) ----
TRADES_LINK = "https://drive.google.com/file/d/1hyM37eafLgvMo8RJDtw9GSEdZ-LQ05ks/view?usp=sharing"
MARKET_LINK = "https://drive.google.com/file/d/1JaNhwQTcYOZ-tpP_ZwHXHtNzo-GpW-TO/view?usp=drive_link"

DEFAULT_ASSET = "GIGA-USD"
REFRESH_INTERVAL = 300  # seconds

# ---- Session state keys ----
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()
if "auto_refresh_enabled" not in st.session_state:
    st.session_state.auto_refresh_enabled = True


# ========= Helpers =========
def _drive_id(url_or_id: str) -> str:
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


def _download_drive_bytes(url_or_id: str) -> bytes | None:
    fid = _drive_id(url_or_id)
    if not fid:
        return None
    base = "https://drive.google.com/uc?export=download"
    try:
        with requests.Session() as s:
            r1 = s.get(base, params={"id": fid}, stream=True, timeout=60)
            if "text/html" in (r1.headers.get("Content-Type") or ""):
                m = re.search(r"confirm=([0-9A-Za-z_-]+)", r1.text)
                if m:
                    r2 = s.get(base, params={"id": fid, "confirm": m.group(1)}, stream=True, timeout=60)
                    r2.raise_for_status()
                    return r2.content
            r1.raise_for_status()
            return r1.content
    except Exception as e:
        st.error(f"Network error downloading file ID {fid}: {e}")
        return None


def _read_parquet_or_csv(b: bytes, label: str) -> pd.DataFrame:
    if not b:
        st.warning(f"{label}: no bytes downloaded")
        return pd.DataFrame()
    if b[:4] != b"PAR1":
        try:
            return pd.read_csv(io.BytesIO(b))
        except Exception:
            st.error(f"{label}: not a Parquet file (magic mismatch) and CSV fallback failed.")
            return pd.DataFrame()
    try:
        return pd.read_parquet(io.BytesIO(b))
    except Exception as e:
        st.error(f"{label}: failed to read Parquet: {e}")
        return pd.DataFrame()


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
        norm = col.lower().replace("_", " ").replace("-", " ")
        if norm in p_up_variations:
            rename_map[col] = "p_up"
        elif norm in p_down_variations:
            rename_map[col] = "p_down"
    if rename_map:
        df = df.rename(columns=rename_map)
    if "p_up" not in df.columns:
        df["p_up"] = np.nan
    if "p_down" not in df.columns:
        df["p_down"] = np.nan
    return df


# IMPORTANT: raw timestamps remain unchanged. Use parsed view ONLY when needed.
def _parsed_ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def seconds_since_last_run() -> int:
    return int(time.time() - st.session_state.get("last_refresh", 0))


def maybe_auto_refresh() -> int:
    elapsed = seconds_since_last_run()
    if st.session_state.get("auto_refresh_enabled", False) and elapsed >= REFRESH_INTERVAL:
        st.session_state.last_refresh = time.time()
        st.cache_data.clear()
        st.rerun()
    return elapsed


# ========= Data loading (cached) =========
@st.cache_data(ttl=60)
def load_data(trades_link: str, market_link: str):
    # TRADES
    trades = pd.DataFrame()
    tb = _download_drive_bytes(trades_link)
    if tb:
        trades = _read_parquet_or_csv(tb, "Trades")

    # MARKET
    market = pd.DataFrame()
    mb = _download_drive_bytes(market_link)
    if mb:
        market = _read_parquet_or_csv(mb, "Market")

    # ---- Normalize trades (no timestamp coercion) ----
    if not trades.empty:
        trades = lower_strip_cols(trades)
        colmap = {}
        if "value" in trades.columns:
            colmap["value"] = "usd_value"
        if "side" in trades.columns and "action" in trades.columns:
            colmap["side"] = "trade_direction"
        elif "side" in trades.columns and "action" not in trades.columns:
            colmap["side"] = "action"
        trades = trades.rename(columns=colmap)

        if "action" in trades.columns:
            trades["unified_action"] = (
                trades["action"]
                .astype(str)
                .str.upper()
                .map({"OPEN": "buy", "CLOSE": "sell"})
                .fillna(trades["action"].astype(str).str.lower())
            )
        elif "trade_direction" in trades.columns:
            trades["unified_action"] = trades["trade_direction"]
        elif "side" in trades.columns:
            trades["unified_action"] = trades["side"]
        else:
            trades["unified_action"] = "unknown"

        if "asset" in trades.columns:
            trades["asset"] = trades["asset"].apply(unify_symbol)

        # Convert numeric columns to Decimal where useful
        for col in ["quantity", "price", "usd_value", "pnl", "pnl_pct"]:
            if col in trades.columns:
                trades[col] = trades[col].apply(lambda x: Decimal(str(x)) if pd.notna(x) else Decimal(0))

    # ---- Normalize market (no timestamp coercion) ----
    if not market.empty:
        market = lower_strip_cols(market)
        if "product_id" in market.columns:
            market = market.rename(columns={"product_id": "asset"})
        if "asset" in market.columns:
            market["asset"] = market["asset"].apply(unify_symbol)

        market = normalize_prob_columns(market)

        # Ensure OHLC numeric for chart
        for col in ["open", "high", "low", "close"]:
            if col in market.columns:
                market[col] = pd.to_numeric(market[col], errors="coerce")

    return trades, market


# ========= P&L / Trades utils =========
def format_timedelta_hhmm(td):
    if pd.isna(td):
        return "N/A"
    total_seconds = int(td.total_seconds())
    m, _ = divmod(total_seconds, 60)
    h, mm = divmod(m, 60)
    return f"{h:02d}:{mm:02d}"


def calculate_pnl_and_metrics(trades_df: pd.DataFrame):
    if trades_df is None or trades_df.empty:
        return {}, pd.DataFrame(), {}

    df = trades_df.copy()
    parsed_ts = _parsed_ts(df["timestamp"])
    df["__parsed_ts__"] = parsed_ts
    df = df.sort_values("__parsed_ts__").drop(columns="__parsed_ts__").reset_index(drop=True)

    pnl_per_asset, positions = {}, {}
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
                realized = (price - avg_cost) * trade_qty
                pnl_per_asset[asset] += realized
                total += realized
                cur_pnl = realized
                if realized > 0:
                    win += 1
                    gp += realized
                else:
                    loss += 1
                    gl += abs(realized)
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

    if not df.empty and "pnl" in df.columns:
        df["asset_cumulative_pnl"] = df.groupby("asset")["pnl"].transform(lambda x: x.cumsum())

    return pnl_per_asset, df, stats


def calculate_open_positions(trades_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df is None or trades_df.empty or market_df is None or market_df.empty:
        return pd.DataFrame()

    positions = {}
    tdf = trades_df.copy()
    tdf["__parsed_ts__"] = _parsed_ts(tdf["timestamp"])
    tdf = tdf.sort_values("__parsed_ts__").drop(columns="__parsed_ts__")

    for _, row in tdf.iterrows():
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
            latest_asset_rows = market_df[market_df["asset"] == asset]
            if not latest_asset_rows.empty:
                # latest by parsed timestamp
                idx = _parsed_ts(latest_asset_rows["timestamp"]).idxmax()
                latest_price = Decimal(str(latest_asset_rows.loc[idx, "close"]))
                avg_entry = data["cost"] / data["quantity"]
                current_value = latest_price * data["quantity"]
                unrealized = current_value - data["cost"]
                open_positions.append(
                    {
                        "Asset": asset,
                        "Quantity": float(data["quantity"]),
                        "Avg. Entry Price": float(avg_entry),
                        "Current Price": float(latest_price),
                        "Current Value ($)": float(current_value),
                        "Unrealized P&L ($)": float(unrealized),
                    }
                )

    return pd.DataFrame(open_positions)


def match_trades_fifo(trades_df: pd.DataFrame):
    if trades_df is None or trades_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    tdf = trades_df.copy()
    tdf["__parsed_ts__"] = _parsed_ts(tdf["timestamp"])

    matched, open_positions = [], []

    for asset, group in tdf.groupby("asset"):
        g = group.sort_values("__parsed_ts__").drop(columns="__parsed_ts__")
        buys = [row.to_dict() for _, row in g[g["unified_action"].isin(["buy", "open"])].iterrows()]
        sells = [row.to_dict() for _, row in g[g["unified_action"].isin(["sell", "close"])].iterrows()]

        for sell in sells:
            sell_ts = pd.to_datetime(sell["timestamp"], errors="coerce")
            sell_qty = sell.get("quantity", Decimal(0))
            while sell_qty > Decimal("1e-9") and buys:
                b0 = buys[0]
                buy_ts = pd.to_datetime(b0["timestamp"], errors="coerce")
                if pd.isna(buy_ts) or pd.isna(sell_ts) or buy_ts >= sell_ts:
                    break
                buy_qty = b0.get("quantity", Decimal(0))
                trade_qty = min(sell_qty, buy_qty)
                if trade_qty > 0:
                    pnl = (sell["price"] - b0["price"]) * trade_qty
                    hold_time = sell_ts - buy_ts
                    matched.append(
                        {
                            "Asset": asset,
                            "Quantity": trade_qty,
                            "Buy Time": buy_ts,
                            "Buy Price": b0["price"],
                            "Sell Time": sell_ts,
                            "Sell Price": sell["price"],
                            "Hold Time": hold_time,
                            "P&L ($)": pnl,
                            "Reason Buy": b0.get("reason"),
                            "Reason Sell": sell.get("reason"),
                        }
                    )
                    sell_qty -= trade_qty
                    buys[0]["quantity"] -= trade_qty
                    if buys[0]["quantity"] < Decimal("1e-9"):
                        buys.pop(0)
        open_positions.extend(buys)

    matched_df = pd.DataFrame(matched) if matched else pd.DataFrame()
    open_df = pd.DataFrame(open_positions) if open_positions else pd.DataFrame()

    if not matched_df.empty:
        buy_cost = matched_df["Buy Price"] * matched_df["Quantity"]
        is_zero = buy_cost < Decimal("1e-18")
        matched_df["P&L %"] = 100 * np.where(is_zero, 0, matched_df["P&L ($)"] / buy_cost)
        matched_df = matched_df.sort_values("Sell Time", ascending=False)

    if not open_df.empty:
        open_df["__parsed_ts__"] = _parsed_ts(open_df["timestamp"])
        open_df = open_df.sort_values("__parsed_ts__", ascending=False).drop(columns="__parsed_ts__")

    return matched_df, open_df


# ========= Load + maybe refresh =========
st.markdown("## Crypto Trading Strategy")
st.caption("ML Signals with Price-Based Exit Logic (timestamps read exactly as stored)")

trades_df, market_df = load_data(TRADES_LINK, MARKET_LINK)
elapsed = maybe_auto_refresh()

# ========= FIXED Debug panel =========
with st.expander("🔎 Data Freshness Debug", expanded=True):
    # Trades
    if not trades_df.empty and "timestamp" in trades_df.columns:
        # Create a working copy with parsed timestamps
        trades_debug = trades_df.copy()
        trades_debug["__parsed_ts__"] = _parsed_ts(trades_debug["timestamp"])
        
        # Remove any rows where timestamp couldn't be parsed
        trades_debug_valid = trades_debug.dropna(subset=["__parsed_ts__"])
        
        if not trades_debug_valid.empty:
            # Sort by parsed timestamp to get the truly latest entry
            trades_debug_sorted = trades_debug_valid.sort_values("__parsed_ts__")
            latest_idx = trades_debug_sorted.index[-1]  # Last row after sorting
            
            tr_raw_latest = trades_debug_sorted.loc[latest_idx, "timestamp"]
            tr_parsed_latest = trades_debug_sorted.loc[latest_idx, "__parsed_ts__"]
            
            st.write(f"**Latest Trade Timestamp (raw):** {tr_raw_latest}")
            st.write(f"**Latest Trade Timestamp (parsed):** {tr_parsed_latest}")
            
            # Show parsing issues if any
            total_trades = len(trades_df)
            valid_trades = len(trades_debug_valid)
            if total_trades != valid_trades:
                st.warning(f"⚠️ {total_trades - valid_trades} trades have unparseable timestamps")
        else:
            st.error("❌ No valid timestamps found in trades data")
            st.write("**Sample raw timestamps:**")
            st.write(trades_df["timestamp"].head().tolist())
    else:
        st.write("**Latest Trade Timestamp (raw):** No trade data")
        st.write("**Latest Trade Timestamp (parsed):** No trade data")
    
    st.write(f"**Total Trades:** {len(trades_df):,}")

    # Market
    if not market_df.empty and "timestamp" in market_df.columns:
        # Create a working copy with parsed timestamps
        market_debug = market_df.copy()
        market_debug["__parsed_ts__"] = _parsed_ts(market_debug["timestamp"])
        
        # Remove any rows where timestamp couldn't be parsed
        market_debug_valid = market_debug.dropna(subset=["__parsed_ts__"])
        
        if not market_debug_valid.empty:
            # Sort by parsed timestamp to get the truly latest entry
            market_debug_sorted = market_debug_valid.sort_values("__parsed_ts__")
            latest_idx = market_debug_sorted.index[-1]  # Last row after sorting
            
            mk_raw_latest = market_debug_sorted.loc[latest_idx, "timestamp"]
            mk_parsed_latest = market_debug_sorted.loc[latest_idx, "__parsed_ts__"]
            
            st.write(f"**Latest Market Timestamp (raw):** {mk_raw_latest}")
            st.write(f"**Latest Market Timestamp (parsed):** {mk_parsed_latest}")
            st.write(f"**Market timestamp dtype:** {market_df['timestamp'].dtype}")
            
            # Show parsing issues if any
            total_market = len(market_df)
            valid_market = len(market_debug_valid)
            if total_market != valid_market:
                st.warning(f"⚠️ {total_market - valid_market} market records have unparseable timestamps")
        else:
            st.error("❌ No valid timestamps found in market data")
            st.write("**Sample raw timestamps:**")
            st.write(market_df["timestamp"].head().tolist())
        
        st.write(f"**Total Market Records:** {len(market_df):,}")
    else:
        st.write("**Latest Market Timestamp (raw):** No market data")
        st.write("**Latest Market Timestamp (parsed):** No market data")

    st.write(f"**Cache age:** {elapsed}s")

# Add additional debugging section
with st.expander("🔍 Deep Dive Debug", expanded=False):
    if not trades_df.empty and "timestamp" in trades_df.columns:
        st.write("**Last 10 Trade Timestamps (raw order):**")
        st.dataframe(trades_df[["timestamp"]].tail(10))
        
        # Show timestamp format analysis
        sample_timestamps = trades_df["timestamp"].dropna().head(5).tolist()
        st.write("**Sample timestamp formats:**")
        for i, ts in enumerate(sample_timestamps):
            st.write(f"{i+1}. `{ts}` (type: {type(ts).__name__})")
        
        # Check for duplicates or sorting issues
        parsed_ts = _parsed_ts(trades_df["timestamp"])
        valid_parsed = parsed_ts.dropna()
        if len(valid_parsed) > 0:
            is_sorted = valid_parsed.is_monotonic_increasing
            st.write(f"**Data chronologically sorted:** {'✅ Yes' if is_sorted else '❌ No'}")
            st.write(f"**Timestamp range:** {valid_parsed.min()} to {valid_parsed.max()}")
    
    if not market_df.empty and "timestamp" in market_df.columns:
        st.write("**Last 10 Market Timestamps (raw order):**")
        st.dataframe(market_df[["timestamp", "asset"]].tail(10))
        
        # Check market data sorting by asset
        for asset in market_df["asset"].unique()[:3]:  # Check first 3 assets
            asset_data = market_df[market_df["asset"] == asset]
            parsed_ts = _parsed_ts(asset_data["timestamp"])
            valid_parsed = parsed_ts.dropna()
            if len(valid_parsed) > 0:
                is_sorted = valid_parsed.is_monotonic_increasing
                st.write(f"**{asset} chronologically sorted:** {'✅ Yes' if is_sorted else '❌ No'}")

# ========= Sidebar =========
with st.sidebar:
    st.markdown("<h1 style='text-align:center;'>Crypto Strategy</h1>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.session_state.auto_refresh_enabled = st.toggle(
            "🔄 Auto-Refresh (5min)", value=st.session_state.get("auto_refresh_enabled", True)
        )
    with c2:
        if st.button("Force Fresh Load"):
            st.cache_data.clear()
            st.session_state.last_refresh = time.time()
            st.rerun()

    if st.session_state.get("auto_refresh_enabled", True):
        remaining = max(0, REFRESH_INTERVAL - seconds_since_last_run())
        m, s = divmod(remaining, 60)
        st.caption(f"⏱️ Next refresh in {m:02d}:{s:02d}")
    else:
        st.caption("Auto-refresh disabled")

    st.markdown("---")
    st.markdown(f"**Trades:** {'✅' if not trades_df.empty else '⚠️'} {len(trades_df):,}")
    st.markdown(f"**Market:** {'✅' if not market_df.empty else '❌'} {len(market_df):,}")
    if not market_df.empty and "asset" in market_df.columns:
        st.markdown(f"**Assets:** {market_df['asset'].nunique():,}")
    
    # Open Positions in Sidebar
    st.markdown("---")
    open_positions_df = calculate_open_positions(trades_df, market_df) if not trades_df.empty else pd.DataFrame()
    if not open_positions_df.empty:
        st.markdown("**Open Positions**")
        for _, pos in open_positions_df.iterrows():
            pnl = pos["Unrealized P&L ($)"]
            color = "#16a34a" if pnl >= 0 else "#ef4444"
            asset_name = pos["Asset"]
            cur_price = pos["Current Price"]
            pf = ".8f" if cur_price < 0.001 else ".6f" if cur_price < 1 else ".2f"
            avg_price = pos["Avg. Entry Price"]
            epf = ".8f" if avg_price < 0.001 else ".6f" if avg_price < 1 else ".2f"
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;'>"
                f"<span style='color:{color};font-weight:700'>{asset_name}</span>"
                f"<span style='font-weight:700'>${cur_price:{pf}}</span></div>",
                unsafe_allow_html=True,
            )
            st.caption(
                f"Qty: {pos['Quantity']:.4f} | Entry: ${avg_price:{epf}} | P&L: ${pnl:.2f}"
            )
    else:
        st.markdown("**Open Positions**")
        st.caption("No open positions")

# ========= Tabs =========
tab1, tab2, tab3 = st.tabs(["📈 Price & Trades", "💰 P&L Analysis", "📜 Trade History"])

# ----- TAB 1: Price & Trades -----
with tab1:
    if market_df.empty or "asset" not in market_df.columns:
        st.warning("Market data not available.")
    else:
        assets = sorted(market_df["asset"].dropna().unique().tolist())
        default_index = assets.index(DEFAULT_ASSET) if DEFAULT_ASSET in assets else 0

        ui1, ui2 = st.columns([3, 2])
        with ui1:
            selected_asset = st.selectbox("Select Asset to View", assets, index=default_index, key="asset_select_main")
        with ui2:
            range_choice = st.selectbox(
                "Select Date Range",
                ["4 hours", "12 hours", "1 day", "3 days", "7 days", "30 days", "All"],
                index=0,
                key="range_select_main",
            )

        asset_market = market_df[market_df["asset"] == selected_asset].copy()
        if asset_market.empty:
            st.warning(f"No market data found for {selected_asset}.")
        else:
            asset_market_sorted = asset_market.sort_values(
                by="timestamp", key=lambda s: _parsed_ts(s)
            )
            end_parsed = _parsed_ts(asset_market_sorted["timestamp"]).max()
            if pd.isna(end_parsed):
                st.warning("Timestamps for this asset could not be parsed.")
            else:
                if range_choice == "4 hours":
                    start_parsed = end_parsed - timedelta(hours=4)
                elif range_choice == "12 hours":
                    start_parsed = end_parsed - timedelta(hours=12)
                elif range_choice == "1 day":
                    start_parsed = end_parsed - timedelta(days=1)
                elif range_choice == "3 days":
                    start_parsed = end_parsed - timedelta(days=3)
                elif range_choice == "7 days":
                    start_parsed = end_parsed - timedelta(days=7)
                elif range_choice == "30 days":
                    start_parsed = end_parsed - timedelta(days=30)
                else:
                    start_parsed = _parsed_ts(asset_market_sorted["timestamp"]).min()

                parsed_col = _parsed_ts(asset_market_sorted["timestamp"])
                vis = asset_market_sorted.loc[(parsed_col >= start_parsed) & (parsed_col <= end_parsed)].copy()

                if vis.empty:
                    st.warning(f"No data for {selected_asset} in the selected date range: {range_choice}.")
                else:
                    last_price = vis["close"].iloc[-1]
                    pf = ",.8f" if last_price < 0.001 else ",.6f" if last_price < 1 else ",.4f"
                    st.metric(f"Last Price for {selected_asset}", f"${last_price:{pf}}")

                    price_range = vis["high"].max() - vis["low"].min()
                    y_min = vis["low"].min() - price_range * 0.15
                    y_max = vis["high"].max() + price_range * 0.05
                    tick_format = "%H:%M" if range_choice in ["4 hours", "12 hours"] else "%m/%d %H:%M"

                    fig = go.Figure()
                    fig.add_trace(
                        go.Candlestick(
                            x=_parsed_ts(vis["timestamp"]),
                            open=vis["open"],
                            high=vis["high"],
                            low=vis["low"],
                            close=vis["close"],
                            name=selected_asset,
                            increasing_line_color="#26a69a",
                            decreasing_line_color="#ef5350",
                            increasing_fillcolor="rgba(38,166,154,0.5)",
                            decreasing_fillcolor="rgba(239,83,80,0.5)",
                            line=dict(width=1),
                        )
                    )

                    # ML signals (if available)
                    if {"p_up", "p_down"}.issubset(vis.columns):
                        sig = vis.dropna(subset=["p_up", "p_down"]).copy()
                        if not sig.empty:
                            sig["confidence"] = (sig["p_up"] - sig["p_down"]).abs()
                            sizes = (sig["confidence"] * 100.0) / 5.0 + 3.0
                            colors = np.where(sig["p_down"] > sig["p_up"], "#ff6b6b", "#51cf66")
                            fig.add_trace(
                                go.Scatter(
                                    x=_parsed_ts(sig["timestamp"]),
                                    y=sig["close"],
                                    mode="markers",
                                    marker=dict(size=sizes, color=colors, opacity=0.7, line=dict(width=1, color="white")),
                                    name="ML Signals",
                                    customdata=list(zip(sig["p_up"], sig["p_down"], sig["confidence"])),
                                    hovertemplate="<b>ML Signal</b><br>Time: %{x|%Y-%m-%d %H:%M}<br>"
                                                  "Price: $%{y:.6f}<br>P(Up): %{customdata[0]:.3f}<br>"
                                                  "P(Down): %{customdata[1]:.3f}<br>"
                                                  "Confidence: %{customdata[2]:.3f}<extra></extra>",
                                )
                            )

                    # Buy/Sell markers from trades
                    if not trades_df.empty:
                        asset_trades = trades_df[trades_df["asset"] == selected_asset].copy()
                        if not asset_trades.empty and "timestamp" in asset_trades.columns:
                            t_parsed = _parsed_ts(asset_trades["timestamp"])
                            mask = (t_parsed >= start_parsed) & (t_parsed <= end_parsed)
                            asset_trades = asset_trades.loc[mask].copy()

                            if not asset_trades.empty:
                                # y marker level slightly above the min
                                marker_y = y_min + (vis["high"].max() - vis["low"].min()) * 0.02

                                buys = asset_trades[asset_trades["unified_action"].str.lower().isin(["buy", "open"])]
                                if not buys.empty:
                                    buy_prices = buys["price"].apply(float)
                                    buy_reasons = buys.get("reason", pd.Series([""] * len(buys))).fillna("")
                                    fig.add_trace(
                                        go.Scatter(
                                            x=_parsed_ts(buys["timestamp"]),
                                            y=[marker_y] * len(buys),
                                            mode="markers",
                                            name="BUY",
                                            marker=dict(symbol="triangle-up", size=14, color="#4caf50", line=dict(width=1, color="white")),
                                            customdata=np.stack((buy_prices, buy_reasons), axis=-1),
                                            hovertemplate="<b>BUY</b> @ $%{customdata[0]:.8f}<br>%{x|%H:%M:%S}"
                                                          "<br>Reason: %{customdata[1]}<extra></extra>",
                                        )
                                    )

                                sells = asset_trades[asset_trades["unified_action"].str.lower().isin(["sell", "close"])]
                                if not sells.empty:
                                    sell_prices = sells["price"].apply(float)
                                    sell_reasons = sells.get("reason", pd.Series([""] * len(sells))).fillna("")
                                    fig.add_trace(
                                        go.Scatter(
                                            x=_parsed_ts(sells["timestamp"]),
                                            y=[marker_y] * len(sells),
                                            mode="markers",
                                            name="SELL",
                                            marker=dict(symbol="triangle-down", size=14, color="#f44336", line=dict(width=1, color="white")),
                                            customdata=np.stack((sell_prices, sell_reasons), axis=-1),
                                            hovertemplate="<b>SELL</b> @ $%{customdata[0]:.8f}<br>%{x|%H:%M:%S}"
                                                          "<br>Reason: %{customdata[1]}<extra></extra>",
                                        )
                                    )

                    fig.update_layout(
                        title=f"{selected_asset} — Price & Trades ({range_choice})",
                        template="plotly_white",
                        xaxis_rangeslider_visible=False,
                        xaxis=dict(
                            title="Time",
                            type="date",
                            tickformat=tick_format,
                            showgrid=True,
                            gridcolor="rgba(128,128,128,0.1)",
                            tickangle=-45,
                        ),
                        yaxis=dict(
                            title="Price (USD)",
                            tickformat=".8f" if last_price < 0.001 else ".6f" if last_price < 1 else ".4f",
                            showgrid=True,
                            gridcolor="rgba(128,128,128,0.1)",
                            range=[y_min, y_max],
                        ),
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                        height=750,
                        margin=dict(l=60, r=20, t=80, b=80),
                        plot_bgcolor="rgba(250,250,250,0.8)",
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        config={
                            "displayModeBar": True,
                            "modeBarButtonsToAdd": ["drawline", "drawopenpath", "drawclosedpath"],
                            "scrollZoom": True,
                        },
                    )

                    # Quick stats for the filtered period
                    asset_trades_period = pd.DataFrame()
                    if not trades_df.empty:
                        at = trades_df[trades_df["asset"] == selected_asset].copy()
                        if not at.empty:
                            t_parsed = _parsed_ts(at["timestamp"])
                            asset_trades_period = at.loc[(t_parsed >= start_parsed) & (t_parsed <= end_parsed)].copy()
                    c1, c2, c3, c4, c5 = st.columns(5)
                    with c1:
                        st.metric("Total Trades", len(asset_trades_period))
                    with c2:
                        st.metric("Buy Orders", len(asset_trades_period[asset_trades_period["unified_action"].str.lower().isin(["buy", "open"])]))
                    with c3:
                        st.metric("Sell Orders", len(asset_trades_period[asset_trades_period["unified_action"].str.lower().isin(["sell", "close"])]))
                    with c4:
                        if "pnl" in asset_trades_period.columns:
                            period_pnl = asset_trades_period["pnl"].sum()
                            st.metric("Period P&L", f"${period_pnl:.6f}")
                    with c5:
                        if len(asset_trades_period) >= 2 and "timestamp" in asset_trades_period.columns:
                            tps = _parsed_ts(asset_trades_period["timestamp"])
                            span = tps.max() - tps.min()
                            st.metric("Trading Span", format_timedelta_hhmm(span))

                    # Watchlist/Open positions
                    st.markdown("---")
                    open_positions_df = calculate_open_positions(trades_df, market_df) if not trades_df.empty else pd.DataFrame()
                    if not open_positions_df.empty:
                        st.markdown("### Open Positions")
                        for _, pos in open_positions_df.iterrows():
                            pnl = pos["Unrealized P&L ($)"]
                            color = "#16a34a" if pnl >= 0 else "#ef4444"
                            asset_name = pos["Asset"]
                            cur_price = pos["Current Price"]
                            pf = ".8f" if cur_price < 0.001 else ".6f" if cur_price < 1 else ".2f"
                            avg_price = pos["Avg. Entry Price"]
                            epf = ".8f" if avg_price < 0.001 else ".6f" if avg_price < 1 else ".2f"
                            st.markdown(
                                f"<div style='display:flex;justify-content:space-between;'>"
                                f"<span style='color:{color};font-weight:700'>{asset_name}</span>"
                                f"<span style='font-weight:700'>${cur_price:{pf}}</span></div>",
                                unsafe_allow_html=True,
                            )
                            st.caption(
                                f"Qty: {pos['Quantity']:.4f} | Entry: ${avg_price:{epf}} | P&L: ${pnl:.2f}"
                            )

                    if not market_df.empty and "asset" in market_df.columns:
                        st.markdown("---")
                        st.markdown("### Watchlist")
                        held = set(open_positions_df["Asset"]) if not open_positions_df.empty else set()
                        for a in sorted(market_df["asset"].dropna().unique()):
                            if a in held:
                                continue
                            rows = market_df[market_df["asset"] == a]
                            if not rows.empty:
                                idx = _parsed_ts(rows["timestamp"]).idxmax()
                                last = rows.loc[idx, "close"]
                                pf = ".8f" if last < 0.001 else ".6f" if last < 1 else ".2f"
                                st.markdown(
                                    f"<div style='display:flex;justify-content:space-between;color:#666'>"
                                    f"<span>{a}</span><span style='font-weight:600'>${last:{pf}}</span></div>",
                                    unsafe_allow_html=True,
                                )

# ----- TAB 2: P&L Analysis -----
with tab2:
    if trades_df.empty or "timestamp" not in trades_df.columns:
        st.warning("No trade data loaded to analyze P&L.")
    else:
        pnl_summary, trades_with_pnl, stats = calculate_pnl_and_metrics(trades_df)

        st.markdown("### Strategy Performance")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Closed Trades", f"{stats.get('total_trades', 0):,}")
        with c2:
            st.metric("Win Rate", f"{stats.get('win_rate', 0):.1f}%")
        with c3:
            pf = stats.get("profit_factor", 0)
            st.metric("Profit Factor", "∞" if isinstance(pf, float) and np.isinf(pf) else f"{pf:.2f}")
        with c4:
            st.metric("Max Drawdown", f"${stats.get('max_drawdown', 0):.2f}")

        if not trades_with_pnl.empty:
            plot_df = trades_with_pnl.copy()
            plot_df["__parsed_ts__"] = _parsed_ts(plot_df["timestamp"])
            plot_df = plot_df.sort_values("__parsed_ts__")
            plot_df["cumulative_pnl"] = plot_df["cumulative_pnl"].apply(float)

            fig_pnl = go.Figure()
            fig_pnl.add_trace(
                go.Scatter(
                    x=plot_df["__parsed_ts__"],
                    y=plot_df["cumulative_pnl"],
                    mode="lines",
                    name="Cumulative P&L",
                    line=dict(width=2),
                )
            )
            fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Break Even")
            fig_pnl.update_layout(
                title="Total Portfolio P&L",
                template="plotly_white",
                yaxis_title="P&L (USD)",
                xaxis_title="Date",
                hovermode="x unified",
            )
            st.plotly_chart(fig_pnl, use_container_width=True)

        st.markdown("---")
        st.markdown("### Realized P&L by Asset")
        if pnl_summary:
            total_pnl = sum(p for p in pnl_summary.values() if pd.notna(p))
            st.metric("Overall P&L", f"${total_pnl:,.2f}")
            for asset, pnl in sorted(pnl_summary.items(), key=lambda kv: kv[1], reverse=True):
                color = "#10b981" if pnl >= 0 else "#ef4444"
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between'>"
                    f"<span>{asset}</span><span style='color:{color};font-weight:600'>${pnl:,.2f}</span></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No realized P&L yet.")

# ----- TAB 3: Trade History -----
with tab3:
    st.markdown("### Trade History")
    if trades_df.empty:
        st.warning("No trade history to display.")
    else:
        matched_df, open_df = match_trades_fifo(trades_df)

        st.markdown("#### Completed Trades (FIFO)")
        if not matched_df.empty:
            display_matched = matched_df.copy()
            for col in ["Quantity", "Buy Price", "Sell Price", "P&L ($)", "P&L %"]:
                if col in display_matched.columns:
                    display_matched[col] = display_matched[col].apply(float)

            display_matched["Buy Time"] = pd.to_datetime(display_matched["Buy Time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            display_matched["Sell Time"] = pd.to_datetime(display_matched["Sell Time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            display_matched["Hold Time"] = display_matched["Hold Time"].apply(format_timedelta_hhmm)

            st.dataframe(
                display_matched[
                    [
                        "Asset",
                        "Quantity",
                        "Buy Time",
                        "Buy Price",
                        "Sell Time",
                        "Sell Price",
                        "Hold Time",
                        "P&L ($)",
                        "P&L %",
                    ]
                ],
                column_config={
                    "Asset": st.column_config.TextColumn(width="small"),
                    "Quantity": st.column_config.NumberColumn(format="%.4f", width="small"),
                    "Buy Price": st.column_config.NumberColumn(format="$%.8f"),
                    "Sell Price": st.column_config.NumberColumn(format="$%.8f"),
                    "P&L ($)": st.column_config.NumberColumn(format="$%.4f"),
                    "P&L %": st.column_config.NumberColumn(format="%.4f%%"),
                },
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No completed (buy/sell) trades found.")

        st.markdown("---")
        st.markdown("#### Open Positions (Unmatched Buys)")
        if not open_df.empty:
            display_open = open_df.copy()
            for col in ["quantity", "price"]:
                if col in display_open.columns:
                    display_open[col] = display_open[col].apply(float)

            display_open["Time"] = pd.to_datetime(display_open["timestamp"], errors="coerce").dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            display_open = display_open.rename(
                columns={"asset": "Asset", "quantity": "Quantity", "price": "Price", "reason": "Reason"}
            )

            st.dataframe(
                display_open[["Time", "Asset", "Quantity", "Price", "Reason"]],
                column_config={
                    "Asset": st.column_config.TextColumn(width="small"),
                    "Quantity": st.column_config.NumberColumn(format="%.4f", width="small"),
                    "Price": st.column_config.NumberColumn(format="$%.8f"),
                },
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No open positions.")

