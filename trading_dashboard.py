import io
import re
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta

# =========================
# HARD-CODED FILE LINKS (two CSVs)
# =========================
TRADES_LINK = "https://drive.google.com/file/d/1zSdFcG4Xlh_iSa180V6LRSeEucAokXYk/view?usp=drive_link"
MARKET_LINK = "https://drive.google.com/file/d/1tvY7CheH_p5f3uaE7VPUYS78hDHNmX_C/view?usp=sharing"  # OHLCV + probs (CSV)

st.set_page_config(page_title="Trading Analytics — Candles + Trades", layout="wide")
LOCAL_TZ = "America/Los_Angeles"
DEFAULT_ASSET = "GIGA-USD"  # canonical name for GIGA only

# =========================
# Helpers
# =========================
def lower_strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.lower().strip() for c in out.columns]
    return out

def to_local_naive(ts):
    """Convert timestamps to LOCAL_TZ and drop tz info (naive)."""
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

def extract_drive_id(url_or_id: str) -> str:
    if not url_or_id:
        return ""
    s = url_or_id.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", s):  # looks like a file id
        return s
    m = re.search(r"/d/([A-Za-z0-9_-]{20,})", s)
    if m: return m.group(1)
    m = re.search(r"[?&]id=([A-Za-z0-9_-]{20,})", s)
    if m: return m.group(1)
    return s

def _drive_confirm_download(session: requests.Session, file_id: str) -> bytes:
    """
    Handle Google Drive's big-file confirm page by extracting the confirm token.
    Returns raw file bytes.
    """
    base = "https://drive.google.com/uc?export=download"
    params = {"id": file_id}
    headers = {"User-Agent": "Mozilla/5.0"}
    r1 = session.get(base, params=params, stream=True, timeout=60, headers=headers)
    r1.raise_for_status()

    ctype = (r1.headers.get("Content-Type") or "").lower()
    if "text/html" not in ctype:
        return r1.content

    html = r1.text
    token = None
    m = re.search(r"confirm=([0-9A-Za-z_-]+)", html)
    if m:
        token = m.group(1)
    else:
        for k, v in r1.cookies.items():
            if k.startswith("download_warning"):
                token = v
                break

    if token:
        params["confirm"] = token
        r2 = session.get(base, params=params, stream=True, timeout=60, headers=headers)
        r2.raise_for_status()
        return r2.content

    return r1.content  # will be detected as HTML later if not the file

def download_drive_csv_bytes(url_or_id: str) -> bytes | None:
    """Robustly download Drive file bytes (handles big-file confirm); returns raw bytes or None."""
    fid = extract_drive_id(url_or_id)
    if not fid:
        return None
    try:
        with requests.Session() as s:
            return _drive_confirm_download(s, fid)
    except Exception:
        try:
            url = f"https://drive.google.com/uc?export=download&id={fid}"
            r = requests.get(url, stream=True, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            return r.content
        except Exception:
            return None

def _looks_like_html(raw: bytes) -> bool:
    head = (raw[:512] or b"").lower()
    return b"<html" in head or b"<!doctype html" in head

def read_csv_best_effort(raw: bytes, label: str) -> pd.DataFrame | None:
    """
    Try multiple parsing strategies to survive Drive HTML + funky CSVs.
    """
    if not raw:
        st.error(f"Failed to download bytes for {label}.")
        return None

    if _looks_like_html(raw):
        st.error(f"Google Drive returned HTML for {label} (permissions or big-file confirm). "
                 f"Set it to 'Anyone with the link: Viewer'.")
        try:
            tables = pd.read_html(io.BytesIO(raw))
            if tables:
                st.caption("Preview of what Drive returned (first table):")
                st.dataframe(tables[0].head(10))
        except Exception:
            pass
        return None

    # Try several parsers/encodings
    candidates = [
        {"encoding": "utf-8"},
        {"encoding": "utf-8", "engine": "python"},
        {"encoding": "utf-8", "engine": "python", "sep": None},  # sniff delimiter
        {"encoding": "latin1"},
        {"encoding": "latin1", "engine": "python"},
        {"encoding": "latin1", "engine": "python", "sep": None},
        {"encoding": "utf-8", "engine": "python", "on_bad_lines": "skip"},
        {"encoding": "latin1", "engine": "python", "on_bad_lines": "skip"},
    ]
    for kwargs in candidates:
        try:
            return pd.read_csv(io.BytesIO(raw), **kwargs)
        except Exception:
            continue

    st.error(f"Failed to parse {label} as CSV even with fallbacks.")
    return None

# ---------- ONLY GIGA gets unified; others stay untouched ----------
def unify_symbol(val: str) -> str:
    """
    Canonicalize asset/product_id names:
    - Any entry that contains 'GIGA' (case/underscore insensitive) => 'GIGA-USD'
    - Everything else (LCX-USD, MNDE-USD, MOG-USD, VVV-USD, CVX*, etc.) is returned EXACTLY as-is.
    """
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
    if "p_up" not in df.columns: df["p_up"] = np.nan
    if "p_down" not in df.columns: df["p_down"] = np.nan
    return df

def calculate_pnl_and_metrics(trades_df):
    if trades_df is None or trades_df.empty:
        return {}, pd.DataFrame(), {}
    pnl_per_asset, positions = {}, {}
    df = trades_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["pnl"] = 0.0
    df["cumulative_pnl"] = 0.0
    total = 0.0
    win = loss = 0
    gp = gl = 0.0
    peak = mdd = 0.0

    for i, row in df.iterrows():
        asset = row["asset"]; action = row["action"]
        price = float(row["price"]); qty = float(row["quantity"])
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

# =========================
# Load both CSVs (cached)
# =========================
@st.cache_data(ttl=600)
def load_data():
    # Download raw bytes
    raw_trades = download_drive_csv_bytes(TRADES_LINK)
    raw_market = download_drive_csv_bytes(MARKET_LINK)

    # Parse CSVs
    trades = read_csv_best_effort(raw_trades, "Trades CSV")
    market = read_csv_best_effort(raw_market, "Market CSV")

    # Normalize trades
    if trades is not None and not trades.empty:
        trades = lower_strip_cols(trades)

        # Choose the identifier column and rename to 'asset'
        if "product_id" in trades.columns and "asset" not in trades.columns:
            trades = trades.rename(columns={"product_id": "asset"})
        # Normalize core columns
        trades = trades.rename(columns={"side": "action", "size": "quantity"})
        # Canonicalize asset names (unify only GIGA -> GIGA-USD)
        trades["asset"] = trades["asset"].apply(unify_symbol)
        # Timestamps
        trades["timestamp"] = to_local_naive(trades["timestamp"])
        if "action" in trades.columns:
            trades["action"] = trades["action"].str.lower()

        # numeric fields for plotting info
        if "quantity" in trades.columns:
            trades["quantity"] = pd.to_numeric(trades["quantity"], errors="coerce")
        if "price" in trades.columns:
            trades["price"] = pd.to_numeric(trades["price"], errors="coerce")
        if "usd_value" in trades.columns:
            trades["usd_value"] = pd.to_numeric(trades["usd_value"], errors="coerce")

        need = ["timestamp", "asset", "action", "price", "quantity"]
        if not all(c in trades.columns for c in need):
            st.error(f"Trades CSV missing columns: {need}")
            trades = None

    # Normalize market (OHLCV + probabilities)
    if market is not None and not market.empty:
        market = lower_strip_cols(market)

        if "product_id" in market.columns and "asset" not in market.columns:
            market = market.rename(columns={"product_id": "asset"})
        # Canonicalize asset names (unify only GIGA -> GIGA-USD)
        market["asset"] = market["asset"].apply(unify_symbol)

        need_m = ["timestamp", "asset", "open", "high", "low", "close"]
        if not all(c in market.columns for c in need_m):
            st.error(f"Market CSV missing columns: {need_m}")
            market = None
        else:
            market["timestamp"] = to_local_naive(market["timestamp"])
            market = normalize_prob_columns(market)
            # Ensure numeric OHLC
            for col in ["open", "high", "low", "close"]:
                market[col] = pd.to_numeric(market[col], errors="coerce")

    # P&L (optional)
    if trades is not None and not trades.empty:
        pnl_summary, trades_with_pnl, stats = calculate_pnl_and_metrics(trades)
    else:
        pnl_summary, trades_with_pnl, stats = {}, pd.DataFrame(), {}

    return trades_with_pnl, pnl_summary, stats, market

trades_df, pnl_summary, summary_stats, market_df = load_data()

st.markdown("## Trading Analytics — Candles + Trades")
st.caption("Select an asset. Candlesticks show P-Up / P-Down in hover. BUY/SELL markers come from the Trades CSV.")

# =========================
# UI: Asset + Range
# =========================
if market_df is None or market_df.empty:
    st.error("Couldn’t load the Market CSV. Make sure it’s shared as ‘Anyone with the link: Viewer’.")
else:
    assets = sorted(market_df["asset"].dropna().unique().tolist())
    if not assets:
        st.error("No assets found in Market CSV.")
    else:
        # Prefer GIGA-USD if present
        default_index = assets.index(DEFAULT_ASSET) if DEFAULT_ASSET in assets else 0
        selected_asset = st.selectbox("Asset", assets, index=default_index)

        colA, colB = st.columns([3, 1])
        with colB:
            range_choice = st.selectbox("Range", ["30 days", "7 days", "1 day", "All"], index=0)

        df = market_df[market_df["asset"] == selected_asset].copy().sort_values("timestamp")
        if df.empty:
            st.warning("No rows for selected asset.")
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

            # ---- Build custom hover for candles
            def fmt4(v):
                try:
                    return f"{float(v):.4f}"
                except Exception:
                    return "—"

            hovertext = [
                "📅 " + row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                + f"<br>Open: ${fmt4(row['open'])}"
                + f"<br>High: ${fmt4(row['high'])}"
                + f"<br>Low: ${fmt4(row['low'])}"
                + f"<br>Close: ${fmt4(row['close'])}"
                + f"<br>P-Up: {fmt4(row.get('p_up'))}"
                + f"<br>P-Down: {fmt4(row.get('p_down'))}"
                for _, row in vis.iterrows()
            ]

            # ---- Base candlestick
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=vis["timestamp"],
                open=vis["open"], high=vis["high"], low=vis["low"], close=vis["close"],
                name=selected_asset,
                text=hovertext,
                hoverinfo="text"
            ))

            # ---- Overlay trades for the selected asset
            asset_trades = pd.DataFrame()
            if trades_df is not None and not trades_df.empty:
                asset_trades = trades_df[trades_df["asset"] == selected_asset].copy()
                # keep only trades in the visible date range
                asset_trades = asset_trades[
                    (asset_trades["timestamp"] >= start_date) & (asset_trades["timestamp"] <= end_date)
                ].sort_values("timestamp")

            if not asset_trades.empty:
                buys = asset_trades[asset_trades["action"] == "buy"]
                sells = asset_trades[asset_trades["action"] == "sell"]

                # Helpful hover text for trades
                def trade_hover(df_):
                    h = []
                    for _, r in df_.iterrows():
                        parts = [
                            "📅 " + r["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                            f"Price: ${fmt4(r.get('price'))}",
                        ]
                        if "quantity" in r and pd.notna(r["quantity"]):
                            parts.append(f"Size: {fmt4(r['quantity'])}")
                        if "usd_value" in r and pd.notna(r["usd_value"]):
                            try:
                                parts.append(f"USD: ${float(r['usd_value']):,.2f}")
                            except Exception:
                                pass
                        if "model" in r and isinstance(r["model"], str):
                            parts.append(f"Model: {r['model']}")
                        if "asset" in r and isinstance(r["asset"], str):
                            parts.append(f"Asset: {r['asset']}")
                        h.append("<br>".join(parts))
                    return h

                if not buys.empty:
                    fig.add_trace(go.Scatter(
                        x=buys["timestamp"],
                        y=buys["price"],
                        mode="markers",
                        name="BUY",
                        marker=dict(symbol="triangle-up", size=12, line=dict(width=1, color="black")),
                        marker_color="green",
                        hoverinfo="text",
                        text=trade_hover(buys),
                    ))

                if not sells.empty:
                    fig.add_trace(go.Scatter(
                        x=sells["timestamp"],
                        y=sells["price"],
                        mode="markers",
                        name="SELL",
                        marker=dict(symbol="triangle-down", size=12, line=dict(width=1, color="black")),
                        marker_color="red",
                        hoverinfo="text",
                        text=trade_hover(sells),
                    ))
            else:
                st.info("No trades for this asset in the selected range.")

            # ---- Layout
            fig.update_layout(
                template="plotly_white",
                xaxis_rangeslider_visible=True,
                hovermode="x unified",
                yaxis_title="Price (USD)",
                title=f"{selected_asset} — Price, Probabilities & Trades",
                legend=dict(orientation="h", y=1.03, x=0.5, xanchor="center"),
            )
            st.plotly_chart(fig, use_container_width=True)

# (Optional) simple portfolio P&L view still available if you want it
if trades_df is not None and not trades_df.empty:
    st.markdown("### Portfolio P&L (Quick View)")
    buys = trades_df[trades_df["action"] == "buy"]
    sells = trades_df[trades_df["action"] == "sell"]
    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Scatter(x=trades_df["timestamp"], y=trades_df["cumulative_pnl"], mode="lines", name="Cumulative P&L"))
    fig_pnl.add_trace(go.Scatter(x=buys["timestamp"], y=buys["cumulative_pnl"], mode="markers", name="Buys"))
    fig_pnl.add_trace(go.Scatter(x=sells["timestamp"], y=sells["cumulative_pnl"], mode="markers", name="Sells"))
    fig_pnl.update_layout(template="plotly_white", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig_pnl, use_container_width=True)
