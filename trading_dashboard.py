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

# ========= Theme Management =========
def apply_theme():
    theme = st.session_state.get("theme", "light")
    if theme == "dark":
        st.markdown("""
        <style>
        .stApp { background-color: #0E1117; color: #FAFAFA; }
        .stSidebar { background-color: #1E1E1E; }
        .metric-card { background-color: #262730; padding: 1rem; border-radius: 0.5rem; border: 1px solid #3B4252; }
        .stExpander { background-color: #262730; border: 1px solid #3B4252; }
        .stSelectbox > div > div { background-color: #262730; color: #FAFAFA; }
        @media (max-width: 768px) {
            .stSidebar { width: 100% !important; }
            .main .block-container { padding-top: 2rem; padding-left: 1rem; padding-right: 1rem; }
            .stMetric { font-size: 0.8rem; }
            .stMetric > div { font-size: 0.7rem; }
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp { background-color: #FFFFFF; color: #262626; }
        .metric-card { background-color: #F8F9FA; padding: 1rem; border-radius: 0.5rem; border: 1px solid #E9ECEF; }
        @media (max-width: 768px) {
            .stSidebar { width: 100% !important; }
            .main .block-container { padding-top: 2rem; padding-left: 1rem; padding-right: 1rem; }
            .stMetric { font-size: 0.8rem; }
            .stMetric > div { font-size: 0.7rem; }
        }
        </style>
        """, unsafe_allow_html=True)

# ---- Google Drive links (update if files move) ----
TRADES_LINK = "https://drive.google.com/file/d/1sZyH06Zy9tN8vOd31JcNDM1JDayImfu3/view?usp=sharing"
MARKET_LINK = "https://drive.google.com/file/d/1JaNhwQTcYOZ-tpP_ZwHXHtNzo-GpW-TO/view?usp=drive_link"

DEFAULT_ASSET = "GIGA-USD"
REFRESH_INTERVAL = 300  # seconds

# ---- Session state keys ----
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()
if "auto_refresh_enabled" not in st.session_state:
    st.session_state.auto_refresh_enabled = True
if "theme" not in st.session_state:
    st.session_state.theme = "light"

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
# === Mirror of the bot's dynamic entry config (tune to match the bot) ===
DYNAMIC_ENTRY_UI = {
    # NOTE: bot has 0.0025 (0.25%) but comments say 0.5%.
    # Set this to whatever the bot actually uses.
    "confirmation_bounce_pct": 0.0025,   # 0.25% bounce above the post-signal low
    "timeout_minutes": 40,               # ~100 cycles at 20s ≈ 33m; use 40m buffer
    "cooldown_minutes": 10,              # avoid double-counting overlapping signals
    "match_window_minutes": 5,           # how close an OPEN must be to count as “taken”
}

def _meets_entry_signal(row, asset):
    """Entry Signal: p_up > p_down AND p_up >= buy_threshold AND conf >= min_confidence"""
    th = ASSET_THRESHOLDS.get(asset)
    if th is None:
        return False, np.nan, np.nan, np.nan
    pu = pd.to_numeric(row.get("p_up"), errors="coerce")
    pdn = pd.to_numeric(row.get("p_down"), errors="coerce")
    if pd.isna(pu) or pd.isna(pdn):
        return False, pu, pdn, np.nan
    conf = abs(pu - pdn)
    ok = (pu > pdn) and (pu >= th["buy_threshold"]) and (conf >= th["min_confidence"])
    return ok, pu, pdn, conf

def _first_confirmation_window(asset_df, start_idx, bounce_pct, timeout_td):
    """
    Given an index where the signal starts, walk forward:
      - track lowest low since signal
      - confirm when close >= low*(1 + bounce_pct)
      - stop if timeout exceeded or signal invalidated (optional)
    Returns (confirm_idx or None, watch_low_at_confirm or None)
    """
    start_ts = pd.to_datetime(asset_df.iloc[start_idx]["timestamp"], errors="coerce")
    if pd.isna(start_ts):
        return None, None
    watch_low = float(asset_df.iloc[start_idx].get("low", np.nan))
    best_low = watch_low if not pd.isna(watch_low) else np.inf

    for i in range(start_idx, len(asset_df)):
        row = asset_df.iloc[i]
        ts = pd.to_datetime(row["timestamp"], errors="coerce")
        if pd.isna(ts):
            continue
        if ts - start_ts > timeout_td:
            return None, None

        # keep updating the low while watching
        cur_low = float(row.get("low", np.nan))
        if not pd.isna(cur_low):
            best_low = min(best_low, cur_low)

        close_px = float(row.get("close", np.nan))
        if not pd.isna(close_px) and best_low not in (None, np.inf, np.nan):
            if close_px >= best_low * (1.0 + bounce_pct):
                return i, best_low

    return None, None

def identify_missed_entries(market_df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Find (asset, signal_time, confirm_time, expected_price, pu, pd, confidence) where
    entry signal + confirmation occurred, but no OPEN/BUY within +/- match_window_minutes.
    """
    if market_df is None or market_df.empty:
        return pd.DataFrame()

    # Prepare trades (only OPEN/BUY)
    opens = pd.DataFrame()
    if trades_df is not None and not trades_df.empty:
        action_col = "action" if "action" in trades_df.columns else ("unified_action" if "unified_action" in trades_df.columns else None)
        if action_col:
            mask = trades_df[action_col].astype(str).str.upper().isin(["OPEN", "BUY"])
            opens = trades_df.loc[mask].copy()
            opens["__t__"] = pd.to_datetime(opens["timestamp"], errors="coerce")
            opens["asset"] = opens["asset"].apply(unify_symbol) if "asset" in opens.columns else opens["asset"]

    bounce_pct = DYNAMIC_ENTRY_UI["confirmation_bounce_pct"]
    timeout_td = pd.Timedelta(minutes=DYNAMIC_ENTRY_UI["timeout_minutes"])
    cooldown_td = pd.Timedelta(minutes=DYNAMIC_ENTRY_UI["cooldown_minutes"])
    match_td = pd.Timedelta(minutes=DYNAMIC_ENTRY_UI["match_window_minutes"])

    out = []
    for asset, g in market_df.groupby("asset"):
        if asset not in ASSET_THRESHOLDS:
            continue
        g = g.copy().sort_values(by="timestamp", key=lambda s: pd.to_datetime(s, errors="coerce")).reset_index(drop=True)
        g["__t__"] = pd.to_datetime(g["timestamp"], errors="coerce")

        i = 0
        last_taken_ts = None
        while i < len(g):
            row = g.iloc[i]
            ok, pu, pdn, conf = _meets_entry_signal(row, asset)
            if not ok:
                i += 1
                continue

            # Found a signal start -> look for confirmation
            confirm_idx, watch_low = _first_confirmation_window(g, i, bounce_pct, timeout_td)
            if confirm_idx is None:
                # no confirmation; advance one step to look for another signal
                i += 1
                continue

            sig_ts = g.iloc[i]["__t__"]
            conf_row = g.iloc[confirm_idx]
            conf_ts = conf_row["__t__"]
            expected_px = float(conf_row.get("close", np.nan))

            # cooldown to avoid overlapping signals spamming
            if last_taken_ts is not None and conf_ts - last_taken_ts < cooldown_td:
                i = confirm_idx + 1
                continue

            # Check if an OPEN exists near confirmation time
            taken = False
            if not opens.empty:
                o = opens[opens["asset"] == asset]
                if not o.empty:
                    delta = (o["__t__"] - conf_ts).abs()
                    taken = (delta <= match_td).any()

            if not taken:
                out.append({
                    "asset": asset,
                    "signal_time": sig_ts,
                    "confirm_time": conf_ts,
                    "expected_price": expected_px,
                    "watch_low_at_confirm": watch_low,
                    "p_up_at_signal": float(pu) if pd.notna(pu) else np.nan,
                    "p_down_at_signal": float(pdn) if pd.notna(pdn) else np.nan,
                    "confidence_at_signal": float(conf) if pd.notna(conf) else np.nan,
                    "bounce_pct_used": bounce_pct,
                    "timeout_minutes_used": DYNAMIC_ENTRY_UI["timeout_minutes"],
                    "match_window_minutes": DYNAMIC_ENTRY_UI["match_window_minutes"],
                })
            else:
                # Treat as handled; set cooldown anchor
                last_taken_ts = conf_ts

            # move index forward beyond confirmation to avoid duplicates
            i = confirm_idx + 1

    missed = pd.DataFrame(out)
    if not missed.empty:
        missed = missed.sort_values("confirm_time", ascending=False)
    return missed

# === Thresholds used by the bot (keep in sync with the bot's CONFIG) ===
ASSET_THRESHOLDS = {
    "CVX-USD": {"buy_threshold": 0.70, "min_confidence": 0.60},
    "MNDE-USD": {"buy_threshold": 0.68, "min_confidence": 0.60},
    "MOG-USD": {"buy_threshold": 0.75, "min_confidence": 0.60},
    "VVV-USD": {"buy_threshold": 0.65, "min_confidence": 0.60},
    "LCX-USD": {"buy_threshold": 0.72, "min_confidence": 0.60},
    "GIGA-USD": {"buy_threshold": 0.73, "min_confidence": 0.60},
}

def _confidence_from_probs(pu, pdn):
    if pd.isna(pu) or pd.isna(pdn):
        return np.nan
    return float(abs(float(pu) - float(pdn)))

def flag_threshold_violations(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Adds per-row:
      - valid_at_open (bool): was OPEN legal at log time per thresholds?
      - violation_reason (str): why not
      - revalidated_hint (bool): heuristic tag if reason mentions re-validation
    """
    if trades.empty:
        return trades.copy()
    df = trades.copy()

    # Ensure numeric
    for col in ["p_up", "p_down", "confidence"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "confidence" not in df.columns:
        df["confidence"] = df.apply(lambda r: _confidence_from_probs(r.get("p_up"), r.get("p_down")), axis=1)

    # Determine action column
    action_col = "action" if "action" in df.columns else ("unified_action" if "unified_action" in df.columns else None)

    valid_flags, reasons = [], []
    for _, row in df.iterrows():
        # Only care about OPEN buys
        is_open = False
        if action_col:
            val = str(row.get(action_col, "")).upper()
            is_open = (val == "OPEN" or val == "BUY")
        if not is_open:
            valid_flags.append(True)
            reasons.append("")
            continue

        asset = row.get("asset")
        th = ASSET_THRESHOLDS.get(asset)
        if th is None:
            valid_flags.append(True)  # Unknown asset: don't penalize
            reasons.append("")
            continue

        pu, pdn, conf = row.get("p_up"), row.get("p_down"), row.get("confidence")
        r = []
        if pd.isna(pu) or pd.isna(pdn) or pd.isna(conf):
            r.append("missing probabilities")
        else:
            if not (pu > pdn): r.append("p_up ≤ p_down")
            if not (pu >= th["buy_threshold"]): r.append(f"p_up {pu:.3f} < buy_threshold {th['buy_threshold']:.2f}")
            if not (conf >= th["min_confidence"]): r.append(f"confidence {conf:.3f} < min_confidence {th['min_confidence']:.2f}")
        valid_flags.append(len(r) == 0)
        reasons.append("; ".join(r))

    df["valid_at_open"] = valid_flags
    df["violation_reason"] = reasons

    reason_text = df.get("reason", pd.Series([""] * len(df))).astype(str).str.lower()
    df["revalidated_hint"] = reason_text.str.contains("re-validated") | reason_text.str.contains("revalidated")

    return df

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
    position_open_times = {}
    tdf = trades_df.copy()
    tdf["__parsed_ts__"] = _parsed_ts(tdf["timestamp"])
    tdf = tdf.sort_values("__parsed_ts__").drop(columns="__parsed_ts__")

    for _, row in tdf.iterrows():
        asset = row.get("asset", "")
        action = str(row.get("unified_action", "")).lower().strip()
        qty = row.get("quantity", Decimal(0))
        price = row.get("price", Decimal(0))
        timestamp = row.get("timestamp", "")

        if asset not in positions:
            positions[asset] = {"quantity": Decimal(0), "cost": Decimal(0)}

        if action in ["buy", "open"]:
            if positions[asset]["quantity"] == 0:
                position_open_times[asset] = timestamp
            positions[asset]["cost"] += qty * price
            positions[asset]["quantity"] += qty
        elif action in ["sell", "close"]:
            if positions[asset]["quantity"] > 0:
                avg_cost = positions[asset]["cost"] / max(positions[asset]["quantity"], Decimal("1e-12"))
                sell_qty = min(qty, positions[asset]["quantity"])
                positions[asset]["cost"] -= avg_cost * sell_qty
                positions[asset]["quantity"] -= sell_qty
                if positions[asset]["quantity"] <= Decimal("1e-9"):
                    if asset in position_open_times:
                        del position_open_times[asset]

    open_positions = []
    for asset, data in positions.items():
        if data["quantity"] > Decimal("1e-9"):
            latest_asset_rows = market_df[market_df["asset"] == asset]
            if not latest_asset_rows.empty:
                idx = _parsed_ts(latest_asset_rows["timestamp"]).idxmax()
                latest_price = Decimal(str(latest_asset_rows.loc[idx, "close"]))
                avg_entry = data["cost"] / data["quantity"]
                current_value = latest_price * data["quantity"]
                unrealized = current_value - data["cost"]

                open_time_str = "N/A"
                if asset in position_open_times:
                    try:
                        open_ts = pd.to_datetime(position_open_times[asset], errors="coerce")
                        if pd.notna(open_ts):
                            open_time_str = open_ts.strftime("%H:%M")
                    except:
                        pass

                open_positions.append(
                    {
                        "Asset": asset,
                        "Quantity": float(data["quantity"]),
                        "Avg. Entry Price": float(avg_entry),
                        "Current Price": float(latest_price),
                        "Current Value ($)": float(current_value),
                        "Unrealized P&L ($)": float(unrealized),
                        "Open Time": open_time_str,
                    }
                )

    return pd.DataFrame(open_positions)

def analyze_trade_performance_by_reason(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze trading performance by signal/reason"""
    if trades_df is None or trades_df.empty or "reason" not in trades_df.columns:
        return pd.DataFrame()

    matched_df, _ = match_trades_fifo(trades_df)
    if matched_df.empty:
        return pd.DataFrame()

    performance_data = []

    if "Reason Buy" in matched_df.columns:
        buy_analysis = matched_df.groupby("Reason Buy").agg({
            "P&L ($)": ["count", "sum", "mean", lambda x: (x > 0).sum()],
            "P&L %": "mean"
        }).round(4)
        buy_analysis.columns = ["Total Trades", "Total P&L ($)", "Avg P&L ($)", "Wins", "Avg P&L (%)"]
        buy_analysis["Win Rate (%)"] = (buy_analysis["Wins"] / buy_analysis["Total Trades"] * 100).round(1)
        buy_analysis["Signal Type"] = "Buy Signal"
        buy_analysis["Reason"] = buy_analysis.index
        buy_analysis = buy_analysis.reset_index(drop=True)
        performance_data.append(buy_analysis)

    if "Reason Sell" in matched_df.columns:
        sell_analysis = matched_df.groupby("Reason Sell").agg({
            "P&L ($)": ["count", "sum", "mean", lambda x: (x > 0).sum()],
            "P&L %": "mean"
        }).round(4)
        sell_analysis.columns = ["Total Trades", "Total P&L ($)", "Avg P&L ($)", "Wins", "Avg P&L (%)"]
        sell_analysis["Win Rate (%)"] = (sell_analysis["Wins"] / sell_analysis["Total Trades"] * 100).round(1)
        sell_analysis["Signal Type"] = "Sell Signal"
        sell_analysis["Reason"] = sell_analysis.index
        sell_analysis = sell_analysis.reset_index(drop=True)
        performance_data.append(sell_analysis)

    if performance_data:
        result = pd.concat(performance_data, ignore_index=True)
        result = result[["Signal Type", "Reason", "Total Trades", "Win Rate (%)",
                        "Total P&L ($)", "Avg P&L ($)", "Avg P&L (%)"]]
        return result.sort_values(["Signal Type", "Total P&L ($)"], ascending=[True, False])

    return pd.DataFrame()

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

# Apply theme
apply_theme()

trades_df, market_df = load_data(TRADES_LINK, MARKET_LINK)
elapsed = maybe_auto_refresh()

# >>> Add validation flags to trades <<<
trades_df = flag_threshold_violations(trades_df) if not trades_df.empty else trades_df

# ========= FIXED Debug panel =========
with st.expander("🔎 Data Freshness Debug", expanded=True):
    # Trades
    if not trades_df.empty and "timestamp" in trades_df.columns:
        trades_debug = trades_df.copy()
        trades_debug["__parsed_ts__"] = _parsed_ts(trades_debug["timestamp"])
        trades_debug_valid = trades_debug.dropna(subset=["__parsed_ts__"])
        if not trades_debug_valid.empty:
            trades_debug_sorted = trades_debug_valid.sort_values("__parsed_ts__")
            latest_idx = trades_debug_sorted.index[-1]
            tr_raw_latest = trades_debug_sorted.loc[latest_idx, "timestamp"]
            tr_parsed_latest = trades_debug_sorted.loc[latest_idx, "__parsed_ts__"]
            st.write(f"**Latest Trade Timestamp (raw):** {tr_raw_latest}")
            st.write(f"**Latest Trade Timestamp (parsed):** {tr_parsed_latest}")
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
        market_debug = market_df.copy()
        market_debug["__parsed_ts__"] = _parsed_ts(market_debug["timestamp"])
        market_debug_valid = market_debug.dropna(subset=["__parsed_ts__"])
        if not market_debug_valid.empty:
            market_debug_sorted = market_debug_valid.sort_values("__parsed_ts__")
            latest_idx = market_debug_sorted.index[-1]
            mk_raw_latest = market_debug_sorted.loc[latest_idx, "timestamp"]
            mk_parsed_latest = market_debug_sorted.loc[latest_idx, "__parsed_ts__"]
            st.write(f"**Latest Market Timestamp (raw):** {mk_raw_latest}")
            st.write(f"**Latest Market Timestamp (parsed):** {mk_parsed_latest}")
            st.write(f"**Market timestamp dtype:** {market_df['timestamp'].dtype}")
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
        sample_timestamps = trades_df["timestamp"].dropna().head(5).tolist()
        st.write("**Sample timestamp formats:**")
        for i, ts in enumerate(sample_timestamps):
            st.write(f"{i+1}. `{ts}` (type: {type(ts).__name__})")
        parsed_ts = _parsed_ts(trades_df["timestamp"])
        valid_parsed = parsed_ts.dropna()
        if len(valid_parsed) > 0:
            is_sorted = valid_parsed.is_monotonic_increasing
            st.write(f"**Data chronologically sorted:** {'✅ Yes' if is_sorted else '❌ No'}")
            st.write(f"**Timestamp range:** {valid_parsed.min()} to {valid_parsed.max()}")
    if not market_df.empty and "timestamp" in market_df.columns:
        st.write("**Last 10 Market Timestamps (raw order):**")
        st.dataframe(market_df[["timestamp", "asset"]].tail(10))
        for asset in market_df["asset"].unique()[:3]:
            asset_data = market_df[market_df["asset"] == asset]
            parsed_ts = _parsed_ts(asset_data["timestamp"])
            valid_parsed = parsed_ts.dropna()
            if len(valid_parsed) > 0:
                is_sorted = valid_parsed.is_monotonic_increasing
                st.write(f"**{asset} chronologically sorted:** {'✅ Yes' if is_sorted else '❌ No'}")

# ========= Sidebar =========
with st.sidebar:
    st.markdown("<h1 style='text-align:center;'>Crypto Strategy</h1>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("🌙" if st.session_state.theme == "light" else "☀️", help="Toggle theme"):
            st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
            st.rerun()
    with col2:
        st.session_state.auto_refresh_enabled = st.toggle(
            "🔄", value=st.session_state.get("auto_refresh_enabled", True), help="Auto-Refresh (5min)"
        )
    with col3:
        if st.button("↻", help="Force refresh"):
            st.cache_data.clear()
            st.session_state.last_refresh = time.time()
            st.rerun()

    if st.session_state.get("auto_refresh_enabled", True):
        elapsed_m, elapsed_s = divmod(elapsed, 60)
        remaining = max(0, REFRESH_INTERVAL - elapsed)
        if remaining > 0:
            remaining_m, remaining_s = divmod(remaining, 60)
            st.caption(f"⏱️ Last refresh: {elapsed_m:02d}:{elapsed_s:02d} ago")
            st.caption(f"🔄 Next refresh: ~{remaining_m:02d}:{remaining_s:02d}")
        else:
            st.caption(f"🔄 Refreshing now...")
    else:
        elapsed_m, elapsed_s = divmod(elapsed, 60)
        st.caption(f"⏸️ Auto-refresh disabled")
        st.caption(f"Last refresh: {elapsed_m:02d}:{elapsed_s:02d} ago")

    st.markdown("---")
    st.markdown(f"**Trades:** {'✅' if not trades_df.empty else '⚠️'} {len(trades_df):,}")
    st.markdown(f"**Market:** {'✅' if not market_df.empty else '❌'} {len(market_df):,}")
    if not market_df.empty and "asset" in market_df.columns:
        st.markdown(f"**Assets:** {market_df['asset'].nunique():,}")

    st.markdown("---")
    open_positions_df = calculate_open_positions(trades_df, market_df) if not trades_df.empty else pd.DataFrame()
    if not open_positions_df.empty:
        st.markdown("**📊 Open Positions**")
        sorted_positions = open_positions_df.sort_values("Unrealized P&L ($)", ascending=False)
        for _, pos in sorted_positions.iterrows():
            pnl = pos["Unrealized P&L ($)"]
            pnl_pct = ((pos["Current Price"] - pos["Avg. Entry Price"]) / pos["Avg. Entry Price"] * 100) if pos["Avg. Entry Price"] != 0 else 0
            if pnl >= 0:
                color = "#16a34a"; pnl_icon = "📈"
            else:
                color = "#ef4444"; pnl_icon = "📉"
            asset_name = pos["Asset"]
            cur_price = pos["Current Price"]
            pf = ".8f" if cur_price < 0.001 else ".6f" if cur_price < 1 else ".4f"
            avg_price = pos["Avg. Entry Price"]
            epf = ".8f" if avg_price < 0.001 else ".6f" if avg_price < 1 else ".4f"
            open_time = pos.get("Open Time", "N/A")
            quantity = pos["Quantity"]
            card_bg = "rgba(22,163,74,0.1)" if pnl >= 0 else "rgba(239,68,68,0.1)"
            st.markdown(
                f"""
                <div style="background-color: {card_bg}; border-left: 4px solid {color}; padding: 12px; margin: 8px 0; border-radius: 4px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <strong style="font-size: 14px;">{asset_name}</strong>
                        <span style="color: {color}; font-weight: bold; font-size: 13px;">{pnl_icon} {pnl_pct:+.1f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span style="font-size: 11px; color: #888; background: rgba(128,128,128,0.1); padding: 2px 6px; border-radius: 3px;">⏰ {open_time}</span>
                        <span style="font-size: 12px; color: #666;">Qty: {quantity:.4f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                        <span style="font-size: 12px; color: #666;">Current: <strong>${cur_price:{pf}}</strong></span>
                        <span style="font-size: 12px; color: #666;">Entry: <strong>${avg_price:{epf}}</strong></span>
                    </div>
                    <div style="text-align: center; padding-top: 5px; border-top: 1px solid rgba(128,128,128,0.2);">
                        <span style="color: {color}; font-weight: bold; font-size: 13px;">P&L: ${pnl:+.4f}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        total_pnl = sorted_positions["Unrealized P&L ($)"].sum()
        total_value = sorted_positions["Current Value ($)"].sum()
        winners = len(sorted_positions[sorted_positions["Unrealized P&L ($)"] > 0])
        total_positions = len(sorted_positions)
        st.markdown("---")
        st.markdown(
            f"""
            <div style='text-align: center; padding: 0.5rem; background-color: rgba(128,128,128,0.1); border-radius: 0.25rem; margin-top: 0.5rem;'>
                <div style='font-size: 0.8rem; color: {"#16a34a" if total_pnl >= 0 else "#ef4444"}; font-weight: 600;'>
                    Total P&L: ${total_pnl:+.2f}
                </div>
                <div style='font-size: 0.7rem; color: #666; margin-top: 0.25rem;'>
                    {winners}/{total_positions} winning | ${total_value:.2f} total value
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown("**📊 Open Positions**")
        st.markdown(
            """
            <div style='text-align: center; padding: 1rem; background-color: rgba(128,128,128,0.05); 
                       border-radius: 0.25rem; color: #666; font-style: italic;'>
                No open positions
            </div>
            """,
            unsafe_allow_html=True
        )

# ========= Tabs =========
tab1, tab2, tab3, tab4 = st.tabs(["📈 Price & Trades", "💰 P&L Analysis", "📜 Trade History", "🚦 Validations"])

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
            asset_market_sorted = asset_market.sort_values(by="timestamp", key=lambda s: _parsed_ts(s))
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

                    # Buy/Sell markers from trades (with validation info)
                    if not trades_df.empty:
                        asset_trades = trades_df[trades_df["asset"] == selected_asset].copy()
                        if not asset_trades.empty and "timestamp" in asset_trades.columns:
                            t_parsed = _parsed_ts(asset_trades["timestamp"])
                            mask = (t_parsed >= start_parsed) & (t_parsed <= end_parsed)
                            asset_trades = asset_trades.loc[mask].copy()

                            if not asset_trades.empty:
                                marker_y = y_min + (vis["high"].max() - vis["low"].min()) * 0.02

                                # BUY markers
                                buys = asset_trades[asset_trades["unified_action"].str.lower().isin(["buy", "open"])].copy()
                                if not buys.empty:
                                    buy_prices = buys["price"].apply(float)
                                    buy_reasons = buys.get("reason", pd.Series([""] * len(buys))).fillna("")

                                    # validation fields for hover + coloring
                                    if "valid_at_open" not in buys.columns:
                                        buys["valid_at_open"] = True
                                    if "violation_reason" not in buys.columns:
                                        buys["violation_reason"] = ""
                                    if "revalidated_hint" not in buys.columns:
                                        buys["revalidated_hint"] = False

                                    buy_valid = buys["valid_at_open"].fillna(True)
                                    buy_viols = buys["violation_reason"].fillna("")
                                    buy_reval = buys["revalidated_hint"].fillna(False)

                                    # colors: green if valid, amber if invalid
                                    buy_colors = np.where(buy_valid, "#4caf50", "#f59e0b")
                                    buy_valid_str = buy_valid.map(lambda x: "Yes" if x else "No")
                                    buy_reval_str = buy_reval.map(lambda x: "Yes" if x else "")
                                    buy_viol_str = buy_viols

                                    fig.add_trace(
                                        go.Scatter(
                                            x=_parsed_ts(buys["timestamp"]),
                                            y=[marker_y] * len(buys),
                                            mode="markers",
                                            name="BUY",
                                            marker=dict(symbol="triangle-up", size=14, color=buy_colors, line=dict(width=1, color="white")),
                                            customdata=np.stack((buy_prices, buy_reasons, buy_valid_str, buy_reval_str, buy_viol_str), axis=-1),
                                            hovertemplate="<b>BUY</b> @ $%{customdata[0]:.8f}<br>%{x|%H:%M:%S}"
                                                          "<br>Reason: %{customdata[1]}"
                                                          "<br>Valid at OPEN: %{customdata[2]}"
                                                          "<br>%{customdata[3]}"  # Re-validated line (empty if not)
                                                          "<br>%{customdata[4]}"  # Violation reason (empty if none)
                                                          "<extra></extra>",
                                        )
                                    )

                                # SELL markers
                                sells = asset_trades[asset_trades["unified_action"].str.lower().isin(["sell", "close"])].copy()
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
                        template="plotly_dark" if st.session_state.theme == "dark" else "plotly_white",
                        xaxis_rangeslider_visible=False,
                        xaxis=dict(
                            title="Time",
                            type="date",
                            tickformat=tick_format,
                            showgrid=True,
                            gridcolor="rgba(59,66,82,0.3)" if st.session_state.theme == "dark" else "rgba(128,128,128,0.1)",
                            tickangle=-45,
                        ),
                        yaxis=dict(
                            title="Price (USD)",
                            tickformat=".8f" if last_price < 0.001 else ".6f" if last_price < 1 else ".4f",
                            showgrid=True,
                            gridcolor="rgba(59,66,82,0.3)" if st.session_state.theme == "dark" else "rgba(128,128,128,0.1)",
                            range=[y_min, y_max],
                        ),
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                        height=750,
                        margin=dict(l=60, r=20, t=80, b=80),
                        plot_bgcolor="rgba(14,17,23,1)" if st.session_state.theme == "dark" else "rgba(250,250,250,0.8)",
                        paper_bgcolor="rgba(14,17,23,1)" if st.session_state.theme == "dark" else "rgba(255,255,255,1)",
                        font_color="#FAFAFA" if st.session_state.theme == "dark" else "#262626",
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

                    buy_orders = 0
                    sell_orders = 0
                    if not asset_trades_period.empty:
                        buy_orders = len(asset_trades_period[asset_trades_period["unified_action"].str.lower().isin(["buy", "open"])])
                        sell_orders = len(asset_trades_period[asset_trades_period["unified_action"].str.lower().isin(["sell", "close"])])

                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("Trade Orders", len(asset_trades_period))
                    with c2:
                        st.metric("Buy Orders", buy_orders)
                    with c3:
                        st.metric("Sell Orders", sell_orders)
                    with c4:
                        if "pnl" in asset_trades_period.columns:
                            period_pnl = asset_trades_period["pnl"].sum()
                            st.metric("Period P&L", f"${period_pnl:.6f}")
                        else:
                            st.metric("Period P&L", "N/A")

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

            chart_bg = "rgba(14,17,23,1)" if st.session_state.theme == "dark" else "rgba(255,255,255,1)"
            chart_grid = "rgba(59,66,82,0.3)" if st.session_state.theme == "dark" else "rgba(128,128,128,0.1)"
            chart_text = "#FAFAFA" if st.session_state.theme == "dark" else "#262626"

            fig_pnl.update_layout(
                title="Total Portfolio P&L",
                template="plotly_dark" if st.session_state.theme == "dark" else "plotly_white",
                yaxis_title="P&L (USD)",
                xaxis_title="Date",
                hovermode="x unified",
                plot_bgcolor=chart_bg,
                paper_bgcolor=chart_bg,
                font_color=chart_text,
                xaxis=dict(gridcolor=chart_grid),
                yaxis=dict(gridcolor=chart_grid),
            )
            st.plotly_chart(fig_pnl, use_container_width=True)

        st.markdown("---")
        col_left, col_right = st.columns([1, 1])
        with col_left:
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
        with col_right:
            st.markdown("### Signal Performance Analysis")
            attribution_df = analyze_trade_performance_by_reason(trades_df)
            if not attribution_df.empty:
                st.dataframe(
                    attribution_df,
                    column_config={
                        "Signal Type": st.column_config.TextColumn(width="small"),
                        "Reason": st.column_config.TextColumn(width="medium"),
                        "Total Trades": st.column_config.NumberColumn(width="small"),
                        "Win Rate (%)": st.column_config.NumberColumn(format="%.1f%%", width="small"),
                        "Total P&L ($)": st.column_config.NumberColumn(format="$%.4f"),
                        "Avg P&L ($)": st.column_config.NumberColumn(format="$%.4f"),
                        "Avg P&L (%)": st.column_config.NumberColumn(format="%.2f%%"),
                    },
                    use_container_width=True,
                    hide_index=True,
                )
                if len(attribution_df) > 0:
                    best_signal = attribution_df.loc[attribution_df["Total P&L ($)"].idxmax()]
                    worst_signal = attribution_df.loc[attribution_df["Total P&L ($)"].idxmin()]
                    st.markdown("**Top Performer:** " +
                                f"{best_signal['Reason']} ({best_signal['Signal Type']}) - " +
                                f"${best_signal['Total P&L ($)']:.2f} / {best_signal['Win Rate (%)']:.1f}% WR")
                    if len(attribution_df) > 1:
                        st.markdown("**Worst Performer:** " +
                                    f"{worst_signal['Reason']} ({worst_signal['Signal Type']}) - " +
                                    f"${worst_signal['Total P&L ($)']:.2f} / {worst_signal['Win Rate (%)']:.1f}% WR")
            else:
                st.info("No signal attribution data available. Make sure your trades have 'reason' field populated.")

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
                    ["Asset","Quantity","Buy Time","Buy Price","Sell Time","Sell Price","Hold Time","P&L ($)","P&L %"]
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
            display_open["Time"] = pd.to_datetime(display_open["timestamp"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
            display_open = display_open.rename(columns={"asset": "Asset", "quantity": "Quantity", "price": "Price", "reason": "Reason"})
            st.dataframe(
                display_open[["Time","Asset","Quantity","Price","Reason"]],
                column_config={
                    "Asset": st.column_config.TextColumn(width="small"),
                    "Quantity": st.column_config.NumberColumn(format="%.4f", width="small"),
                    "Price": st.column_config.NumberColumn(format="$%.8f"),
                },
                use_container_width=True,
                hide_index=True,
            )
            st.info("Note: validity checks apply to OPEN actions. See the '🚦 Validations' tab for flagged entries.")
        else:
            st.info("No open positions.")

# ----- TAB 4: Validations -----
with tab4:
    st.markdown("### Threshold Validations at Execution Time (OPEN orders)")
    if trades_df.empty:
        st.info("No trades loaded.")
    else:
        # Determine which column indicates action
        action_col = "action" if "action" in trades_df.columns else ("unified_action" if "unified_action" in trades_df.columns else None)
        if action_col:
            opens_mask = trades_df[action_col].astype(str).str.upper().isin(["OPEN", "BUY"])
        else:
            opens_mask = pd.Series([False] * len(trades_df), index=trades_df.index)

        opens = trades_df.loc[opens_mask].copy()
        if opens.empty:
            st.info("No OPEN orders in the log.")
        else:
            total_opens = len(opens)
            invalid = opens[~opens["valid_at_open"]]
            valid = opens[opens["valid_at_open"]]

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("OPEN Orders", f"{total_opens:,}")
            with c2:
                st.metric("Valid at OPEN", f"{len(valid):,}")
            with c3:
                st.metric("Flagged (Invalid)", f"{len(invalid):,}")

            if not invalid.empty:
                show_cols = ["timestamp","asset","price","p_up","p_down","confidence","reason","violation_reason"]
                show_cols = [c for c in show_cols if c in invalid.columns]
                st.markdown("#### ❗ Flagged OPENs")
                st.dataframe(
                    invalid[show_cols].sort_values("timestamp"),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.success("No violations detected 🎉")

            st.markdown("---")
            st.caption("Open = buy execution records; validity is evaluated using the thresholds configured for each asset and the probabilities logged with the trade at that time.")

