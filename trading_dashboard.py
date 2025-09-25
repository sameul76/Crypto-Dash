# streamlit_app_pst.py
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

# ---- Google Drive links ----
TRADES_LINK = "https://drive.google.com/file/d/1GDBkqqjPTER8rBCfD82bIYXsN8iVng2f/view?usp=drive_link"
MARKET_LINK = "https://drive.google.com/file/d/1PGHNRVyAPPp9U_2MXxxAU_jkDu7pi_1f/view?usp=sharing"

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
    m = re.search(r"/d/([A-Za-z0-9_-]{20,})", s);  m = m or re.search(r"[?&]id=([A-Za-z0-9_-]{20,})", s)
    return (m.group(1) if m else s)

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
            st.error(f"{label}: not a Parquet file and CSV fallback failed.")
            return pd.DataFrame()
    try:
        return pd.read_parquet(io.BytesIO(b))
    except Exception as e:
        st.error(f"{label}: failed to read Parquet: {e}")
        return pd.DataFrame()

def lower_strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(); out.columns = [c.lower().strip() for c in out.columns]; return out

def unify_symbol(val: str) -> str:
    if not isinstance(val, str): return val
    s = val.strip().upper().replace("_", "-")
    if "GIGA" in s: return "GIGA-USD"
    return s

def normalize_prob_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    p_up_variations = {"p_up", "p-up", "pup", "pup prob", "p up"}
    p_down_variations = {"p_down", "p-down", "pdown", "pdown prob", "p down"}
    for col in df.columns:
        norm = col.lower().replace("_", " ").replace("-", " ")
        if norm in p_up_variations: rename_map[col] = "p_up"
        elif norm in p_down_variations: rename_map[col] = "p_down"
    if rename_map: df = df.rename(columns=rename_map)
    if "p_up" not in df.columns: df["p_up"] = np.nan
    if "p_down" not in df.columns: df["p_down"] = np.nan
    return df

# Parse to pandas Timestamps (may be tz-aware or naive)
def _parsed_ts(s: pd.Series) -> pd.Series:
    # raw parse, no forced timezone
    return pd.to_datetime(s, errors="coerce")


# tz-aware UTC (used internally for comparisons if needed)
def _series_to_utc(s: pd.Series) -> pd.Series:
    """
    Logic/comparison helper: interpret NAIVE timestamps as PST first, then convert to UTC.
    (Previously we assumed naive==UTC, which shifted PST data by 8 hours.)
    """
    ts = pd.to_datetime(s, errors="coerce")
    try:
        if getattr(ts.dt, "tz", None) is None:
            # naive PST -> UTC
            return ts.dt.tz_localize("America/Los_Angeles").dt.tz_convert("UTC")
        else:
            # aware -> UTC
            return ts.dt.tz_convert("UTC")
    except Exception:
        return ts  # last resort

# === NEW: tz-aware PST for all UI displays ===
def _series_to_pst(s: pd.Series) -> pd.Series:
    """
    Display helper: interpret NAIVE timestamps as PST, and convert aware timestamps to PST.
    This matches your fetcher, which saves timestamps as NAIVE PST.
    """
    ts = pd.to_datetime(s, errors="coerce")
    try:
        if getattr(ts.dt, "tz", None) is None:
            # naive -> treat as PST
            return ts.dt.tz_localize("America/Los_Angeles")
        else:
            # aware -> convert to PST
            return ts.dt.tz_convert("America/Los_Angeles")
    except Exception:
        return ts  # last resort
        
def _pick_ts_pst(df: pd.DataFrame, fallback_col: str = "timestamp") -> pd.Series:
    if "timestamp_pst" in df.columns:
        return _series_to_pst(df["timestamp_pst"])
    return _series_to_pst(df[fallback_col]) if fallback_col in df.columns else pd.Series([], dtype="datetime64[ns, America/Los_Angeles]")

def seconds_since_last_run() -> int:
    return int(time.time() - st.session_state.get("last_refresh", 0))

def maybe_auto_refresh() -> int:
    elapsed = seconds_since_last_run()
    if st.session_state.get("auto_refresh_enabled", False) and elapsed >= REFRESH_INTERVAL:
        st.session_state.last_refresh = time.time()
        st.cache_data.clear()
        st.rerun()
    return elapsed

# === Thresholds used by the bot (keep in sync with bot CONFIG) ===
ASSET_THRESHOLDS = {
    "CVX-USD": {"buy_threshold": 0.70, "min_confidence": 0.60, "sell_threshold": 0.30},
    "MNDE-USD": {"buy_threshold": 0.68, "min_confidence": 0.60, "sell_threshold": 0.32},
    "MOG-USD": {"buy_threshold": 0.75, "min_confidence": 0.60, "sell_threshold": 0.25},
    "VVV-USD": {"buy_threshold": 0.65, "min_confidence": 0.60, "sell_threshold": 0.35},
    "LCX-USD": {"buy_threshold": 0.72, "min_confidence": 0.60, "sell_threshold": 0.28},
    "GIGA-USD": {"buy_threshold": 0.73, "min_confidence": 0.60, "sell_threshold": 0.27},
}

def _confidence_from_probs(pu, pdn):
    if pd.isna(pu) or pd.isna(pdn): return np.nan
    return float(abs(float(pu) - float(pdn)))

# NEW: Position tracking function
def track_positions_over_time(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Track position quantities over time for each asset.
    Returns DataFrame with columns: timestamp, asset, position_qty
    """
    if trades_df.empty:
        return pd.DataFrame(columns=['timestamp', 'asset', 'position_qty'])
    
    position_history = []
    
    for asset in trades_df['asset'].unique():
        asset_trades = trades_df[trades_df['asset'] == asset].copy()
        asset_trades['__parsed_ts__'] = _parsed_ts(asset_trades['timestamp'])
        asset_trades = asset_trades.sort_values('__parsed_ts__')
        
        current_qty = Decimal(0)
        
        for _, trade in asset_trades.iterrows():
            action = str(trade.get('unified_action', '')).lower().strip()
            qty = trade.get('quantity', Decimal(0))
            
            if action in ['buy', 'open']:
                current_qty += qty
            elif action in ['sell', 'close']:
                current_qty -= qty
                current_qty = max(current_qty, Decimal(0))  # Can't go negative
            
            position_history.append({
                'timestamp': trade['timestamp'],
                'asset': asset,
                'position_qty': float(current_qty)
            })
    
    return pd.DataFrame(position_history)

# NEW: Get position status at specific timestamp
def get_position_at_time(position_history: pd.DataFrame, asset: str, timestamp: pd.Timestamp) -> float:
    """Get position quantity for an asset at a specific timestamp."""
    if position_history.empty:
        return 0.0
    
    asset_positions = position_history[position_history['asset'] == asset].copy()
    if asset_positions.empty:
        return 0.0
    
    asset_positions['__parsed_ts__'] = _series_to_utc(asset_positions['timestamp'])
    asset_positions = asset_positions.sort_values('__parsed_ts__')
    
    # Find the last position update before or at the given timestamp
    mask = asset_positions['__parsed_ts__'] <= timestamp
    if mask.any():
        return asset_positions.loc[mask, 'position_qty'].iloc[-1]
    else:
        return 0.0

# NEW: Identify missed buy signals (when in position)
def identify_missed_buys(market_df: pd.DataFrame, trades_df: pd.DataFrame, position_history: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Find missed buy opportunities when in position (for adding to position)."""
    if market_df is None or market_df.empty:
        return pd.DataFrame()
    
    missed_buys = []
    match_td = pd.Timedelta(minutes=int(cfg["match_window_minutes"]))
    
    # Get executed trades for comparison
    executed_buys = pd.DataFrame()
    if not trades_df.empty:
        action_col = "action" if "action" in trades_df.columns else "unified_action"
        buy_mask = trades_df[action_col].astype(str).str.upper().isin(["OPEN", "BUY"])
        executed_buys = trades_df.loc[buy_mask].copy()
        if not executed_buys.empty:
            executed_buys["__t__"] = _series_to_utc(executed_buys["timestamp"])
    
    for asset in market_df['asset'].unique():
        if asset not in ASSET_THRESHOLDS:
            continue
        
        asset_data = market_df[market_df['asset'] == asset].copy()
        asset_data['__t__'] = _series_to_utc(asset_data['timestamp'])
        asset_data = asset_data.sort_values('__t__')
        
        th = ASSET_THRESHOLDS[asset]
        
        for _, row in asset_data.iterrows():
            # Check if this is a buy signal
            pu = pd.to_numeric(row.get("p_up"), errors="coerce")
            pdn = pd.to_numeric(row.get("p_down"), errors="coerce")
            
            if pd.isna(pu) or pd.isna(pdn):
                continue
            
            conf = abs(pu - pdn)
            is_buy_signal = (pu > pdn) and (pu >= th["buy_threshold"]) and (conf >= th["min_confidence"])
            
            if not is_buy_signal:
                continue
            
            # Check if we were in a position at this time
            signal_time = row['__t__']
            position_qty = get_position_at_time(position_history, asset, signal_time)
            
            if position_qty <= 0:  # Not in position, don't mark as missed buy
                continue
            
            # Check if there was an executed buy within the match window
            executed = False
            if not executed_buys.empty:
                asset_buys = executed_buys[executed_buys['asset'] == asset]
                if not asset_buys.empty:
                    time_diffs = (asset_buys['__t__'] - signal_time).abs()
                    executed = (time_diffs <= match_td).any()
            
            if not executed:
                missed_buys.append({
                    'asset': asset,
                    'timestamp': row['timestamp'],  # keep original; display layer will convert to PST
                    'price': float(row.get('close', 0)),
                    'p_up': float(pu),
                    'p_down': float(pdn),
                    'confidence': float(conf),
                    'signal_type': 'missed_buy'
                })
    
    return pd.DataFrame(missed_buys)

# NEW: Identify missed sell signals (when in position)
def identify_missed_sells(market_df: pd.DataFrame, trades_df: pd.DataFrame, position_history: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Find missed sell opportunities when in position."""
    if market_df is None or market_df.empty:
        return pd.DataFrame()
    
    missed_sells = []
    match_td = pd.Timedelta(minutes=int(cfg["match_window_minutes"]))
    
    # Get executed sells for comparison
    executed_sells = pd.DataFrame()
    if not trades_df.empty:
        action_col = "action" if "action" in trades_df.columns else "unified_action"
        sell_mask = trades_df[action_col].astype(str).str.upper().isin(["CLOSE", "SELL"])
        executed_sells = trades_df.loc[sell_mask].copy()
        if not executed_sells.empty:
            executed_sells["__t__"] = _series_to_utc(executed_sells["timestamp"])
    
    for asset in market_df['asset'].unique():
        if asset not in ASSET_THRESHOLDS:
            continue
        
        asset_data = market_df[market_df['asset'] == asset].copy()
        asset_data['__t__'] = _series_to_utc(asset_data['timestamp'])
        asset_data = asset_data.sort_values('__t__')
        
        th = ASSET_THRESHOLDS[asset]
        
        for _, row in asset_data.iterrows():
            # Check if this is a sell signal
            pu = pd.to_numeric(row.get("p_up"), errors="coerce")
            pdn = pd.to_numeric(row.get("p_down"), errors="coerce")
            
            if pd.isna(pu) or pd.isna(pdn):
                continue
            
            conf = abs(pu - pdn)
            is_sell_signal = (pdn > pu) and (pdn >= th["sell_threshold"]) and (conf >= th["min_confidence"])
            
            if not is_sell_signal:
                continue
            
            # Check if we were in a position at this time
            signal_time = row['__t__']
            position_qty = get_position_at_time(position_history, asset, signal_time)
            
            if position_qty <= 0:  # Not in position, don't mark as missed sell
                continue
            
            # Check if there was an executed sell within the match window
            executed = False
            if not executed_sells.empty:
                asset_sells = executed_sells[executed_sells['asset'] == asset]
                if not asset_sells.empty:
                    time_diffs = (asset_sells['__t__'] - signal_time).abs()
                    executed = (time_diffs <= match_td).any()
            
            if not executed:
                missed_sells.append({
                    'asset': asset,
                    'timestamp': row['timestamp'],  # keep original; display layer will convert to PST
                    'price': float(row.get('close', 0)),
                    'p_up': float(pu),
                    'p_down': float(pdn),
                    'confidence': float(conf),
                    'signal_type': 'missed_sell'
                })
    
    return pd.DataFrame(missed_sells)

def flag_threshold_violations(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty: return trades.copy()
    df = trades.copy()
    for col in ["p_up", "p_down", "confidence"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "confidence" not in df.columns:
        df["confidence"] = df.apply(lambda r: _confidence_from_probs(r.get("p_up"), r.get("p_down")), axis=1)
    action_col = "action" if "action" in df.columns else ("unified_action" if "unified_action" in df.columns else None)

    val_flags, reasons = [], []
    for _, row in df.iterrows():
        is_open = False
        if action_col:
            val = str(row.get(action_col, "")).upper()
            is_open = (val == "OPEN" or val == "BUY")
        if not is_open:
            val_flags.append(True); reasons.append(""); continue

        asset = row.get("asset"); th = ASSET_THRESHOLDS.get(asset)
        if th is None:
            val_flags.append(True); reasons.append(""); continue
        pu, pdn, conf = row.get("p_up"), row.get("p_down"), row.get("confidence")
        r = []
        if pd.isna(pu) or pd.isna(pdn) or pd.isna(conf):
            r.append("missing probabilities")
        else:
            if not (pu > pdn): r.append("p_up ≤ p_down")
            if not (pu >= th["buy_threshold"]): r.append(f"p_up {pu:.3f} < buy_threshold {th['buy_threshold']:.2f}")
            if not (conf >= th["min_confidence"]): r.append(f"confidence {conf:.3f} < min_confidence {th['min_confidence']:.2f}")
        val_flags.append(len(r) == 0); reasons.append("; ".join(r))
    df["valid_at_open"] = val_flags
    df["violation_reason"] = reasons
    reason_text = df.get("reason", pd.Series([""] * len(df))).astype(str).str.lower()
    df["revalidated_hint"] = reason_text.str.contains("re-validated") | reason_text.str.contains("revalidated")
    return df

# === Dynamic entry config (for what-if analysis) ===
DYNAMIC_ENTRY_UI = {
    "confirmation_bounce_pct": 0.0025,   # 0.25% bounce
    "timeout_minutes": 40,               # ~100 cycles @ 20s ≈ 33m; buffer
    "cooldown_minutes": 10,
    "match_window_minutes": 5,
}

# ========= Data loading (cached) =========
@st.cache_data(ttl=60)
def load_data(trades_link: str, market_link: str):
    trades = pd.DataFrame();  tb = _download_drive_bytes(trades_link)
    if tb: trades = _read_parquet_or_csv(tb, "Trades")

    market = pd.DataFrame();  mb = _download_drive_bytes(market_link)
    if mb: market = _read_parquet_or_csv(mb, "Market")

    if not trades.empty:
        trades = lower_strip_cols(trades)
        colmap = {}
        if "value" in trades.columns: colmap["value"] = "usd_value"
        if "side" in trades.columns and "action" in trades.columns: colmap["side"] = "trade_direction"
        elif "side" in trades.columns and "action" not in trades.columns: colmap["side"] = "action"
        trades = trades.rename(columns=colmap)

        if "action" in trades.columns:
            trades["unified_action"] = (
                trades["action"].astype(str).str.upper().map({"OPEN": "buy", "CLOSE": "sell"})
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

        for col in ["quantity", "price", "usd_value", "pnl", "pnl_pct"]:
            if col in trades.columns:
                trades[col] = trades[col].apply(lambda x: Decimal(str(x)) if pd.notna(x) else Decimal(0))

    if not market.empty:
        market = lower_strip_cols(market)
        if "product_id" in market.columns: market = market.rename(columns={"product_id": "asset"})
        if "asset" in market.columns: market["asset"] = market["asset"].apply(unify_symbol)
        market = normalize_prob_columns(market)
        for col in ["open", "high", "low", "close"]:
            if col in market.columns:
                market[col] = pd.to_numeric(market[col], errors="coerce")

    return trades, market

# ========= P&L / Trades utils =========
def format_timedelta_hhmm(td):
    if pd.isna(td): return "N/A"
    total_seconds = int(td.total_seconds()); m, _ = divmod(total_seconds, 60); h, mm = divmod(m, 60)
    return f"{h:02d}:{mm:02d}"

def calculate_pnl_and_metrics(trades_df: pd.DataFrame):
    if trades_df is None or trades_df.empty: return {}, pd.DataFrame(), {}
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
                if realized > 0: win += 1; gp += realized
                else: loss += 1; gl += abs(realized)
                positions[asset]["cost"] -= avg_cost * trade_qty
                positions[asset]["quantity"] -= trade_qty

        df.loc[i, "pnl"] = cur_pnl
        df.loc[i, "cumulative_pnl"] = total
        peak = max(peak, total); mdd = max(mdd, peak - total)

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

def match_trades_fifo(trades_df: pd.DataFrame):
    if trades_df is None or trades_df.empty: return pd.DataFrame(), pd.DataFrame()
    tdf = trades_df.copy(); tdf["__parsed_ts__"] = _parsed_ts(tdf["timestamp"])
    matched, open_df = [], []
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
                if pd.isna(buy_ts) or pd.isna(sell_ts) or buy_ts >= sell_ts: break
                buy_qty = b0.get("quantity", Decimal(0))
                trade_qty = min(sell_qty, buy_qty)
                if trade_qty > 0:
                    pnl = (sell["price"] - b0["price"]) * trade_qty
                    hold_time = sell_ts - buy_ts
                    matched.append({
                        "Asset": asset, "Quantity": trade_qty,
                        "Buy Time": buy_ts, "Buy Price": b0["price"],
                        "Sell Time": sell_ts, "Sell Price": sell["price"],
                        "Hold Time": hold_time, "P&L ($)": pnl,
                        "Reason Buy": b0.get("reason"), "Reason Sell": sell.get("reason"),
                    })
                    sell_qty -= trade_qty
                    buys[0]["quantity"] -= trade_qty
                    if buys[0]["quantity"] < Decimal("1e-9"): buys.pop(0)
        open_df.extend(buys)

    matched_df = pd.DataFrame(matched) if matched else pd.DataFrame()
    open_df = pd.DataFrame(open_df) if open_df else pd.DataFrame()
    if not matched_df.empty:
        buy_cost = matched_df["Buy Price"] * matched_df["Quantity"]
        is_zero = buy_cost < Decimal("1e-18")
        matched_df["P&L %"] = 100 * np.where(is_zero, 0, matched_df["P&L ($)"] / buy_cost)
        matched_df = matched_df.sort_values("Sell Time", ascending=False)
    if not open_df.empty:
        open_df["__parsed_ts__"] = _parsed_ts(open_df["timestamp"])
        open_df = open_df.sort_values("__parsed_ts__", ascending=False).drop(columns="__parsed_ts__")
    return matched_df, open_df

def calculate_open_positions(trades_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df is None or trades_df.empty or market_df is None or market_df.empty: return pd.DataFrame()
    positions = {}; position_open_times = {}
    tdf = trades_df.copy(); tdf["__parsed_ts__"] = _parsed_ts(tdf["timestamp"])
    tdf = tdf.sort_values("__parsed_ts__").drop(columns="__parsed_ts__")

    for _, row in tdf.iterrows():
        asset = row.get("asset", ""); action = str(row.get("unified_action", "")).lower().strip()
        qty = row.get("quantity", Decimal(0)); price = row.get("price", Decimal(0)); timestamp = row.get("timestamp", "")
        if asset not in positions: positions[asset] = {"quantity": Decimal(0), "cost": Decimal(0)}
        if action in ["buy", "open"]:
            if positions[asset]["quantity"] == 0: position_open_times[asset] = timestamp
            positions[asset]["cost"] += qty * price; positions[asset]["quantity"] += qty
        elif action in ["sell", "close"]:
            if positions[asset]["quantity"] > 0:
                avg_cost = positions[asset]["cost"] / max(positions[asset]["quantity"], Decimal("1e-12"))
                sell_qty = min(qty, positions[asset]["quantity"])
                positions[asset]["cost"] -= avg_cost * sell_qty; positions[asset]["quantity"] -= sell_qty
                if positions[asset]["quantity"] <= Decimal("1e-9"):
                    if asset in position_open_times: del position_open_times[asset]

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
                        open_ts = pd.to_datetime(position_open_times[asset], errors="coerce", utc=True).tz_convert("America/Los_Angeles")
                        if pd.notna(open_ts): open_time_str = open_ts.strftime("%H:%M")
                    except: pass
                open_positions.append({
                    "Asset": asset, "Quantity": float(data["quantity"]),
                    "Avg. Entry Price": float(avg_entry), "Current Price": float(latest_price),
                    "Current Value ($)": float(current_value), "Unrealized P&L ($)": float(unrealized),
                    "Open Time": open_time_str,
                })
    return pd.DataFrame(open_positions)

# ========= Load + maybe refresh =========
st.markdown("## Crypto Trading Strategy")
st.caption("ML Signals with Price-Based Entry/Exit (displayed in PST)")
apply_theme()

trades_df, market_df = load_data(TRADES_LINK, MARKET_LINK)
elapsed = maybe_auto_refresh()

# >>> Validation flags <<<
trades_df = flag_threshold_violations(trades_df) if not trades_df.empty else trades_df

# >>> Track positions and identify missed buys/sells <<<
position_history = track_positions_over_time(trades_df) if not trades_df.empty else pd.DataFrame()
missed_buys_df = identify_missed_buys(market_df, trades_df, position_history, DYNAMIC_ENTRY_UI)
missed_sells_df = identify_missed_sells(market_df, trades_df, position_history, DYNAMIC_ENTRY_UI)

# ========= Sidebar =========
with st.sidebar:
    st.markdown("<h1 style='text-align:center;'>Crypto Strategy</h1>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("🌙" if st.session_state.theme == "light" else "☀️", help="Toggle theme"):
            st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"; st.rerun()
    with col2:
        st.session_state.auto_refresh_enabled = st.toggle("🔄", value=st.session_state.get("auto_refresh_enabled", True), help="Auto-Refresh (5min)")
    with col3:
        if st.button("↻", help="Force refresh"):
            st.cache_data.clear(); st.session_state.last_refresh = time.time(); st.rerun()

def _series_to_pst(s: pd.Series) -> pd.Series:
    """
    Interpret NAIVE timestamps as PST and convert tz-aware timestamps to PST.
    This matches your fetcher, which writes NAIVE PST timestamps.
    """
    ts = pd.to_datetime(s, errors="coerce")
    try:
        # If the series is naive -> localize to PST; otherwise convert to PST
        if getattr(ts.dt, "tz", None) is None:
            return ts.dt.tz_localize("America/Los_Angeles")
        else:
            return ts.dt.tz_convert("America/Los_Angeles")
    except Exception:
        return ts  # fallback

def _pick_ts_pst(df: pd.DataFrame) -> pd.Series:
    """
    Prefer a 'timestamp_pst' column if present (already PST),
    otherwise use 'timestamp' and normalize to PST.
    """
    if "timestamp_pst" in df.columns:
        return _series_to_pst(df["timestamp_pst"])
    return _series_to_pst(df["timestamp"])

# ==== SIDEBAR FRESHNESS (replace your whole block with this) ====

# Data freshness / alignment based on DATA ONLY — PST displays (no lag banner)
if not market_df.empty and "timestamp" in market_df.columns:
    mk_ts_pst = _pick_ts_pst(market_df)
    mk_min_pst, mk_max_pst = mk_ts_pst.min(), mk_ts_pst.max()
    st.caption(
        f"📈 Market window (PST): "
        f"{mk_min_pst.strftime('%Y-%m-%d %H:%M:%S %Z')} → {mk_max_pst.strftime('%Y-%m-%d %H:%M:%S %Z')}"
    )

    if not trades_df.empty and "timestamp" in trades_df.columns:
        tr_ts_pst = _series_to_pst(trades_df["timestamp"])
        tr_min_pst, tr_max_pst = tr_ts_pst.min(), tr_ts_pst.max()
        st.caption(
            f"🧾 Trades window (PST): "
            f"{tr_min_pst.strftime('%Y-%m-%d %H:%M:%S %Z')} → {tr_max_pst.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        )

    st.markdown("---")
    st.markdown(f"**Trades:** {'✅' if not trades_df.empty else '⚠️'} {len(trades_df):,}")
    st.markdown(f"**Market:** {'✅' if not market_df.empty else '❌'} {len(market_df):,}")
    if not market_df.empty and "asset" in market_df.columns:
        st.markdown(f"**Assets:** {market_df['asset'].nunique():,}")

    st.markdown("---")
    st.markdown("**🔧 Tuning (what-if)**")
    tw_col1, tw_col2 = st.columns(2)
    with tw_col1:
        ui_bounce = st.number_input("Bounce (pct)", min_value=0.0005, max_value=0.01,
                                    value=DYNAMIC_ENTRY_UI["confirmation_bounce_pct"], step=0.0001, format="%.4f")
    with tw_col2:
        ui_timeout_m = st.number_input("Timeout (min)", min_value=7, max_value=180,
                                       value=DYNAMIC_ENTRY_UI["timeout_minutes"], step=1)
    ui_match_m = st.number_input("Match window (±min)", 1, 20, DYNAMIC_ENTRY_UI["match_window_minutes"], 1)

    DYNAMIC_ENTRY_TRY = dict(DYNAMIC_ENTRY_UI)
    DYNAMIC_ENTRY_TRY["confirmation_bounce_pct"] = float(ui_bounce)
    DYNAMIC_ENTRY_TRY["timeout_minutes"] = int(ui_timeout_m)
    DYNAMIC_ENTRY_TRY["match_window_minutes"] = int(ui_match_m)

    # Compute WHAT-IF missed buys/sells with updated config
    missed_buys_try = identify_missed_buys(market_df, trades_df, position_history, DYNAMIC_ENTRY_TRY)
    missed_sells_try = identify_missed_sells(market_df, trades_df, position_history, DYNAMIC_ENTRY_TRY)

# ========= Debug panel =========
with st.expander("🔎 Data Freshness Debug", expanded=True):
    # Unified windows (PST, data-driven)
    if not market_df.empty and "timestamp" in market_df.columns:
        st.write(f"**Market window (PST):** {_pick_ts_pst(market_df).min()} → {_pick_ts_pst(market_df).max()}")
    if not trades_df.empty and "timestamp" in trades_df.columns:
        st.write(f"**Trades window (PST):** {_series_to_pst(trades_df['timestamp']).min()} → {_series_to_pst(trades_df['timestamp']).max()}")

    # Trades freshness detail (PST)
    if not trades_df.empty and "timestamp" in trades_df.columns:
        trades_debug = trades_df.copy(); trades_debug["__parsed_ts__"] = _series_to_pst(trades_debug["timestamp"])
        trades_valid = trades_debug.dropna(subset=["__parsed_ts__"])
        if not trades_valid.empty:
            trades_sorted = trades_valid.sort_values("__parsed_ts__")
            latest_idx = trades_sorted.index[-1]
            st.write(f"**Latest Trade Timestamp (raw):** {trades_sorted.loc[latest_idx,'timestamp']}")
            st.write(f"**Latest Trade Timestamp (PST):** {trades_sorted.loc[latest_idx,'__parsed_ts__']}")
            if len(trades_df) != len(trades_valid):
                st.warning(f"⚠️ {len(trades_df) - len(trades_valid)} trades have unparseable timestamps")
        else:
            st.error("❌ No valid timestamps found in trades data")
            st.write("**Sample raw timestamps:**")
            st.write(trades_df["timestamp"].head().tolist())
    else:
        st.write("**Latest Trade Timestamp:** No trade data")
    st.write(f"**Total Trades:** {len(trades_df):,}")

    # Market freshness detail (PST)
    if not market_df.empty and "timestamp" in market_df.columns:
        market_debug = market_df.copy(); market_debug["__parsed_ts__"] = _pick_ts_pst(market_debug)
        market_valid = market_debug.dropna(subset=["__parsed_ts__"])
        if not market_valid.empty:
            market_sorted = market_valid.sort_values("__parsed_ts__")
            latest_idx = market_sorted.index[-1]
            st.write(f"**Latest Market Timestamp (raw):** {market_sorted.loc[latest_idx,'timestamp']}")
            st.write(f"**Latest Market Timestamp (PST):** {market_sorted.loc[latest_idx,'__parsed_ts__']}")
            st.write(f"**Market timestamp dtype:** {market_df['timestamp'].dtype}")
            if len(market_df) != len(market_valid):
                st.warning(f"⚠️ {len(market_df) - len(market_valid)} market records have unparseable timestamps")
        else:
            st.error("❌ No valid timestamps found in market data")
            st.write("**Sample raw timestamps:**")
            st.write(market_df["timestamp"].head().tolist())
        st.write(f"**Total Market Records:** {len(market_df):,}")
    else:
        st.write("**Market data:** None")
    st.write(f"**Cache age:** {elapsed}s")

# ========= Tabs =========
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Price & Trades", "💰 P&L Analysis", "📜 Trade History", "🚦 Validations",
    "🚫 Missed Entries", "📊 Overall Stats"
])

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
            range_choice = st.selectbox("Select Date Range",
                ["4 hours","12 hours","1 day","3 days","7 days","30 days","All"], index=0, key="range_select_main")

        asset_market = market_df[market_df["asset"] == selected_asset].copy()
        if asset_market.empty:
            st.warning(f"No market data found for {selected_asset}.")
        else:
            # prefer timestamp_pst for UI, else convert timestamp -> PST
            src_ts = asset_market["timestamp_pst"] if "timestamp_pst" in asset_market.columns else asset_market["timestamp"]
            asset_ts = _series_to_pst(src_ts)
            asset_market_sorted = asset_market.assign(__t__=asset_ts).sort_values("__t__")
            end_parsed = asset_ts.max()
            if pd.isna(end_parsed): st.warning("Timestamps could not be parsed.")
            else:
                if range_choice == "4 hours": start_parsed = end_parsed - timedelta(hours=4)
                elif range_choice == "12 hours": start_parsed = end_parsed - timedelta(hours=12)
                elif range_choice == "1 day": start_parsed = end_parsed - timedelta(days=1)
                elif range_choice == "3 days": start_parsed = end_parsed - timedelta(days=3)
                elif range_choice == "7 days": start_parsed = end_parsed - timedelta(days=7)
                elif range_choice == "30 days": start_parsed = end_parsed - timedelta(days=30)
                else: start_parsed = asset_ts.min()

                vis_mask = (asset_market_sorted["__t__"] >= start_parsed) & (asset_market_sorted["__t__"] <= end_parsed)
                vis = asset_market_sorted.loc[vis_mask].copy()

                if vis.empty: st.warning(f"No data for {selected_asset} in range {range_choice}.")
                else:
                    last_price = vis["close"].iloc[-1]
                    pf = ",.8f" if last_price < 0.001 else ",.6f" if last_price < 1 else ",.4f"
                    st.metric(f"Last Price for {selected_asset}", f"${last_price:{pf}}")

                    price_range = vis["high"].max() - vis["low"].min()
                    y_min = vis["low"].min() - price_range * 0.15
                    y_max = vis["high"].max() + price_range * 0.05
                    tick_format = "%H:%M" if range_choice in ["4 hours", "12 hours"] else "%m/%d %H:%M"

                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=vis["__t__"],
                        open=vis["open"], high=vis["high"], low=vis["low"], close=vis["close"],
                        name=selected_asset, increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
                        increasing_fillcolor="rgba(38,166,154,0.5)", decreasing_fillcolor="rgba(239,83,80,0.5)",
                        line=dict(width=1)
                    ))

                    # ML signals overlay
                    sig_cols_ok = {"p_up", "p_down"}.issubset(vis.columns)
                    if sig_cols_ok:
                        sig = vis.dropna(subset=["p_up", "p_down"]).copy()
                        if not sig.empty:
                            sig["confidence"] = (sig["p_up"] - sig["p_down"]).abs()
                            sizes = (sig["confidence"] * 100.0) / 5.0 + 3.0
                            colors = np.where(sig["p_down"] > sig["p_up"], "#ff6b6b", "#51cf66")
                            fig.add_trace(go.Scatter(
                                x=sig["__t__"], y=sig["close"], mode="markers", name="ML Signals",
                                marker=dict(size=sizes, color=colors, opacity=0.7, line=dict(width=1, color="white")),
                                customdata=list(zip(sig["p_up"], sig["p_down"], sig["confidence"])),
                                hovertemplate="<b>ML Signal</b><br>Time: %{x|%Y-%m-%d %H:%M}<br>"
                                              "Price: $%{y:.6f}<br>P(Up): %{customdata[0]:.3f}<br>"
                                              "P(Down): %{customdata[1]:.3f}<br>"
                                              "Confidence: %{customdata[2]:.3f}<extra></extra>",
                            ))

                    # Buy/Sell markers from trades
                    if not trades_df.empty:
                        asset_trades = trades_df[trades_df["asset"] == selected_asset].copy()
                        if not asset_trades.empty and "timestamp" in asset_trades.columns:
                            t_ts = _series_to_pst(asset_trades["timestamp"])
                            mask = (t_ts >= start_parsed) & (t_ts <= end_parsed)
                            asset_trades = asset_trades.loc[mask].copy()

                            if not asset_trades.empty:
                                marker_y = y_min + (vis["high"].max() - vis["low"].min()) * 0.02
                                buys = asset_trades[asset_trades["unified_action"].str.lower().isin(["buy","open"])].copy()
                                if not buys.empty:
                                    if "valid_at_open" not in buys.columns: buys["valid_at_open"] = True
                                    if "violation_reason" not in buys.columns: buys["violation_reason"] = ""
                                    if "revalidated_hint" not in buys.columns: buys["revalidated_hint"] = False
                                    buy_prices = buys["price"].apply(float)
                                    buy_reasons = buys.get("reason", pd.Series([""] * len(buys))).fillna("")
                                    buy_colors = np.where(buys["valid_at_open"].fillna(True), "#4caf50", "#f59e0b")
                                    buy_valid_str = buys["valid_at_open"].map(lambda x: "Yes" if x else "No")
                                    buy_reval_str = buys["revalidated_hint"].map(lambda x: "Re-validated bounce: Yes" if x else "")
                                    buy_viol_str = buys["violation_reason"].fillna("")
                                    fig.add_trace(go.Scatter(
                                        x=_series_to_pst(buys["timestamp"]), y=[marker_y] * len(buys), mode="markers", name="BUY",
                                        marker=dict(symbol="triangle-up", size=14, color=buy_colors, line=dict(width=1, color="white")),
                                        customdata=np.stack((buy_prices, buy_reasons, buy_valid_str, buy_reval_str, buy_viol_str), axis=-1),
                                        hovertemplate="<b>BUY</b> @ $%{customdata[0]:.8f}<br>%{x|%H:%M:%S}"
                                                      "<br>Reason: %{customdata[1]}<br>Valid at OPEN: %{customdata[2]}"
                                                      "<br>%{customdata[3]}<br>%{customdata[4]}<extra></extra>",
                                    ))

                                sells = asset_trades[asset_trades["unified_action"].str.lower().isin(["sell","close"])].copy()
                                if not sells.empty:
                                    sell_prices = sells["price"].apply(float)
                                    sell_reasons = sells.get("reason", pd.Series([""] * len(sells))).fillna("")
                                    fig.add_trace(go.Scatter(
                                        x=_series_to_pst(sells["timestamp"]), y=[marker_y] * len(sells), mode="markers", name="SELL",
                                        marker=dict(symbol="triangle-down", size=14, color="#f44336", line=dict(width=1, color="white")),
                                        customdata=np.stack((sell_prices, sell_reasons), axis=-1),
                                        hovertemplate="<b>SELL</b> @ $%{customdata[0]:.8f}<br>%{x|%H:%M:%S}"
                                                      "<br>Reason: %{customdata[1]}<extra></extra>",
                                    ))

                    # NEW: Overlay missed buys when in position
                    if not missed_buys_df.empty:
                        asset_missed_buys = missed_buys_df[missed_buys_df['asset'] == selected_asset].copy()
                        if not asset_missed_buys.empty:
                            asset_missed_buys['__t__'] = _series_to_pst(asset_missed_buys['timestamp'])
                            fig.add_trace(go.Scatter(
                                x=asset_missed_buys['__t__'], y=asset_missed_buys['price'],
                                mode="markers", name="Missed BUY (add to position)",
                                marker=dict(symbol="circle-open", size=12, color="#4caf50", line=dict(width=2, color="#4caf50")),
                                customdata=np.stack((
                                    asset_missed_buys['p_up'], asset_missed_buys['p_down'],
                                    asset_missed_buys['confidence'], asset_missed_buys['price']
                                ), axis=-1),
                                hovertemplate="<b>Missed BUY (add to position)</b><br>%{x|%Y-%m-%d %H:%M}<br>"
                                              "Price: $%{customdata[3]:.8f}<br>"
                                              "P(Up): %{customdata[0]:.3f}<br>"
                                              "P(Down): %{customdata[1]:.3f}<br>"
                                              "Confidence: %{customdata[2]:.3f}<extra></extra>",
                            ))

                    # NEW: Overlay missed sells when in position
                    if not missed_sells_df.empty:
                        asset_missed_sells = missed_sells_df[missed_sells_df['asset'] == selected_asset].copy()
                        if not asset_missed_sells.empty:
                            asset_missed_sells['__t__'] = _series_to_pst(asset_missed_sells['timestamp'])
                            fig.add_trace(go.Scatter(
                                x=asset_missed_sells['__t__'], y=asset_missed_sells['price'],
                                mode="markers", name="Missed SELL (in position)",
                                marker=dict(symbol="square-open", size=12, color="#f44336", line=dict(width=2, color="#f44336")),
                                customdata=np.stack((
                                    asset_missed_sells['p_up'], asset_missed_sells['p_down'],
                                    asset_missed_sells['confidence'], asset_missed_sells['price']
                                ), axis=-1),
                                hovertemplate="<b>Missed SELL (in position)</b><br>%{x|%Y-%m-%d %H:%M}<br>"
                                              "Price: $%{customdata[3]:.8f}<br>"
                                              "P(Up): %{customdata[0]:.3f}<br>"
                                              "P(Down): %{customdata[1]:.3f}<br>"
                                              "Confidence: %{customdata[2]:.3f}<extra></extra>",
                            ))

                    fig.update_layout(
                        title=f"{selected_asset} — Price & Trades ({range_choice})",
                        template="plotly_dark" if st.session_state.theme == "dark" else "plotly_white",
                        xaxis_rangeslider_visible=False,
                        xaxis=dict(title="Time (PST)", type="date", tickformat=tick_format,
                                   showgrid=True, gridcolor="rgba(59,66,82,0.3)" if st.session_state.theme == "dark" else "rgba(128,128,128,0.1)",
                                   tickangle=-45),
                        yaxis=dict(title="Price (USD)", tickformat=".8f" if last_price < 0.001 else ".6f" if last_price < 1 else ".4f",
                                   showgrid=True, gridcolor="rgba(59,66,82,0.3)" if st.session_state.theme == "dark" else "rgba(128,128,128,0.1)",
                                   range=[y_min, y_max]),
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                        height=750, margin=dict(l=60, r=20, t=80, b=80),
                        plot_bgcolor="rgba(14,17,23,1)" if st.session_state.theme == "dark" else "rgba(250,250,250,0.8)",
                        paper_bgcolor="rgba(14,17,23,1)" if st.session_state.theme == "dark" else "rgba(255,255,255,1)",
                        font_color="#FAFAFA" if st.session_state.theme == "dark" else "#262626",
                    )
                    st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": True, "modeBarButtonsToAdd": ["drawline","drawopenpath","drawclosedpath"], "scrollZoom": True})

                    # Quick counts in view window
                    miss_buys_now = 0 if missed_buys_df.empty else len(missed_buys_df[missed_buys_df["asset"]==selected_asset])
                    miss_sells_now = 0 if missed_sells_df.empty else len(missed_sells_df[missed_sells_df["asset"]==selected_asset])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption(f"Missed BUYs (add to position): **{miss_buys_now}**")
                    with col2:
                        st.caption(f"Missed SELLs (in position): **{miss_sells_now}**")

# ----- TAB 2: P&L Analysis -----
with tab2:
    if trades_df.empty or "timestamp" not in trades_df.columns:
        st.warning("No trade data loaded to analyze P&L.")
    else:
        pnl_summary, trades_with_pnl, stats = calculate_pnl_and_metrics(trades_df)
        st.markdown("### Strategy Performance")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Closed Trades", f"{stats.get('total_trades', 0):,}")
        with c2: st.metric("Win Rate", f"{stats.get('win_rate', 0):.1f}%")
        with c3:
            pf = stats.get("profit_factor", 0)
            st.metric("Profit Factor", "∞" if isinstance(pf, float) and np.isinf(pf) else f"{pf:.2f}")
        with c4: st.metric("Max Drawdown", f"${stats.get('max_drawdown', 0):.2f}")

        if not trades_with_pnl.empty:
            plot_df = trades_with_pnl.copy()
            plot_df["__parsed_ts__"] = _series_to_pst(plot_df["timestamp"])
            plot_df = plot_df.sort_values("__parsed_ts__")
            plot_df["cumulative_pnl"] = plot_df["cumulative_pnl"].apply(float)
            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Scatter(x=plot_df["__parsed_ts__"], y=plot_df["cumulative_pnl"], mode="lines", name="Cumulative P&L", line=dict(width=2)))
            fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Break Even")
            chart_bg = "rgba(14,17,23,1)" if st.session_state.theme == "dark" else "rgba(255,255,255,1)"
            chart_grid = "rgba(59,66,82,0.3)" if st.session_state.theme == "dark" else "rgba(128,128,128,0.1)"
            chart_text = "#FAFAFA" if st.session_state.theme == "dark" else "#262626"
            fig_pnl.update_layout(title="Total Portfolio P&L (PST)", template="plotly_dark" if st.session_state.theme == "dark" else "plotly_white",
                                  yaxis_title="P&L (USD)", xaxis_title="Date (PST)", hovermode="x unified",
                                  plot_bgcolor=chart_bg, paper_bgcolor=chart_bg, font_color=chart_text,
                                  xaxis=dict(gridcolor=chart_grid), yaxis=dict(gridcolor=chart_grid))
            st.plotly_chart(fig_pnl, use_container_width=True)

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
            for col in ["Quantity","Buy Price","Sell Price","P&L ($)","P&L %"]:
                if col in display_matched.columns: display_matched[col] = display_matched[col].apply(float)
            # Ensure PST when formatting
            display_matched["Buy Time"]  = pd.to_datetime(display_matched["Buy Time"],  utc=True).dt.tz_convert("America/Los_Angeles").dt.strftime("%Y-%m-%d %H:%M:%S")
            display_matched["Sell Time"] = pd.to_datetime(display_matched["Sell Time"], utc=True).dt.tz_convert("America/Los_Angeles").dt.strftime("%Y-%m-%d %H:%M:%S")
            display_matched["Hold Time"] = display_matched["Hold Time"].apply(format_timedelta_hhmm)
            st.dataframe(
                display_matched[["Asset","Quantity","Buy Time","Buy Price","Sell Time","Sell Price","Hold Time","P&L ($)","P&L %"]],
                column_config={
                    "Asset": st.column_config.TextColumn(width="small"),
                    "Quantity": st.column_config.NumberColumn(format="%.4f", width="small"),
                    "Buy Price": st.column_config.NumberColumn(format="$%.8f"),
                    "Sell Price": st.column_config.NumberColumn(format="$%.8f"),
                    "P&L ($)": st.column_config.NumberColumn(format="$%.4f"),
                    "P&L %": st.column_config.NumberColumn(format="%.4f%%"),
                }, use_container_width=True, hide_index=True
            )
        else:
            st.info("No completed (buy/sell) trades found.")
        st.markdown("---")
        st.markdown("#### Open Positions (Unmatched Buys)")
        if not open_df.empty:
            display_open = open_df.copy()
            for col in ["quantity","price"]:
                if col in display_open.columns: display_open[col] = display_open[col].apply(float)
            display_open["Time"] = pd.to_datetime(display_open["timestamp"], errors="coerce", utc=True).dt.tz_convert("America/Los_Angeles").dt.strftime("%Y-%m-%d %H:%M:%S")
            display_open = display_open.rename(columns={"asset":"Asset","quantity":"Quantity","price":"Price","reason":"Reason"})
            st.dataframe(
                display_open[["Time","Asset","Quantity","Price","Reason"]],
                column_config={
                    "Asset": st.column_config.TextColumn(width="small"),
                    "Quantity": st.column_config.NumberColumn(format="%.4f", width="small"),
                    "Price": st.column_config.NumberColumn(format="$%.8f"),
                }, use_container_width=True, hide_index=True
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
        action_col = "action" if "action" in trades_df.columns else ("unified_action" if "unified_action" in trades_df.columns else None)
        opens_mask = trades_df[action_col].astype(str).str.upper().isin(["OPEN","BUY"]) if action_col else pd.Series([False]*len(trades_df), index=trades_df.index)
        opens = trades_df.loc[opens_mask].copy()
        if opens.empty:
            st.info("No OPEN orders in the log.")
        else:
            total_opens = len(opens)
            invalid = opens[~opens["valid_at_open"]]; valid = opens[opens["valid_at_open"]]
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("OPEN Orders", f"{total_opens:,}")
            with c2: st.metric("Valid at OPEN", f"{len(valid):,}")
            with c3: st.metric("Flagged (Invalid)", f"{len(invalid):,}")
            with c4:
                vr = 0 if total_opens == 0 else (len(valid) / total_opens) * 100
                st.metric("Validity Rate", f"{vr:.1f}%")
            if not invalid.empty:
                show_cols = [c for c in ["timestamp","asset","price","p_up","p_down","confidence","reason","violation_reason"] if c in invalid.columns]
                # Show PST in table by adding a display column (optional)
                invalid_disp = invalid.copy()
                invalid_disp["timestamp_pst"] = _series_to_pst(invalid_disp["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S %Z")
                st.markdown("#### ❗ Flagged OPENs")
                st.dataframe(invalid_disp[["timestamp_pst"] + [c for c in show_cols if c != "timestamp"]].sort_values("timestamp_pst"),
                             use_container_width=True, hide_index=True)
            else:
                st.success("No violations detected 🎉")
            st.caption("Open = buy execution records; validity is evaluated against per-asset thresholds and logged probabilities at that time.")

# ----- TAB 5: Missed Entries -----
with tab5:
    st.markdown("### Position-Aware Missed Signals")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Missed BUYs (add to position)", f"{len(missed_buys_df):,}")
    with col2:
        st.metric("Missed SELLs (in position)", f"{len(missed_sells_df):,}")
    with col3:
        try_buys = len(missed_buys_try) if 'missed_buys_try' in locals() else 0
        st.metric("Missed BUYs (what-if)", f"{try_buys:,}")
    with col4:
        try_sells = len(missed_sells_try) if 'missed_sells_try' in locals() else 0
        st.metric("Missed SELLs (what-if)", f"{try_sells:,}")

    # Filters
    all_assets = set()
    if not missed_buys_df.empty:
        all_assets.update(missed_buys_df['asset'].unique())
    if not missed_sells_df.empty:
        all_assets.update(missed_sells_df['asset'].unique())
    
    if all_assets:
        assets_list = sorted(list(all_assets))
        sel_assets = st.multiselect("Filter by asset", assets_list, default=assets_list)
    else:
        sel_assets = []

    # Display missed buys table
    if not missed_buys_df.empty:
        st.markdown("#### Missed BUY Signals (when in position - add to position)")
        filtered_buys = missed_buys_df[missed_buys_df['asset'].isin(sel_assets)] if sel_assets else missed_buys_df
        display_buys = filtered_buys.copy()
        # render timestamps in PST
        display_buys['timestamp'] = _series_to_pst(display_buys['timestamp']).dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(
            display_buys[['asset', 'timestamp', 'price', 'p_up', 'p_down', 'confidence']],
            column_config={
                'price': st.column_config.NumberColumn(format="$%.8f"),
                'p_up': st.column_config.NumberColumn(format="%.3f"),
                'p_down': st.column_config.NumberColumn(format="%.3f"),
                'confidence': st.column_config.NumberColumn(format="%.3f"),
            },
            use_container_width=True,
            hide_index=True
        )

    # Display missed sells table
    if not missed_sells_df.empty:
        st.markdown("#### Missed SELL Signals (when in position)")
        filtered_sells = missed_sells_df[missed_sells_df['asset'].isin(sel_assets)] if sel_assets else missed_sells_df
        display_sells = filtered_sells.copy()
        display_sells['timestamp'] = _series_to_pst(display_sells['timestamp']).dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(
            display_sells[['asset', 'timestamp', 'price', 'p_up', 'p_down', 'confidence']],
            column_config={
                'price': st.column_config.NumberColumn(format="$%.8f"),
                'p_up': st.column_config.NumberColumn(format="%.3f"),
                'p_down': st.column_config.NumberColumn(format="%.3f"),
                'confidence': st.column_config.NumberColumn(format="%.3f"),
            },
            use_container_width=True,
            hide_index=True
        )

    # Download CSV
    if not missed_buys_df.empty or not missed_sells_df.empty:
        combined_missed = []
        if not missed_buys_df.empty:
            buys_export = missed_buys_df.copy()
            buys_export['signal_type'] = 'missed_buy'
            combined_missed.append(buys_export)
        if not missed_sells_df.empty:
            sells_export = missed_sells_df.copy()
            sells_export['signal_type'] = 'missed_sell'
            combined_missed.append(sells_export)
        
        if combined_missed:
            export_df = pd.concat(combined_missed, ignore_index=True)
            export_df['timestamp'] = _series_to_pst(export_df['timestamp']).dt.strftime("%Y-%m-%d %H:%M:%S")
            csv = export_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Missed Signals CSV", data=csv, file_name="missed_signals.csv", mime="text/csv")

# ----- TAB 6: Overall Stats -----
with tab6:
    st.markdown("### 📊 Overall Stats (Position-Aware Analysis)")
    colA, colB, colC, colD = st.columns(4)
    
    with colA: st.metric("Missed BUYs (add to pos)", f"{len(missed_buys_df):,}")
    with colB: st.metric("Missed SELLs (in pos)", f"{len(missed_sells_df):,}")
    with colC: st.metric("Total Position-Aware", f"{len(missed_buys_df) + len(missed_sells_df):,}")
    
    # Validity rate for OPENs
    action_col = "action" if "action" in trades_df.columns else ("unified_action" if "unified_action" in trades_df.columns else None)
    if not trades_df.empty and action_col:
        opens_mask = trades_df[action_col].astype(str).str.upper().isin(["OPEN","BUY"])
        opens = trades_df.loc[opens_mask].copy()
        total_opens = len(opens); valid_opens = len(opens[opens["valid_at_open"]]) if "valid_at_open" in opens.columns else total_opens
        vr = 0 if total_opens == 0 else (valid_opens / total_opens) * 100
    else:
        total_opens, valid_opens, vr = 0, 0, 0
    with colD: st.metric("OPEN Validity Rate", f"{vr:.1f}%")

    st.markdown("---")
    
    # Per-asset breakdown for position-aware missed signals
    if not missed_buys_df.empty or not missed_sells_df.empty:
        st.markdown("#### Position-Aware Missed Signals by Asset")
        
        buy_counts = missed_buys_df.groupby('asset').size().rename('missed_buys') if not missed_buys_df.empty else pd.Series(name='missed_buys', dtype=int)
        sell_counts = missed_sells_df.groupby('asset').size().rename('missed_sells') if not missed_sells_df.empty else pd.Series(name='missed_sells', dtype=int)
        
        combined = pd.concat([buy_counts, sell_counts], axis=1).fillna(0).astype(int)
        combined['total_missed'] = combined['missed_buys'] + combined['missed_sells']
        combined = combined.sort_values('total_missed', ascending=False)
        
        st.dataframe(
            combined.reset_index(),
            column_config={
                'missed_buys': st.column_config.NumberColumn(format="%d"),
                'missed_sells': st.column_config.NumberColumn(format="%d"),
                'total_missed': st.column_config.NumberColumn(format="%d"),
            },
            use_container_width=True,
            hide_index=True
        )

    # What-if comparison if available
    if 'missed_buys_try' in locals() and 'missed_sells_try' in locals():
        st.markdown("---")
        st.markdown("#### Current vs What-If Comparison")
        
        current_total = len(missed_buys_df) + len(missed_sells_df)
        try_total = len(missed_buys_try) + len(missed_sells_try)
        improvement = current_total - try_total
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Total", f"{current_total:,}")
        with col2:
            st.metric("What-If Total", f"{try_total:,}")
        with col3:
            st.metric("Improvement", f"{improvement:+d}")

    st.markdown("---")
    st.caption(
        "The position-aware analysis shows missed BUY signals only when you were in a position "
        "(opportunities to add to positions), and missed SELL signals only when you were in a position. "
        "This provides insights for position sizing and adding to existing holdings based on strong signals."
    )

# ====== Sidebar: Open Positions compact ======
with st.sidebar:
    st.markdown("---")
    open_positions_df = calculate_open_positions(trades_df, market_df) if not trades_df.empty else pd.DataFrame()
    if not open_positions_df.empty:
        st.markdown("**📊 Open Positions**")
        sorted_positions = open_positions_df.sort_values("Unrealized P&L ($)", ascending=False)
        for _, pos in sorted_positions.iterrows():
            pnl = pos["Unrealized P&L ($)"]
            pnl_pct = ((pos["Current Price"] - pos["Avg. Entry Price"]) / pos["Avg. Entry Price"] * 100) if pos["Avg. Entry Price"] != 0 else 0
            color = "#16a34a" if pnl >= 0 else "#ef4444"; pnl_icon = "📈" if pnl >= 0 else "📉"
            asset_name = pos["Asset"]; cur_price = pos["Current Price"]
            pf = ".8f" if cur_price < 0.001 else ".6f" if cur_price < 1 else ".4f"
            avg_price = pos["Avg. Entry Price"]; epf = ".8f" if avg_price < 0.001 else ".6f" if avg_price < 1 else ".4f"
            open_time = pos.get("Open Time", "N/A"); quantity = pos["Quantity"]
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
                """, unsafe_allow_html=True
            )
        total_pnl = sorted_positions["Unrealized P&L ($)"].sum()
        total_value = sorted_positions["Current Value ($)"].sum()
        winners = len(sorted_positions[sorted_positions["Unrealized P&L ($)"] > 0]); total_positions = len(sorted_positions)
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
            """, unsafe_allow_html=True
        )
    else:
        st.markdown("**📊 Open Positions**")
        st.markdown(
            """
            <div style='text-align: center; padding: 1rem; background-color: rgba(128,128,128,0.05); 
                       border-radius: 0.25rem; color: #666; font-style: italic;'>
                No open positions
            </div>
            """, unsafe_allow_html=True
        )

