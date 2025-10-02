# trading_dashboard.py
import io
import re
import time
from datetime import timedelta
from decimal import Decimal, getcontext

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# =====================[ TIME FIX PATCH — BEGIN ]=====================

TZ_PST = "America/Los_Angeles"

def _ensure_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df['timestamp'] is tz-aware UTC (datetime64[ns, UTC]).
    Also (re)build df['timestamp_pst'] as tz-aware PST datetimes (not strings),
    so .dt ops & comparisons work everywhere.
    """
    if df is None or len(df) == 0:
        return df

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    elif "timestamp_pst" in df.columns:
        ts = pd.to_datetime(df["timestamp_pst"], errors="coerce")
        if getattr(ts.dt, "tz", None) is None:
            ts = ts.dt.tz_localize(TZ_PST).dt.tz_convert("UTC")
        else:
            ts = ts.dt.tz_convert("UTC")
        df["timestamp"] = ts
    else:
        return df

    df["timestamp_pst"] = df["timestamp"].dt.tz_convert(TZ_PST)

    cols = list(df.columns)
    if "timestamp" in cols and "timestamp_pst" in cols:
        c = cols.pop(cols.index("timestamp_pst"))
        cols.insert(cols.index("timestamp") + 1, c)
        df = df[cols]
    return df

def _as_utc(ts_like):
    t = pd.to_datetime(ts_like, errors="coerce")
    if t is pd.NaT:
        return t
    if getattr(t, "tzinfo", None) is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")

def normalize_all_times(*dfs):
    return [(_ensure_time_columns(df.copy()) if df is not None else df) for df in dfs]

def get_position_at_time(position_history: pd.DataFrame, asset: str, signal_time) -> float:
    ph = position_history
    if ph is None or ph.empty:
        return 0.0

    if "asset" in ph.columns:
        ph = ph[ph["asset"] == asset].copy()
        if ph.empty:
            return 0.0

    ph = _ensure_time_columns(ph)
    ph["timestamp"] = pd.to_datetime(ph["timestamp"], utc=True, errors="coerce")

    t_utc = _as_utc(signal_time)
    if pd.isna(t_utc):
        return 0.0

    mask = ph["timestamp"] <= t_utc
    if not mask.any():
        return 0.0

    for cand in ("position_qty", "quantity", "qty", "position_size"):
        if cand in ph.columns:
            return float(ph.loc[mask, cand].iloc[-1])
    return 0.0

# ======================[ TIME FIX PATCH — END ]======================

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

# ---- Data sources ----
TRADES_LINK = "https://drive.google.com/file/d/1cSeYf4kWJA49lMdQ1_yQ3hRui6wdrLD-/view?usp=sharing"
MARKET_LINK = "https://drive.google.com/file/d/1s1_TQqyAhE337jvz544egecvRGOjaBwx/view?usp=sharing"

DEFAULT_ASSET = "GIGA-USD"

# ---- Session state ----
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()
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

def seconds_since_last_run() -> int:
    return int(time.time() - st.session_state.get("last_refresh", 0))

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

# ========= Data loading (cached) =========
@st.cache_data(ttl=60)
def load_data(trades_link: str, market_link: str):
    trades = pd.DataFrame()
    tb = _download_drive_bytes(trades_link)
    if tb:
        trades = _read_parquet_or_csv(tb, "Trades")

    market = pd.DataFrame()
    mb = _download_drive_bytes(market_link)
    if mb:
        market = _read_parquet_or_csv(mb, "Market")

    # ---- TRADES ----
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

        # ensure datetime type if stringy
        if "timestamp_pst" in trades.columns and not pd.api.types.is_datetime64_any_dtype(trades["timestamp_pst"]):
            trades["timestamp_pst"] = pd.to_datetime(trades["timestamp_pst"], errors="coerce")

    # ---- MARKET ----
    if not market.empty:
        market = lower_strip_cols(market)
        if "product_id" in market.columns:
            market = market.rename(columns={"product_id": "asset"})
        if "asset" in market.columns:
            market["asset"] = market["asset"].apply(unify_symbol)
        market = normalize_prob_columns(market)
        for col in ["open", "high", "low", "close"]:
            if col in market.columns:
                market[col] = pd.to_numeric(market[col], errors="coerce")

        # ✅ fixed syntax: ensure parse to datetime
        if "timestamp_pst" in market.columns and not pd.api.types.is_datetime64_any_dtype(market["timestamp_pst"]):
            market["timestamp_pst"] = pd.to_datetime(market["timestamp_pst"], errors="coerce")

    # Normalize to UTC + PST tz-aware columns
    trades, market = normalize_all_times(trades, market)
    return trades, market

# ========= Upload override helper =========
def _read_uploaded_market(uploaded) -> pd.DataFrame:
    """Read an uploaded CSV or Parquet, normalize columns & times to match app."""
    if uploaded is None:
        return pd.DataFrame()
    name = (uploaded.name or "").lower()
    try:
        if name.endswith(".parquet") or name.endswith(".pq"):
            df = pd.read_parquet(uploaded)
        else:
            df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read uploaded market file: {e}")
        return pd.DataFrame()

    df = lower_strip_cols(df)
    if "product_id" in df.columns and "asset" not in df.columns:
        df = df.rename(columns={"product_id": "asset"})
    if "asset" in df.columns:
        df["asset"] = df["asset"].apply(unify_symbol)

    # normalize OHLC to numeric
    for c in ["open", "high", "low", "close"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # normalize probabilities
    df = normalize_prob_columns(df)

    # ensure timestamp_pst exists and is tz-aware PST, then make __x__ (tz-naive PST) for Plotly
    if "timestamp_pst" in df.columns:
        ts = pd.to_datetime(df["timestamp_pst"], errors="coerce")
        try:
            if getattr(ts.dt, "tz", None) is None:
                ts = ts.dt.tz_localize(TZ_PST)
            else:
                ts = ts.dt.tz_convert(TZ_PST)
        except Exception:
            ts = pd.to_datetime(df["timestamp_pst"], errors="coerce", utc=True).dt.tz_convert(TZ_PST)
        df["timestamp_pst"] = ts
        df["timestamp"] = ts.dt.tz_convert("UTC")
    elif "timestamp" in df.columns:
        tsu = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["timestamp"] = tsu
        df["timestamp_pst"] = tsu.dt.tz_convert(TZ_PST)
    else:
        st.error("Uploaded file is missing a recognizable time column (timestamp or timestamp_pst).")
        return pd.DataFrame()

    # tz-naive PST clone for Plotly only
    df["__x__"] = df["timestamp_pst"].dt.tz_localize(None)
    return df

# ========= P&L / Trades utils =========
def format_timedelta_hhmm(td):
    if pd.isna(td): return "N/A"
    total_seconds = int(td.total_seconds()); m, _ = divmod(total_seconds, 60); h, mm = divmod(m, 60)
    return f"{h:02d}:{mm:02d}"

def calculate_pnl_and_metrics(trades_df: pd.DataFrame):
    if trades_df is None or trades_df.empty or "timestamp_pst" not in trades_df.columns: 
        return {}, pd.DataFrame(), {}
    df = trades_df.copy().sort_values("timestamp_pst").reset_index(drop=True)

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

def summarize_realized_pnl(trades_with_pnl: pd.DataFrame):
    if trades_with_pnl is None or trades_with_pnl.empty or "pnl" not in trades_with_pnl.columns:
        return 0.0, pd.DataFrame(columns=["Asset", "Realized P&L ($)"])
    df = trades_with_pnl.copy()
    df["pnl_float"] = df["pnl"].apply(lambda x: float(x) if pd.notna(x) else 0.0)
    per_asset = (
        df.groupby("asset")["pnl_float"].sum().reset_index()
          .rename(columns={"asset": "Asset", "pnl_float": "Realized P&L ($)"})
          .sort_values("Realized P&L ($)", ascending=False)
    )
    overall = float(per_asset["Realized P&L ($)"].sum()) if not per_asset.empty else 0.0
    return overall, per_asset

def match_trades_fifo(trades_df: pd.DataFrame):
    if trades_df is None or trades_df.empty or "timestamp_pst" not in trades_df.columns: 
        return pd.DataFrame(), pd.DataFrame()
    tdf = trades_df.copy()
    matched, open_df = [], []
    for asset, group in tdf.groupby("asset"):
        g = group.sort_values("timestamp_pst")
        buys = [row.to_dict() for _, row in g[g["unified_action"].isin(["buy", "open"])].iterrows()]
        sells = [row.to_dict() for _, row in g[g["unified_action"].isin(["sell", "close"])].iterrows()]
        for sell in sells:
            sell_ts = sell["timestamp_pst"]
            sell_qty = sell.get("quantity", Decimal(0))
            while sell_qty > Decimal("1e-9") and buys:
                b0 = buys[0]
                buy_ts = b0["timestamp_pst"]
                if buy_ts >= sell_ts: break
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
        open_df = open_df.sort_values("timestamp_pst", ascending=False)
    return matched_df, open_df

def calculate_open_positions(trades_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    if (trades_df is None or trades_df.empty or "timestamp_pst" not in trades_df.columns or 
        market_df is None or market_df.empty or "timestamp_pst" not in market_df.columns): 
        return pd.DataFrame()
    positions = {}; position_open_times = {}
    tdf = trades_df.copy().sort_values("timestamp_pst")

    for _, row in tdf.iterrows():
        asset = row.get("asset", ""); action = str(row.get("unified_action", "")).lower().strip()
        qty = row.get("quantity", Decimal(0)); price = row.get("price", Decimal(0)); timestamp_pst = row.get("timestamp_pst", "")
        if asset not in positions: positions[asset] = {"quantity": Decimal(0), "cost": Decimal(0)}
        if action in ["buy", "open"]:
            if positions[asset]["quantity"] == 0: position_open_times[asset] = timestamp_pst
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
                idx = latest_asset_rows["timestamp_pst"].idxmax()
                latest_price = Decimal(str(latest_asset_rows.loc[idx, "close"]))
                avg_entry = data["cost"] / data["quantity"]
                current_value = latest_price * data["quantity"]
                unrealized = current_value - data["cost"]
                open_time_str = "N/A"
                if asset in position_open_times:
                    try:
                        open_ts = position_open_times[asset]
                        if pd.notna(open_ts): open_time_str = open_ts.strftime("%H:%M")
                    except: pass
                open_positions.append({
                    "Asset": asset, "Quantity": float(data["quantity"]),
                    "Avg. Entry Price": float(avg_entry), "Current Price": float(latest_price),
                    "Current Value ($)": float(current_value), "Unrealized P&L ($)": float(unrealized),
                    "Open Time": open_time_str,
                })
    return pd.DataFrame(open_positions)

def track_positions_over_time(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty or "timestamp_pst" not in trades_df.columns:
        return pd.DataFrame(columns=['timestamp_pst', 'asset', 'position_qty'])
    position_history = []
    for asset in trades_df['asset'].unique():
        asset_trades = trades_df[trades_df['asset'] == asset].copy().sort_values('timestamp_pst')
        current_qty = Decimal(0)
        for _, trade in asset_trades.iterrows():
            action = str(trade.get('unified_action', '')).lower().strip()
            qty = trade.get('quantity', Decimal(0))
            if action in ['buy', 'open']:
                current_qty += qty
            elif action in ['sell', 'close']:
                current_qty -= qty
                current_qty = max(current_qty, Decimal(0))
            position_history.append({
                'timestamp_pst': trade['timestamp_pst'],
                'asset': asset,
                'position_qty': float(current_qty)
            })
    return pd.DataFrame(position_history)

def identify_missed_buys(market_df: pd.DataFrame, trades_df: pd.DataFrame, position_history: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if market_df is None or market_df.empty or "timestamp_pst" not in market_df.columns:
        return pd.DataFrame()
    missed_buys = []
    match_td = pd.Timedelta(minutes=int(cfg["match_window_minutes"]))
    executed_buys = pd.DataFrame()
    if not trades_df.empty and "timestamp_pst" in trades_df.columns:
        action_col = "action" if "action" in trades_df.columns else "unified_action"
        buy_mask = trades_df[action_col].astype(str).str.upper().isin(["OPEN", "BUY"])
        executed_buys = trades_df.loc[buy_mask].copy()
    
    for asset in market_df['asset'].unique():
        if asset not in ASSET_THRESHOLDS:
            continue
        asset_data = market_df[market_df['asset'] == asset].copy().sort_values('timestamp_pst')
        th = ASSET_THRESHOLDS[asset]
        for _, row in asset_data.iterrows():
            pu = pd.to_numeric(row.get("p_up"), errors="coerce")
            pdn = pd.to_numeric(row.get("p_down"), errors="coerce")
            if pd.isna(pu) or pd.isna(pdn):
                continue
            conf = abs(pu - pdn)
            is_buy_signal = (pu > pdn) and (pu >= th["buy_threshold"]) and (conf >= th["min_confidence"])
            if not is_buy_signal:
                continue

            signal_time = row['timestamp_pst']
            position_qty = get_position_at_time(position_history, asset, signal_time)
            if position_qty > 0:
                continue  # already in position
            
            executed = False
            if not executed_buys.empty:
                asset_buys = executed_buys[executed_buys['asset'] == asset]
                if not asset_buys.empty:
                    time_diffs = (asset_buys['timestamp_pst'] - signal_time).abs()
                    executed = (time_diffs <= match_td).any()
            if not executed:
                missed_buys.append({
                    'asset': asset,
                    'timestamp_pst': row['timestamp_pst'],
                    'price': float(row.get('close', 0)),
                    'p_up': float(pu),
                    'p_down': float(pdn),
                    'confidence': float(conf),
                    'signal_type': 'missed_buy'
                })
    return pd.DataFrame(missed_buys)

def identify_missed_sells(market_df: pd.DataFrame, trades_df: pd.DataFrame, position_history: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if market_df is None or market_df.empty or "timestamp_pst" not in market_df.columns:
        return pd.DataFrame()
    missed_sells = []
    match_td = pd.Timedelta(minutes=int(cfg["match_window_minutes"]))
    executed_sells = pd.DataFrame()
    if not trades_df.empty and "timestamp_pst" in trades_df.columns:
        action_col = "action" if "action" in trades_df.columns else "unified_action"
        sell_mask = trades_df[action_col].astype(str).str.upper().isin(["CLOSE", "SELL"])
        executed_sells = trades_df.loc[sell_mask].copy()
    
    for asset in market_df['asset'].unique():
        if asset not in ASSET_THRESHOLDS:
            continue
        asset_data = market_df[market_df['asset'] == asset].copy().sort_values('timestamp_pst')
        th = ASSET_THRESHOLDS[asset]
        for _, row in asset_data.iterrows():
            pu = pd.to_numeric(row.get("p_up"), errors="coerce")
            pdn = pd.to_numeric(row.get("p_down"), errors="coerce")
            if pd.isna(pu) or pd.isna(pdn):
                continue
            conf = abs(pu - pdn)
            is_sell_signal = (pdn > pu) and (pdn >= th["sell_threshold"]) and (conf >= th["min_confidence"])
            if not is_sell_signal:
                continue
            signal_time = row['timestamp_pst']
            position_qty = get_position_at_time(position_history, asset, signal_time)
            if position_qty <= 0:
                continue
            executed = False
            if not executed_sells.empty:
                asset_sells = executed_sells[executed_sells['asset'] == asset]
                if not asset_sells.empty:
                    time_diffs = (asset_sells['timestamp_pst'] - signal_time).abs()
                    executed = (time_diffs <= match_td).any()
            if not executed:
                missed_sells.append({
                    'asset': asset,
                    'timestamp_pst': row['timestamp_pst'],
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
    "confirmation_bounce_pct": 0.0025,
    "timeout_minutes": 40,
    "cooldown_minutes": 10,
    "match_window_minutes": 5,
}

# ========= UI header & theme =========
st.markdown("## Crypto Trading Strategy")
st.caption("ML Signals with Price-Based Entry/Exit (displayed in PST)")
apply_theme()

# ========= Sidebar controls (theme + refresh + upload override) =========
with st.sidebar:
    st.markdown("<h1 style='text-align:center;'>Crypto Strategy</h1>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("🌙" if st.session_state.theme == "light" else "☀️", help="Toggle theme"):
            st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
            st.rerun()
    with col2:
        if st.button("↻ Refresh data", help="Clear cache and reload"):
            st.cache_data.clear()
            st.session_state.last_refresh = time.time()
            st.rerun()

    st.markdown("---")
    st.markdown("**Data override**")
    uploaded_market = st.file_uploader("Upload market CSV/Parquet to override", type=["csv", "parquet", "pq"], key="upload_market")

# ========= Load data =========
trades_df, market_df = load_data(TRADES_LINK, MARKET_LINK)
elapsed = seconds_since_last_run()

# If a file is uploaded, completely override market_df
if uploaded_market is not None:
    override_df = _read_uploaded_market(uploaded_market)
    if not override_df.empty:
        market_df = override_df
        st.cache_data.clear()
        st.session_state.last_refresh = time.time()
        st.info("Using uploaded market data (overrides Google Drive).")

# Ensure we have __x__ for Plotly if the source was Drive
if not market_df.empty and "__x__" not in market_df.columns and "timestamp_pst" in market_df.columns:
    t = pd.to_datetime(market_df["timestamp_pst"], errors="coerce")
    try:
        if getattr(t.dt, "tz", None) is None:
            t = t.dt.tz_localize(TZ_PST)
        else:
            t = t.dt.tz_convert(TZ_PST)
    except Exception:
        t = pd.to_datetime(market_df["timestamp_pst"], errors="coerce", utc=True).dt.tz_convert(TZ_PST)
    market_df["timestamp_pst"] = t
    market_df["__x__"] = t.dt.tz_localize(None)

# >>> Validation flags <<<
trades_df = flag_threshold_violations(trades_df) if not trades_df.empty else trades_df

# >>> Track positions and identify missed buys/sells <<<
position_history = track_positions_over_time(trades_df) if not trades_df.empty else pd.DataFrame()
missed_buys_df = identify_missed_buys(market_df, trades_df, position_history, DYNAMIC_ENTRY_UI) if not market_df.empty else pd.DataFrame()
missed_sells_df = identify_missed_sells(market_df, trades_df, position_history, DYNAMIC_ENTRY_UI) if not market_df.empty else pd.DataFrame()

# ========= Sidebar summaries =========
with st.sidebar:
    st.markdown("### ML Signals with Price-Based Entry/Exit (displayed in PST)")
    if not market_df.empty and "timestamp_pst" in market_df.columns:
        valid_market_times = market_df["timestamp_pst"].dropna()
        if not valid_market_times.empty:
            st.caption(f"📈 Market window (PST): {valid_market_times.min()} → {valid_market_times.max()}")
    if not trades_df.empty and "timestamp_pst" in trades_df.columns:
        valid_trade_times = trades_df["timestamp_pst"].dropna()
        if not valid_trade_times.empty:
            st.caption(f"🧾 Trades window (PST): {valid_trade_times.min()} → {valid_trade_times.max()}")

    st.markdown("---")
    st.markdown(f"**Trades:** {'✅' if not trades_df.empty else '⚠️'} {len(trades_df):,}")
    st.markdown(f"**Market:** {'✅' if not market_df.empty else '❌'} {len(market_df):,}")
    if not market_df.empty and "asset" in market_df.columns:
        st.markdown(f"**Assets:** {market_df['asset'].nunique():,}")

# ========= Debug panel =========
with st.expander("🔎 Data Freshness Debug", expanded=True):
    if not market_df.empty and "timestamp_pst" in market_df.columns:
        st.write(f"**Market window (PST):** {market_df['timestamp_pst'].min()} → {market_df['timestamp_pst'].max()}")
        st.write(f"**Total Market Records:** {len(market_df):,}")
    else:
        st.write("**Market data:** missing or no timestamp_pst")
    if not trades_df.empty and "timestamp_pst" in trades_df.columns:
        st.write(f"**Trades window (PST):** {trades_df['timestamp_pst'].min()} → {trades_df['timestamp_pst'].max()}")
        st.write(f"**Total Trades:** {len(trades_df):,}")
        trades_valid = trades_df.dropna(subset=["timestamp_pst"])
        if not trades_valid.empty:
            trades_sorted = trades_valid.sort_values("timestamp_pst")
            st.write(f"**Latest Trade Timestamp (PST):** {trades_sorted['timestamp_pst'].iloc[-1]}")
    else:
        st.write("**Trades data:** missing or no timestamp_pst")
    st.write(f"**Cache age:** {elapsed}s")

# ========= Tabs =========
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Price & Trades", "💰 P&L Analysis", "📜 Trade History", "🚦 Validations",
    "🚫 Missed Entries", "📊 Overall Stats"
])

# ----- TAB 1: Price & Trades -----
with tab1:
    if (market_df.empty) or ("asset" not in market_df.columns) or market_df["asset"].dropna().empty:
        st.warning("Market data not available or missing assets.")
    else:
        assets = sorted(market_df["asset"].dropna().unique().tolist())
        default_index = assets.index(DEFAULT_ASSET) if DEFAULT_ASSET in assets else 0
        selected_asset = st.selectbox("Select Asset to View", assets, index=min(default_index, len(assets)-1), key="asset_select_main")
        selected_asset = st.session_state.get("asset_select_main", selected_asset)

        asset_market_raw = market_df[market_df["asset"] == selected_asset].copy()
        if "__x__" not in asset_market_raw.columns and "timestamp_pst" in asset_market_raw.columns:
            # safety: create plotting x if missing
            t = pd.to_datetime(asset_market_raw["timestamp_pst"], errors="coerce")
            try:
                if getattr(t.dt, "tz", None) is None:
                    t = t.dt.tz_localize(TZ_PST)
                else:
                    t = t.dt.tz_convert(TZ_PST)
            except Exception:
                t = pd.to_datetime(asset_market_raw["timestamp_pst"], errors="coerce", utc=True).dt.tz_convert(TZ_PST)
            asset_market_raw["__x__"] = t.dt.tz_localize(None)

        asset_market_raw = asset_market_raw.sort_values("__x__")

        # Split rows with complete OHLC vs incomplete (candlestick requires all)
        ohlc_cols = ["open","high","low","close"]
        complete_mask = asset_market_raw[ohlc_cols].notna().all(axis=1)
        asset_candles = asset_market_raw.loc[complete_mask].copy()
        asset_incomplete = asset_market_raw.loc[~complete_mask].copy()

        # Metric
        last_close = asset_market_raw["close"].dropna().iloc[-1] if asset_market_raw["close"].notna().any() else np.nan
        if pd.notna(last_close):
            pf = ",.8f" if last_close < 0.001 else ",.6f" if last_close < 1 else ",.4f"
            st.metric(f"Last Price for {selected_asset}", f"${last_close:{pf}}")

        fig = go.Figure()

        # 1) Candlesticks
        if not asset_candles.empty:
            fig.add_trace(go.Candlestick(
                x=asset_candles["__x__"],
                open=asset_candles["open"], high=asset_candles["high"],
                low=asset_candles["low"], close=asset_candles["close"],
                name=f"{selected_asset} (candles)",
                increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
                increasing_fillcolor="rgba(38,166,154,0.5)", decreasing_fillcolor="rgba(239,83,80,0.5)",
                line=dict(width=1)
            ))

        # 2) Fallback line for incomplete OHLC (keeps newest visible)
        if not asset_incomplete.empty and asset_incomplete[["__x__","close"]].dropna().shape[0] > 0:
            tmp = asset_incomplete.dropna(subset=["__x__","close"])
            fig.add_trace(go.Scatter(
                x=tmp["__x__"], y=tmp["close"],
                mode="lines+markers", name="Price (incomplete OHLC)",
                line=dict(width=1, dash="dot"),
                hovertemplate="<b>Price (incomplete OHLC)</b><br>%{x|%Y-%m-%d %H:%M}<br>Close: $%{y:.8f}<extra></extra>",
            ))
            st.info("Newest rows have missing O/H/L/C; showing a dashed close-price line for those rows.")

        # 3) ML signals
        if {"p_up", "p_down", "close", "__x__"}.issubset(asset_market_raw.columns):
            sig = asset_market_raw.dropna(subset=["__x__","p_up","p_down","close"]).copy()
            if not sig.empty:
                sig["p_up"] = pd.to_numeric(sig["p_up"], errors="coerce")
                sig["p_down"] = pd.to_numeric(sig["p_down"], errors="coerce")
                sig = sig.dropna(subset=["p_up","p_down"])
                sig["confidence"] = (sig["p_up"] - sig["p_down"]).abs()
                sizes = (sig["confidence"] * 100.0) / 5.0 + 3.0
                colors = np.where(sig["p_down"] > sig["p_up"], "#ff6b6b", "#51cf66")
                fig.add_trace(go.Scatter(
                    x=sig["__x__"], y=sig["close"], mode="markers", name="ML Signals",
                    marker=dict(size=sizes, color=colors, opacity=0.7, line=dict(width=1, color="white")),
                    customdata=list(zip(sig["p_up"], sig["p_down"], sig["confidence"])),
                    hovertemplate="<b>ML Signal</b><br>%{x|%Y-%m-%d %H:%M}<br>Close: $%{y:.6f}"
                                  "<br>P(Up): %{customdata[0]:.3f}"
                                  "<br>P(Down): %{customdata[1]:.3f}"
                                  "<br>Confidence: %{customdata[2]:.3f}<extra></extra>",
                ))

        # 4) Trade markers — convert to the same tz-naive PST x
        marker_y = None
        yref = asset_candles["low"] if not asset_candles.empty else asset_market_raw["close"].dropna()
        if not yref.empty:
            ymin, ymax = float(yref.min()), float(yref.max())
            marker_y = ymin - max(1e-12, (ymax - ymin)) * 0.02

        if marker_y is not None and (not trades_df.empty) and ("timestamp_pst" in trades_df.columns):
            tdf = trades_df[trades_df["asset"] == selected_asset].copy()
            if not tdf.empty:
                tx = pd.to_datetime(tdf["timestamp_pst"], errors="coerce")
                try:
                    if getattr(tx.dt, "tz", None) is None:
                        tx = tx.dt.tz_localize(TZ_PST)
                    else:
                        tx = tx.dt.tz_convert(TZ_PST)
                except Exception:
                    tx = pd.to_datetime(tdf["timestamp_pst"], errors="coerce", utc=True).dt.tz_convert(TZ_PST)
                tdf["__x__"] = tx.dt.tz_localize(None)

                buys = tdf[tdf["unified_action"].str.lower().isin(["buy","open"])]
                sells = tdf[tdf["unified_action"].str.lower().isin(["sell","close"])]

                if not buys.empty:
                    fig.add_trace(go.Scatter(
                        x=buys["__x__"], y=[marker_y]*len(buys), mode="markers", name="BUY",
                        marker=dict(symbol="triangle-up", size=14, color="#4caf50", line=dict(width=1, color="white")),
                        hovertemplate="<b>BUY</b><br>%{x|%H:%M:%S}<extra></extra>",
                    ))
                if not sells.empty:
                    fig.add_trace(go.Scatter(
                        x=sells["__x__"], y=[marker_y]*len(sells), mode="markers", name="SELL",
                        marker=dict(symbol="triangle-down", size=14, color="#f44336", line=dict(width=1, color="white")),
                        hovertemplate="<b>SELL</b><br>%{x|%H:%M:%S}<extra></extra>",
                    ))

        # 5) Layout + rangeslider
        fig.update_layout(
            title=f"{selected_asset} — Full History",
            template="plotly_dark" if st.session_state.theme == "dark" else "plotly_white",
            xaxis=dict(
                title="Time (PST)",
                type="date",
                rangeslider=dict(visible=True),
                rangeselector=dict(
                    buttons=[
                        dict(count=4, label="4h", step="hour", stepmode="backward"),
                        dict(count=12, label="12h", step="hour", stepmode="backward"),
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(count=3, label="3d", step="day", stepmode="backward"),
                        dict(count=7, label="7d", step="day", stepmode="backward"),
                        dict(step="all", label="All"),
                    ]
                ),
            ),
            yaxis=dict(title="Price (USD)"),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            height=780, margin=dict(l=60, r=20, t=80, b=80),
            plot_bgcolor="rgba(14,17,23,1)" if st.session_state.theme == "dark" else "rgba(250,250,250,0.8)",
            paper_bgcolor="rgba(14,17,23,1)" if st.session_state.theme == "dark" else "rgba(255,255,255,1)",
            font_color="#FAFAFA" if st.session_state.theme == "dark" else "#262626",
        )

        st.plotly_chart(fig, use_container_width=True,
            config={"displayModeBar": True, "modeBarButtonsToAdd": ["drawline","drawopenpath","drawclosedpath"], "scrollZoom": True})

        with st.expander("📎 Latest rows (quick check)"):
            tail_cols = [c for c in ["__x__","timestamp_pst","open","high","low","close","p_up","p_down"] if c in asset_market_raw.columns]
            st.dataframe(asset_market_raw[tail_cols].tail(15), use_container_width=True)

# ----- TAB 2: P&L Analysis -----
with tab2:
    if trades_df.empty or "timestamp_pst" not in trades_df.columns:
        st.warning("No trade data loaded to analyze P&L or missing timestamp_pst column.")
    else:
        pnl_summary, trades_with_pnl, stats = calculate_pnl_and_metrics(trades_df)
        overall_realized, per_asset_pnl = summarize_realized_pnl(trades_with_pnl)

        st.markdown("### Strategy Performance")
        c0, c1, c2, c3, c4 = st.columns(5)
        with c0: st.metric("Overall Realized P&L", f"${overall_realized:+.2f}")
        with c1: st.metric("Closed Trades", f"{stats.get('total_trades', 0):,}")
        with c2: st.metric("Win Rate", f"{stats.get('win_rate', 0):.1f}%")
        with c3:
            pf = stats.get("profit_factor", 0)
            st.metric("Profit Factor", "∞" if isinstance(pf, float) and np.isinf(pf) else f"{pf:.2f}")
        with c4: st.metric("Max Drawdown", f"${stats.get('max_drawdown', 0):.2f}")

        if not trades_with_pnl.empty:
            plot_df = trades_with_pnl.copy().sort_values("timestamp_pst")
            plot_df["cumulative_pnl"] = plot_df["cumulative_pnl"].apply(float)
            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Scatter(x=plot_df["timestamp_pst"], y=plot_df["cumulative_pnl"], mode="lines", name="Cumulative P&L", line=dict(width=2)))
            fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Break Even")
            chart_bg = "rgba(14,17,23,1)" if st.session_state.theme == "dark" else "rgba(255,255,255,1)"
            chart_grid = "rgba(59,66,82,0.3)" if st.session_state.theme == "dark" else "rgba(128,128,128,0.1)"
            chart_text = "#FAFAFA" if st.session_state.theme == "dark" else "#262626"
            fig_pnl.update_layout(title="Total Portfolio P&L (PST)", template="plotly_dark" if st.session_state.theme == "dark" else "plotly_white",
                                  yaxis_title="P&L (USD)", xaxis_title="Date (PST)", hovermode="x unified",
                                  plot_bgcolor=chart_bg, paper_bgcolor=chart_bg, font_color=chart_text,
                                  xaxis=dict(gridcolor=chart_grid), yaxis=dict(gridcolor=chart_grid),
                                  xaxis_rangeslider_visible=True)
            st.plotly_chart(fig_pnl, use_container_width=True)

        st.markdown("#### Per-Asset Realized P&L")
        if not per_asset_pnl.empty:
            st.dataframe(
                per_asset_pnl,
                column_config={
                    "Asset": st.column_config.TextColumn(width="small"),
                    "Realized P&L ($)": st.column_config.NumberColumn(format="$%.2f"),
                },
                use_container_width=True,
                hide_index=True,
            )

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=per_asset_pnl["Asset"],
                y=per_asset_pnl["Realized P&L ($)"],
                name="Realized P&L ($)"
            ))
            fig_bar.add_hline(y=0, line_dash="dash", line_color="gray")
            chart_bg = "rgba(14,17,23,1)" if st.session_state.theme == "dark" else "rgba(255,255,255,1)"
            chart_text = "#FAFAFA" if st.session_state.theme == "dark" else "#262626"
            fig_bar.update_layout(
                title="Per-Asset Realized P&L",
                template="plotly_dark" if st.session_state.theme == "dark" else "plotly_white",
                yaxis_title="P&L (USD)", xaxis_title="Asset",
                hovermode="x unified",
                plot_bgcolor=chart_bg, paper_bgcolor=chart_bg, font_color=chart_text,
                height=420
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No realized P&L yet (need at least one completed sell to compute).")

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
            display_matched["Buy Time"]  = display_matched["Buy Time"].dt.strftime("%Y-%m-%d %H:%M:%S")
            display_matched["Sell Time"] = display_matched["Sell Time"].dt.strftime("%Y-%m-%d %H:%M:%S")
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
            display_open["Time"] = display_open["timestamp_pst"].dt.strftime("%Y-%m-%d %H:%M:%S")
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
                invalid_disp = invalid.copy()
                if "timestamp_pst" in invalid_disp.columns:
                    invalid_disp["timestamp_pst_str"] = invalid_disp["timestamp_pst"].dt.strftime("%Y-%m-%d %H:%M:%S")
                    show_cols = [c for c in ["asset","price","p_up","p_down","confidence","reason","violation_reason"] if c in invalid_disp.columns]
                    display_cols = ["timestamp_pst_str"] + show_cols
                    sort_col = "timestamp_pst_str"
                else:
                    show_cols = [c for c in ["asset","price","p_up","p_down","confidence","reason","violation_reason"] if c in invalid_disp.columns]
                    display_cols = show_cols; sort_col = show_cols[0] if show_cols else None
                
                st.markdown("#### ❗ Flagged OPENs")
                if display_cols and sort_col:
                    st.dataframe(invalid_disp[display_cols].sort_values(sort_col), use_container_width=True, hide_index=True)
                else:
                    st.dataframe(invalid_disp, use_container_width=True, hide_index=True)
            else:
                st.success("No violations detected 🎉")
            st.caption("Open = buy execution records; validity is evaluated against per-asset thresholds and logged probabilities at that time.")

# ----- TAB 5: Missed Entries -----
with tab5:
    st.markdown("### Position-Aware Missed Signals")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Missed BUYs (add to position)", f"{len(missed_buys_df):,}")
    with col2: st.metric("Missed SELLs (in position)", f"{len(missed_sells_df):,}")
    with col3:
        try_buys = len(missed_buys_df) if 'missed_buys_try' not in locals() else len(missed_buys_try)
        st.metric("Missed BUYs (what-if)", f"{try_buys:,}")
    with col4:
        try_sells = len(missed_sells_df) if 'missed_sells_try' not in locals() else len(missed_sells_try)
        st.metric("Missed SELLs (what-if)", f"{try_sells:,}")

    all_assets = set()
    if not missed_buys_df.empty: all_assets.update(missed_buys_df['asset'].unique())
    if not missed_sells_df.empty: all_assets.update(missed_sells_df['asset'].unique())
    sel_assets = st.multiselect("Filter by asset", sorted(list(all_assets)), default=sorted(list(all_assets))) if all_assets else []

    if not missed_buys_df.empty:
        st.markdown("#### Missed BUY Signals (when in position - add to position)")
        filtered_buys = missed_buys_df[missed_buys_df['asset'].isin(sel_assets)] if sel_assets else missed_buys_df
        display_buys = filtered_buys.copy()
        if "timestamp_pst" in display_buys.columns:
            display_buys['timestamp'] = display_buys['timestamp_pst'].dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(display_buys[['asset', 'timestamp', 'price', 'p_up', 'p_down', 'confidence']],
                     column_config={'price': st.column_config.NumberColumn(format="$%.8f"),
                                    'p_up': st.column_config.NumberColumn(format="%.3f"),
                                    'p_down': st.column_config.NumberColumn(format="%.3f"),
                                    'confidence': st.column_config.NumberColumn(format="%.3f")},
                     use_container_width=True, hide_index=True)

    if not missed_sells_df.empty:
        st.markdown("#### Missed SELL Signals (when in position)")
        filtered_sells = missed_sells_df[missed_sells_df['asset'].isin(sel_assets)] if sel_assets else missed_sells_df
        display_sells = filtered_sells.copy()
        if "timestamp_pst" in display_sells.columns:
            display_sells['timestamp'] = display_sells['timestamp_pst'].dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(display_sells[['asset', 'timestamp', 'price', 'p_up', 'p_down', 'confidence']],
                     column_config={'price': st.column_config.NumberColumn(format="$%.8f"),
                                    'p_up': st.column_config.NumberColumn(format="%.3f"),
                                    'p_down': st.column_config.NumberColumn(format="%.3f"),
                                    'confidence': st.column_config.NumberColumn(format="%.3f")},
                     use_container_width=True, hide_index=True)

    if not missed_buys_df.empty or not missed_sells_df.empty:
        combined_missed = []
        if not missed_buys_df.empty:
            buys_export = missed_buys_df.copy(); buys_export['signal_type'] = 'missed_buy'
            if "timestamp_pst" in buys_export.columns: buys_export['timestamp'] = buys_export['timestamp_pst'].dt.strftime("%Y-%m-%d %H:%M:%S")
            combined_missed.append(buys_export)
        if not missed_sells_df.empty:
            sells_export = missed_sells_df.copy(); sells_export['signal_type'] = 'missed_sell'
            if "timestamp_pst" in sells_export.columns: sells_export['timestamp'] = sells_export['timestamp_pst'].dt.strftime("%Y-%m-%d %H:%M:%S")
            combined_missed.append(sells_export)
        if combined_missed:
            export_df = pd.concat(combined_missed, ignore_index=True)
            st.download_button("⬇️ Download Missed Signals CSV",
                               data=export_df.to_csv(index=False).encode("utf-8"),
                               file_name="missed_signals.csv", mime="text/csv")

# ----- TAB 6: Overall Stats -----
with tab6:
    st.markdown("### 📊 Overall Stats (Position-Aware Analysis)")
    colA, colB, colC, colD = st.columns(4)
    with colA: st.metric("Missed BUYs (add to pos)", f"{len(missed_buys_df):,}")
    with colB: st.metric("Missed SELLs (in pos)", f"{len(missed_sells_df):,}")
    with colC: st.metric("Total Position-Aware", f"{len(missed_buys_df) + len(missed_sells_df):,}")
    action_col = "action" if "action" in trades_df.columns else ("unified_action" if "unified_action" in trades_df.columns else None)
    if not trades_df.empty and action_col:
        opens_mask = trades_df[action_col].astype(str).str.upper().isin(["OPEN","BUY"])
        opens = trades_df.loc[opens_mask].copy()
        total_opens = len(opens); valid_opens = len(opens["valid_at_open"]) if "valid_at_open" in opens.columns else total_opens
        vr = 0 if total_opens == 0 else (valid_opens / total_opens) * 100
    else:
        total_opens, valid_opens, vr = 0, 0, 0
    with colD: st.metric("OPEN Validity Rate", f"{vr:.1f}%")

    st.markdown("---")
    if not missed_buys_df.empty or not missed_sells_df.empty:
        st.markdown("#### Position-Aware Missed Signals by Asset")
        buy_counts = missed_buys_df.groupby('asset').size().rename('missed_buys') if not missed_buys_df.empty else pd.Series(name='missed_buys', dtype=int)
        sell_counts = missed_sells_df.groupby('asset').size().rename('missed_sells') if not missed_sells_df.empty else pd.Series(name='missed_sells', dtype=int)
        combined = pd.concat([buy_counts, sell_counts], axis=1).fillna(0).astype(int)
        combined['total_missed'] = combined['missed_buys'] + combined['missed_sells']
        combined = combined.sort_values('total_missed', ascending=False)
        st.dataframe(combined.reset_index(), use_container_width=True, hide_index=True)

    if 'missed_buys_try' in locals() and 'missed_sells_try' in locals():
        st.markdown("---")
        st.markdown("#### Current vs What-If Comparison")
        current_total = len(missed_buys_df) + len(missed_sells_df)
        try_total = len(missed_buys_try) + len(missed_sells_try)
        improvement = current_total - try_total
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Current Total", f"{current_total:,}")
        with col2: st.metric("What-If Total", f"{try_total:,}")
        with col3: st.metric("Improvement", f"{improvement:+d}")

    st.markdown("---")
    st.caption(
        "The position-aware analysis shows missed BUY signals only when you were in a position "
        "(opportunities to add to positions), and missed SELL signals only when you were in a position. "
        "This provides insights for position sizing and adding to existing holdings based on strong signals."
    )

# ====== Sidebar: Open Positions compact ======
with st.sidebar:
    st.markdown("---")
    open_positions_df = calculate_open_positions(trades_df, market_df) if (not trades_df.empty and not market_df.empty) else pd.DataFrame()
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
