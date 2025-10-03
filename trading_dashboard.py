"""
Crypto Trading Strategy Dashboard
Single-file version with responsive charts and optimized performance
"""

import io
import re
import time
from decimal import Decimal, getcontext
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# =====================[ CONFIGURATION ]=====================
st.set_page_config(page_title="Crypto Trading Strategy", layout="wide")
getcontext().prec = 30

TZ_PST = "America/Los_Angeles"
DEFAULT_ASSET = "GIGA-USD"

TRADES_LINK = "https://drive.google.com/file/d/1cSeYf4kWJA49lMdQ1_yQ3hRui6wdrLD-/view?usp=drive_link"
MARKET_LINK = "https://drive.google.com/file/d/1cCECtOBtPAQBvbmpfjbvn7sdUSi0qgK5/view?usp=sharing"

ASSET_THRESHOLDS = {
    "CVX-USD": {"buy_threshold": 0.70, "min_confidence": 0.60, "sell_threshold": 0.30},
    "MNDE-USD": {"buy_threshold": 0.68, "min_confidence": 0.60, "sell_threshold": 0.32},
    "MOG-USD": {"buy_threshold": 0.75, "min_confidence": 0.60, "sell_threshold": 0.25},
    "VVV-USD": {"buy_threshold": 0.65, "min_confidence": 0.60, "sell_threshold": 0.35},
    "LCX-USD": {"buy_threshold": 0.72, "min_confidence": 0.60, "sell_threshold": 0.28},
    "GIGA-USD": {"buy_threshold": 0.73, "min_confidence": 0.60, "sell_threshold": 0.27},
}

# Responsive chart heights (mobile-first approach)
CHART_HEIGHTS = {
    'mobile': 450,      # Fits phone screens
    'tablet': 600,      # Tablets
    'desktop': 750,     # Desktop
    'compact': 400,     # Compact views
}
DEFAULT_CHART_HEIGHT = CHART_HEIGHTS['mobile']

# =====================[ UTILITY FUNCTIONS ]=====================

def get_responsive_chart_height(size: str = 'default') -> int:
    """Get appropriate chart height for different contexts."""
    if size == 'default':
        return DEFAULT_CHART_HEIGHT
    return CHART_HEIGHTS.get(size, DEFAULT_CHART_HEIGHT)

def format_price(price: float) -> str:
    """Format price with appropriate precision."""
    if price < 0.001:
        return f"${price:.8f}"
    elif price < 1:
        return f"${price:.6f}"
    else:
        return f"${price:.4f}"

def format_timedelta_hhmm(td):
    """Format timedelta as HH:MM."""
    if pd.isna(td):
        return "N/A"
    total_seconds = int(td.total_seconds())
    m, _ = divmod(total_seconds, 60)
    h, mm = divmod(m, 60)
    return f"{h:02d}:{mm:02d}"

def seconds_since_last_run() -> int:
    """Calculate seconds since last refresh."""
    return int(time.time() - st.session_state.get("last_refresh", 0))

# =====================[ DATA LOADING ]=====================

def _extract_file_id(url_or_id: str) -> str:
    """Extract Google Drive file ID from URL."""
    if not url_or_id:
        return ""
    patterns = [
        r'/d/([a-zA-Z0-9_-]{25,})',
        r'id=([a-zA-Z0-9_-]{25,})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    return url_or_id.strip()

def _download_drive_bytes(url_or_id: str, label: str = "File") -> bytes | None:
    """Download from Google Drive."""
    file_id = _extract_file_id(url_or_id)
    if not file_id:
        st.error(f"{label}: Invalid file ID")
        return None
    
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    try:
        session = requests.Session()
        response = session.get(download_url, stream=True, timeout=30)
        
        # Handle virus scan confirmation
        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break
        
        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(download_url, params=params, stream=True, timeout=30)
        
        if response.status_code != 200:
            st.error(f"{label}: HTTP {response.status_code}")
            return None
        
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' in content_type:
            st.error(f"{label}: Got HTML. Check sharing permissions.")
            return None
        
        data = response.content
        if len(data) == 0:
            st.error(f"{label}: Downloaded 0 bytes")
            return None
        
        return data
    except requests.exceptions.Timeout:
        st.error(f"{label}: Download timeout")
        return None
    except Exception as e:
        st.error(f"{label}: Download failed - {e}")
        return None

def _read_parquet_or_csv(b: bytes, label: str) -> pd.DataFrame:
    """Read bytes as parquet or CSV."""
    if not b:
        st.warning(f"{label}: no bytes")
        return pd.DataFrame()
    if b[:4] != b"PAR1":
        try:
            return pd.read_csv(io.BytesIO(b))
        except Exception:
            st.error(f"{label}: not Parquet and CSV failed")
            return pd.DataFrame()
    try:
        return pd.read_parquet(io.BytesIO(b))
    except Exception as e:
        st.error(f"{label}: failed to read: {e}")
        return pd.DataFrame()

def _ensure_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure timezone-aware timestamp columns."""
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
    
    # Reorder columns
    cols = list(df.columns)
    if "timestamp" in cols and "timestamp_pst" in cols:
        c = cols.pop(cols.index("timestamp_pst"))
        cols.insert(cols.index("timestamp") + 1, c)
        df = df[cols]
    
    return df

def _unify_symbol(val: str) -> str:
    """Unify asset symbol format."""
    if not isinstance(val, str):
        return val
    s = val.strip().upper().replace("_", "-")
    if "GIGA" in s:
        return "GIGA-USD"
    return s

def normalize_dataframe(df: pd.DataFrame, is_trades: bool = False) -> pd.DataFrame:
    """Centralized normalization for all dataframes."""
    if df.empty:
        return df
    
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]
    df = _ensure_time_columns(df)
    
    if is_trades:
        colmap = {}
        if "value" in df.columns:
            colmap["value"] = "usd_value"
        if "side" in df.columns and "action" in df.columns:
            colmap["side"] = "trade_direction"
        elif "side" in df.columns:
            colmap["side"] = "action"
        
        df = df.rename(columns=colmap)
        
        if "action" in df.columns:
            df["unified_action"] = (
                df["action"].astype(str).str.upper()
                .map({"OPEN": "buy", "CLOSE": "sell"})
                .fillna(df["action"].astype(str).str.lower())
            )
        
        for col in ["quantity", "price", "usd_value", "pnl", "pnl_pct"]:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: Decimal(str(x)) if pd.notna(x) else Decimal(0)
                )
    else:
        if "product_id" in df.columns:
            df = df.rename(columns={"product_id": "asset"})
        
        rename_map = {}
        for col in df.columns:
            norm = col.lower().replace("_", " ").replace("-", " ")
            if norm in ["p up", "pup", "p_up"]:
                rename_map[col] = "p_up"
            elif norm in ["p down", "pdown", "p_down"]:
                rename_map[col] = "p_down"
        
        if rename_map:
            df = df.rename(columns=rename_map)
        
        if "p_up" not in df.columns:
            df["p_up"] = np.nan
        if "p_down" not in df.columns:
            df["p_down"] = np.nan
        
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    
    if "asset" in df.columns:
        df["asset"] = df["asset"].apply(_unify_symbol)
    
    return df

@st.cache_data(ttl=300, show_spinner=False)
def load_data(trades_link: str, market_link: str):
    """Load and normalize data from Google Drive."""
    trades = pd.DataFrame()
    tb = _download_drive_bytes(trades_link, "Trades")
    if tb:
        trades = _read_parquet_or_csv(tb, "Trades")
        trades = normalize_dataframe(trades, is_trades=True)
    
    market = pd.DataFrame()
    mb = _download_drive_bytes(market_link, "Market")
    if mb:
        market = _read_parquet_or_csv(mb, "Market")
        market = normalize_dataframe(market, is_trades=False)
    
    return trades, market

def read_uploaded_market(uploaded) -> pd.DataFrame:
    """Read uploaded market file."""
    if uploaded is None:
        return pd.DataFrame()
    
    name = (uploaded.name or "").lower()
    try:
        if name.endswith(".parquet") or name.endswith(".pq"):
            df = pd.read_parquet(uploaded)
        else:
            df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read: {e}")
        return pd.DataFrame()
    
    return normalize_dataframe(df, is_trades=False)

# =====================[ CALCULATIONS ]=====================

def calculate_pnl_and_metrics(trades_df: pd.DataFrame):
    """Calculate P&L and trading statistics."""
    if trades_df is None or trades_df.empty or "timestamp_pst" not in trades_df.columns:
        return {}, pd.DataFrame(), {}
    
    df = trades_df.copy().sort_values("timestamp_pst").reset_index(drop=True)
    
    pnl_per_asset, positions = {}, {}
    df["pnl"], df["cumulative_pnl"] = Decimal(0), Decimal(0)
    total, win, loss = Decimal(0), 0, 0
    gp, gl = Decimal(0), Decimal(0)
    peak, mdd = Decimal(0), Decimal(0)
    
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
    
    return pnl_per_asset, df, stats

def summarize_realized_pnl(trades_with_pnl: pd.DataFrame):
    """Summarize realized P&L per asset."""
    if trades_with_pnl is None or trades_with_pnl.empty or "pnl" not in trades_with_pnl.columns:
        return 0.0, pd.DataFrame(columns=["Asset", "Realized P&L ($)"])
    
    df = trades_with_pnl.copy()
    df["pnl_float"] = df["pnl"].apply(lambda x: float(x) if pd.notna(x) else 0.0)
    
    per_asset = (
        df.groupby("asset")["pnl_float"]
        .sum()
        .reset_index()
        .rename(columns={"asset": "Asset", "pnl_float": "Realized P&L ($)"})
        .sort_values("Realized P&L ($)", ascending=False)
    )
    
    overall = float(per_asset["Realized P&L ($)"].sum()) if not per_asset.empty else 0.0
    return overall, per_asset

def match_trades_fifo(trades_df: pd.DataFrame):
    """Match buy/sell trades using FIFO."""
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
                if buy_ts >= sell_ts:
                    break
                
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
                    })
                    
                    sell_qty -= trade_qty
                    buys[0]["quantity"] -= trade_qty
                    if buys[0]["quantity"] < Decimal("1e-9"):
                        buys.pop(0)
        
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
    """Calculate current open positions."""
    if (trades_df is None or trades_df.empty or "timestamp_pst" not in trades_df.columns or
        market_df is None or market_df.empty or "timestamp_pst" not in market_df.columns):
        return pd.DataFrame()
    
    positions = {}
    position_open_times = {}
    tdf = trades_df.copy().sort_values("timestamp_pst")
    
    for _, row in tdf.iterrows():
        asset = row.get("asset", "")
        action = str(row.get("unified_action", "")).lower().strip()
        qty = row.get("quantity", Decimal(0))
        price = row.get("price", Decimal(0))
        timestamp_pst = row.get("timestamp_pst", "")
        
        if asset not in positions:
            positions[asset] = {"quantity": Decimal(0), "cost": Decimal(0)}
        
        if action in ["buy", "open"]:
            if positions[asset]["quantity"] == 0:
                position_open_times[asset] = timestamp_pst
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
                idx = latest_asset_rows["timestamp_pst"].idxmax()
                latest_price = Decimal(str(latest_asset_rows.loc[idx, "close"]))
                avg_entry = data["cost"] / data["quantity"]
                current_value = latest_price * data["quantity"]
                unrealized = current_value - data["cost"]
                
                open_time_str = "N/A"
                if asset in position_open_times:
                    try:
                        open_ts = position_open_times[asset]
                        if pd.notna(open_ts):
                            open_time_str = open_ts.strftime("%H:%M")
                    except:
                        pass
                
                open_positions.append({
                    "Asset": asset,
                    "Quantity": float(data["quantity"]),
                    "Avg. Entry Price": float(avg_entry),
                    "Current Price": float(latest_price),
                    "Current Value ($)": float(current_value),
                    "Unrealized P&L ($)": float(unrealized),
                    "Open Time": open_time_str,
                })
    
    return pd.DataFrame(open_positions)

def flag_threshold_violations(trades: pd.DataFrame) -> pd.DataFrame:
    """Flag trades that violated thresholds."""
    if trades.empty:
        return trades.copy()
    
    df = trades.copy()
    
    for col in ["p_up", "p_down", "confidence"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    if "confidence" not in df.columns:
        df["confidence"] = df.apply(
            lambda r: abs(r.get("p_up", 0) - r.get("p_down", 0))
            if pd.notna(r.get("p_up")) and pd.notna(r.get("p_down"))
            else np.nan,
            axis=1
        )
    
    action_col = "action" if "action" in df.columns else "unified_action"
    val_flags, reasons = [], []
    
    for _, row in df.iterrows():
        is_open = False
        if action_col:
            val = str(row.get(action_col, "")).upper()
            is_open = val in ["OPEN", "BUY"]
        
        if not is_open:
            val_flags.append(True)
            reasons.append("")
            continue
        
        asset = row.get("asset")
        th = ASSET_THRESHOLDS.get(asset)
        
        if th is None:
            val_flags.append(True)
            reasons.append("")
            continue
        
        pu = row.get("p_up")
        pdn = row.get("p_down")
        conf = row.get("confidence")
        
        r = []
        if pd.isna(pu) or pd.isna(pdn) or pd.isna(conf):
            r.append("missing probabilities")
        else:
            if not (pu > pdn):
                r.append("p_up ≤ p_down")
            if not (pu >= th["buy_threshold"]):
                r.append(f"p_up {pu:.3f} < buy_threshold {th['buy_threshold']:.2f}")
            if not (conf >= th["min_confidence"]):
                r.append(f"confidence {conf:.3f} < min_confidence {th['min_confidence']:.2f}")
        
        val_flags.append(len(r) == 0)
        reasons.append("; ".join(r))
    
    df["valid_at_open"] = val_flags
    df["violation_reason"] = reasons
    
    return df

# =====================[ VISUALIZATIONS ]=====================

def create_price_chart(asset_market: pd.DataFrame, asset_trades: pd.DataFrame, 
                       selected_asset: str, theme: str) -> go.Figure:
    """Create price chart with responsive height."""
    asset_market = asset_market.copy()
    if "timestamp_pst" in asset_market.columns:
        asset_market["__x__"] = pd.to_datetime(asset_market["timestamp_pst"]).dt.tz_localize(None)
    
    need_cols = ["__x__", "open", "high", "low", "close"]
    present_cols = [c for c in need_cols if c in asset_market.columns]
    asset_candles = asset_market.dropna(subset=present_cols).copy()
    
    if not asset_candles.empty:
        asset_candles = asset_candles.drop_duplicates(subset="__x__", keep="last")
    
    fig = go.Figure()
    
    if not asset_candles.empty:
        fig.add_trace(go.Candlestick(
            x=asset_candles["__x__"], open=asset_candles["open"],
            high=asset_candles["high"], low=asset_candles["low"],
            close=asset_candles["close"], name=selected_asset,
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
            increasing_fillcolor="rgba(38,166,154,0.5)",
            decreasing_fillcolor="rgba(239,83,80,0.5)", line=dict(width=1)
        ))
    
    if {"p_up", "p_down", "close", "__x__"}.issubset(asset_market.columns):
        sig = asset_market.dropna(subset=["__x__", "p_up", "p_down", "close"]).copy()
        if not sig.empty:
            sig["p_up"] = pd.to_numeric(sig["p_up"], errors="coerce")
            sig["p_down"] = pd.to_numeric(sig["p_down"], errors="coerce")
            sig = sig.dropna(subset=["p_up", "p_down"])
            sig["confidence"] = (sig["p_up"] - sig["p_down"]).abs()
            
            sizes = (sig["confidence"] * 100.0) / 5.0 + 3.0
            colors = np.where(sig["p_down"] > sig["p_up"], "#ff6b6b", "#51cf66")
            
            fig.add_trace(go.Scatter(
                x=sig["__x__"], y=sig["close"], mode="markers", name="ML Signals",
                marker=dict(size=sizes, color=colors, opacity=0.7, line=dict(width=1, color="white")),
                customdata=list(zip(sig["p_up"], sig["p_down"], sig["confidence"])),
                hovertemplate=(
                    "<b>ML Signal</b><br>%{x|%Y-%m-%d %H:%M}<br>Close: $%{y:.6f}<br>"
                    "P(Up): %{customdata[0]:.3f}<br>P(Down): %{customdata[1]:.3f}<br>"
                    "Confidence: %{customdata[2]:.3f}<extra></extra>"
                ),
            ))
    
    marker_y = None
    yref = asset_candles["low"] if not asset_candles.empty else asset_market.get("close", pd.Series())
    if not yref.empty:
        ymin, ymax = float(yref.min()), float(yref.max())
        marker_y = ymin - max(1e-12, (ymax - ymin)) * 0.02
    
    if marker_y is not None and not asset_trades.empty:
        tdf = asset_trades.copy()
        if "timestamp_pst" in tdf.columns:
            tdf["__x__"] = pd.to_datetime(tdf["timestamp_pst"]).dt.tz_localize(None)
            
            buys = tdf[tdf["unified_action"].str.lower().isin(["buy", "open"])]
            sells = tdf[tdf["unified_action"].str.lower().isin(["sell", "close"])]
            
            if not buys.empty:
                fig.add_trace(go.Scatter(
                    x=buys["__x__"], y=[marker_y] * len(buys), mode="markers", name="BUY",
                    marker=dict(symbol="triangle-up", size=14, color="#4caf50", line=dict(width=1, color="white")),
                    hovertemplate="<b>BUY</b><br>%{x|%H:%M:%S}<extra></extra>",
                ))
            
            if not sells.empty:
                fig.add_trace(go.Scatter(
                    x=sells["__x__"], y=[marker_y] * len(sells), mode="markers", name="SELL",
                    marker=dict(symbol="triangle-down", size=14, color="#f44336", line=dict(width=1, color="white")),
                    hovertemplate="<b>SELL</b><br>%{x|%H:%M:%S}<extra></extra>",
                ))
    
    chart_height = get_responsive_chart_height()
    
    fig.update_layout(
        title=f"{selected_asset} — Full History",
        template="plotly_dark" if theme == "dark" else "plotly_white",
        xaxis=dict(
            title="Time (PST)", type="date", rangeslider=dict(visible=True),
            rangeselector=dict(buttons=[
                dict(count=4, label="4h", step="hour", stepmode="backward"),
                dict(count=12, label="12h", step="hour", stepmode="backward"),
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(count=3, label="3d", step="day", stepmode="backward"),
                dict(count=7, label="7d", step="day", stepmode="backward"),
                dict(step="all", label="All"),
            ])
        ),
        yaxis=dict(title="Price (USD)"), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=chart_height, margin=dict(l=60, r=20, t=80, b=80),
        plot_bgcolor="rgba(14,17,23,1)" if theme == "dark" else "rgba(250,250,250,0.8)",
        paper_bgcolor="rgba(14,17,23,1)" if theme == "dark" else "rgba(255,255,255,1)",
        font_color="#FAFAFA" if theme == "dark" else "#262626",
    )
    
    return fig

def create_pnl_chart(trades_with_pnl: pd.DataFrame, theme: str) -> go.Figure:
    """Create cumulative P&L chart with responsive height."""
    plot_df = trades_with_pnl.copy().sort_values("timestamp_pst")
    plot_df["cumulative_pnl"] = plot_df["cumulative_pnl"].apply(float)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df["timestamp_pst"], y=plot_df["cumulative_pnl"],
        mode="lines", name="Cumulative P&L", line=dict(width=2)
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Break Even")
    
    chart_height = get_responsive_chart_height()
    chart_bg = "rgba(14,17,23,1)" if theme == "dark" else "rgba(255,255,255,1)"
    chart_grid = "rgba(59,66,82,0.3)" if theme == "dark" else "rgba(128,128,128,0.1)"
    chart_text = "#FAFAFA" if theme == "dark" else "#262626"
    
    fig.update_layout(
        title="Total Portfolio P&L (PST)", template="plotly_dark" if theme == "dark" else "plotly_white",
        yaxis_title="P&L (USD)", xaxis_title="Date (PST)", hovermode="x unified",
        plot_bgcolor=chart_bg, paper_bgcolor=chart_bg, font_color=chart_text,
        xaxis=dict(gridcolor=chart_grid, rangeslider=dict(visible=True)),
        yaxis=dict(gridcolor=chart_grid), height=chart_height,
    )
    
    return fig

def create_pnl_bar_chart(per_asset_pnl: pd.DataFrame, theme: str) -> go.Figure:
    """Create per-asset P&L bar chart."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=per_asset_pnl["Asset"], y=per_asset_pnl["Realized P&L ($)"],
        name="Realized P&L ($)"
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    chart_height = get_responsive_chart_height('compact')
    chart_bg = "rgba(14,17,23,1)" if theme == "dark" else "rgba(255,255,255,1)"
    chart_text = "#FAFAFA" if theme == "dark" else "#262626"
    
    fig.update_layout(
        title="Per-Asset Realized P&L", template="plotly_dark" if theme == "dark" else "plotly_white",
        yaxis_title="P&L (USD)", xaxis_title="Asset", hovermode="x unified",
        plot_bgcolor=chart_bg, paper_bgcolor=chart_bg, font_color=chart_text, height=chart_height
    )
    
    return fig

# =====================[ SESSION STATE & THEME ]=====================

def init_session_state():
    """Initialize session state."""
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()
    if "theme" not in st.session_state:
        st.session_state.theme = "light"
    if "selected_asset" not in st.session_state:
        st.session_state.selected_asset = DEFAULT_ASSET

def apply_theme():
    """Apply theme CSS with mobile responsive styles."""
    theme = st.session_state.theme
    if theme == "dark":
        st.markdown("""
        <style>
        .stApp { background-color: #0E1117; color: #FAFAFA; }
        .stSidebar { background-color: #1E1E1E; }
        @media (max-width: 768px) {
            .stSidebar { width: 100% !important; }
            .main .block-container { padding: 2rem 1rem; }
            .stMetric { font-size: 0.9rem; }
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp { background-color: #FFFFFF; color: #262626; }
        @media (max-width: 768px) {
            .stSidebar { width: 100% !important; }
            .main .block-container { padding: 2rem 1rem; }
            .stMetric { font-size: 0.9rem; }
        }
        </style>
        """, unsafe_allow_html=True)

def toggle_theme():
    """Toggle theme without reloading data."""
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

def refresh_data():
    """Clear cache and reload."""
    st.cache_data.clear()
    st.session_state.last_refresh = time.time()
    for key in ['cache_key', 'pnl_summary', 'trades_with_pnl', 'stats', 'open_positions']:
        if key in st.session_state:
            del st.session_state[key]

def compute_and_cache_metrics(trades_df: pd.DataFrame, market_df: pd.DataFrame):
    """Compute metrics once and cache in session state."""
    cache_key = f"{len(trades_df)}_{len(market_df)}"
    
    if 'cache_key' not in st.session_state or st.session_state.cache_key != cache_key:
        with st.spinner("Computing metrics..."):
            trades_df = flag_threshold_violations(trades_df) if not trades_df.empty else trades_df
            pnl_summary, trades_with_pnl, stats = calculate_pnl_and_metrics(trades_df)
            open_positions = calculate_open_positions(trades_df, market_df)
            overall_realized, per_asset_pnl = summarize_realized_pnl(trades_with_pnl)
            
            st.session_state.cache_key = cache_key
            st.session_state.trades_df = trades_df
            st.session_state.pnl_summary = pnl_summary
            st.session_state.trades_with_pnl = trades_with_pnl
            st.session_state.stats = stats
            st.session_state.open_positions = open_positions
            st.session_state.overall_realized = overall_realized
            st.session_state.per_asset_pnl = per_asset_pnl
    
    return (
        st.session_state.trades_df, st.session_state.pnl_summary,
        st.session_state.trades_with_pnl, st.session_state.stats,
        st.session_state.open_positions, st.session_state.overall_realized,
        st.session_state.per_asset_pnl,
    )

# =====================[ MAIN APP ]=====================

init_session_state()
st.markdown("## Crypto Trading Strategy")
st.caption("ML Signals with Price-Based Entry/Exit (PST)")
apply_theme()

# Sidebar
with st.sidebar:
    st.markdown("<h1 style='text-align:center;'>Crypto Strategy</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.button("🌙" if st.session_state.theme == "light" else "☀️",
                 help="Toggle theme", on_click=toggle_theme, key="theme_toggle")
    with col2:
        st.button("↻ Refresh data", help="Clear cache and reload",
                 on_click=refresh_data, key="refresh_button")
    
    st.markdown("---")
    uploaded_market = st.file_uploader("Upload market data (optional)",
                                      type=["csv", "parquet", "pq"], key="upload_market")

# Load data
with st.spinner("Loading data..."):
    trades_df, market_df = load_data(TRADES_LINK, MARKET_LINK)

if uploaded_market is not None:
    override_df = read_uploaded_market(uploaded_market)
    if not override_df.empty:
        market_df = override_df
        refresh_data()
        st.info("Using uploaded market data.")

# Compute metrics (cached)
if not trades_df.empty and not market_df.empty:
    (trades_df, pnl_summary, trades_with_pnl, stats, 
     open_positions, overall_realized, per_asset_pnl) = compute_and_cache_metrics(trades_df, market_df)
else:
    trades_df = flag_threshold_violations(trades_df) if not trades_df.empty else trades_df
    pnl_summary, trades_with_pnl, stats = {}, pd.DataFrame(), {}
    open_positions = pd.DataFrame()
    overall_realized, per_asset_pnl = 0.0, pd.DataFrame()

# Sidebar summaries
with st.sidebar:
    st.markdown("### Data Status")
    if not market_df.empty and "timestamp_pst" in market_df.columns:
        times = market_df["timestamp_pst"].dropna()
        if not times.empty:
            st.caption(f"📈 Market: {times.min().strftime('%m/%d %H:%M')} → {times.max().strftime('%m/%d %H:%M')}")
    
    if not trades_df.empty and "timestamp_pst" in trades_df.columns:
        times = trades_df["timestamp_pst"].dropna()
        if not times.empty:
            st.caption(f"🧾 Trades: {times.min().strftime('%m/%d %H:%M')} → {times.max().strftime('%m/%d %H:%M')}")
    
    st.markdown("---")
    st.markdown(f"**Trades:** {'✅' if not trades_df.empty else '⚠️'} {len(trades_df):,}")
    st.markdown(f"**Market:** {'✅' if not market_df.empty else '❌'} {len(market_df):,}")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Price", "💰 P&L", "📜 History", "🚦 Validations"])

with tab1:
    if market_df.empty:
        st.warning("Market data not available.")
    else:
        assets = sorted(market_df["asset"].dropna().unique().tolist())
        selected_asset = st.selectbox("Select Asset", assets,
            index=assets.index(st.session_state.selected_asset) if st.session_state.selected_asset in assets else 0)
        st.session_state.selected_asset = selected_asset
        
        asset_market = market_df[market_df["asset"] == selected_asset].copy()
        asset_trades = trades_df[trades_df["asset"] == selected_asset].copy() if not trades_df.empty else pd.DataFrame()
        
        if not asset_market.empty and "close" in asset_market.columns:
            last_close = asset_market["close"].dropna().iloc[-1]
            if pd.notna(last_close):
                st.metric(f"Last Price for {selected_asset}", format_price(float(last_close)))
        
        fig = create_price_chart(asset_market, asset_trades, selected_asset, st.session_state.theme)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

with tab2:
    if trades_df.empty:
        st.warning("No trade data.")
    else:
        st.markdown("### Strategy Performance")
        c0, c1, c2, c3, c4 = st.columns(5)
        with c0:
            st.metric("Realized P&L", f"${overall_realized:+.2f}")
        with c1:
            st.metric("Closed Trades", f"{stats.get('total_trades', 0):,}")
        with c2:
            st.metric("Win Rate", f"{stats.get('win_rate', 0):.1f}%")
        with c3:
            pf = stats.get("profit_factor", 0)
            st.metric("Profit Factor", "∞" if pf == float("inf") else f"{pf:.2f}")
        with c4:
            st.metric("Max Drawdown", f"${stats.get('max_drawdown', 0):.2f}")
        
        if not trades_with_pnl.empty:
            fig_pnl = create_pnl_chart(trades_with_pnl, st.session_state.theme)
            st.plotly_chart(fig_pnl, use_container_width=True)
        
        st.markdown("#### Per-Asset P&L")
        if not per_asset_pnl.empty:
            st.dataframe(per_asset_pnl, use_container_width=True, hide_index=True)
            fig_bar = create_pnl_bar_chart(per_asset_pnl, st.session_state.theme)
            st.plotly_chart(fig_bar, use_container_width=True)

with tab3:
    if trades_df.empty:
        st.warning("No trades.")
    else:
        matched_df, open_df = match_trades_fifo(trades_df)
        
        st.markdown("#### Completed Trades (FIFO)")
        if not matched_df.empty:
            display = matched_df.copy()
            for col in ["Quantity", "Buy Price", "Sell Price", "P&L ($)", "P&L %"]:
                if col in display.columns:
                    display[col] = display[col].apply(float)
            display["Buy Time"] = display["Buy Time"].dt.strftime("%Y-%m-%d %H:%M:%S")
            display["Sell Time"] = display["Sell Time"].dt.strftime("%Y-%m-%d %H:%M:%S")
            display["Hold Time"] = display["Hold Time"].apply(format_timedelta_hhmm)
            st.dataframe(display[["Asset", "Quantity", "Buy Time", "Buy Price", 
                                 "Sell Time", "Sell Price", "Hold Time", "P&L ($)", "P&L %"]],
                        use_container_width=True, hide_index=True, height=400)
        else:
            st.info("No completed trades.")

with tab4:
    if trades_df.empty:
        st.info("No trades.")
    else:
        action_col = "action" if "action" in trades_df.columns else "unified_action"
        opens = trades_df[trades_df[action_col].astype(str).str.upper().isin(["OPEN", "BUY"])].copy()
        
        if not opens.empty:
            total = len(opens)
            valid = opens[opens["valid_at_open"]]
            invalid = opens[~opens["valid_at_open"]]
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("OPEN Orders", f"{total:,}")
            with c2:
                st.metric("Valid", f"{len(valid):,}")
            with c3:
                st.metric("Flagged", f"{len(invalid):,}")
            with c4:
                vr = (len(valid) / total * 100) if total > 0 else 0
                st.metric("Validity Rate", f"{vr:.1f}%")
            
            if not invalid.empty:
                st.markdown("#### ❗ Flagged OPENs")
                st.dataframe(invalid[["timestamp_pst", "asset", "price", "p_up", 
                                    "p_down", "confidence", "violation_reason"]],
                           use_container_width=True, hide_index=True, height=400)

# Sidebar open positions
with st.sidebar:
    st.markdown("---")
    if not open_positions.empty:
        st.markdown("**📊 Open Positions**")
        sorted_pos = open_positions.sort_values("Unrealized P&L ($)", ascending=False)
        
        for _, pos in sorted_pos.iterrows():
            pnl = pos["Unrealized P&L ($)"]
            pnl_pct = ((pos["Current Price"] - pos["Avg. Entry Price"]) / 
                      pos["Avg. Entry Price"] * 100) if pos["Avg. Entry Price"] != 0 else 0
            color = "#16a34a" if pnl >= 0 else "#ef4444"
            
            st.markdown(f"""
            <div style="background-color: {'rgba(22,163,74,0.1)' if pnl >= 0 else 'rgba(239,68,68,0.1)'}; 
                       border-left: 4px solid {color}; padding: 12px; margin: 8px 0; border-radius: 4px;">
                <div style="display: flex; justify-content: space-between;">
                    <strong>{pos['Asset']}</strong>
                    <span style="color: {color}; font-weight: bold;">{'📈' if pnl >= 0 else '📉'} {pnl_pct:+.1f}%</span>
                </div>
                <div style="font-size: 12px; color: #666; margin-top: 8px;">
                    Current: <strong>{format_price(pos['Current Price'])}</strong> | 
                    Entry: <strong>{format_price(pos['Avg. Entry Price'])}</strong>
                </div>
                <div style="text-align: center; padding-top: 5px; margin-top: 5px; 
                           border-top: 1px solid rgba(128,128,128,0.2);">
                    <span style="color: {color}; font-weight: bold;">P&L: ${pnl:+.4f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        total_pnl = sorted_pos["Unrealized P&L ($)"].sum()
        st.markdown(f"""
        <div style='text-align: center; padding: 0.5rem; background-color: rgba(128,128,128,0.1); 
                   border-radius: 0.25rem; margin-top: 0.5rem;'>
            <div style='font-size: 0.8rem; color: {"#16a34a" if total_pnl >= 0 else "#ef4444"}; font-weight: 600;'>
                Total P&L: ${total_pnl:+.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("**📊 Open Positions**")
        st.markdown("<div style='text-align: center; padding: 1rem; color: #666; font-style: italic;'>No open positions</div>",
                   unsafe_allow_html=True)
