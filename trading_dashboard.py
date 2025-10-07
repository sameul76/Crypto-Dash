"""
Crypto Trading Strategy Dashboard with Manual Trading
Single-file version with responsive charts and Pi bot integration
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

# ========= Configuration =========
st.set_page_config(page_title="Crypto Trading Strategy", layout="wide")
getcontext().prec = 30

# Data sources
TRADES_LINK = "https://drive.google.com/file/d/1cSeYf4kWJA49lMdQ1_yQ3hRui6wdrLD-/view?usp=drive_link"
MARKET_LINK = "https://drive.google.com/file/d/1QnlKihJu4Mc51B1wX1Tu84lKjwGmpXdU/view?usp=sharing"

DEFAULT_ASSET = "GIGA-USD"
TZ_PST = "America/Los_Angeles"

# Asset-specific thresholds
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
    'mobile': 450,
    'tablet': 600,
    'desktop': 750,
    'compact': 400,
}
DEFAULT_CHART_HEIGHT = CHART_HEIGHTS['mobile']

# ========= Session State Initialization =========
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "selected_asset" not in st.session_state:
    st.session_state.selected_asset = DEFAULT_ASSET

# ========= Utility Functions =========
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

# ========= Theme Management =========
def apply_theme():
    """Apply theme-specific CSS with mobile responsiveness."""
    theme = st.session_state.theme
    if theme == "dark":
        st.markdown("""
        <style>
        .stApp { background-color: #0E1117; color: #FAFAFA; }
        .stSidebar { background-color: #1E1E1E; }
        .metric-card { 
            background-color: #262730; 
            padding: 1rem; 
            border-radius: 0.5rem; 
            border: 1px solid #3B4252; 
        }
        .stExpander { background-color: #262730; border: 1px solid #3B4252; }
        
        @media (max-width: 768px) {
            .stSidebar { width: 100% !important; }
            .main .block-container { 
                padding-top: 2rem; 
                padding-left: 1rem; 
                padding-right: 1rem; 
            }
            .stMetric { font-size: 0.9rem; }
            .stMetric > div { font-size: 0.8rem; }
            .element-container { margin-bottom: 0.5rem; }
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp { background-color: #FFFFFF; color: #262626; }
        .metric-card { 
            background-color: #F8F9FA; 
            padding: 1rem; 
            border-radius: 0.5rem; 
            border: 1px solid #E9ECEF; 
        }
        
        @media (max-width: 768px) {
            .stSidebar { width: 100% !important; }
            .main .block-container { 
                padding-top: 2rem; 
                padding-left: 1rem; 
                padding-right: 1rem; 
            }
            .stMetric { font-size: 0.9rem; }
            .stMetric > div { font-size: 0.8rem; }
            .element-container { margin-bottom: 0.5rem; }
        }
        </style>
        """, unsafe_allow_html=True)

def toggle_theme():
    """Toggle theme without reloading data."""
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

def refresh_data():
    """Clear cache and reload data."""
    st.cache_data.clear()
    st.session_state.last_refresh = time.time()
    for key in ['cache_key', 'pnl_summary', 'trades_with_pnl', 'stats', 'open_positions']:
        if key in st.session_state:
            del st.session_state[key]

# ========= Data Loading Functions =========
def _extract_file_id(url_or_id: str) -> str:
    """Extract Google Drive file ID from URL."""
    if not url_or_id:
        return ""
    patterns = [r'/d/([a-zA-Z0-9_-]{25,})', r'id=([a-zA-Z0-9_-]{25,})']
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
            st.error(f"{label}: Got HTML instead of file. Check sharing permissions.")
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
        st.warning(f"{label}: no bytes downloaded")
        return pd.DataFrame()
    if b[:4] != b"PAR1":
        try:
            return pd.read_csv(io.BytesIO(b))
        except Exception:
            st.error(f"{label}: not Parquet and CSV fallback failed")
            return pd.DataFrame()
    try:
        return pd.read_parquet(io.BytesIO(b))
    except Exception as e:
        st.error(f"{label}: failed to read Parquet: {e}")
        return pd.DataFrame()

def _ensure_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure timestamp columns are properly timezone-aware."""
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
    """Load and normalize trades and market data."""
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
    """Read uploaded market file and normalize."""
    if uploaded is None:
        return pd.DataFrame()
    
    name = (uploaded.name or "").lower()
    try:
        if name.endswith(".parquet") or name.endswith(".pq"):
            df = pd.read_parquet(uploaded)
        else:
            df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
        return pd.DataFrame()
    
    return normalize_dataframe(df, is_trades=False)

# ========= Trading Functions =========
def execute_trade(asset: str, action: str, quantity: float, pi_host: str):
    """Send trade request to Pi bot."""
    try:
        url = f"http://{pi_host}:5000/trade"
        
        response = requests.post(
            url,
            json={
                "asset": asset,
                "action": action,
                "quantity": quantity
            },
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "message": data.get('message', 'Trade queued on bot'),
                "trade_id": data.get('trade_id')
            }
        else:
            error_data = response.json()
            return {
                "success": False,
                "message": error_data.get('message', 'Unknown error')
            }
    
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "message": "Bot not responding (timeout). Check if bot is running."
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "message": f"Cannot reach bot at {pi_host}:5000. Check IP address and firewall."
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error: {str(e)}"
        }

def show_trading_panel(market_df: pd.DataFrame, open_positions: pd.DataFrame):
    """Display trading interface with buy/sell buttons."""
    st.markdown("---")
    st.markdown("### 🎯 Quick Trade")
    
    # Get Pi host from secrets
    try:
        pi_host = st.secrets.get("PI_HOST", "")
    except:
        pi_host = ""
    
    if not pi_host:
        st.warning("⚠️ Add PI_HOST to .streamlit/secrets.toml to enable trading")
        st.code("""
# .streamlit/secrets.toml
PI_HOST = "localhost"  # if dashboard on same Pi
# or
PI_HOST = "192.168.1.100"  # your Pi's IP
        """)
        return
    
    # Check bot connection
    try:
        health_check = requests.get(f"http://{pi_host}:5000/health", timeout=3)
        if health_check.status_code == 200:
            bot_status = health_check.json()
            st.success(f"✅ Bot online ({bot_status.get('mode', 'unknown')} mode)")
        else:
            st.error(f"❌ Bot unreachable at {pi_host}")
            return
    except:
        st.error(f"❌ Cannot connect to bot at {pi_host}:5000")
        return
    
    if market_df.empty or "asset" not in market_df.columns:
        st.info("No market data available for trading.")
        return
    
    assets = sorted(market_df["asset"].dropna().unique().tolist())
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        trade_asset = st.selectbox("Asset to Trade", assets, key="trade_asset_select")
    
    # Get current price and position
    asset_market = market_df[market_df["asset"] == trade_asset]
    current_price = None
    if not asset_market.empty and "close" in asset_market.columns:
        current_price = float(asset_market["close"].dropna().iloc[-1])
    
    current_position = 0.0
    if not open_positions.empty:
        pos = open_positions[open_positions["Asset"] == trade_asset]
        if not pos.empty:
            current_position = float(pos["Quantity"].iloc[0])
    
    with col2:
        st.metric("Current Price", format_price(current_price) if current_price else "N/A")
    with col3:
        st.metric("Position", f"{current_position:.4f}")
    
    # Trade quantity input
    col1, col2 = st.columns(2)
    
    with col1:
        quantity = st.number_input(
            "Quantity",
            min_value=0.0,
            max_value=10000.0,
            value=1.0,
            step=0.1,
            format="%.4f",
            key="trade_quantity",
            help="Enter 0 to use maximum available (90% of balance for BUY, full position for SELL)"
        )
    
    with col2:
        order_type = st.selectbox("Order Type", ["Market"], key="order_type")
    
    # Calculate estimated cost/proceeds
    if current_price and quantity > 0:
        estimated_value = quantity * current_price
        st.caption(f"Estimated value: ${estimated_value:.2f}")
    
    # Trade buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        buy_clicked = st.button("🟢 BUY", type="primary", use_container_width=True, key="buy_button")
    
    with col2:
        sell_clicked = st.button("🔴 SELL", type="secondary", use_container_width=True, key="sell_button")
    
    # Execute trades
    if buy_clicked:
        if current_price is None:
            st.error("Cannot execute: No current price data")
        else:
            st.session_state.pending_trade = {
                "action": "BUY",
                "asset": trade_asset,
                "quantity": quantity,
                "price": current_price
            }
    
    if sell_clicked:
        if current_price is None:
            st.error("Cannot execute: No current price data")
        elif current_position == 0:
            st.error("Cannot SELL: No position in this asset")
        elif quantity > 0 and quantity > current_position:
            st.error(f"Cannot SELL {quantity}: Only {current_position:.4f} available")
        else:
            st.session_state.pending_trade = {
                "action": "SELL",
                "asset": trade_asset,
                "quantity": quantity if quantity > 0 else current_position,
                "price": current_price
            }
    
    # Show confirmation dialog if trade is pending
    if "pending_trade" in st.session_state:
        trade = st.session_state.pending_trade
        
        st.markdown("---")
        st.warning(f"⚠️ Confirm {trade['action']} Order")
        
        qty_text = f"{trade['quantity']:.4f}" if trade['quantity'] > 0 else "MAX"
        est_total = trade['quantity'] * trade['price'] if trade['quantity'] > 0 else "~90% balance" if trade['action'] == "BUY" else f"~{current_position * trade['price']:.2f}"
        
        st.markdown(f"""
        **Asset:** {trade['asset']}  
        **Action:** {trade['action']}  
        **Quantity:** {qty_text}  
        **Price:** {format_price(trade['price'])}  
        **Estimated Total:** ${est_total if isinstance(est_total, str) else f'{est_total:.2f}'}
        """)
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("✅ Confirm", type="primary", key="confirm_trade"):
                result = execute_trade(
                    trade['asset'],
                    trade['action'],
                    trade['quantity'],
                    pi_host
                )
                
                if result['success']:
                    st.success(f"✅ {result['message']}")
                    st.info("Trade queued on bot. Will execute on next cycle (~20 seconds)")
                    del st.session_state.pending_trade
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error(f"❌ {result['message']}")
                    del st.session_state.pending_trade
        
        with col2:
            if st.button("❌ Cancel", key="cancel_trade"):
                del st.session_state.pending_trade
                st.rerun()

# ========= Calculation Functions =========
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
                avg_cost = positions[asset]["cost"] / max(
                    positions[asset]["quantity"], Decimal("1e-12")
                )
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
        df["asset_cumulative_pnl"] = df.groupby("asset")["pnl"].transform(
            lambda x: x.cumsum()
        )
    
    return pnl_per_asset, df, stats

def summarize_realized_pnl(trades_with_pnl: pd.DataFrame):
    """Summarize realized P&L per asset."""
    if (trades_with_pnl is None or trades_with_pnl.empty or 
        "pnl" not in trades_with_pnl.columns):
        return 0.0, pd.DataFrame(columns=["Asset", "Realized P&L ($)"])
    
    df = trades_with_pnl.copy()
    df["pnl_float"] = df["pnl"].apply(
        lambda x: float(x) if pd.notna(x) else 0.0
    )
    
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
    if (trades_df is None or trades_df.empty or 
        "timestamp_pst" not in trades_df.columns):
        return pd.DataFrame(), pd.DataFrame()
    
    tdf = trades_df.copy()
    matched, open_df = [], []
    
    for asset, group in tdf.groupby("asset"):
        g = group.sort_values("timestamp_pst")
        buys = [
            row.to_dict() 
            for _, row in g[g["unified_action"].isin(["buy", "open"])].iterrows()
        ]
        sells = [
            row.to_dict() 
            for _, row in g[g["unified_action"].isin(["sell", "close"])].iterrows()
        ]
        
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
        matched_df["P&L %"] = 100 * np.where(
            is_zero, 0, matched_df["P&L ($)"] / buy_cost
        )
        matched_df = matched_df.sort_values("Sell Time", ascending=False)
    
    if not open_df.empty:
        open_df = open_df.sort_values("timestamp_pst", ascending=False)
    
    return matched_df, open_df

def calculate_open_positions(
    trades_df: pd.DataFrame,
    market_df: pd.DataFrame
) -> pd.DataFrame:
    """Calculate current open positions with unrealized P&L."""
    if (trades_df is None or trades_df.empty or 
        "timestamp_pst" not in trades_df.columns or
        market_df is None or market_df.empty or 
        "timestamp_pst" not in market_df.columns):
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
                avg_cost = positions[asset]["cost"] / max(
                    positions[asset]["quantity"], Decimal("1e-12")
                )
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
    """Flag trades that violated thresholds at execution."""
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
                r.append(
                    f"p_up {pu:.3f} < buy_threshold {th['buy_threshold']:.2f}"
                )
            if not (conf >= th["min_confidence"]):
                r.append(
                    f"confidence {conf:.3f} < min_confidence {th['min_confidence']:.2f}"
                )
        
        val_flags.append(len(r) == 0)
        reasons.append("; ".join(r))
    
    df["valid_at_open"] = val_flags
    df["violation_reason"] = reasons
    
    return df

# ========= Visualization Functions =========
def create_price_chart(
    asset_market: pd.DataFrame,
    asset_trades: pd.DataFrame,
    selected_asset: str,
    theme: str
) -> go.Figure:
    """Create price chart with candles, signals, and trade markers."""
    asset_market = asset_market.copy()
    if "timestamp_pst" in asset_market.columns:
        asset_market["__x__"] = pd.to_datetime(
            asset_market["timestamp_pst"]
        ).dt.tz_localize(None)
    
    need_cols = ["__x__", "open", "high", "low", "close"]
    present_cols = [c for c in need_cols if c in asset_market.columns]
    asset_candles = asset_market.dropna(subset=present_cols).copy()
    
    if not asset_candles.empty:
        asset_candles = asset_candles.drop_duplicates(subset="__x__", keep="last")
    
    fig = go.Figure()
    
    if not asset_candles.empty:
        fig.add_trace(go.Candlestick(
            x=asset_candles["__x__"],
            open=asset_candles["open"],
            high=asset_candles["high"],
            low=asset_candles["low"],
            close=asset_candles["close"],
            name=f"{selected_asset}",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            increasing_fillcolor="rgba(38,166,154,0.5)",
            decreasing_fillcolor="rgba(239,83,80,0.5)",
            line=dict(width=1)
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
                x=sig["__x__"],
                y=sig["close"],
                mode="markers",
                name="ML Signals",
                marker=dict(
                    size=sizes,
                    color=colors,
                    opacity=0.7,
                    line=dict(width=1, color="white")
                ),
                customdata=list(zip(sig["p_up"], sig["p_down"], sig["confidence"])),
                hovertemplate=(
                    "<b>ML Signal</b><br>"
                    "%{x|%Y-%m-%d %H:%M}<br>"
                    "Close: $%{y:.6f}<br>"
                    "P(Up): %{customdata[0]:.3f}<br>"
                    "P(Down): %{customdata[1]:.3f}<br>"
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
                    x=buys["__x__"],
                    y=[marker_y] * len(buys),
                    mode="markers",
                    name="BUY",
                    marker=dict(
                        symbol="triangle-up",
                        size=14,
                        color="#4caf50",
                        line=dict(width=1, color="white")
                    ),
                    hovertemplate="<b>BUY</b><br>%{x|%H:%M:%S}<extra></extra>",
                ))
            
            if not sells.empty:
                fig.add_trace(go.Scatter(
                    x=sells["__x__"],
                    y=[marker_y] * len(sells),
                    mode="markers",
                    name="SELL",
                    marker=dict(
                        symbol="triangle-down",
                        size=14,
                        color="#f44336",
                        line=dict(width=1, color="white")
                    ),
                    hovertemplate="<b>SELL</b><br>%{x|%H:%M:%S}<extra></extra>",
                ))
    
    chart_height = get_responsive_chart_height()
    
    fig.update_layout(
        title=f"{selected_asset} — Full History",
        template="plotly_dark" if theme == "dark" else "plotly_white",
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
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        height=chart_height,
        margin=dict(l=60, r=20, t=80, b=80),
        plot_bgcolor="rgba(14,17,23,1)" if theme == "dark" else "rgba(250,250,250,0.8)",
        paper_bgcolor="rgba(14,17,23,1)" if theme == "dark" else "rgba(255,255,255,1)",
        font_color="#FAFAFA" if theme == "dark" else "#262626",
    )
    
    return fig

def create_pnl_chart(trades_with_pnl: pd.DataFrame, theme: str) -> go.Figure:
    """Create cumulative P&L chart."""
    plot_df = trades_with_pnl.copy().sort_values("timestamp_pst")
    plot_df["cumulative_pnl"] = plot_df["cumulative_pnl"].apply(float)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df["timestamp_pst"],
        y=plot_df["cumulative_pnl"],
        mode="lines",
        name="Cumulative P&L",
        line=dict(width=2)
    ))
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        annotation_text="Break Even"
    )
    
    chart_height = get_responsive_chart_height()
    
    chart_bg = "rgba(14,17,23,1)" if theme == "dark" else "rgba(255,255,255,1)"
    chart_grid = "rgba(59,66,82,0.3)" if theme == "dark" else "rgba(128,128,128,0.1)"
    chart_text = "#FAFAFA" if theme == "dark" else "#262626"
    
    fig.update_layout(
        title="Total Portfolio P&L (PST)",
        template="plotly_dark" if theme == "dark" else "plotly_white",
        yaxis_title="P&L (USD)",
        xaxis_title="Date (PST)",
        hovermode="x unified",
        plot_bgcolor=chart_bg,
        paper_bgcolor=chart_bg,
        font_color=chart_text,
        xaxis=dict(gridcolor=chart_grid, rangeslider=dict(visible=True)),
        yaxis=dict(gridcolor=chart_grid),
        height=chart_height,
    )
    
    return fig

def create_pnl_bar_chart(per_asset_pnl: pd.DataFrame, theme: str) -> go.Figure:
    """Create per-asset P&L bar chart."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=per_asset_pnl["Asset"],
        y=per_asset_pnl["Realized P&L ($)"],
        name="Realized P&L ($)"
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    chart_height = get_responsive_chart_height('compact')
    
    chart_bg = "rgba(14,17,23,1)" if theme == "dark" else "rgba(255,255,255,1)"
    chart_text = "#FAFAFA" if theme == "dark" else "#262626"
    
    fig.update_layout(
        title="Per-Asset Realized P&L",
        template="plotly_dark" if theme == "dark" else "plotly_white",
        yaxis_title="P&L (USD)",
        xaxis_title="Asset",
        hovermode="x unified",
        plot_bgcolor=chart_bg,
        paper_bgcolor=chart_bg,
        font_color=chart_text,
        height=chart_height
    )
    
    return fig

# ========= Metrics Caching =========
def compute_and_cache_metrics(trades_df: pd.DataFrame, market_df: pd.DataFrame):
    """Compute expensive metrics once and cache in session state."""
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
        st.session_state.trades_df,
        st.session_state.pnl_summary,
        st.session_state.trades_with_pnl,
        st.session_state.stats,
        st.session_state.open_positions,
        st.session_state.overall_realized,
        st.session_state.per_asset_pnl,
    )

# ========= MAIN APP =========
st.markdown("## Crypto Trading Strategy")
st.caption("ML Signals with Price-Based Entry/Exit (displayed in PST)")
apply_theme()

# Sidebar
with st.sidebar:
    st.markdown("<h1 style='text-align:center;'>Crypto Strategy</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.button(
            "🌙" if st.session_state.theme == "light" else "☀️",
            help="Toggle theme",
            on_click=toggle_theme,
            key="theme_toggle"
        )
    with col2:
        st.button(
            "↻ Refresh data",
            help="Clear cache and reload",
            on_click=refresh_data,
            key="refresh_button"
        )
    
    st.markdown("---")
    st.markdown("**Data override**")
    uploaded_market = st.file_uploader(
        "Upload market CSV/Parquet to override",
        type=["csv", "parquet", "pq"],
        key="upload_market"
    )

# Load data
with st.spinner("Loading data from Google Drive..."):
    trades_df, market_df = load_data(TRADES_LINK, MARKET_LINK)

# Handle upload
if uploaded_market is not None:
    override_df = read_uploaded_market(uploaded_market)
    if not override_df.empty:
        market_df = override_df
        refresh_data()
        st.info("Using uploaded market data.")

# Compute metrics
if not trades_df.empty and not market_df.empty:
    (trades_df, pnl_summary, trades_with_pnl, stats, 
     open_positions, overall_realized, per_asset_pnl) = compute_and_cache_metrics(trades_df, market_df)
else:
    trades_df = flag_threshold_violations(trades_df) if not trades_df.empty else trades_df
    pnl_summary, trades_with_pnl, stats = {}, pd.DataFrame(), {}
    open_positions = pd.DataFrame()
    overall_realized, per_asset_pnl = 0.0, pd.DataFrame()

elapsed = seconds_since_last_run()

# Sidebar summary
with st.sidebar:
    st.markdown("### Data Status")
    
    if not market_df.empty and "timestamp_pst" in market_df.columns:
        valid_market_times = market_df["timestamp_pst"].dropna()
        if not valid_market_times.empty:
            st.caption(
                f"📈 Market: {valid_market_times.min().strftime('%m/%d %H:%M')} → "
                f"{valid_market_times.max().strftime('%m/%d %H:%M')}"
            )
    
    if not trades_df.empty and "timestamp_pst" in trades_df.columns:
        valid_trade_times = trades_df["timestamp_pst"].dropna()
        if not valid_trade_times.empty:
            st.caption(
                f"🧾 Trades: {valid_trade_times.min().strftime('%m/%d %H:%M')} → "
                f"{valid_trade_times.max().strftime('%m/%d %H:%M')}"
            )
    
    st.markdown("---")
    st.markdown(f"**Trades:** {'✅' if not trades_df.empty else '⚠️'} {len(trades_df):,}")
    st.markdown(f"**Market:** {'✅' if not market_df.empty else '❌'} {len(market_df):,}")
    if not market_df.empty and "asset" in market_df.columns:
        st.markdown(f"**Assets:** {market_df['asset'].nunique():,}")

# Debug panel
with st.expander("🔎 Data Freshness Debug", expanded=False):
    if not market_df.empty and "timestamp_pst" in market_df.columns:
        st.write(f"**Market:** {market_df['timestamp_pst'].min()} → {market_df['timestamp_pst'].max()}")
        st.write(f"**Records:** {len(market_df):,}")
    
    if not trades_df.empty and "timestamp_pst" in trades_df.columns:
        st.write(f"**Trades:** {trades_df['timestamp_pst'].min()} → {trades_df['timestamp_pst'].max()}")
        st.write(f"**Total:** {len(trades_df):,}")
    
    st.write(f"**Cache age:** {elapsed}s")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Price & Trades",
    "💰 P&L Analysis",
    "📜 Trade History",
    "🚦 Validations"
])

# TAB 1: Price & Trades
with tab1:
    if market_df.empty or "asset" not in market_df.columns:
        st.warning("Market data not available.")
    else:
        assets = sorted(market_df["asset"].dropna().unique().tolist())
        default_index = assets.index(st.session_state.selected_asset) if st.session_state.selected_asset in assets else 0
        
        selected_asset = st.selectbox(
            "Select Asset",
            assets,
            index=min(default_index, len(assets)-1),
            key="asset_select_main"
        )
        st.session_state.selected_asset = selected_asset
        
        asset_market = market_df[market_df["asset"] == selected_asset].copy()
        asset_trades = trades_df[trades_df["asset"] == selected_asset].copy() if not trades_df.empty else pd.DataFrame()
        
        if not asset_market.empty and "close" in asset_market.columns:
            last_close = asset_market["close"].dropna().iloc[-1]
            if pd.notna(last_close):
                st.metric(f"Last Price for {selected_asset}", format_price(float(last_close)))
        
        fig = create_price_chart(asset_market, asset_trades, selected_asset, st.session_state.theme)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})
        
        # Add trading panel
        show_trading_panel(market_df, open_positions)
        
        with st.expander("📎 Latest rows"):
            display_cols = [c for c in ["timestamp_pst", "open", "high", "low", "close", "p_up", "p_down"] 
                          if c in asset_market.columns]
            st.dataframe(asset_market[display_cols].tail(15), use_container_width=True)

# TAB 2: P&L Analysis
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
            pf_display = "∞" if isinstance(pf, float) and pf == float("inf") else f"{pf:.2f}"
            st.metric("Profit Factor", pf_display)
        with c4:
            st.metric("Max Drawdown", f"${stats.get('max_drawdown', 0):.2f}")
        
        if not trades_with_pnl.empty:
            fig_pnl = create_pnl_chart(trades_with_pnl, st.session_state.theme)
            st.plotly_chart(fig_pnl, use_container_width=True)
        
        st.markdown("#### Per-Asset P&L")
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
            
            fig_bar = create_pnl_bar_chart(per_asset_pnl, st.session_state.theme)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No realized P&L yet.")

# TAB 3: Trade History
with tab3:
    st.markdown("### Trade History")
    
    if trades_df.empty:
        st.warning("No trades.")
    else:
        matched_df, open_df = match_trades_fifo(trades_df)
        
        st.markdown("#### Completed Trades (FIFO)")
        if not matched_df.empty:
            display_matched = matched_df.copy()
            for col in ["Quantity", "Buy Price", "Sell Price", "P&L ($)", "P&L %"]:
                if col in display_matched.columns:
                    display_matched[col] = display_matched[col].apply(float)
            
            display_matched["Buy Time"] = display_matched["Buy Time"].dt.strftime("%Y-%m-%d %H:%M:%S")
            display_matched["Sell Time"] = display_matched["Sell Time"].dt.strftime("%Y-%m-%d %H:%M:%S")
            display_matched["Hold Time"] = display_matched["Hold Time"].apply(format_timedelta_hhmm)
            
            st.dataframe(
                display_matched[["Asset", "Quantity", "Buy Time", "Buy Price", "Sell Time", "Sell Price", "Hold Time", "P&L ($)", "P&L %"]],
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
                height=400
            )
        else:
            st.info("No completed trades.")
        
        st.markdown("---")
        st.markdown("#### Open Positions")
        if not open_df.empty:
            display_open = open_df.copy()
            for col in ["quantity", "price"]:
                if col in display_open.columns:
                    display_open[col] = display_open[col].apply(float)
            
            display_open["Time"] = display_open["timestamp_pst"].dt.strftime("%Y-%m-%d %H:%M:%S")
            display_open = display_open.rename(columns={
                "asset": "Asset",
                "quantity": "Quantity",
                "price": "Price",
                "reason": "Reason"
            })
            
            st.dataframe(
                display_open[["Time", "Asset", "Quantity", "Price", "Reason"]],
                column_config={
                    "Asset": st.column_config.TextColumn(width="small"),
                    "Quantity": st.column_config.NumberColumn(format="%.4f", width="small"),
                    "Price": st.column_config.NumberColumn(format="$%.8f"),
                },
                use_container_width=True,
                hide_index=True,
                height=400
            )
        else:
            st.info("No open positions.")

# TAB 4: Validations
with tab4:
    st.markdown("### Threshold Validations")
    
    if trades_df.empty:
        st.info("No trades.")
    else:
        action_col = "action" if "action" in trades_df.columns else "unified_action"
        opens_mask = trades_df[action_col].astype(str).str.upper().isin(["OPEN", "BUY"])
        opens = trades_df.loc[opens_mask].copy()
        
        if opens.empty:
            st.info("No OPEN orders.")
        else:
            total_opens = len(opens)
            invalid = opens[~opens["valid_at_open"]]
            valid = opens[opens["valid_at_open"]]
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("OPEN Orders", f"{total_opens:,}")
            with c2:
                st.metric("Valid", f"{len(valid):,}")
            with c3:
                st.metric("Flagged", f"{len(invalid):,}")
            with c4:
                vr = (len(valid) / total_opens * 100) if total_opens > 0 else 0
                st.metric("Validity Rate", f"{vr:.1f}%")
            
            if not invalid.empty:
                st.markdown("#### Flagged OPENs")
                invalid_disp = invalid.copy()
                
                if "timestamp_pst" in invalid_disp.columns:
                    invalid_disp["timestamp_pst_str"] = invalid_disp["timestamp_pst"].dt.strftime("%Y-%m-%d %H:%M:%S")
                    show_cols = ["timestamp_pst_str"] + [
                        c for c in ["asset", "price", "p_up", "p_down", "confidence", "reason", "violation_reason"] 
                        if c in invalid_disp.columns
                    ]
                    st.dataframe(
                        invalid_disp[show_cols].sort_values("timestamp_pst_str"),
                        use_container_width=True,
                        hide_index=True,
                        height=400
                    )
                else:
                    st.dataframe(invalid_disp, use_container_width=True, hide_index=True)
            else:
                st.success("No violations detected")

# Sidebar: Open Positions
with st.sidebar:
    st.markdown("---")
    
    if not open_positions.empty:
        st.markdown("**📊 Open Positions**")
        sorted_positions = open_positions.sort_values("Unrealized P&L ($)", ascending=False)
        
        for _, pos in sorted_positions.iterrows():
            pnl = pos["Unrealized P&L ($)"]
            pnl_pct = (
                (pos["Current Price"] - pos["Avg. Entry Price"]) / pos["Avg. Entry Price"] * 100
            ) if pos["Avg. Entry Price"] != 0 else 0
            
            color = "#16a34a" if pnl >= 0 else "#ef4444"
            pnl_icon = "📈" if pnl >= 0 else "📉"
            card_bg = "rgba(22,163,74,0.1)" if pnl >= 0 else "rgba(239,68,68,0.1)"
            
            st.markdown(
                f"""
                <div style="background-color: {card_bg}; border-left: 4px solid {color}; 
                           padding: 12px; margin: 8px 0; border-radius: 4px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <strong style="font-size: 14px;">{pos['Asset']}</strong>
                        <span style="color: {color}; font-weight: bold; font-size: 13px;">
                            {pnl_icon} {pnl_pct:+.1f}%
                        </span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span style="font-size: 11px; color: #888; background: rgba(128,128,128,0.1); 
                                   padding: 2px 6px; border-radius: 3px;">
                            ⏰ {pos.get('Open Time', 'N/A')}
                        </span>
                        <span style="font-size: 12px; color: #666;">
                            Qty: {pos['Quantity']:.4f}
                        </span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                        <span style="font-size: 12px; color: #666;">
                            Current: <strong>{format_price(pos['Current Price'])}</strong>
                        </span>
                        <span style="font-size: 12px; color: #666;">
                            Entry: <strong>{format_price(pos['Avg. Entry Price'])}</strong>
                        </span>
                    </div>
                    <div style="text-align: center; padding-top: 5px; 
                               border-top: 1px solid rgba(128,128,128,0.2);">
                        <span style="color: {color}; font-weight: bold; font-size: 13px;">
                            P&L: ${pnl:+.4f}
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        total_pnl = sorted_positions["Unrealized P&L ($)"].sum()
        total_value = sorted_positions["Current Value ($)"].sum()
        winners = len(sorted_positions[sorted_positions["Unrealized P&L ($)"] > 0])
        
        st.markdown("---")
        st.markdown(
            f"""
            <div style='text-align: center; padding: 0.5rem; 
                       background-color: rgba(128,128,128,0.1); 
                       border-radius: 0.25rem;'>
                <div style='font-size: 0.8rem; 
                           color: {"#16a34a" if total_pnl >= 0 else "#ef4444"}; 
                           font-weight: 600;'>
                    Total P&L: ${total_pnl:+.2f}
                </div>
                <div style='font-size: 0.7rem; color: #666; margin-top: 0.25rem;'>
                    {winners}/{len(sorted_positions)} winning | ${total_value:.2f} total
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown("**📊 Open Positions**")
        st.markdown(
            """
            <div style='text-align: center; padding: 1rem; 
                       background-color: rgba(128,128,128,0.05); 
                       border-radius: 0.25rem; color: #666; font-style: italic;'>
                No open positions
            </div>
            """,
            unsafe_allow_html=True
        )


