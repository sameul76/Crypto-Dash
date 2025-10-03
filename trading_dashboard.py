"""
Crypto Trading Strategy Dashboard - Main Application
Streamlit app with responsive charts and modular architecture.
"""

import time
import streamlit as st
import pandas as pd

# Module imports (these would be separate files)
from config import ASSET_THRESHOLDS, DEFAULT_ASSET, TRADES_LINK, MARKET_LINK
from data_loader import load_data, read_uploaded_market
from calculations import (
    calculate_pnl_and_metrics,
    summarize_realized_pnl,
    match_trades_fifo,
    calculate_open_positions,
    flag_threshold_violations,
    format_timedelta_hhmm,
)
from visualizations import create_price_chart, create_pnl_chart, create_pnl_bar_chart
from utils import format_price

# ========= App Setup =========
st.set_page_config(page_title="Crypto Trading Strategy", layout="wide")

# ========= Session State Initialization =========
def init_session_state():
    """Initialize all session state variables."""
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()
    if "theme" not in st.session_state:
        st.session_state.theme = "light"
    if "selected_asset" not in st.session_state:
        st.session_state.selected_asset = DEFAULT_ASSET

init_session_state()

# ========= Theme Management =========
def apply_theme():
    """Apply theme-specific CSS with responsive mobile styles."""
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
        
        /* Mobile responsive styles */
        @media (max-width: 768px) {
            .stSidebar { width: 100% !important; }
            .main .block-container { 
                padding-top: 2rem; 
                padding-left: 1rem; 
                padding-right: 1rem; 
            }
            /* Make metrics more compact on mobile */
            .stMetric { font-size: 0.9rem; }
            .stMetric > div { font-size: 0.8rem; }
            /* Reduce spacing between elements */
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
        
        /* Mobile responsive styles */
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
    """Toggle between light and dark theme without reloading data."""
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

def refresh_data():
    """Clear cache and reload data."""
    st.cache_data.clear()
    st.session_state.last_refresh = time.time()
    # Clear computed data from session state
    for key in ['cache_key', 'pnl_summary', 'trades_with_pnl', 'stats', 'open_positions']:
        if key in st.session_state:
            del st.session_state[key]

# ========= Helper Functions =========
def seconds_since_last_run() -> int:
    """Calculate seconds since last data refresh."""
    return int(time.time() - st.session_state.get("last_refresh", 0))

def compute_and_cache_metrics(trades_df: pd.DataFrame, market_df: pd.DataFrame):
    """
    Compute expensive metrics once and cache in session state.
    Only recomputes if data has changed.
    """
    # Create a cache key based on data shape
    cache_key = f"{len(trades_df)}_{len(market_df)}"
    
    if 'cache_key' not in st.session_state or st.session_state.cache_key != cache_key:
        with st.spinner("Computing metrics..."):
            # Flag violations
            trades_df = flag_threshold_violations(trades_df) if not trades_df.empty else trades_df
            
            # Calculate P&L
            pnl_summary, trades_with_pnl, stats = calculate_pnl_and_metrics(trades_df)
            
            # Calculate positions
            open_positions = calculate_open_positions(trades_df, market_df)
            
            # Summarize P&L
            overall_realized, per_asset_pnl = summarize_realized_pnl(trades_with_pnl)
            
            # Cache results
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

# ========= UI Header & Theme =========
st.markdown("## Crypto Trading Strategy")
st.caption("ML Signals with Price-Based Entry/Exit (displayed in PST)")
apply_theme()

# ========= Sidebar Controls =========
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

# ========= Load Data =========
with st.spinner("Loading data from Google Drive..."):
    trades_df, market_df = load_data(TRADES_LINK, MARKET_LINK)

# Handle uploaded override
if uploaded_market is not None:
    override_df = read_uploaded_market(uploaded_market)
    if not override_df.empty:
        market_df = override_df
        refresh_data()
        st.info("Using uploaded market data (overrides Google Drive).")

# ========= Compute Metrics (Cached) =========
if not trades_df.empty and not market_df.empty:
    (trades_df, pnl_summary, trades_with_pnl, stats, 
     open_positions, overall_realized, per_asset_pnl) = compute_and_cache_metrics(trades_df, market_df)
else:
    trades_df = flag_threshold_violations(trades_df) if not trades_df.empty else trades_df
    pnl_summary, trades_with_pnl, stats = {}, pd.DataFrame(), {}
    open_positions = pd.DataFrame()
    overall_realized, per_asset_pnl = 0.0, pd.DataFrame()

elapsed = seconds_since_last_run()

# ========= Sidebar Summaries =========
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

# ========= Debug Panel =========
with st.expander("🔎 Data Freshness Debug", expanded=False):
    if not market_df.empty and "timestamp_pst" in market_df.columns:
        st.write(f"**Market window:** {market_df['timestamp_pst'].min()} → {market_df['timestamp_pst'].max()}")
        st.write(f"**Records:** {len(market_df):,}")
    else:
        st.write("**Market data:** missing or no timestamp_pst")
    
    if not trades_df.empty and "timestamp_pst" in trades_df.columns:
        st.write(f"**Trades window:** {trades_df['timestamp_pst'].min()} → {trades_df['timestamp_pst'].max()}")
        st.write(f"**Total:** {len(trades_df):,}")
    else:
        st.write("**Trades data:** missing or no timestamp_pst")
    
    st.write(f"**Cache age:** {elapsed}s")

# ========= Tabs =========
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Price & Trades",
    "💰 P&L Analysis",
    "📜 Trade History",
    "🚦 Validations"
])

# ========= TAB 1: Price & Trades =========
with tab1:
    if market_df.empty or "asset" not in market_df.columns or market_df["asset"].dropna().empty:
        st.warning("Market data not available or missing assets.")
    else:
        assets = sorted(market_df["asset"].dropna().unique().tolist())
        default_index = assets.index(st.session_state.selected_asset) if st.session_state.selected_asset in assets else 0
        
        selected_asset = st.selectbox(
            "Select Asset to View",
            assets,
            index=min(default_index, len(assets)-1),
            key="asset_select_main"
        )
        st.session_state.selected_asset = selected_asset
        
        # Filter data for selected asset
        asset_market = market_df[market_df["asset"] == selected_asset].copy()
        asset_trades = trades_df[trades_df["asset"] == selected_asset].copy() if not trades_df.empty else pd.DataFrame()
        
        # Display last price
        if not asset_market.empty and "close" in asset_market.columns:
            last_close = asset_market["close"].dropna().iloc[-1]
            if pd.notna(last_close):
                st.metric(f"Last Price for {selected_asset}", format_price(float(last_close)))
        
        # Create and display chart with responsive height
        fig = create_price_chart(asset_market, asset_trades, selected_asset, st.session_state.theme)
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": True, "scrollZoom": True}
        )
        
        # Show latest data
        with st.expander("📎 Latest rows (quick check)"):
            display_cols = [c for c in ["timestamp_pst", "open", "high", "low", "close", "p_up", "p_down"] 
                          if c in asset_market.columns]
            st.dataframe(asset_market[display_cols].tail(15), use_container_width=True)

# ========= TAB 2: P&L Analysis =========
with tab2:
    if trades_df.empty or "timestamp_pst" not in trades_df.columns:
        st.warning("No trade data loaded to analyze P&L.")
    else:
        st.markdown("### Strategy Performance")
        
        # Use responsive column layout
        c0, c1, c2, c3, c4 = st.columns(5)
        with c0:
            st.metric("Overall Realized P&L", f"${overall_realized:+.2f}")
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
        
        # Cumulative P&L chart with responsive height
        if not trades_with_pnl.empty:
            fig_pnl = create_pnl_chart(trades_with_pnl, st.session_state.theme)
            st.plotly_chart(fig_pnl, use_container_width=True)
        
        # Per-asset P&L
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
            
            # Bar chart with responsive height
            fig_bar = create_pnl_bar_chart(per_asset_pnl, st.session_state.theme)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No realized P&L yet (need at least one completed sell).")

# ========= TAB 3: Trade History =========
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
                height=400  # Scrollable on mobile
            )
        else:
            st.info("No completed trades found.")
        
        st.markdown("---")
        st.markdown("#### Open Positions (Unmatched Buys)")
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

# ========= TAB 4: Validations =========
with tab4:
    st.markdown("### Threshold Validations at Execution Time")
    
    if trades_df.empty:
        st.info("No trades loaded.")
    else:
        action_col = "action" if "action" in trades_df.columns else "unified_action"
        opens_mask = trades_df[action_col].astype(str).str.upper().isin(["OPEN", "BUY"])
        opens = trades_df.loc[opens_mask].copy()
        
        if opens.empty:
            st.info("No OPEN orders in the log.")
        else:
            total_opens = len(opens)
            invalid = opens[~opens["valid_at_open"]]
            valid = opens[opens["valid_at_open"]]
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("OPEN Orders", f"{total_opens:,}")
            with c2:
                st.metric("Valid at OPEN", f"{len(valid):,}")
            with c3:
                st.metric("Flagged (Invalid)", f"{len(invalid):,}")
            with c4:
                vr = (len(valid) / total_opens * 100) if total_opens > 0 else 0
                st.metric("Validity Rate", f"{vr:.1f}%")
            
            if not invalid.empty:
                st.markdown("#### ❗ Flagged OPENs")
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
            
            st.caption("Validity is evaluated against per-asset thresholds and logged probabilities at execution time.")

# ========= Sidebar: Open Positions =========
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
            
            asset_name = pos["Asset"]
            cur_price = pos["Current Price"]
            avg_price = pos["Avg. Entry Price"]
            
            st.markdown(
                f"""
                <div style="background-color: {card_bg}; border-left: 4px solid {color}; 
                           padding: 12px; margin: 8px 0; border-radius: 4px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <strong style="font-size: 14px;">{asset_name}</strong>
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
                            Current: <strong>{format_price(cur_price)}</strong>
                        </span>
                        <span style="font-size: 12px; color: #666;">
                            Entry: <strong>{format_price(avg_price)}</strong>
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
        total_positions = len(sorted_positions)
        
        st.markdown("---")
        st.markdown(
            f"""
            <div style='text-align: center; padding: 0.5rem; 
                       background-color: rgba(128,128,128,0.1); 
                       border-radius: 0.25rem; margin-top: 0.5rem;'>
                <div style='font-size: 0.8rem; 
                           color: {"#16a34a" if total_pnl >= 0 else "#ef4444"}; 
                           font-weight: 600;'>
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
            <div style='text-align: center; padding: 1rem; 
                       background-color: rgba(128,128,128,0.05); 
                       border-radius: 0.25rem; color: #666; font-style: italic;'>
                No open positions
            </div>
            """,
            unsafe_allow_html=True
        )
