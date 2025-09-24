import io
import re
import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from datetime import datetime, timedelta
import pytz
from decimal import Decimal, getcontext

# Set precision for Decimal calculations
getcontext().prec = 30

# Note: Reading Parquet files with pandas requires the 'pyarrow' and 'pytz' libraries.
# pip install streamlit pandas plotly pyarrow requests pytz
import pyarrow  # noqa: F401 (ensures pyarrow engine is available)

# =========================
# App Configuration
# =========================
st.set_page_config(page_title="Crypto Trading Strategy", layout="wide")

# =========================
# Auto-Refresh Setup
# =========================
REFRESH_INTERVAL = 300  # 5 minutes in seconds

# Initialize session state for auto-refresh
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()
if 'auto_refresh_enabled' not in st.session_state:
    st.session_state.auto_refresh_enabled = True

DEFAULT_ASSET = "GIGA-USD"

# =========================
# Google Drive Links
# =========================
# Trade log file (trades_log.parquet) - UPDATED
TRADES_LINK = "https://drive.google.com/file/d/1hyM37eafLgvMo8RJDtw9GSEdZ-LQ05ks/view?usp=sharing"
# Features/market data file exported by your OHLCV service (parquet or csv)
MARKET_LINK = "https://drive.google.com/file/d/17ASJZw2zZ0oZweuiN62uDx97tRP9fxF1/view?usp=sharing"

# =========================
# Helpers — data processing
# =========================
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
    # Keep existing column names, but normalize common variants for p_up/p_down.
    rename_map = {}
    p_up_variations = {"p_up", "p-up", "pup", "pup prob", "p up"}
    p_down_variations = {"p_down", "p-down", "pdown", "pdown prob", "p down"}
    for col in df.columns:
        col_lower_
