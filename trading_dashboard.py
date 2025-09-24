#!/usr/bin/env python3
"""
Advanced Crypto Trading Bot
VERSION: 2.5.0 (Uses fetcher 'confidence' + clean fallbacks)

Core logic:
- Entry (longs only by default):
    p_up > p_down  AND
    p_up >= buy_threshold  AND
    confidence >= min_confidence
- Exit (in addition to price-based TP/SL/Trailing/Emergency):
    p_up <= exit_threshold  OR
    confidence < min_confidence

Notes:
- 'confidence' is now consumed from the data file (top - second-best probability),
  produced by the OHLCV service. If it's missing, we fall back to computing it
  from [p_down, p_hold, p_up] present in the row.
- Timestamps in the data are NAIVE PST; we treat them as plain datetimes.
"""

import os
import sys
import time
import json
import logging
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import hmac
import hashlib
import base64
from urllib.parse import urlencode

# =========================
# CONFIGURATION
# =========================

BASE = Path(__file__).parent.resolve()
LOGS_DIR = BASE / "logs" / "trading"
DATA_DIR = BASE / "Data"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    # ============== Trading Mode ==============
    "paper_trading": True,                 # False for live trading
    "bypass_api_in_paper_mode": True,      # No Coinbase API calls in paper mode
    "simulate_slippage": True,
    "slippage_bps": 5,                     # 5 bps = 0.05%

    # ============== Trading Assets ==============
    "assets": ["CVX-USD", "MNDE-USD", "MOG-USD", "VVV-USD", "LCX-USD", "GIGA-USD"],
    "safe_haven": "PAXG-USD",

    # ============== Initial Balances ==============
    "initial_balance_per_asset": 100.0,

    # ============== Per-Asset Configs (defaults) ==============
    "asset_configs": {
        "CVX-USD": {"buy_threshold": 0.70, "exit_threshold": 0.65, "min_confidence": 0.60,
                    "max_position_pct": 0.95, "trailing_stop": True, "volatility_multiplier": 1.2},
        "MNDE-USD": {"buy_threshold": 0.68, "exit_threshold": 0.62, "min_confidence": 0.60,
                     "max_position_pct": 0.90, "trailing_stop": True, "volatility_multiplier": 1.5},
        "MOG-USD": {"buy_threshold": 0.75, "exit_threshold": 0.62, "min_confidence": 0.65,
                    "max_position_pct": 0.85, "trailing_stop": True, "volatility_multiplier": 2.0},
        "VVV-USD": {"buy_threshold": 0.65, "exit_threshold": 0.60, "min_confidence": 0.55,
                    "max_position_pct": 0.95, "trailing_stop": False, "volatility_multiplier": 1.3},
        "LCX-USD": {"buy_threshold": 0.72, "exit_threshold": 0.68, "min_confidence": 0.60,
                    "max_position_pct": 0.92, "trailing_stop": True, "volatility_multiplier": 1.4},
        "GIGA-USD": {"buy_threshold": 0.73, "exit_threshold": 0.65, "min_confidence": 0.62,
                     "max_position_pct": 0.88, "trailing_stop": True, "volatility_multiplier": 1.8},
    },

    # ============== Dynamic TP/SL Settings (global defaults) ==============
    "tp_sl_config": {
        "base_take_profit_pct": 0.05,     # 5%
        "base_stop_loss_pct": 0.03,       # 3%
        "atr_lookback": 14,
        "confidence_boost_factor": 0.5,   # widens TP, tightens SL with higher confidence
        "trailing_stop_trigger": 0.02,    # 2% profit activates trailing
        "trailing_stop_distance": 0.015,  # 1.5% trail
        "max_take_profit": 0.15,          # 15% cap
        "max_stop_loss": 0.08,            # 8% cap
    },

    # ============== Risk Management ==============
    "emergency_stop_loss": 0.10,          # -10% hard stop from entry

    # ============== Data & Timing ==============
    "data_file": DATA_DIR / "latest_features.parquet",
    "portfolio_file": LOGS_DIR / "portfolio.json",
    "trades_log": Path("/home/samuel/Desktop/Bot/GIGA/Data") / "trades_log.parquet",
    "trading_cycle_seconds": 20,
    "data_freshness_limit": 28800,        # 8 hours (relaxed for dev)

    # ============== API Settings ==============
    "api_base": "https://api.exchange.coinbase.com",
    "sandbox": True,
    "request_timeout": 10,

    # ============== Overrides file (hot reload) ==============
    "asset_overrides_file": BASE / "asset_overrides.json",
    "reload_asset_overrides_each_cycle": True,
}

DEFAULT_ASSET_CONFIG = {
    "buy_threshold": 0.70,
    "exit_threshold": 0.65,
    "min_confidence": 0.60,
    "volatility_multiplier": 1.5,
    "max_position_pct": 0.90,
    "trailing_stop": True,
}

API_CREDENTIALS = {
    "key": os.getenv("COINBASE_API_KEY", ""),
    "secret": os.getenv("COINBASE_API_SECRET", ""),
    "passphrase": os.getenv("COINBASE_API_PASSPHRASE", ""),
}

if CONFIG["sandbox"]:
    CONFIG["api_base"] = "https://api-public.sandbox.exchange.coinbase.com"

# =========================
# LOGGING
# =========================

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(LOGS_DIR / "trading_bot.log")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logging.info("🚀 Trading Bot Starting")

# =========================
# CONFIG MANAGER (per-asset overrides)
# =========================

class ConfigManager:
    def __init__(self, root_cfg: dict):
        self.root = root_cfg
        self._overrides = {}
        self.load_overrides()

    def load_overrides(self):
        path = self.root.get("asset_overrides_file")
        self._overrides = {}
        if not path:
            return
        try:
            if Path(path).exists():
                with open(path, "r") as f:
                    self._overrides = json.load(f) or {}
                logging.info(f"🔁 Loaded overrides from {path}")
        except Exception as e:
            logging.warning(f"Failed to load overrides: {e}")

    @staticmethod
    def _deep_merge(base: dict, add: dict) -> dict:
        out = dict(base or {})
        for k, v in (add or {}).items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = ConfigManager._deep_merge(out[k], v)
            else:
                out[k] = v
        return out

    def get_asset_cfg(self, asset: str) -> dict:
        in_code = self.root.get("asset_configs", {}).get(asset, {})
        ov_asset = (self._overrides.get("asset_configs", {}) or {}).get(asset, {})
        return self._deep_merge(DEFAULT_ASSET_CONFIG, self._deep_merge(in_code, ov_asset))

    def get_tp_sl_cfg(self, asset: str) -> dict:
        base = dict(self.root.get("tp_sl_config", {}))
        ov_all = (self._overrides.get("tp_sl_overrides", {}) or {}).get("*", {})
        ov_asset = (self._overrides.get("tp_sl_overrides", {}) or {}).get(asset, {})
        return self._deep_merge(base, self._deep_merge(ov_all, ov_asset))

    def maybe_reload_each_cycle(self):
        if self.root.get("reload_asset_overrides_each_cycle", False):
            self.load_overrides()

CONFIG_MGR = ConfigManager(CONFIG)

# =========================
# COINBASE API CLIENT
# =========================

class CoinbaseClient:
    def __init__(self, api_key: str, secret: str, passphrase: str, sandbox: bool = True, paper_mode: bool = False):
        self.paper_mode = paper_mode
        if not paper_mode:
            self.api_key = api_key
            self.secret = secret
            self.passphrase = passphrase
            self.base_url = CONFIG["api_base"]
            logging.info("💰 Live API client initialized")
        else:
            self.api_key = None
            self.secret = None
            self.passphrase = None
            self.base_url = None
            logging.info("📄 Paper mode - API bypassed")

    def _create_signature(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        if self.paper_mode:
            return "paper_trading_signature"
        message = timestamp + method + path + body
        hmac_key = base64.b64decode(self.secret)
        signature = hmac.new(hmac_key, message.encode(), hashlib.sha256)
        return base64.b64encode(signature.digest()).decode()

    def _make_request(self, method: str, path: str, params: dict = None, data: dict = None) -> dict:
        if self.paper_mode:
            if "/ticker" in path:
                return {"price": "0.0", "paper_mode": True}
            elif "/orders" in path:
                return {"id": f"paper_order_{int(time.time())}", "status": "done", "paper_mode": True}
            return {"paper_mode": True}

        if not all([self.api_key, self.secret, self.passphrase]):
            raise ValueError("Missing API credentials for live trading")

        timestamp = str(time.time())
        url = f"{self.base_url}{path}"
        if params:
            q = urlencode(params)
            url += f"?{q}"
            path += f"?{q}"
        body = json.dumps(data) if data else ""
        signature = self._create_signature(timestamp, method.upper(), path, body)
        headers = {
            "CB-ACCESS-KEY": self.api_key,
            "CB-ACCESS-SIGN": signature,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "CB-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json"
        }
        try:
            resp = requests.request(method, url, headers=headers, data=body if data else None,
                                    timeout=CONFIG["request_timeout"])
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logging.error(f"API request failed: {e}")
            return {}

    def get_product_ticker(self, product_id: str) -> dict:
        if self.paper_mode:
            return {"price": "0.0", "paper_mode": True}
        return self._make_request("GET", f"/products/{product_id}/ticker")

    def place_market_order(self, side: str, product_id: str, size: Optional[str] = None, funds: Optional[str] = None) -> dict:
        if self.paper_mode:
            logging.info(f"📄 PAPER ORDER: {side.upper()} {product_id} size={size} funds={funds}")
            return {"id": f"paper_{side}_{product_id}_{int(time.time())}", "status": "done",
                    "side": side, "product_id": product_id, "size": size, "funds": funds,
                    "paper_mode": True, "no_api_call": True}
        data = {"side": side, "product_id": product_id, "type": "market"}
        if size: data["size"] = size
        elif funds: data["funds"] = funds
        else: raise ValueError("Must specify either size or funds")
        return self._make_request("POST", "/orders", data=data)

# =========================
# PAPER TRADING PRICE MANAGER
# =========================

class PaperTradingPriceManager:
    def __init__(self):
        self.current_prices = {}
        self.last_update = None

    def update_prices_from_data(self, df: pd.DataFrame):
        try:
            latest = df.sort_values("timestamp").groupby("product_id").tail(1)
            for _, row in latest.iterrows():
                asset = row["product_id"]
                price = float(row.get("close", 0.0))
                if price > 0:
                    self.current_prices[asset] = price
            self.last_update = datetime.now()
        except Exception as e:
            logging.error(f"Failed to update prices: {e}")

    def get_price(self, asset: str) -> Optional[float]:
        return self.current_prices.get(asset)

    def apply_slippage(self, price: float, side: str) -> float:
        if not CONFIG.get("simulate_slippage", False):
            return price
        slip = CONFIG.get("slippage_bps", 5) / 10000.0
        return price * (1 + slip) if side.lower() == "buy" else price * (1 - slip)

# =========================
# ENHANCED PORTFOLIO
# =========================

class EnhancedPortfolio:
    def __init__(self, portfolio_file: Path, config_mgr: ConfigManager):
        self.file = portfolio_file
        self.config_mgr = config_mgr
        self.positions = self._load_portfolio()
        self.tpsl_calculator = DynamicTPSLCalculator(self.config_mgr)

    def _load_portfolio(self) -> dict:
        if self.file.exists():
            try:
                with open(self.file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load portfolio: {e}")

        initial = {"created_at": datetime.now().isoformat(),
                   "total_initial_value": len(CONFIG["assets"]) * CONFIG["initial_balance_per_asset"],
                   "assets": {}}
        for asset in CONFIG["assets"]:
            initial["assets"][asset] = {
                "quantity": 0.0,
                "entry_price": 0.0,
                "current_value": 0.0,
                "allocated_balance": CONFIG["initial_balance_per_asset"],
                "in_position": False,
                "position_side": None,
                "take_profit_price": 0.0,
                "stop_loss_price": 0.0,
                "highest_price": 0.0,
                "lowest_price": 0.0,
                "trailing_stop_active": False,
                "total_pnl": 0.0,
                "win_trades": 0,
                "lose_trades": 0,
                "total_trades": 0,
                "last_trade_date": None,
                "last_trade_time": None
            }
        self._save_portfolio(initial)
        return initial

    def _save_portfolio(self, portfolio: dict = None):
        try:
            with open(self.file, 'w') as f:
                json.dump(self.positions if portfolio is None else portfolio, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save portfolio: {e}")

    def get_asset_info(self, asset: str) -> dict:
        return self.positions["assets"].get(asset, {})

    def can_trade(self, asset: str) -> Tuple[bool, str]:
        info = self.get_asset_info(asset)
        if not info: return False, "Asset not found"
        if info.get("in_position", False):
            return False, f"Already in {info.get('position_side','?')} position"
        return True, "OK"

    def open_position(self, asset: str, side: str, quantity: float, entry_price: float,
                      asset_data: pd.Series, probabilities: dict):
        info = self.get_asset_info(asset)
        levels = self.tpsl_calculator.calculate_levels(asset, entry_price, side, asset_data, probabilities)
        now_iso = datetime.now().isoformat()
        info.update({
            "quantity": quantity,
            "entry_price": entry_price,
            "current_value": quantity * entry_price,
            "in_position": True,
            "position_side": side,
            "take_profit_price": levels["take_profit_price"],
            "stop_loss_price": levels["stop_loss_price"],
            "highest_price": entry_price,
            "lowest_price": entry_price,
            "trailing_stop_active": False,
            "entry_time": now_iso,
            "last_trade_time": now_iso,
            "last_trade_date": datetime.now().date().isoformat(),
            "total_trades": info.get("total_trades", 0) + 1
        })
        self._save_portfolio()
        logging.info(f"📈 {asset} OPEN {side.upper()} qty={quantity:.6f} @ {entry_price:.6f} | "
                     f"TP={levels['take_profit_price']:.6f} SL={levels['stop_loss_price']:.6f}")

    def close_position(self, asset: str, exit_price: float, reason: str) -> dict:
        info = self.get_asset_info(asset)
        if not info.get("in_position", False):
            return {"success": False, "reason": "No position"}
        entry = info["entry_price"]; qty = info["quantity"]; side = info["position_side"]
        pnl = qty * ((exit_price - entry) if side == "buy" else (entry - exit_price))
        pnl_pct = ((exit_price - entry) / entry) if side == "buy" else ((entry - exit_price) / entry)
        info["total_pnl"] += pnl
        info["allocated_balance"] += pnl
        if pnl > 0: info["win_trades"] += 1
        else: info["lose_trades"] += 1
        info.update({"quantity": 0.0, "entry_price": 0.0, "current_value": 0.0,
                     "in_position": False, "position_side": None,
                     "take_profit_price": 0.0, "stop_loss_price": 0.0,
                     "highest_price": 0.0, "lowest_price": 0.0,
                     "trailing_stop_active": False,
                     "last_trade_time": datetime.now().isoformat()})
        self._save_portfolio()
        logging.info(f"🔴 {asset} CLOSE reason='{reason}' exit={exit_price:.6f} "
                     f"P&L=${pnl:.2f} ({pnl_pct:.2%}) bal=${info['allocated_balance']:.2f}")
        return {"success": True, "pnl": pnl, "pnl_pct": pnl_pct, "entry_price": entry,
                "exit_price": exit_price, "quantity": qty, "side": side}

    def update_trailing_stops(self, asset: str, current_price: float):
        info = self.get_asset_info(asset)
        if not info.get("in_position", False): return
        side = info["position_side"]; entry = info["entry_price"]
        a_cfg = self.config_mgr.get_asset_cfg(asset)
        if not a_cfg.get("trailing_stop", False): return
        ts_cfg = self.config_mgr.get_tp_sl_cfg(asset)

        if side == "buy":
            if current_price > info["highest_price"]:
                info["highest_price"] = current_price
            profit_pct = (current_price - entry) / entry
            if profit_pct >= ts_cfg["trailing_stop_trigger"]:
                info["trailing_stop_active"] = True
            if info["trailing_stop_active"]:
                new_sl = info["highest_price"] * (1 - ts_cfg["trailing_stop_distance"])
                if new_sl > info["stop_loss_price"]:
                    info["stop_loss_price"] = new_sl
        else:
            if current_price < info["lowest_price"]:
                info["lowest_price"] = current_price
            profit_pct = (entry - current_price) / entry
            if profit_pct >= ts_cfg["trailing_stop_trigger"]:
                info["trailing_stop_active"] = True
            if info["trailing_stop_active"]:
                new_sl = info["lowest_price"] * (1 + ts_cfg["trailing_stop_distance"])
                if new_sl < info["stop_loss_price"]:
                    info["stop_loss_price"] = new_sl
        self._save_portfolio()

    def check_exit_conditions(self, asset: str, current_price: float) -> Tuple[bool, str]:
        info = self.get_asset_info(asset)
        if not info.get("in_position", False): return False, "No position"
        side = info["position_side"]; entry = info["entry_price"]
        tp = info["take_profit_price"]; sl = info["stop_loss_price"]

        emergency_pct = CONFIG["emergency_stop_loss"]
        if side == "buy":
            if current_price <= entry * (1 - emergency_pct):
                return True, f"Emergency SL {emergency_pct:.0%}"
            if current_price >= tp:
                return True, f"Take Profit @ {tp:.6f}"
            if current_price <= sl:
                return True, f"Stop Loss @ {sl:.6f}"
        else:
            if current_price >= entry * (1 + emergency_pct):
                return True, f"Emergency SL {emergency_pct:.0%}"
            if current_price <= tp:
                return True, f"Take Profit @ {tp:.6f}"
            if current_price >= sl:
                return True, f"Stop Loss @ {sl:.6f}"
        return False, "Hold"

# =========================
# DYNAMIC TP/SL
# =========================

class DynamicTPSLCalculator:
    def __init__(self, config_mgr: ConfigManager):
        self.cfg_mgr = config_mgr

    def calculate_levels(self, asset: str, entry_price: float, side: str,
                         asset_data: pd.Series, probabilities: dict) -> dict:
        a_cfg = self.cfg_mgr.get_asset_cfg(asset)
        t_cfg = self.cfg_mgr.get_tp_sl_cfg(asset)

        base_tp = float(t_cfg["base_take_profit_pct"])
        base_sl = float(t_cfg["base_stop_loss_pct"])
        max_tp = float(t_cfg["max_take_profit"])
        max_sl = float(t_cfg["max_stop_loss"])
        boost = float(t_cfg.get("confidence_boost_factor", 0.5))

        atr_rel = float(asset_data.get("ATRr_14", 0.02))
        vol_factor = max(0.5, min(3.0, atr_rel * 50.0))
        p_up = float(probabilities.get("p_up", 0))
        p_down = float(probabilities.get("p_down", 0))
        p_hold = float(probabilities.get("p_hold", 0))
        # Use provided confidence if present; else fallback to proper top-second calc
        conf = float(probabilities.get("confidence", _calc_conf_from_probs(p_down, p_hold, p_up)))
        conf_factor = max(0.1, min(2.0, (conf - 0.33) / 0.67))
        asset_mult = float(a_cfg.get("volatility_multiplier", 1.5))

        if side == "buy":
            tp_pct = base_tp * (1 + conf_factor * boost) * vol_factor * asset_mult
            tp_pct = min(tp_pct, max_tp)
            take_profit_price = entry_price * (1 + tp_pct)

            sl_pct = base_sl * vol_factor * asset_mult / max(0.5, conf_factor)
            sl_pct = min(sl_pct, max_sl)
            stop_loss_price = entry_price * (1 - sl_pct)
        else:
            tp_pct = base_tp * (1 + conf_factor * boost) * vol_factor * asset_mult
            tp_pct = min(tp_pct, max_tp)
            take_profit_price = entry_price * (1 - tp_pct)

            sl_pct = base_sl * vol_factor * asset_mult / max(0.5, conf_factor)
            sl_pct = min(sl_pct, max_sl)
            stop_loss_price = entry_price * (1 + sl_pct)

        return {
            "take_profit_price": take_profit_price,
            "stop_loss_price": stop_loss_price,
            "take_profit_pct": tp_pct,
            "stop_loss_pct": sl_pct
        }

# =========================
# CONFIDENCE UTIL
# =========================

def _calc_conf_from_probs(p_down: float, p_hold: float, p_up: float) -> float:
    """Top-minus-second confidence from three-class probabilities."""
    arr = np.array([p_down, p_hold, p_up], dtype=float)
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0)
    best = np.max(arr)
    second = np.partition(arr, -2)[-2]
    return float(max(0.0, min(1.0, best - second)))

# =========================
# ADVANCED TRADING BOT
# =========================

class AdvancedTradingBot:
    def __init__(self):
        self.paper_mode = CONFIG.get("paper_trading", True)
        if self.paper_mode:
            self.client = CoinbaseClient("", "", "", sandbox=True, paper_mode=True)
            self.price_manager = PaperTradingPriceManager()
        else:
            self.client = CoinbaseClient(**API_CREDENTIALS, sandbox=CONFIG["sandbox"], paper_mode=False)
            self.price_manager = None

        self.config_mgr = CONFIG_MGR
        self.portfolio = EnhancedPortfolio(CONFIG["portfolio_file"], self.config_mgr)
        self.trades_log = CONFIG["trades_log"]
        self.latest_snapshot: Optional[pd.DataFrame] = None

        # Ensure log path exists and file seeded
        self.trades_log.parent.mkdir(parents=True, exist_ok=True)
        if not self.trades_log.exists():
            pd.DataFrame(columns=[
                "timestamp","asset","action","side","quantity","price","value",
                "reason","p_up","p_down","p_hold","confidence",
                "take_profit","stop_loss","pnl","pnl_pct","balance_after","trading_mode"
            ]).to_parquet(self.trades_log, index=False)
            logging.info(f"📊 Created trades log: {self.trades_log}")

    # ---------- helpers ----------

    def _log_trade(self, asset: str, action: str, side: str, quantity: float,
                   price: float, reason: str, probabilities: dict,
                   tp_price: float = 0, sl_price: float = 0,
                   pnl: float = 0, pnl_pct: float = 0):
        try:
            ai = self.portfolio.get_asset_info(asset)
            p_up = float(probabilities.get("p_up", 0))
            p_down = float(probabilities.get("p_down", 0))
            p_hold = float(probabilities.get("p_hold", 0))
            conf = float(probabilities.get("confidence", _calc_conf_from_probs(p_down, p_hold, p_up)))
            trade_data = {
                "timestamp": datetime.now().isoformat(),
                "asset": asset,
                "action": action,
                "side": side,
                "quantity": quantity,
                "price": price,
                "value": quantity * price,
                "reason": reason,
                "p_up": p_up,
                "p_down": p_down,
                "p_hold": p_hold,
                "confidence": conf,
                "take_profit": tp_price,
                "stop_loss": sl_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "balance_after": ai.get("allocated_balance", 0),
                "trading_mode": "PAPER" if self.paper_mode else "LIVE"
            }
            existing = pd.read_parquet(self.trades_log) if self.trades_log.exists() else pd.DataFrame()
            out = pd.concat([existing, pd.DataFrame([trade_data])], ignore_index=True) if not existing.empty else pd.DataFrame([trade_data])
            out.to_parquet(self.trades_log, index=False)
            logging.info(f"📝 Logged trade: {action} {asset} @ {price:.6f}")
        except Exception as e:
            logging.error(f"Failed to log trade: {e}")

    def _get_current_price(self, product_id: str) -> Optional[float]:
        try:
            if self.paper_mode and self.price_manager:
                return self.price_manager.get_price(product_id)
            t = self.client.get_product_ticker(product_id)
            if t and "price" in t:
                return float(t["price"])
        except Exception as e:
            logging.warning(f"Price fetch failed for {product_id}: {e}")
        return None

    def _read_latest_data(self) -> Optional[pd.DataFrame]:
        """
        Reads latest_features.parquet and ensures a 'confidence' column exists.
        If absent, computes confidence = top - second among [p_down,p_hold,p_up].
        Also creates a per-asset latest snapshot for quick lookups.
        """
        try:
            f = CONFIG["data_file"]
            if not Path(f).exists():
                logging.warning(f"Data file not found: {f}")
                return None
            df = pd.read_parquet(f)

            # Normalize timestamp (NAIVE PST in file; treat as naive)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                if df["timestamp"].notna().any():
                    age_sec = (datetime.now() - df["timestamp"].max()).total_seconds()
                    if age_sec > CONFIG["data_freshness_limit"]:
                        logging.warning(f"Data stale: {age_sec:.0f}s old")
                        return None

            # Ensure confidence exists
            needed = {"p_down", "p_hold", "p_up"}
            if "confidence" not in df.columns or df["confidence"].isna().all():
                if needed.issubset(df.columns):
                    probs = df[["p_down","p_hold","p_up"]].to_numpy(dtype=float)
                    best = np.nanmax(probs, axis=1)
                    second = np.partition(probs, -2, axis=1)[:, -2]
                    df["confidence"] = np.clip(best - second, 0.0, 1.0)
                else:
                    # last-ditch: if only up/down exist
                    if {"p_up","p_down"}.issubset(df.columns):
                        df["confidence"] = (df["p_up"] - df["p_down"]).abs()
                    else:
                        df["confidence"] = 0.0

            # Build latest snapshot by product
            try:
                snap = df.sort_values("timestamp").groupby("product_id", as_index=False).tail(1).set_index("product_id")
                self.latest_snapshot = snap
            except Exception:
                self.latest_snapshot = None
            return df
        except Exception as e:
            logging.error(f"Failed reading data: {e}")
            return None

    # ---------- trading logic ----------

    def _should_open_position(self, asset: str, asset_data: pd.Series) -> Tuple[bool, str, dict]:
        cfg = self.config_mgr.get_asset_cfg(asset)
        p_up = float(asset_data.get("p_up", 0.0))
        p_down = float(asset_data.get("p_down", 0.0))
        p_hold = float(asset_data.get("p_hold", 0.0))
        confidence = float(asset_data.get("confidence", _calc_conf_from_probs(p_down, p_hold, p_up)))
        probs = {"p_up": p_up, "p_down": p_down, "p_hold": p_hold, "confidence": confidence}

        # Already in position?
        can, why = self.portfolio.can_trade(asset)
        if not can:
            return False, why, probs

        # Entry rule uses the real 'confidence' now
        if (p_up > p_down) and (p_up >= cfg["buy_threshold"]) and (confidence >= cfg.get("min_confidence", 0.60)):
            return True, "BUY", probs

        return False, f"No entry (p_up={p_up:.3f}, p_down={p_down:.3f}, conf={confidence:.3f})", probs

    def _execute_open_position(self, asset: str, side: str, asset_data: pd.Series, probabilities: dict) -> bool:
        try:
            price = self._get_current_price(asset)
            if not price:
                logging.warning(f"No price for {asset}")
                return False
            exec_price = self.price_manager.apply_slippage(price, side) if (self.paper_mode and self.price_manager) else price

            ai = self.portfolio.get_asset_info(asset)
            a_cfg = self.config_mgr.get_asset_cfg(asset)
            avail = ai["allocated_balance"]
            max_value = avail * a_cfg.get("max_position_pct", 0.9)
            if max_value < 10:
                logging.warning(f"{asset}: low balance ${avail:.2f}")
                return False
            qty = max_value / exec_price

            # Place (simulate or live)
            _ = self.client.place_market_order(side, asset, funds=str(max_value))

            # Update portfolio & log
            self.portfolio.open_position(asset, side, qty, exec_price, asset_data, probabilities)
            ai = self.portfolio.get_asset_info(asset)
            self._log_trade(asset, "OPEN", side, qty, exec_price,
                            f"Signal {side.upper()} p_up={probabilities.get('p_up',0):.3f}",
                            probabilities, tp_price=ai["take_profit_price"], sl_price=ai["stop_loss_price"])
            return True
        except Exception as e:
            logging.error(f"Open position failed {asset}: {e}", exc_info=True)
            return False

    def _probability_exit_signal(self, asset: str) -> Tuple[bool, str]:
        if self.latest_snapshot is None or asset not in self.latest_snapshot.index:
            return False, "No snapshot"
        row = self.latest_snapshot.loc[asset]
        cfg = self.config_mgr.get_asset_cfg(asset)
        p_up = float(row.get("p_up", 0.0))
        p_down = float(row.get("p_down", 0.0))
        p_hold = float(row.get("p_hold", 0.0))
        conf = float(row.get("confidence", _calc_conf_from_probs(p_down, p_hold, p_up)))

        if p_up <= cfg["exit_threshold"]:
            return True, f"Prob exit: p_up {p_up:.3f} ≤ exit_threshold {cfg['exit_threshold']:.2f}"
        if conf < cfg.get("min_confidence", 0.60):
            return True, f"Prob exit: confidence {conf:.3f} < min_confidence {cfg['min_confidence']:.2f}"
        return False, "Hold"

    def _check_and_close_positions(self):
        for asset in CONFIG["assets"]:
            ai = self.portfolio.get_asset_info(asset)
            if not ai.get("in_position", False):
                continue

            price = self._get_current_price(asset)
            if not price:
                continue

            # update trailing
            self.portfolio.update_trailing_stops(asset, price)

            # price-based exits first
            should, reason = self.portfolio.check_exit_conditions(asset, price)

            # probability-based exit if price didn't trigger
            if not should:
                pexit, preason = self._probability_exit_signal(asset)
                if pexit:
                    should, reason = True, preason

            if should:
                exit_side = "sell" if ai["position_side"] == "buy" else "buy"
                exec_price = self.price_manager.apply_slippage(price, exit_side) if (self.paper_mode and self.price_manager) else price
                result = self.portfolio.close_position(asset, exec_price, reason)
                if result.get("success", False):
                    _ = self.client.place_market_order(exit_side, asset, size=str(result["quantity"]))
                    self._log_trade(asset, "CLOSE", result["side"], result["quantity"], exec_price,
                                    reason, {}, pnl=result["pnl"], pnl_pct=result["pnl_pct"])

    def _process_asset_signals(self, df: pd.DataFrame):
        # scan a small lookback for each asset to catch recent valid signals (not just latest)
        recent = df.groupby("product_id").tail(20)
        for asset, grp in recent.groupby("product_id"):
            if asset not in CONFIG["assets"]:
                continue
            for _, row in grp.sort_values("timestamp", ascending=False).iterrows():
                ok, side, probs = self._should_open_position(asset, row)
                if ok and side in ("BUY",):
                    # age check: optional — keep generous (<= 120 min)
                    signal_time = pd.to_datetime(row["timestamp"])
                    age_min = (pd.Timestamp.now() - signal_time).total_seconds() / 60.0
                    if age_min <= 120:
                        logging.info(f"🎯 {asset} {side}: p_up={probs['p_up']:.3f}, conf={probs['confidence']:.3f}, {age_min:.1f} min ago")
                        self._execute_open_position(asset, side.lower(), row, probs)
                        break  # one open per asset per cycle

    # ---------- main cycle ----------

    def run_trading_cycle(self):
        try:
            # hot reload overrides
            self.config_mgr.maybe_reload_each_cycle()

            df = self._read_latest_data()
            if df is None or df.empty:
                logging.warning("No valid data for trading")
                return

            if self.paper_mode and self.price_manager:
                self.price_manager.update_prices_from_data(df)

            # exits then entries
            self._check_and_close_positions()
            self._process_asset_signals(df)

            # status
            total_value = 0.0
            active = 0
            total_pnl = 0.0
            for asset in CONFIG["assets"]:
                ai = self.portfolio.get_asset_info(asset)
                bal = ai["allocated_balance"]
                total_value += bal
                total_pnl += ai.get("total_pnl", 0.0)
                if ai.get("in_position", False):
                    active += 1
                    cur = self._get_current_price(asset) or 0.0
                    entry = ai["entry_price"]; qty = ai["quantity"]
                    unreal = qty * ((cur - entry) if ai["position_side"] == "buy" else (entry - cur))
                    logging.info(f"💼 {asset}: bal=${bal:.2f} | {ai['position_side'].upper()} @ {entry:.6f} | UPNL=${unreal:.2f}")
            mode = "📄" if self.paper_mode else "💰"
            logging.info(f"{mode} Portfolio=${total_value:.2f} | Total P&L=${total_pnl:.2f} | Active={active}")

        except Exception as e:
            logging.error(f"Trading cycle error: {e}", exc_info=True)

    def start(self):
        mode_text = "PAPER" if self.paper_mode else "LIVE"
        logging.info(f"🚀 Starting bot ({mode_text})")
        logging.info(f"Assets: {CONFIG['assets']}")
        logging.info(f"Cycle: {CONFIG['trading_cycle_seconds']}s")
        if not self.paper_mode:
            logging.info(f"Sandbox: {CONFIG['sandbox']}")

        cycle = 0
        while True:
            try:
                cycle += 1
                logging.info(f"--- Cycle #{cycle} ---")
                self.run_trading_cycle()
                time.sleep(CONFIG["trading_cycle_seconds"])
            except KeyboardInterrupt:
                logging.info("🛑 Stopped by user")
                break
            except Exception as e:
                logging.error(f"Unexpected error: {e}", exc_info=True)
                time.sleep(60)

# =========================
# MAIN
# =========================

def main():
    setup_logging()
    paper = CONFIG.get("paper_trading", True)
    bypass = CONFIG.get("bypass_api_in_paper_mode", True)
    if paper and bypass:
        logging.info("📄 Pure paper mode: no Coinbase API calls; using fetcher data for prices.")
    elif paper and not bypass:
        logging.info("📄 Paper mode with API prices.")
        if not all(API_CREDENTIALS.values()):
            logging.error("Missing API credentials for paper+API mode.")
            return
    else:
        logging.info("💰 LIVE mode")
        if not all(API_CREDENTIALS.values()):
            logging.error("Missing API credentials for LIVE trading.")
            return

    bot = AdvancedTradingBot()
    bot.start()

if __name__ == "__main__":
    main()
