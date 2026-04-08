"""
Binance Connector
=================
Wraps the Binance REST API for both:
  - Spot Testnet  : https://testnet.binance.vision
  - Futures Testnet: https://testnet.binancefuture.com
  - Live Spot     : https://api.binance.com
  - Live Futures  : https://fapi.binance.com

Set environment variables:
    BINANCE_API_KEY     Your Binance (testnet or live) API key
    BINANCE_API_SECRET  Your Binance API secret
    BINANCE_TESTNET     "true" (default) | "false"
    TRADING_MODE        "SPOT" | "FUTURES"
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
import urllib.parse
from typing import Any, Dict, List, Optional

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_KEY    = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")
TESTNET    = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
MODE       = os.getenv("TRADING_MODE", "SPOT").upper()

if TESTNET:
    SPOT_BASE    = "https://testnet.binance.vision"
    FUTURES_BASE = "https://testnet.binancefuture.com"
else:
    SPOT_BASE    = "https://api.binance.com"
    FUTURES_BASE = "https://fapi.binance.com"

BASE_URL = FUTURES_BASE if MODE == "FUTURES" else SPOT_BASE

TIMEOUT = 10


# ---------------------------------------------------------------------------
# Signature helpers
# ---------------------------------------------------------------------------

def _sign(params: Dict[str, Any]) -> Dict[str, Any]:
    query = urllib.parse.urlencode(params)
    sig = hmac.new(
        API_SECRET.encode("utf-8"),
        query.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    params["signature"] = sig
    return params


def _headers() -> Dict[str, str]:
    return {"X-MBX-APIKEY": API_KEY}


def _get(path: str, params: Dict[str, Any] = None, signed: bool = False) -> Any:
    if params is None:
        params = {}
    if signed:
        params["timestamp"] = int(time.time() * 1000)
        params = _sign(params)
    resp = requests.get(BASE_URL + path, params=params, headers=_headers(), timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _post(path: str, params: Dict[str, Any]) -> Any:
    params["timestamp"] = int(time.time() * 1000)
    params = _sign(params)
    resp = requests.post(
        BASE_URL + path,
        params=params,
        headers=_headers(),
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def _delete(path: str, params: Dict[str, Any]) -> Any:
    params["timestamp"] = int(time.time() * 1000)
    params = _sign(params)
    resp = requests.delete(
        BASE_URL + path,
        params=params,
        headers=_headers(),
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Market Data  (no auth needed)
# ---------------------------------------------------------------------------

def get_price(symbol: str) -> float:
    """Get current best price for a symbol."""
    data = _get("/api/v3/ticker/price", {"symbol": symbol})
    return float(data["price"])


def get_klines(symbol: str, interval: str = "1m", limit: int = 100) -> List[Dict]:
    """
    Returns list of OHLCV dicts.
    interval: 1m, 3m, 5m, 15m, 30m, 1h, 4h, 1d
    """
    if MODE == "FUTURES":
        path = "/fapi/v1/klines"
    else:
        path = "/api/v3/klines"

    raw = _get(path, {"symbol": symbol, "interval": interval, "limit": limit})
    candles = []
    for c in raw:
        candles.append({
            "timestamp": int(c[0]),
            "open":      float(c[1]),
            "high":      float(c[2]),
            "low":       float(c[3]),
            "close":     float(c[4]),
            "volume":    float(c[5]),
        })
    return candles


def get_orderbook(symbol: str, limit: int = 5) -> Dict:
    data = _get("/api/v3/depth", {"symbol": symbol, "limit": limit})
    return {
        "bids": [[float(p), float(q)] for p, q in data["bids"]],
        "asks": [[float(p), float(q)] for p, q in data["asks"]],
    }


def get_exchange_info(symbol: str) -> Dict:
    data = _get("/api/v3/exchangeInfo", {"symbol": symbol})
    for s in data["symbols"]:
        if s["symbol"] == symbol:
            return s
    raise ValueError(f"Symbol {symbol} not found")


def get_lot_size(symbol: str) -> Dict[str, float]:
    """Returns minQty, maxQty, stepSize for the symbol."""
    info = get_exchange_info(symbol)
    for f in info["filters"]:
        if f["filterType"] == "LOT_SIZE":
            return {
                "minQty":  float(f["minQty"]),
                "maxQty":  float(f["maxQty"]),
                "stepSize": float(f["stepSize"]),
            }
    return {"minQty": 0.001, "maxQty": 9999, "stepSize": 0.001}


# ---------------------------------------------------------------------------
# Account  (auth required)
# ---------------------------------------------------------------------------

def get_account() -> Dict:
    if MODE == "FUTURES":
        return _get("/fapi/v2/account", signed=True)
    return _get("/api/v3/account", signed=True)


def get_balance(asset: str = "USDT") -> float:
    acc = get_account()
    if MODE == "FUTURES":
        for b in acc.get("assets", []):
            if b["asset"] == asset:
                return float(b["availableBalance"])
    else:
        for b in acc.get("balances", []):
            if b["asset"] == asset:
                return float(b["free"])
    return 0.0


def get_open_orders(symbol: str) -> List[Dict]:
    if MODE == "FUTURES":
        return _get("/fapi/v1/openOrders", {"symbol": symbol}, signed=True)
    return _get("/api/v3/openOrders", {"symbol": symbol}, signed=True)


def get_open_positions(symbol: str) -> Optional[Dict]:
    """Futures only — returns current position for symbol."""
    if MODE != "FUTURES":
        return None
    data = _get("/fapi/v2/positionRisk", {"symbol": symbol}, signed=True)
    for p in data:
        if p["symbol"] == symbol and float(p["positionAmt"]) != 0:
            return p
    return None


# ---------------------------------------------------------------------------
# Order placement
# ---------------------------------------------------------------------------

def _round_qty(qty: float, step_size: float) -> float:
    """Round quantity to valid step size."""
    import math
    precision = max(0, round(-math.log10(step_size)))
    return round(round(qty / step_size) * step_size, precision)


def place_market_order(
    symbol: str,
    side: str,          # "BUY" | "SELL"
    usdt_amount: float,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
) -> Dict:
    """
    Places a market order for `usdt_amount` USDT worth of `symbol`.
    Optionally places OCO stop-loss + take-profit orders after fill.
    Returns order response dict.
    """
    price = get_price(symbol)
    lot   = get_lot_size(symbol)
    qty   = _round_qty(usdt_amount / price, lot["stepSize"])
    qty   = max(qty, lot["minQty"])

    path = "/fapi/v1/order" if MODE == "FUTURES" else "/api/v3/order"

    params: Dict[str, Any] = {
        "symbol":   symbol,
        "side":     side,
        "type":     "MARKET",
        "quantity": qty,
    }
    if MODE == "FUTURES":
        params["positionSide"] = "LONG" if side == "BUY" else "SHORT"

    order = _post(path, params)

    # Place stop-loss / take-profit if requested
    sl_order = tp_order = None
    if stop_loss or take_profit:
        sl_order, tp_order = place_sl_tp(symbol, side, qty, stop_loss, take_profit)

    return {
        "order":    order,
        "sl_order": sl_order,
        "tp_order": tp_order,
        "fill_price": price,
        "quantity": qty,
    }


def place_sl_tp(
    symbol: str,
    entry_side: str,    # side of original order
    qty: float,
    stop_loss: Optional[float],
    take_profit: Optional[float],
) -> tuple:
    """
    Place stop-loss and take-profit orders.
    For spot: uses OCO order.
    For futures: uses STOP_MARKET + TAKE_PROFIT_MARKET.
    """
    exit_side = "SELL" if entry_side == "BUY" else "BUY"
    sl_order = tp_order = None

    if MODE == "FUTURES":
        path = "/fapi/v1/order"
        if stop_loss:
            sl_order = _post(path, {
                "symbol":       symbol,
                "side":         exit_side,
                "type":         "STOP_MARKET",
                "stopPrice":    round(stop_loss, 2),
                "quantity":     qty,
                "reduceOnly":   "true",
            })
        if take_profit:
            tp_order = _post(path, {
                "symbol":       symbol,
                "side":         exit_side,
                "type":         "TAKE_PROFIT_MARKET",
                "stopPrice":    round(take_profit, 2),
                "quantity":     qty,
                "reduceOnly":   "true",
            })
    else:
        # Spot — OCO order requires both SL and TP
        if stop_loss and take_profit:
            limit_price = round(take_profit, 2)
            stop_price  = round(stop_loss * 1.001, 2)   # slightly above SL
            stop_limit  = round(stop_loss, 2)
            oco = _post("/api/v3/order/oco", {
                "symbol":             symbol,
                "side":               exit_side,
                "quantity":           qty,
                "price":              limit_price,
                "stopPrice":          stop_price,
                "stopLimitPrice":     stop_limit,
                "stopLimitTimeInForce": "GTC",
            })
            sl_order = tp_order = oco
        elif stop_loss:
            sl_order = _post("/api/v3/order", {
                "symbol":       symbol,
                "side":         exit_side,
                "type":         "STOP_LOSS_LIMIT",
                "quantity":     qty,
                "price":        round(stop_loss, 2),
                "stopPrice":    round(stop_loss * 1.001, 2),
                "timeInForce":  "GTC",
            })
        elif take_profit:
            tp_order = _post("/api/v3/order", {
                "symbol":      symbol,
                "side":        exit_side,
                "type":        "LIMIT",
                "quantity":    qty,
                "price":       round(take_profit, 2),
                "timeInForce": "GTC",
            })

    return sl_order, tp_order


def cancel_all_orders(symbol: str) -> List[Dict]:
    """Cancel all open orders for a symbol."""
    open_orders = get_open_orders(symbol)
    cancelled = []
    for o in open_orders:
        path = "/fapi/v1/order" if MODE == "FUTURES" else "/api/v3/order"
        try:
            r = _delete(path, {"symbol": symbol, "orderId": o["orderId"]})
            cancelled.append(r)
        except Exception as e:
            print(f"  [warn] Could not cancel order {o['orderId']}: {e}")
    return cancelled


def close_position_market(symbol: str, qty: float, side: str) -> Dict:
    """Close an open position at market price."""
    exit_side = "SELL" if side == "BUY" else "BUY"
    path = "/fapi/v1/order" if MODE == "FUTURES" else "/api/v3/order"
    params: Dict[str, Any] = {
        "symbol":   symbol,
        "side":     exit_side,
        "type":     "MARKET",
        "quantity": qty,
    }
    if MODE == "FUTURES":
        params["reduceOnly"] = "true"
    return _post(path, params)
