"""
Technical Indicators
====================
Pure-Python implementations (no TA-Lib dependency).
All functions accept a list of floats (prices/volumes) and return a float.
"""

from __future__ import annotations
from typing import List, Optional
import math


def _ema(values: List[float], period: int) -> List[float]:
    if len(values) < period:
        return [float("nan")] * len(values)
    k = 2.0 / (period + 1)
    result = [float("nan")] * (period - 1)
    result.append(sum(values[:period]) / period)  # seed with SMA
    for v in values[period:]:
        result.append(v * k + result[-1] * (1 - k))
    return result


def _sma(values: List[float], period: int) -> List[float]:
    result = [float("nan")] * (period - 1)
    for i in range(period, len(values) + 1):
        result.append(sum(values[i - period:i]) / period)
    return result


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def rsi(closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    # Wilder smoothing
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

def macd(
    closes: List[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Returns (macd_line, signal_line, histogram)."""
    if len(closes) < slow + signal:
        return None, None, None
    ema_fast   = _ema(closes, fast)
    ema_slow   = _ema(closes, slow)
    macd_line  = [
        (f - s) if not (math.isnan(f) or math.isnan(s)) else float("nan")
        for f, s in zip(ema_fast, ema_slow)
    ]
    valid_macd = [v for v in macd_line if not math.isnan(v)]
    if len(valid_macd) < signal:
        return None, None, None
    sig_line   = _ema(valid_macd, signal)
    sig_val    = sig_line[-1]
    macd_val   = valid_macd[-1]
    hist_val   = macd_val - sig_val
    return round(macd_val, 6), round(sig_val, 6), round(hist_val, 6)


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

def bollinger_bands(
    closes: List[float],
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Returns (upper, middle, lower)."""
    if len(closes) < period:
        return None, None, None
    window = closes[-period:]
    mid    = sum(window) / period
    std    = math.sqrt(sum((x - mid) ** 2 for x in window) / period)
    upper  = mid + std_dev * std
    lower  = mid - std_dev * std
    return round(upper, 4), round(mid, 4), round(lower, 4)


# ---------------------------------------------------------------------------
# EMA (single value)
# ---------------------------------------------------------------------------

def ema_value(closes: List[float], period: int) -> Optional[float]:
    vals = _ema(closes, period)
    v = vals[-1] if vals else None
    if v is None or math.isnan(v):
        return None
    return round(v, 4)


# ---------------------------------------------------------------------------
# ATR (Average True Range)
# ---------------------------------------------------------------------------

def atr(
    highs: List[float],
    lows:  List[float],
    closes: List[float],
    period: int = 14,
) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    tr_list = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i]  - closes[i - 1]),
        )
        tr_list.append(tr)
    if len(tr_list) < period:
        return None
    # Wilder smoothing
    atr_val = sum(tr_list[:period]) / period
    for tr in tr_list[period:]:
        atr_val = (atr_val * (period - 1) + tr) / period
    return round(atr_val, 6)


# ---------------------------------------------------------------------------
# Volume SMA
# ---------------------------------------------------------------------------

def volume_sma(volumes: List[float], period: int = 20) -> Optional[float]:
    if len(volumes) < period:
        return None
    return round(sum(volumes[-period:]) / period, 2)


# ---------------------------------------------------------------------------
# ADX (Average Directional Index)
# ---------------------------------------------------------------------------

def adx(
    highs:  List[float],
    lows:   List[float],
    closes: List[float],
    period: int = 14,
) -> Optional[float]:
    if len(closes) < period * 2:
        return None
    dm_plus, dm_minus, tr_list = [], [], []
    for i in range(1, len(closes)):
        up   = highs[i]  - highs[i - 1]
        down = lows[i - 1] - lows[i]
        dm_plus.append(up   if up > down and up > 0   else 0)
        dm_minus.append(down if down > up and down > 0 else 0)
        tr_list.append(max(
            highs[i] - lows[i],
            abs(highs[i]  - closes[i - 1]),
            abs(lows[i]   - closes[i - 1]),
        ))
    def wilder(lst, p):
        val = sum(lst[:p]) / p
        result = [val]
        for v in lst[p:]:
            val = (val * (p - 1) + v) / p
            result.append(val)
        return result
    atr_w   = wilder(tr_list,   period)
    dmp_w   = wilder(dm_plus,   period)
    dmm_w   = wilder(dm_minus,  period)
    dx_list = []
    for a, p, m in zip(atr_w, dmp_w, dmm_w):
        if a == 0:
            dx_list.append(0)
        else:
            di_plus  = 100 * p / a
            di_minus = 100 * m / a
            dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus + 1e-9)
            dx_list.append(dx)
    if len(dx_list) < period:
        return None
    adx_val = sum(dx_list[:period]) / period
    for dx in dx_list[period:]:
        adx_val = (adx_val * (period - 1) + dx) / period
    return round(adx_val, 2)


# ---------------------------------------------------------------------------
# Composite: compute all indicators from candle lists
# ---------------------------------------------------------------------------

def compute_indicators(candles: list) -> dict:
    """
    candles: list of dicts with keys open/high/low/close/volume
    Returns a flat dict of all indicator values.
    """
    if len(candles) < 2:
        return {}

    closes  = [c["close"]  for c in candles]
    highs   = [c["high"]   for c in candles]
    lows    = [c["low"]    for c in candles]
    volumes = [c["volume"] for c in candles]

    macd_v, macd_sig, macd_h = macd(closes)
    bb_u, bb_m, bb_l          = bollinger_bands(closes)

    return {
        "rsi_14":        rsi(closes),
        "macd":          macd_v,
        "macd_signal":   macd_sig,
        "macd_hist":     macd_h,
        "bb_upper":      bb_u,
        "bb_middle":     bb_m,
        "bb_lower":      bb_l,
        "ema_9":         ema_value(closes, 9),
        "ema_21":        ema_value(closes, 21),
        "ema_50":        ema_value(closes, 50),
        "atr_14":        atr(highs, lows, closes),
        "volume_sma_20": volume_sma(volumes),
        "adx_14":        adx(highs, lows, closes),
    }
