"""
OpenEnv Typed Models — Crypto Trading AI Agent
Pydantic models for Observation, Action, Reward, and State.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Side(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT  = "LIMIT"

class PositionStatus(str, Enum):
    OPEN   = "OPEN"
    CLOSED = "CLOSED"
    NONE   = "NONE"

class TradingMode(str, Enum):
    SPOT    = "SPOT"
    FUTURES = "FUTURES"


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

class Candle(BaseModel):
    timestamp: int           # Unix ms
    open:  float
    high:  float
    low:   float
    close: float
    volume: float

class Indicators(BaseModel):
    """Technical indicator snapshot at current bar."""
    rsi_14:         Optional[float] = None   # 0–100
    macd:           Optional[float] = None
    macd_signal:    Optional[float] = None
    macd_hist:      Optional[float] = None
    bb_upper:       Optional[float] = None
    bb_middle:      Optional[float] = None
    bb_lower:       Optional[float] = None
    ema_9:          Optional[float] = None
    ema_21:         Optional[float] = None
    ema_50:         Optional[float] = None
    atr_14:         Optional[float] = None   # Average True Range
    volume_sma_20:  Optional[float] = None
    adx_14:         Optional[float] = None   # Trend strength

class Position(BaseModel):
    symbol:       str
    side:         Side
    entry_price:  float
    quantity:     float
    stop_loss:    float
    take_profit:  float
    opened_at:    int          # Unix ms
    pnl:          float = 0.0
    status:       PositionStatus = PositionStatus.OPEN

class Trade(BaseModel):
    """Completed (closed) trade record."""
    symbol:       str
    side:         Side
    entry_price:  float
    exit_price:   float
    quantity:     float
    pnl:          float
    pnl_pct:      float
    reason:       str           # "take_profit" | "stop_loss" | "signal" | "manual"
    opened_at:    int
    closed_at:    int
    duration_ms:  int


# ---------------------------------------------------------------------------
# OpenEnv: Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    symbol:             str
    current_price:      float
    candles:            List[Candle]        # last N candles (e.g. 50)
    indicators:         Indicators
    position:           Optional[Position]  # None if flat
    balance_usdt:       float
    equity:             float               # balance + unrealised PnL
    available_actions:  List[str]
    step_count:         int = 0
    task_id:            str = ""
    task_description:   str = ""
    context:            Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# OpenEnv: Action
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """
    action_type:
        buy       — open a long position (market buy)
        sell      — open a short position (futures) or close long (spot)
        close     — close current open position at market
        hold      — do nothing this step
        set_sl    — update stop-loss on current position
        set_tp    — update take-profit on current position
    """
    action_type:  str
    symbol:       Optional[str]  = None
    quantity:     Optional[float] = None    # USDT amount to use (None = use default risk %)
    stop_loss:    Optional[float] = None    # absolute price
    take_profit:  Optional[float] = None    # absolute price
    reason:       Optional[str]  = None     # agent's reasoning (logged)


# ---------------------------------------------------------------------------
# OpenEnv: Reward
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    value:         float = 0.0       # [-1, 1]
    done:          bool  = False
    partial_score: float = 0.0       # cumulative performance [0, 1]
    info:          Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# OpenEnv: EnvironmentState
# ---------------------------------------------------------------------------

class EnvironmentState(BaseModel):
    task_id:           str
    symbol:            str
    trading_mode:      str
    step_count:        int
    max_steps:         int
    balance_usdt:      float
    initial_balance:   float
    position:          Optional[Position] = None
    trade_history:     List[Trade]        = Field(default_factory=list)
    actions_taken:     List[Dict[str, Any]] = Field(default_factory=list)
    candles:           List[Candle]        = Field(default_factory=list)
    indicators:        Optional[Indicators] = None
    episode_done:      bool  = False
    total_reward:      float = 0.0
    win_trades:        int   = 0
    loss_trades:       int   = 0
    grader_data:       Dict[str, Any] = Field(default_factory=dict)
