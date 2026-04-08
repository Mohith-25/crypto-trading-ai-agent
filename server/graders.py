"""
Trading Task Graders
====================
Deterministic graders that score [0.0, 1.0] based on trading performance.
"""

from __future__ import annotations
import math
from typing import Dict
from server.models import EnvironmentState


# ---------------------------------------------------------------------------
# Task 1 — Spot Scalping (Easy)
# Score based on: win rate, positive PnL, risk discipline
# ---------------------------------------------------------------------------

def grade_task1(state: EnvironmentState) -> float:
    """
    Easy: Score the agent on basic spot scalping.
    - 40%: PnL ratio (equity vs initial)
    - 40%: Win rate of closed trades
    - 20%: Risk discipline (SL always set)
    """
    trades = state.trade_history
    eq_ratio  = state.balance_usdt / state.initial_balance   # > 1 = profit

    # A — Profitability (0.4)
    pnl_score = min(1.0, max(0.0, (eq_ratio - 0.9) / 0.2))   # 0% = 0.5, +10% = 1.0, -10% = 0.0

    # B — Win rate (0.4)
    if trades:
        win_rate  = sum(1 for t in trades if t.pnl > 0) / len(trades)
    else:
        win_rate  = 0.0
    win_score = win_rate

    # C — Risk discipline: SL always placed (0.2)
    actions = state.actions_taken
    buy_actions  = [a for a in actions if a.get("action_type") == "buy"]
    sl_set       = [a for a in buy_actions if a.get("stop_loss") is not None]
    risk_score   = (len(sl_set) / len(buy_actions)) if buy_actions else 0.0

    total = 0.4 * pnl_score + 0.4 * win_score + 0.2 * risk_score
    return round(total, 4)


# ---------------------------------------------------------------------------
# Task 2 — Swing Trading (Medium)
# Score based on: Sharpe-like ratio, max drawdown avoidance, trade quality
# ---------------------------------------------------------------------------

def grade_task2(state: EnvironmentState) -> float:
    """
    Medium: Score swing trading performance.
    - 35%: Net PnL
    - 35%: Avoid large drawdown (max single-loss < 3% of equity)
    - 30%: TP/SL ratio (avg TP hit rate vs SL hit rate)
    """
    trades = state.trade_history

    # A — Net PnL (0.35)
    eq_ratio  = state.balance_usdt / state.initial_balance
    pnl_score = min(1.0, max(0.0, (eq_ratio - 0.85) / 0.3))  # +15% = 1.0, -15% = 0.0

    # B — Drawdown control (0.35)
    if trades:
        max_loss_pct = max((abs(t.pnl_pct) for t in trades if t.pnl < 0), default=0)
        # Penalise if any single trade lost > 3% of entry
        dd_score = 1.0 if max_loss_pct <= 3.0 else max(0.0, 1.0 - (max_loss_pct - 3.0) / 5.0)
    else:
        dd_score = 0.5  # neutral if no trades

    # C — TP vs SL ratio (0.30)
    tp_hits = sum(1 for t in trades if t.reason == "take_profit")
    sl_hits = sum(1 for t in trades if t.reason == "stop_loss")
    total_closed = tp_hits + sl_hits
    if total_closed > 0:
        ratio_score = tp_hits / total_closed
    else:
        ratio_score = 0.5

    total = 0.35 * pnl_score + 0.35 * dd_score + 0.30 * ratio_score
    return round(total, 4)


# ---------------------------------------------------------------------------
# Task 3 — Full AI Trading (Hard)
# Score on: PnL, Sharpe ratio, drawdown, win streak, risk management
# ---------------------------------------------------------------------------

def grade_task3(state: EnvironmentState) -> float:
    """
    Hard: Comprehensive trading score.
    - 30%: Net PnL
    - 25%: Sharpe-like ratio (PnL consistency)
    - 25%: Drawdown control
    - 20%: Trade discipline (SL+TP always set, no overtrading)
    """
    trades  = state.trade_history
    actions = state.actions_taken

    # A — Net PnL (0.30)
    eq_ratio  = state.balance_usdt / state.initial_balance
    pnl_score = min(1.0, max(0.0, (eq_ratio - 0.80) / 0.40))  # +20% = 1.0

    # B — Sharpe-like (PnL consistency) (0.25)
    if len(trades) >= 3:
        pnls  = [t.pnl_pct for t in trades]
        mean  = sum(pnls) / len(pnls)
        std   = math.sqrt(sum((p - mean) ** 2 for p in pnls) / len(pnls)) + 1e-9
        sharpe = mean / std
        # Normalise: sharpe >= 1.0 → score 1.0; negative → 0.0
        sharpe_score = min(1.0, max(0.0, (sharpe + 0.5) / 1.5))
    else:
        sharpe_score = 0.5

    # C — Drawdown control (0.25)
    if trades:
        cum_pnl   = 0.0
        peak      = 0.0
        max_dd    = 0.0
        for t in trades:
            cum_pnl += t.pnl_pct
            peak     = max(peak, cum_pnl)
            dd       = peak - cum_pnl
            max_dd   = max(max_dd, dd)
        dd_score = max(0.0, 1.0 - max_dd / 10.0)   # 10% DD → 0.0
    else:
        dd_score = 0.5

    # D — Trade discipline (0.20)
    buy_actions = [a for a in actions if a.get("action_type") in ("buy", "sell")]
    with_sl     = [a for a in buy_actions if a.get("stop_loss")   is not None]
    with_tp     = [a for a in buy_actions if a.get("take_profit") is not None]
    if buy_actions:
        sl_ratio   = len(with_sl) / len(buy_actions)
        tp_ratio   = len(with_tp) / len(buy_actions)
        disc_score = (sl_ratio + tp_ratio) / 2.0
    else:
        disc_score = 0.5

    total = 0.30 * pnl_score + 0.25 * sharpe_score + 0.25 * dd_score + 0.20 * disc_score
    return round(total, 4)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

GRADERS: Dict[str, callable] = {
    "task1": grade_task1,
    "task2": grade_task2,
    "task3": grade_task3,
}


def run_grader(task_id: str, state: EnvironmentState) -> float:
    grader = GRADERS.get(task_id)
    if grader is None:
        raise ValueError(f"Unknown task_id: {task_id!r}. Valid: {list(GRADERS)}")
    return grader(state)
