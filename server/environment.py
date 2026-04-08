"""
CryptoTradingEnv — OpenEnv-compliant Crypto Trading Environment
Implements step() / reset() / state() API.
Connects to Binance (testnet or live) via binance_connector.
"""

from __future__ import annotations

import copy
import os
import time
from typing import Any, Dict, Optional, Tuple

from server.models import (
    Action, Candle, EnvironmentState, Indicators,
    Observation, Position, PositionStatus, Reward, Side, Trade,
)
from server.indicators import compute_indicators
from server.graders import run_grader
import server.binance_connector as binance

# ---------------------------------------------------------------------------
# Task metadata
# ---------------------------------------------------------------------------

TASK_META: Dict[str, Dict[str, Any]] = {
    "task1": {
        "description": (
            "EASY — Spot Scalping: Trade BTC/USDT on 1-minute candles. "
            "Automatically buy when indicators signal oversold (RSI < 30, price below BB lower), "
            "set stop-loss and take-profit, then close position when TP or SL is hit. "
            "Goal: achieve positive PnL with at least 50% win rate."
        ),
        "symbol":    "BTCUSDT",
        "interval":  "1m",
        "max_steps": 60,
        "candle_limit": 60,
        "risk_pct":  0.01,     # 1% of balance per trade
        "sl_pct":    0.005,    # 0.5% stop-loss
        "tp_pct":    0.015,    # 1.5% take-profit (3:1 RRR)
    },
    "task2": {
        "description": (
            "MEDIUM — Swing Trading: Trade ETH/USDT on 15-minute candles. "
            "Use MACD crossovers + EMA trend filter. "
            "Hold positions longer (hours), target 2–5% take-profit, "
            "stop-loss at 1%. Avoid overtrading — quality over quantity."
        ),
        "symbol":    "ETHUSDT",
        "interval":  "15m",
        "max_steps": 100,
        "candle_limit": 100,
        "risk_pct":  0.02,     # 2% of balance per trade
        "sl_pct":    0.01,     # 1% stop-loss
        "tp_pct":    0.03,     # 3% take-profit
    },
    "task3": {
        "description": (
            "HARD — Full AI Trading: Trade BTC/USDT on 5-minute candles. "
            "Use all indicators (RSI, MACD, BB, EMA, ATR, ADX) to make intelligent "
            "buy/sell decisions. Manage risk strictly: always set SL+TP, "
            "size positions dynamically using ATR, avoid drawdowns > 5%. "
            "Target Sharpe ratio > 1.0 over the episode."
        ),
        "symbol":    "BTCUSDT",
        "interval":  "5m",
        "max_steps": 200,
        "candle_limit": 100,
        "risk_pct":  0.015,
        "sl_pct":    0.008,
        "tp_pct":    0.025,
    },
}

VALID_ACTIONS = ["buy", "sell", "close", "hold", "set_sl", "set_tp"]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class CryptoTradingEnv:
    """
    OpenEnv Crypto Trading Environment.

    Connects to Binance (testnet by default).
    Automatically executes SL/TP orders via Binance OCO/conditional orders.
    """

    def __init__(self) -> None:
        self._state: Optional[EnvironmentState] = None
        self._meta:  Optional[Dict[str, Any]]   = None
        self._use_live_binance: bool = True

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "task1") -> Observation:
        """Reset environment for given task. Fetches live data from Binance."""
        if task_id not in TASK_META:
            raise ValueError(f"Unknown task_id: {task_id!r}. Valid: {list(TASK_META)}")

        self._meta = TASK_META[task_id]
        symbol = self._meta["symbol"]

        # Fetch initial candles + balance
        candles_raw = self._fetch_candles(symbol)
        balance     = self._fetch_balance()
        indicators  = compute_indicators(candles_raw)

        candle_objs = [Candle(**c) for c in candles_raw]
        ind_obj     = Indicators(**{k: v for k, v in indicators.items() if v is not None})

        self._state = EnvironmentState(
            task_id=task_id,
            symbol=symbol,
            trading_mode=os.getenv("TRADING_MODE", "SPOT"),
            step_count=0,
            max_steps=self._meta["max_steps"],
            balance_usdt=balance,
            initial_balance=balance,
            position=None,
            trade_history=[],
            actions_taken=[],
            candles=candle_objs,
            indicators=ind_obj,
            episode_done=False,
            total_reward=0.0,
            win_trades=0,
            loss_trades=0,
            grader_data={},
        )

        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.episode_done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._state.step_count += 1

        # Refresh candles + check if open position hit SL/TP
        self._refresh_market_data()
        self._check_sl_tp_hit()

        # Execute agent action
        reward_value, info = self._execute_action(action)

        # Check termination
        done = (
            self._state.step_count >= self._state.max_steps
            or self._state.balance_usdt <= self._state.initial_balance * 0.85  # 15% DD kill
        )

        partial_score = run_grader(self._state.task_id, self._state)

        if done:
            self._state.episode_done = True
            # Close open position on episode end
            if self._state.position:
                self._close_position_internal("episode_end")
            if partial_score >= 0.75:
                reward_value += 0.3
            elif partial_score >= 0.50:
                reward_value += 0.1
            info["final_score"] = partial_score

        # Refresh balance
        self._state.balance_usdt = self._fetch_balance()

        # Compute equity
        equity = self._compute_equity()

        reward_value = max(-1.0, min(1.0, reward_value))
        self._state.total_reward += reward_value

        reward = Reward(
            value=reward_value,
            done=done,
            partial_score=partial_score,
            info=info,
        )

        obs = self._build_observation()
        return obs, reward, done, info

    def state(self) -> EnvironmentState:
        if self._state is None:
            raise RuntimeError("Call reset() before state().")
        return copy.deepcopy(self._state)

    def grade(self) -> float:
        if self._state is None:
            raise RuntimeError("Call reset() before grade().")
        return run_grader(self._state.task_id, self._state)

    # ------------------------------------------------------------------
    # Market data helpers
    # ------------------------------------------------------------------

    def _fetch_candles(self, symbol: str = None) -> list:
        s = symbol or self._state.symbol
        limit = self._meta.get("candle_limit", 100)
        interval = self._meta.get("interval", "1m")
        try:
            return binance.get_klines(s, interval, limit)
        except Exception as e:
            print(f"  [warn] Binance candle fetch failed: {e}")
            return self._mock_candles()

    def _fetch_balance(self) -> float:
        try:
            return binance.get_balance("USDT")
        except Exception as e:
            print(f"  [warn] Binance balance fetch failed: {e}")
            if self._state:
                return self._state.balance_usdt
            return 1000.0   # demo fallback

    def _refresh_market_data(self) -> None:
        candles_raw = self._fetch_candles()
        indicators  = compute_indicators(candles_raw)
        self._state.candles    = [Candle(**c) for c in candles_raw]
        self._state.indicators = Indicators(**{k: v for k, v in indicators.items() if v is not None})

    def _mock_candles(self) -> list:
        """Fallback mock candles when Binance is unreachable."""
        import random
        price = 65000.0
        candles = []
        ts = int(time.time() * 1000) - 3600000
        for _ in range(60):
            o = price
            h = price * (1 + random.uniform(0, 0.003))
            l = price * (1 - random.uniform(0, 0.003))
            c = random.uniform(l, h)
            candles.append({
                "timestamp": ts, "open": o, "high": h,
                "low": l, "close": c, "volume": random.uniform(10, 100),
            })
            price = c
            ts += 60000
        return candles

    def _current_price(self) -> float:
        try:
            return binance.get_price(self._state.symbol)
        except Exception:
            if self._state.candles:
                return self._state.candles[-1].close
            return 65000.0

    def _compute_equity(self) -> float:
        equity = self._state.balance_usdt
        pos = self._state.position
        if pos and pos.status == PositionStatus.OPEN:
            price = self._current_price()
            if pos.side == Side.BUY:
                unrealised = (price - pos.entry_price) * pos.quantity
            else:
                unrealised = (pos.entry_price - price) * pos.quantity
            equity += unrealised
        return equity

    # ------------------------------------------------------------------
    # SL / TP check (simulated on top of Binance orders)
    # ------------------------------------------------------------------

    def _check_sl_tp_hit(self) -> None:
        """Check if current price has triggered SL or TP on open position."""
        pos = self._state.position
        if not pos or pos.status != PositionStatus.OPEN:
            return

        price = self._current_price()

        if pos.side == Side.BUY:
            if price <= pos.stop_loss:
                self._close_position_internal("stop_loss")
                return
            if price >= pos.take_profit:
                self._close_position_internal("take_profit")
                return
        else:  # SHORT
            if price >= pos.stop_loss:
                self._close_position_internal("stop_loss")
                return
            if price <= pos.take_profit:
                self._close_position_internal("take_profit")
                return

    def _close_position_internal(self, reason: str) -> None:
        """Close position and record trade."""
        pos   = self._state.position
        price = self._current_price()

        if pos.side == Side.BUY:
            pnl = (price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - price) * pos.quantity

        pnl_pct = (pnl / (pos.entry_price * pos.quantity)) * 100

        trade = Trade(
            symbol=pos.symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=price,
            quantity=pos.quantity,
            pnl=round(pnl, 4),
            pnl_pct=round(pnl_pct, 4),
            reason=reason,
            opened_at=pos.opened_at,
            closed_at=int(time.time() * 1000),
            duration_ms=int(time.time() * 1000) - pos.opened_at,
        )
        self._state.trade_history.append(trade)
        if pnl > 0:
            self._state.win_trades += 1
        else:
            self._state.loss_trades += 1

        # Update balance
        self._state.balance_usdt += pnl
        self._state.position = None

        # Cancel any remaining SL/TP orders on Binance
        try:
            binance.cancel_all_orders(self._state.symbol)
        except Exception:
            pass

        print(f"  [trade closed] reason={reason} pnl={pnl:+.4f} ({pnl_pct:+.2f}%)")

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def _execute_action(self, action: Action) -> Tuple[float, Dict[str, Any]]:
        self._state.actions_taken.append(action.model_dump())

        if action.action_type == "buy":
            return self._do_buy(action)
        elif action.action_type == "sell":
            return self._do_sell(action)
        elif action.action_type == "close":
            return self._do_close(action)
        elif action.action_type == "hold":
            return 0.0, {"info": "hold"}
        elif action.action_type == "set_sl":
            return self._do_set_sl(action)
        elif action.action_type == "set_tp":
            return self._do_set_tp(action)
        return -0.02, {"error": f"Unknown action_type: {action.action_type!r}"}

    def _do_buy(self, action: Action) -> Tuple[float, Dict[str, Any]]:
        if self._state.position:
            return -0.05, {"error": "Already in a position. Close first."}

        price = self._current_price()
        risk_pct = self._meta["risk_pct"]
        usdt_amt = self._state.balance_usdt * risk_pct * 10   # 10x risk_pct as trade size
        usdt_amt = min(usdt_amt, self._state.balance_usdt * 0.95)

        # Compute SL / TP
        sl = action.stop_loss
        tp = action.take_profit
        if sl is None:
            sl_pct = self._meta["sl_pct"]
            # ATR-based if available
            ind = self._state.indicators
            if ind and ind.atr_14:
                sl = round(price - 2.0 * ind.atr_14, 2)
            else:
                sl = round(price * (1 - sl_pct), 2)
        if tp is None:
            tp_pct = self._meta["tp_pct"]
            if ind and ind.atr_14:
                tp = round(price + 3.0 * ind.atr_14, 2)
            else:
                tp = round(price * (1 + tp_pct), 2)

        # Place order on Binance
        try:
            result = binance.place_market_order(
                symbol=self._state.symbol,
                side="BUY",
                usdt_amount=usdt_amt,
                stop_loss=sl,
                take_profit=tp,
            )
            qty        = result["quantity"]
            fill_price = result["fill_price"]
        except Exception as e:
            # Simulate locally if Binance unavailable
            qty        = round(usdt_amt / price, 6)
            fill_price = price
            print(f"  [warn] Binance order failed, simulating: {e}")

        self._state.position = Position(
            symbol=self._state.symbol,
            side=Side.BUY,
            entry_price=fill_price,
            quantity=qty,
            stop_loss=sl,
            take_profit=tp,
            opened_at=int(time.time() * 1000),
        )
        self._state.balance_usdt -= fill_price * qty

        return 0.02, {
            "action": "buy",
            "symbol": self._state.symbol,
            "price":  fill_price,
            "qty":    qty,
            "sl":     sl,
            "tp":     tp,
        }

    def _do_sell(self, action: Action) -> Tuple[float, Dict[str, Any]]:
        """Sell = short (futures) or close long (spot)."""
        mode = os.getenv("TRADING_MODE", "SPOT")
        if mode == "SPOT":
            # On spot, sell means close a long
            return self._do_close(action)

        if self._state.position:
            return -0.05, {"error": "Already in a position. Close first."}

        price    = self._current_price()
        risk_pct = self._meta["risk_pct"]
        usdt_amt = self._state.balance_usdt * risk_pct * 10
        sl = action.stop_loss or round(price * (1 + self._meta["sl_pct"]), 2)
        tp = action.take_profit or round(price * (1 - self._meta["tp_pct"]), 2)

        try:
            result = binance.place_market_order(
                symbol=self._state.symbol,
                side="SELL",
                usdt_amount=usdt_amt,
                stop_loss=sl,
                take_profit=tp,
            )
            qty        = result["quantity"]
            fill_price = result["fill_price"]
        except Exception as e:
            qty        = round(usdt_amt / price, 6)
            fill_price = price
            print(f"  [warn] Binance short failed, simulating: {e}")

        self._state.position = Position(
            symbol=self._state.symbol,
            side=Side.SELL,
            entry_price=fill_price,
            quantity=qty,
            stop_loss=sl,
            take_profit=tp,
            opened_at=int(time.time() * 1000),
        )
        return 0.02, {"action": "sell/short", "price": fill_price, "qty": qty, "sl": sl, "tp": tp}

    def _do_close(self, action: Action) -> Tuple[float, Dict[str, Any]]:
        if not self._state.position:
            return -0.05, {"error": "No open position to close."}
        self._close_position_internal(action.reason or "signal")
        return 0.01, {"action": "close"}

    def _do_set_sl(self, action: Action) -> Tuple[float, Dict[str, Any]]:
        if not self._state.position:
            return -0.02, {"error": "No open position."}
        if not action.stop_loss:
            return -0.02, {"error": "set_sl requires stop_loss value."}
        self._state.position.stop_loss = action.stop_loss
        return 0.01, {"action": "set_sl", "new_sl": action.stop_loss}

    def _do_set_tp(self, action: Action) -> Tuple[float, Dict[str, Any]]:
        if not self._state.position:
            return -0.02, {"error": "No open position."}
        if not action.take_profit:
            return -0.02, {"error": "set_tp requires take_profit value."}
        self._state.position.take_profit = action.take_profit
        return 0.01, {"action": "set_tp", "new_tp": action.take_profit}

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        equity = self._compute_equity()
        last50 = self._state.candles[-50:] if len(self._state.candles) >= 50 else self._state.candles
        return Observation(
            symbol=self._state.symbol,
            current_price=last50[-1].close if last50 else 0.0,
            candles=last50,
            indicators=self._state.indicators or Indicators(),
            position=self._state.position,
            balance_usdt=self._state.balance_usdt,
            equity=equity,
            available_actions=VALID_ACTIONS,
            step_count=self._state.step_count,
            task_id=self._state.task_id,
            task_description=self._meta["description"],
            context={
                "remaining_steps":  self._state.max_steps - self._state.step_count,
                "total_trades":     len(self._state.trade_history),
                "win_trades":       self._state.win_trades,
                "loss_trades":      self._state.loss_trades,
                "total_pnl_usdt":   round(self._state.balance_usdt - self._state.initial_balance, 4),
                "pnl_pct":          round((self._state.balance_usdt / self._state.initial_balance - 1) * 100, 2),
            },
        )
