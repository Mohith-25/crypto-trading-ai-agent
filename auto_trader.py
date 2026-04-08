"""
auto_trader.py — Standalone Continuous AI Trader
=================================================
Runs DIRECTLY against Binance (no OpenEnv server needed).
This is the "always-on" version for real/testnet trading.

Usage:
    export BINANCE_API_KEY="your_key"
    export BINANCE_API_SECRET="your_secret"
    export BINANCE_TESTNET="true"         # "false" for live
    export TRADING_MODE="SPOT"            # or FUTURES
    export API_BASE_URL="https://router.huggingface.co/v1"
    export MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
    export HF_TOKEN="hf_your_token"
    export SYMBOL="BTCUSDT"
    export INTERVAL="5m"
    export RISK_PCT="0.02"               # 2% balance per trade
    export SL_PCT="0.008"                # 0.8% stop-loss
    export TP_PCT="0.024"                # 2.4% take-profit (3:1 RRR)
    export MAX_TRADES="20"               # stop after N trades
    export CANDLE_LIMIT="100"

    python auto_trader.py
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYMBOL        = os.getenv("SYMBOL",       "BTCUSDT")
INTERVAL      = os.getenv("INTERVAL",     "5m")
RISK_PCT      = float(os.getenv("RISK_PCT",  "0.02"))
SL_PCT        = float(os.getenv("SL_PCT",   "0.008"))
TP_PCT        = float(os.getenv("TP_PCT",   "0.024"))
MAX_TRADES    = int(os.getenv("MAX_TRADES", "20"))
CANDLE_LIMIT  = int(os.getenv("CANDLE_LIMIT", "100"))
SLEEP_SEC     = int(os.getenv("SLEEP_SEC",   "60"))   # seconds between checks

API_BASE_URL  = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME    = os.getenv("MODEL_NAME",   "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN      = os.getenv("HF_TOKEN",     "")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")

# Import connector from server package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server.binance_connector as binance
from server.indicators import compute_indicators

# ---------------------------------------------------------------------------
# State tracking
# ---------------------------------------------------------------------------

class TraderState:
    def __init__(self):
        self.position:     Optional[Dict] = None
        self.trade_log:    List[Dict]     = []
        self.total_pnl:    float          = 0.0
        self.win_trades:   int            = 0
        self.loss_trades:  int            = 0
        self.running:      bool           = True
        self.initial_balance: float       = 0.0

state = TraderState()


def handle_signal(sig, frame):
    print("\n\n  [signal] Shutting down gracefully...")
    state.running = False

signal.signal(signal.SIGINT,  handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


# ---------------------------------------------------------------------------
# Market + indicator helpers
# ---------------------------------------------------------------------------

def get_market_data() -> Dict[str, Any]:
    candles    = binance.get_klines(SYMBOL, INTERVAL, CANDLE_LIMIT)
    price      = float(candles[-1]["close"])
    indicators = compute_indicators(candles)
    balance    = binance.get_balance("USDT")
    return {
        "price":      price,
        "candles":    candles[-10:],   # last 10 for prompt
        "indicators": indicators,
        "balance":    balance,
    }


# ---------------------------------------------------------------------------
# Position monitoring
# ---------------------------------------------------------------------------

def check_sl_tp(market: Dict[str, Any]) -> Optional[str]:
    """Returns 'stop_loss' | 'take_profit' | None."""
    pos   = state.position
    price = market["price"]
    if not pos:
        return None

    if pos["side"] == "BUY":
        if price <= pos["stop_loss"]:
            return "stop_loss"
        if price >= pos["take_profit"]:
            return "take_profit"
    else:   # SHORT
        if price >= pos["stop_loss"]:
            return "stop_loss"
        if price <= pos["take_profit"]:
            return "take_profit"
    return None


def close_position(reason: str, market: Dict[str, Any]) -> None:
    pos   = state.position
    price = market["price"]
    side  = pos["side"]

    print(f"\n  {'🔴' if reason == 'stop_loss' else '🟢'} Closing position — {reason.upper()}")
    print(f"     Entry: {pos['entry_price']:.2f}  |  Exit: {price:.2f}")

    try:
        binance.cancel_all_orders(SYMBOL)
        binance.close_position_market(SYMBOL, pos["quantity"], side)
    except Exception as e:
        print(f"  [warn] Close order failed: {e}")

    if side == "BUY":
        pnl = (price - pos["entry_price"]) * pos["quantity"]
    else:
        pnl = (pos["entry_price"] - price) * pos["quantity"]

    pnl_pct = (pnl / (pos["entry_price"] * pos["quantity"])) * 100

    print(f"     PnL: {pnl:+.4f} USDT ({pnl_pct:+.2f}%)")

    state.trade_log.append({
        "symbol":      SYMBOL,
        "side":        side,
        "entry":       pos["entry_price"],
        "exit":        price,
        "qty":         pos["quantity"],
        "pnl":         round(pnl, 4),
        "pnl_pct":     round(pnl_pct, 4),
        "reason":      reason,
        "opened_at":   pos["opened_at"],
        "closed_at":   int(time.time()),
    })

    state.total_pnl += pnl
    if pnl > 0:
        state.win_trades += 1
    else:
        state.loss_trades += 1

    state.position = None
    print_summary()


# ---------------------------------------------------------------------------
# LLM decision
# ---------------------------------------------------------------------------

SYSTEM = """You are an expert crypto trading AI. Analyze market data and indicators.
Respond ONLY with valid JSON. No prose.
Format:
{
  "decision": "BUY" | "SELL" | "HOLD",
  "stop_loss_pct": <float 0.3–0.8>,
  "take_profit_pct": <float 0.5–2.0>,
  "confidence": <int 1-10>,
  "reason": "<explain confluence>"
}
Rules:
- MANDATORY CONFLUENCE: You may ONLY open a trade (confidence >= 8) if AT LEAST TWO indicators agree (e.g. MACD bullish AND RSI oversold).
- If indicators are mixed or market is chopping sideways, set confidence to 0 and HOLD.
- Only BUY/SELL when confidence >= 8
- Very strict risk management: tight stop loss, quick take profit."""


def ask_llm(market: Dict[str, Any]) -> Dict[str, Any]:
    ind   = market["indicators"]
    price = market["price"]
    bal   = market["balance"]

    e9  = ind.get("ema_9",  0) or 0
    e21 = ind.get("ema_21", 0) or 0
    trend = "UPTREND ▲" if e9 > e21 else "DOWNTREND ▼" if e21 > e9 else "NEUTRAL"

    pos_str = "FLAT"
    if state.position:
        p = state.position
        pos_str = (f"{p['side']} @ {p['entry_price']:.2f}  "
                   f"SL:{p['stop_loss']:.2f}  TP:{p['take_profit']:.2f}")

    prompt = f"""Symbol: {SYMBOL}  |  Price: {price:.2f}  |  Trend: {trend}
Position: {pos_str}
Balance: {bal:.4f} USDT  |  Trades: {len(state.trade_log)} ({state.win_trades}W/{state.loss_trades}L)  |  PnL: {state.total_pnl:+.4f}

INDICATORS:
  RSI(14):    {ind.get('rsi_14', 'n/a')}
  MACD:       {ind.get('macd', 'n/a')}  Signal: {ind.get('macd_signal', 'n/a')}  Hist: {ind.get('macd_hist', 'n/a')}
  BB:         {ind.get('bb_lower', 'n/a')} | {ind.get('bb_middle', 'n/a')} | {ind.get('bb_upper', 'n/a')}
  EMA9/21/50: {ind.get('ema_9', 'n/a')} / {ind.get('ema_21', 'n/a')} / {ind.get('ema_50', 'n/a')}
  ATR(14):    {ind.get('atr_14', 'n/a')}
  ADX(14):    {ind.get('adx_14', 'n/a')}

What is your trading decision?"""

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.1,
            max_tokens=300,
        )
        raw = resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [LLM error] {e}")
        return {"decision": "HOLD", "stop_loss_pct": SL_PCT * 100, "take_profit_pct": TP_PCT * 100,
                "confidence": 0, "reason": "LLM error"}

    # Parse
    raw = raw.strip()
    if raw.startswith("```"):
        raw = "\n".join(l for l in raw.split("\n") if not l.startswith("```"))
    try:
        return json.loads(raw)
    except Exception:
        import re
        m = re.search(r"\{.*?\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {"decision": "HOLD", "stop_loss_pct": 1.0, "take_profit_pct": 2.0, "confidence": 0, "reason": "parse error"}


# ---------------------------------------------------------------------------
# Open position
# ---------------------------------------------------------------------------

def open_position(decision: Dict, market: Dict) -> None:
    side  = decision["decision"]          # "BUY" | "SELL"
    price = market["price"]
    bal   = market["balance"]

    sl_pct = float(decision.get("stop_loss_pct",  SL_PCT * 100)) / 100
    tp_pct = float(decision.get("take_profit_pct", TP_PCT * 100)) / 100

    if side == "BUY":
        sl = round(price * (1 - sl_pct), 2)
        tp = round(price * (1 + tp_pct), 2)
    else:  # SELL / SHORT
        sl = round(price * (1 + sl_pct), 2)
        tp = round(price * (1 - tp_pct), 2)

    # ATR-based sizing if available
    atr = market["indicators"].get("atr_14")
    if atr:
        risk_amount = bal * RISK_PCT
        risk_per_unit = abs(price - sl)
        qty = round(risk_amount / risk_per_unit, 6) if risk_per_unit > 0 else round(bal * RISK_PCT / price, 6)
        usdt_amt = qty * price
    else:
        usdt_amt = bal * RISK_PCT * 10
        qty = round(usdt_amt / price, 6)

    usdt_amt = min(usdt_amt, bal * 0.95)

    print(f"\n  {'🟢' if side == 'BUY' else '🔴'} Opening {side} @ {price:.2f}")
    print(f"     Qty: {qty:.6f} | USDT: {usdt_amt:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")
    print(f"     Reason: {decision.get('reason', '-')}")

    try:
        result = binance.place_market_order(
            symbol=SYMBOL,
            side=side,
            usdt_amount=usdt_amt,
            stop_loss=sl,
            take_profit=tp,
        )
        fill = result.get("fill_price", price)
        qty  = result.get("quantity",   qty)
    except Exception as e:
        print(f"  [warn] Binance order error: {e}. Simulating locally.")
        fill = price

    state.position = {
        "side":        side,
        "entry_price": fill,
        "quantity":    qty,
        "stop_loss":   sl,
        "take_profit": tp,
        "opened_at":   int(time.time()),
    }


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def print_summary() -> None:
    total = state.win_trades + state.loss_trades
    wr    = (state.win_trades / total * 100) if total > 0 else 0
    print(f"\n  ┌─ SUMMARY ─────────────────────────────────────────┐")
    print(f"  │ Trades:    {total:3d}  ({state.win_trades}W / {state.loss_trades}L)  WinRate: {wr:.1f}%      │")
    print(f"  │ Total PnL: {state.total_pnl:+.4f} USDT                         │")
    print(f"  └───────────────────────────────────────────────────┘")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "="*60)
    print("  Crypto AI Auto-Trader — Starting")
    print("="*60)
    print(f"  Symbol:    {SYMBOL}")
    print(f"  Interval:  {INTERVAL}")
    print(f"  Risk/Trade: {RISK_PCT*100:.1f}% balance")
    print(f"  SL:        {SL_PCT*100:.2f}%  |  TP: {TP_PCT*100:.2f}%")
    print(f"  Max Trades: {MAX_TRADES}")
    print(f"  LLM:       {MODEL_NAME}")
    print(f"  Testnet:   {os.getenv('BINANCE_TESTNET', 'true')}")
    print("="*60)

    state.initial_balance = binance.get_balance("USDT")
    print(f"\n  Initial Balance: {state.initial_balance:.4f} USDT\n")

    iteration = 0

    while state.running:
        iteration += 1
        print(f"\n  ── Iteration {iteration:04d} @ {time.strftime('%Y-%m-%d %H:%M:%S')} ──")

        # 1. Fetch market data
        try:
            market = get_market_data()
        except Exception as e:
            print(f"  [error] Market data fetch failed: {e}")
            time.sleep(10)
            continue

        price = market["price"]
        ind   = market["indicators"]
        print(f"  Price: {price:.2f}  RSI: {ind.get('rsi_14', 'n/a')}  "
              f"MACD: {ind.get('macd', 'n/a')}  ADX: {ind.get('adx_14', 'n/a')}")

        # 2. Check SL/TP on open position
        if state.position:
            hit = check_sl_tp(market)
            if hit:
                close_position(hit, market)
                # Save trade log after each close
                with open("trade_log.json", "w") as f:
                    json.dump(state.trade_log, f, indent=2)
                if len(state.trade_log) >= MAX_TRADES:
                    print(f"\n  Max trades ({MAX_TRADES}) reached. Stopping.")
                    state.running = False
                    break

        # 3. Ask LLM if no position open
        if not state.position:
            print(f"  [LLM] Querying {MODEL_NAME}...")
            decision = ask_llm(market)
            print(f"  [LLM] decision={decision.get('decision')} "
                  f"confidence={decision.get('confidence')} "
                  f"reason={decision.get('reason', '-')}")

            d = decision.get("decision", "HOLD")
            c = int(decision.get("confidence", 0))

            if d in ("BUY", "SELL") and c >= 8:
                # Post-LLM Safety
                ema50 = ind.get("ema_50")
                if d == "BUY" and ema50 and price < (ema50 * 0.998):
                    print("  [VETO] Blocked BUY in strong downtrend (Price < EMA50)")
                elif d == "SELL" and ema50 and price > (ema50 * 1.002):
                    print("  [VETO] Blocked SELL in strong uptrend (Price > EMA50)")
                else:
                    open_position(decision, market)
            else:
                print(f"  [HOLD] Confidence too low ({c}) or HOLD signal.")

        # 4. Sleep until next candle
        print(f"  Sleeping {SLEEP_SEC}s...")
        time.sleep(SLEEP_SEC)

    # Final report
    print("\n" + "="*60)
    print("  AUTO-TRADER STOPPED — FINAL REPORT")
    print("="*60)
    final_bal = binance.get_balance("USDT")
    print(f"  Initial Balance: {state.initial_balance:.4f} USDT")
    print(f"  Final Balance:   {final_bal:.4f} USDT")
    print(f"  Net PnL:         {final_bal - state.initial_balance:+.4f} USDT ({(final_bal/state.initial_balance-1)*100:+.2f}%)")
    print_summary()

    with open("trade_log.json", "w") as f:
        json.dump(state.trade_log, f, indent=2)
    print("\n  Trade log saved to trade_log.json")


if __name__ == "__main__":
    main()
