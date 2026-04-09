"""
AI Trading Agent
================
LLM-powered trading agent that:
  1. Reads live candle data + technical indicators from the environment
  2. Reasons about market conditions using an LLM (via OpenAI client)
  3. Automatically executes BUY / SELL / HOLD with SL and TP
  4. Monitors positions and closes them when SL or TP is hit
  5. Maintains a full trade log and reports performance

Environment variables required:
    API_BASE_URL        LLM endpoint  (e.g. https://router.huggingface.co/v1)
    MODEL_NAME          LLM model id  (e.g. meta-llama/Meta-Llama-3-8B-Instruct)
    HF_TOKEN            HuggingFace / LLM API key
    BINANCE_API_KEY     Binance testnet or live API key
    BINANCE_API_SECRET  Binance API secret
    BINANCE_TESTNET     "true" (default) | "false"
    TRADING_MODE        "SPOT" | "FUTURES"
    ENV_BASE_URL        OpenEnv server URL (default: http://localhost:7860)
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL  = os.getenv("API_BASE_URL",  "https://router.huggingface.co/v1")
MODEL_NAME    = os.getenv("MODEL_NAME",    "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN      = os.getenv("HF_TOKEN",      os.getenv("API_KEY", ""))
ENV_BASE_URL  = os.getenv("ENV_BASE_URL",  "http://localhost:7860")

TEMPERATURE   = 0.1
MAX_TOKENS    = 600
MAX_STEPS     = 50      # per task
SLEEP_BETWEEN = 2.0    # seconds between steps (respects rate limits)

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "dummy",
)

# ---------------------------------------------------------------------------
# Environment HTTP helpers
# ---------------------------------------------------------------------------

def env_reset(task_id: str) -> Dict[str, Any]:
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(f"{ENV_BASE_URL}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()

def env_grade() -> Dict[str, Any]:
    r = requests.get(f"{ENV_BASE_URL}/grade", timeout=30)
    r.raise_for_status()
    return r.json()

def env_state() -> Dict[str, Any]:
    r = requests.get(f"{ENV_BASE_URL}/state", timeout=30)
    r.raise_for_status()
    return r.json()

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert cryptocurrency trading AI agent.
You analyze real-time market data and technical indicators to make profitable trading decisions.

RULES:
1. You MUST respond ONLY with a valid JSON object - no prose, no markdown.
2. Always include stop_loss and take_profit when opening a position.
3. Never open a position if one is already open (check position field).
4. MANDATORY CONFLUENCE: You may ONLY open a trade if AT LEAST TWO indicators agree (e.g., MACD is bullish AND RSI is oversold). If indicators are mixed or market is chopping sideways, you MUST output "hold".
5. Use indicators to justify your decision:
   - RSI < 30 = oversold - consider BUY
   - RSI > 70 = overbought - consider SELL (futures) or HOLD (spot)
   - MACD hist positive and rising - bullish
   - MACD hist negative and falling - bearish
   - EMA9 > EMA21 - uptrend
   - EMA9 < EMA21 - downtrend
   - ADX > 25 - strong trend, follow it
   - ADX < 20 - weak trend, DO NOT TRADE (hold)

RESPONSE FORMAT:
{
  "action_type": "buy" | "sell" | "close" | "hold" | "set_sl" | "set_tp",
  "stop_loss": <float or null>,
  "take_profit": <float or null>,
  "reason": "<explain the technical confluence>"
}

Risk management:
- stop_loss: max 0.8% below entry (scalping). Preserve capital.
- take_profit: ensure realistically fast profits; aim for ~1.5 to 2.0x reward to risk.
- Do NOT hesitate to "close" a position early if momentum flips against you completely."""


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _fmt_indicators(ind: Dict) -> str:
    lines = []
    if ind.get("rsi_14"):       lines.append(f"  RSI(14):     {ind['rsi_14']:.1f}")
    if ind.get("macd"):         lines.append(f"  MACD:        {ind['macd']:.4f}")
    if ind.get("macd_signal"):  lines.append(f"  MACD Signal: {ind['macd_signal']:.4f}")
    if ind.get("macd_hist"):    lines.append(f"  MACD Hist:   {ind['macd_hist']:.4f} {'UP' if ind['macd_hist'] > 0 else 'DOWN'}")
    if ind.get("bb_upper"):     lines.append(f"  BB Upper:    {ind['bb_upper']:.2f}")
    if ind.get("bb_middle"):    lines.append(f"  BB Mid:      {ind['bb_middle']:.2f}")
    if ind.get("bb_lower"):     lines.append(f"  BB Lower:    {ind['bb_lower']:.2f}")
    if ind.get("ema_9"):        lines.append(f"  EMA(9):      {ind['ema_9']:.2f}")
    if ind.get("ema_21"):       lines.append(f"  EMA(21):     {ind['ema_21']:.2f}")
    if ind.get("ema_50"):       lines.append(f"  EMA(50):     {ind['ema_50']:.2f}")
    if ind.get("atr_14"):       lines.append(f"  ATR(14):     {ind['atr_14']:.4f}")
    if ind.get("adx_14"):       lines.append(f"  ADX(14):     {ind['adx_14']:.1f}")
    return "\n".join(lines) if lines else "  (no indicators)"


def _fmt_position(pos: Optional[Dict]) -> str:
    if not pos:
        return "  FLAT - no open position"
    side  = pos.get("side", "?")
    ep    = pos.get("entry_price", 0)
    qty   = pos.get("quantity", 0)
    sl    = pos.get("stop_loss", "?")
    tp    = pos.get("take_profit", "?")
    pnl   = pos.get("pnl", 0)
    return (
        f"  Side:        {side}\n"
        f"  Entry:       {ep:.2f}\n"
        f"  Quantity:    {qty:.6f}\n"
        f"  Stop-Loss:   {sl}\n"
        f"  Take-Profit: {tp}\n"
        f"  Unrealised:  {pnl:+.4f} USDT"
    )


def _recent_candles_summary(candles: List[Dict], n: int = 5) -> str:
    recent = candles[-n:] if len(candles) >= n else candles
    lines  = ["  Timestamp            Open      High      Low       Close     Volume"]
    for c in recent:
        ts = time.strftime("%Y-%m-%d %H:%M", time.gmtime(c["timestamp"] / 1000))
        lines.append(
            f"  {ts}   {c['open']:>8.2f}  {c['high']:>8.2f}  {c['low']:>8.2f}  {c['close']:>8.2f}  {c['volume']:>10.2f}"
        )
    return "\n".join(lines)


def build_prompt(obs: Dict[str, Any], history: List[str]) -> str:
    ctx  = obs.get("context", {})
    ind  = obs.get("indicators", {})
    pos  = obs.get("position")
    cans = obs.get("candles", [])

    trend = "UNKNOWN"
    e9  = ind.get("ema_9")
    e21 = ind.get("ema_21")
    if e9 and e21:
        trend = "UPTREND (UP)" if e9 > e21 else "DOWNTREND (DOWN)"

    return f"""TASK: {obs.get('task_description', '')}

MARKET SNAPSHOT
  Symbol:        {obs.get('symbol', '?')}
  Current Price: {obs.get('current_price', 0):.2f} USDT
  Trend:         {trend}

TECHNICAL INDICATORS
{_fmt_indicators(ind)}

RECENT CANDLES (last 5)
{_recent_candles_summary(cans)}

CURRENT POSITION
{_fmt_position(pos)}

ACCOUNT
  Balance:       {obs.get('balance_usdt', 0):.4f} USDT
  Equity:        {obs.get('equity', 0):.4f} USDT
  PnL:           {ctx.get('total_pnl_usdt', 0):+.4f} USDT ({ctx.get('pnl_pct', 0):+.2f}%)
  Trades:        {ctx.get('total_trades', 0)} total | {ctx.get('win_trades', 0)}W / {ctx.get('loss_trades', 0)}L
  Step:          {obs.get('step_count', 0)} / remaining {ctx.get('remaining_steps', '?')}

RECENT DECISIONS
{chr(10).join(history[-4:]) or '  (none yet)'}

What is your next trading decision? Respond with a single JSON object."""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(messages: List[Dict]) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [LLM error] {e}")
        return '{"action_type": "hold", "reason": "LLM error fallback"}'


def parse_action(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = [l for l in raw.split("\n") if not l.startswith("```")]
        raw = "\n".join(lines)
    try:
        action = json.loads(raw)
        if "action_type" not in action:
            action["action_type"] = "hold"
        return action
    except Exception:
        import re
        m = re.search(r"\{.*?\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {"action_type": "hold", "reason": "parse fallback"}


# ---------------------------------------------------------------------------
# Rule-based signal layer (augments LLM decisions)
# ---------------------------------------------------------------------------

def rule_signal(obs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Fast rule-based safety checks.
    Returns a forced action if critical conditions are met,
    otherwise returns None (LLM decides).
    """
    ind = obs.get("indicators", {})
    pos = obs.get("position")
    price = obs.get("current_price", 0)

    # Force close if position but RSI extreme against us
    if pos:
        side = pos.get("side")
        rsi  = ind.get("rsi_14")
        if side == "BUY" and rsi and rsi > 80:
            return {"action_type": "close", "reason": "RSI overbought - force close long"}
        if side == "SELL" and rsi and rsi < 20:
            return {"action_type": "close", "reason": "RSI oversold - force close short"}

    return None  # let LLM decide


# ---------------------------------------------------------------------------
# Run one task episode
# ---------------------------------------------------------------------------

def run_episode(task_id: str) -> float:
    print(f"\n{'='*65}")
    print(f"  TASK: {task_id.upper()}")
    print(f"{'='*65}")
    print(f"[START] task={task_id}", flush=True)

    obs       = env_reset(task_id)
    messages  = [{"role": "system", "content": SYSTEM_PROMPT}]
    history:  List[str] = []

    print(f"  Symbol : {obs.get('symbol')}")
    print(f"  Balance: {obs.get('balance_usdt', 0):.2f} USDT")
    print(f"  Task   : {obs.get('task_description', '')[:100]}...")

    actual_steps = 0
    for step_num in range(MAX_STEPS):
        actual_steps = step_num + 1
        print(f"\n  -- Step {step_num+1:03d} ------------------------------------------")
        print(f"  Price: {obs.get('current_price', 0):.2f}  |  "
              f"Balance: {obs.get('balance_usdt', 0):.4f}  |  "
              f"PnL: {obs.get('context', {}).get('pnl_pct', 0):+.2f}%")

        # 1. Check rule-based override
        forced = rule_signal(obs)
        if forced:
            action = forced
            print(f"  [RULE] {action}")
        else:
            # 2. Ask LLM
            user_msg = build_prompt(obs, history)
            messages_call = messages + [{"role": "user", "content": user_msg}]
            raw  = call_llm(messages_call)
            action = parse_action(raw)

            # Post-LLM safety veto
            ind = obs.get("indicators", {})
            ema50 = ind.get("ema_50")
            price = obs.get("current_price", 0)
            if action.get("action_type") == "buy" and ema50 and price < (ema50 * 0.998):
                action = {"action_type": "hold", "reason": "SAFETY VETO: Blocked BUY in downtrend (Price < EMA50)"}
            if action.get("action_type") == "sell" and ema50 and price > (ema50 * 1.002):
                action = {"action_type": "hold", "reason": "SAFETY VETO: Blocked SELL in uptrend (Price > EMA50)"}


            # Append to running context (keep short)
            messages.append({"role": "user",      "content": user_msg})
            messages.append({"role": "assistant", "content": raw})
            if len(messages) > 13:   # system + 6 turns
                messages = messages[:1] + messages[-12:]

        print(f"  [ACTION] {json.dumps(action)}")
        history.append(f"step={step_num+1} | {json.dumps(action)}")

        # 3. Execute in environment
        try:
            result = env_step(action)
        except Exception as e:
            print(f"  [step error] {e}")
            break

        reward = result.get("reward", 0.0)
        score  = result.get("partial_score", 0.0)
        done   = result.get("done", False)
        info   = result.get("info", {})
        obs    = result.get("observation", obs)

        print(f"  reward={reward:+.3f}  score={score:.3f}  done={done}")
        print(f"[STEP] step={step_num+1} reward={reward}", flush=True)
        if info:
            for k, v in info.items():
                print(f"    {k}: {v}")

        if done:
            break

        time.sleep(SLEEP_BETWEEN)

    # Final grade
    grade = env_grade()
    print(f"\n  +- FINAL RESULTS ({task_id}) -----------------------------+")
    print(f"  | Score:   {grade.get('score', 0):.4f}                              |")
    print(f"  | Trades:  {grade.get('trades', 0)} ({grade.get('win', 0)}W / {grade.get('loss', 0)}L)                       |")
    print(f"  | PnL:     {grade.get('pnl_pct', 0):+.2f}%                              |")
    print(f"  +---------------------------------------------------------+")
    print(f"[END] task={task_id} score={grade.get('score', 0.0)} steps={actual_steps}", flush=True)

    return grade.get("score", 0.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "="*65)
    print("  OpenEnv Crypto Trading AI Agent - Baseline Inference")
    print("="*65)
    print(f"  LLM:      {MODEL_NAME}")
    print(f"  Endpoint: {API_BASE_URL}")
    print(f"  Env:      {ENV_BASE_URL}")
    print(f"  Mode:     {os.getenv('TRADING_MODE', 'SPOT')} | Testnet: {os.getenv('BINANCE_TESTNET', 'true')}")

    if not HF_TOKEN:
        print("\n  WARNING: HF_TOKEN not set - LLM calls may fail.")

    scores: Dict[str, float] = {}
    for task_id in ["task1", "task2", "task3"]:
        scores[task_id] = run_episode(task_id)

    avg = sum(scores.values()) / len(scores)
    print("\n" + "="*65)
    print("  BASELINE SCORES")
    print("="*65)
    labels = {"task1": "Easy  (1m scalp)", "task2": "Medium (15m swing)", "task3": "Hard   (5m full AI)"}
    for tid, sc in scores.items():
        print(f"  {labels[tid]:25s}: {sc:.4f}")
    print(f"  {'Average':25s}: {avg:.4f}")
    print("="*65)

    with open("baseline_results.json", "w") as f:
        json.dump({"model": MODEL_NAME, "scores": scores, "average": avg}, f, indent=2)
    print("\n  Results saved to baseline_results.json")


if __name__ == "__main__":
    main()
