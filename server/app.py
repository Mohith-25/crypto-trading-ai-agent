"""
FastAPI Server — OpenEnv Crypto Trading
Exposes /reset, /step, /state, /grade, /health endpoints.
"""

from __future__ import annotations
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from server.environment import CryptoTradingEnv
from server.models import Action, Reward

app = FastAPI(
    title="OpenEnv Crypto Trading AI Agent",
    description=(
        "An OpenEnv-compliant RL environment for AI-powered crypto trading. "
        "Connects to Binance (testnet/live) and automatically executes "
        "buy/sell orders with stop-loss and take-profit."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_env = CryptoTradingEnv()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "task1"

class StepRequest(BaseModel):
    action_type:  str
    symbol:       Optional[str]   = None
    quantity:     Optional[float] = None
    stop_loss:    Optional[float] = None
    take_profit:  Optional[float] = None
    reason:       Optional[str]   = None

class StepResponse(BaseModel):
    observation:   Dict[str, Any]
    reward:        float
    done:          bool
    partial_score: float
    info:          Dict[str, Any]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "OpenEnv Crypto Trading AI Agent",
        "version": "1.0.0",
        "binance_testnet": os.getenv("BINANCE_TESTNET", "true"),
        "trading_mode":    os.getenv("TRADING_MODE", "SPOT"),
        "tasks": [
            "task1 — BTC/USDT 1m scalping (easy)",
            "task2 — ETH/USDT 15m swing (medium)",
            "task3 — BTC/USDT 5m full AI (hard)",
        ],
        "endpoints": ["/reset", "/step", "/state", "/grade", "/health"],
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset(req: ResetRequest = None):
    if req is None:
        req = ResetRequest()
    obs = _env.reset(task_id=req.task_id)
    return obs.model_dump()

@app.post("/step")
def step(req: StepRequest) -> StepResponse:
    action = Action(
        action_type=req.action_type,
        symbol=req.symbol,
        quantity=req.quantity,
        stop_loss=req.stop_loss,
        take_profit=req.take_profit,
        reason=req.reason,
    )
    try:
        obs, reward_obj, done, info = _env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return StepResponse(
        observation=obs.model_dump(),
        reward=reward_obj.value,
        done=done,
        partial_score=reward_obj.partial_score,
        info=info,
    )

@app.get("/state")
def state():
    try:
        return _env.state().model_dump()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.get("/grade")
def grade():
    try:
        score = _env.grade()
        s     = _env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "task_id": s.task_id,
        "score":   score,
        "trades":  len(s.trade_history),
        "win":     s.win_trades,
        "loss":    s.loss_trades,
        "pnl_pct": round((s.balance_usdt / s.initial_balance - 1) * 100, 2),
    }

@app.post("/grade")
def grade_post():
    return grade()

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
