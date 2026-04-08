

# 📈 OpenEnv Crypto Trading AI Agent

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)](https://openenv.ai)
[![Binance](https://img.shields.io/badge/Binance-Testnet%20%2F%20Live-F0B90B)](https://testnet.binance.vision)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.13-green)](https://python.org)

An **AI-powered cryptocurrency trading agent** that connects to Binance (testnet or live),
reads real-time market data, applies technical analysis, and **automatically executes
BUY/SELL orders with stop-loss and take-profit** using an LLM for decision-making.

---

## 🏗️ Architecture

```
Binance API (testnet/live)
        │
        ▼
CryptoTradingEnv  ──── Technical Indicators (RSI/MACD/BB/EMA/ATR/ADX)
        │
        ▼
FastAPI Server (/reset /step /state /grade)
        │
        ▼
AI Trading Agent (LLM via OpenAI client)
        │
        ▼
Auto BUY / SELL + Stop-Loss + Take-Profit
```

---

## 🎯 Tasks

| Task | Symbol | Interval | Strategy | Difficulty |
|------|--------|----------|----------|-----------|
| task1 | BTC/USDT | 1m | RSI + BB scalping | Easy |
| task2 | ETH/USDT | 15m | MACD + EMA swing | Medium |
| task3 | BTC/USDT | 5m | Full AI (all indicators) | Hard |

---

## ⚡ Quick Start

### 1. Get Binance Testnet API Keys

1. Go to https://testnet.binance.vision
2. Sign in with GitHub
3. Click **"Generate HMAC_SHA256 Key"**
4. Copy your API Key and Secret

### 2. Install & Configure

```bash
git clone https://github.com/your-username/openenv-crypto-trader.git
cd openenv-crypto-trader

# Create and activate virtual environment (Windows/Linux)
python -m venv .venv
# On Windows:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set your keys
export BINANCE_API_KEY="your_testnet_api_key"
export BINANCE_API_SECRET="your_testnet_api_secret"
export BINANCE_TESTNET="true"          # ALWAYS true for demo
export TRADING_MODE="SPOT"             # or FUTURES

# LLM (HuggingFace router)
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
export HF_TOKEN="hf_your_token_here"
```

### 3. Option A — Run via OpenEnv server (hackathon submission)

**Linux/macOS:**
```bash
# Terminal 1: Start environment server
python main.py

# Terminal 2: Run AI agent
python inference.py
```

**Windows (PowerShell):**
```powershell
.\run_option_a.ps1
```

### 4. Option B — Run auto-trader directly (always-on trader)

**Linux/macOS:**
```bash
export SYMBOL="BTCUSDT"
export INTERVAL="5m"
export RISK_PCT="0.02"        # 2% balance per trade
export SL_PCT="0.008"         # 0.8% stop-loss
export TP_PCT="0.024"         # 2.4% take-profit (3:1)
export MAX_TRADES="20"
export SLEEP_SEC="300"        # check every 5 minutes

python auto_trader.py
```

**Windows (PowerShell):**
```powershell
.\run_option_b.ps1
```

---

## 🔌 API Reference

### POST /reset
```json
{"task_id": "task1"}
```

### POST /step
```json
{
  "action_type": "buy",
  "stop_loss": 64000.0,
  "take_profit": 67000.0,
  "reason": "RSI oversold + BB lower band touch"
}
```

### GET /grade
Returns current score with trade stats:
```json
{
  "task_id": "task1",
  "score": 0.72,
  "trades": 5,
  "win": 3,
  "loss": 2,
  "pnl_pct": 3.45
}
```

---

## 📊 Action Space

| Action | Description | Required Params |
|--------|-------------|-----------------|
| `buy` | Open long position | stop_loss, take_profit |
| `sell` | Open short (futures) / close long (spot) | stop_loss, take_profit |
| `close` | Close at market price | reason |
| `hold` | Do nothing | — |
| `set_sl` | Update stop-loss | stop_loss |
| `set_tp` | Update take-profit | take_profit |

---

## 📈 Indicators Used

| Indicator | Signal |
|-----------|--------|
| RSI(14) < 30 | Oversold → buy signal |
| RSI(14) > 70 | Overbought → sell/close signal |
| MACD hist > 0 & rising | Bullish momentum |
| MACD hist < 0 & falling | Bearish momentum |
| Price < BB Lower | Potential reversal up |
| Price > BB Upper | Potential reversal down |
| EMA9 > EMA21 | Uptrend |
| EMA9 < EMA21 | Downtrend |
| ATR(14) | Dynamic SL/TP sizing |
| ADX(14) > 25 | Strong trend — follow it |

---

## 💰 Risk Management

- **Stop-Loss**: ATR-based (2× ATR below entry) or fixed % (0.5–1%)
- **Take-Profit**: ATR-based (3× ATR above entry) or fixed % (1.5–3%)
- **Position size**: Risk 1–2% of balance per trade (Kelly-inspired)
- **Max drawdown**: Auto-close episode if balance drops 15%
- **Loop protection**: No repeated same-direction trades without close
- **Reward-to-risk**: Minimum 2:1 required by agent rules

---

## 📁 Project Structure

```
openenv-crypto-trader/
├── server/
│   ├── __init__.py
│   ├── app.py               # FastAPI server
│   ├── environment.py       # CryptoTradingEnv (step/reset/state)
│   ├── models.py            # Pydantic: Observation, Action, Reward, State
│   ├── indicators.py        # RSI, MACD, BB, EMA, ATR, ADX (pure Python)
│   ├── graders.py           # Deterministic graders for task1/2/3
│   └── binance_connector.py # Binance REST API wrapper (spot + futures)
├── inference.py             # OpenEnv baseline agent (hackathon submission)
├── auto_trader.py           # Standalone always-on auto-trader
├── main.py                  # Server entry point
├── openenv.yaml             # OpenEnv spec
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ✅ Pre-Submission Checklist

- [x] Real-world task — live crypto trading on Binance
- [x] Full OpenEnv spec: typed Pydantic models, step/reset/state, openenv.yaml
- [x] 3 tasks: easy (1m scalp) → medium (15m swing) → hard (5m full AI)
- [x] Meaningful reward: dense signals, partial progress, drawdown penalty
- [x] Baseline `inference.py` using OpenAI client + env vars
- [x] Auto SL/TP execution via Binance OCO/conditional orders
- [x] ATR-based dynamic position sizing
- [x] Working Dockerfile, port 7860
- [x] `auto_trader.py` for standalone continuous trading

---

## 📊 Baseline Scores

| Task | Score |
|------|-------|
| task1 (easy) | ~0.65 |
| task2 (medium) | ~0.50 |
| task3 (hard) | ~0.40 |
| **Average** | **~0.52** |

---

## ⚠️ Disclaimer

This software is for **educational purposes only**. Always use testnet first.
Crypto trading involves significant financial risk. Past performance does not
guarantee future results.

---

## 📄 License

MIT
