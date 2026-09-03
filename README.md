# 📈 Quant Trading System

A Gradio-powered algorithmic trading pipeline. It ingests **5-minute OHLCV** data from **multiple brokers — Dhan, Zerodha, Binance, and TradingView** — engineers a rich set of technical features, runs a time-series engine (**ARIMA + GARCH + Kalman filter**), trains a gradient-boosting classifier (**XGBoost / LightGBM**) for next-bar direction, blends everything into a composite **signal score**, and lets a **risk engine** produce an actionable trade plan (entry / stop-loss / target / position size).

Sandbox mode provides a seamless **yfinance** fallback so the full pipeline runs end-to-end with a single `Run Pipeline ▶` click — no broker credentials required.

```
Broker 5-min OHLCV (Dhan · Zerodha · Binance · TradingView)
   │
   ▼
Feature Engineering (EMA169/ATR/VWAP/RSI/Vol-ratio/Swings/BOS-CHoCH/Patterns/Momentum)
   │
   ▼
Time-Series Engine (ARIMA + GARCH + Kalman)
   │
   ▼
Quant ML Engine (XGBoost / LightGBM → P(UP) / P(DOWN))
   │
   ▼
Signal Score (−1 to +1)
   │
   ▼
Risk Engine (entry / SL / target / position size, R:R & max-loss capped)
   │
   ▼
(broker order — sandbox is a no-op)
```

---

## ✨ Features

- **Gradio UI** — **broker selector** (Dhan / Zerodha / Binance / TradingView), symbol / exchange / risk / ML-engine controls plus live status, trade plan, price chart, indicator panel and recent data tables.
- **Connection status pill** — a green **● CONNECTED** / red **● NOT CONNECTED** indicator next to the credentials that updates live as you switch broker or toggle sandbox mode.
- **Multi-broker support** — a common `BrokerClient` interface in `broker_base.py`:
  - **Dhan** (`dhan_client.py`) — Indian equities/F&O via DhanHQ
  - **Zerodha** (`zerodha_client.py`) — Indian equities/F&O/commodities via Kite Connect
  - **Binance** (`binance_client.py`) — crypto spot via python-binance (testnet supported)
  - **TradingView** (`tradingview_client.py`) — charting-platform alerts → webhook order forwarding (data via yfinance)
- **Feature engineering** (`features.py`):
  - EMA(169) + slope · ATR / volatility · session-anchored VWAP · RSI · volume ratio
  - Swing high/low, **BOS / CHoCH** market-structure flags
  - Candle patterns (bullish/bearish engulfing, doji, hammer) · returns / momentum
- **Time-series engine** (`ts_engine.py`):
  - ARIMA(1,0,1) next-return forecast
  - GARCH(1,1) volatility forecast
  - Local-level Kalman filter for trend + slope
  - Each component degrades gracefully if its library is unavailable.
- **Quant ML engine** (`ml_engine.py`): XGBoost / LightGBM (with a sklearn GradientBoosting fallback) predicting P(UP) / P(DOWN) on the final bar, plus feature importances.
- **Risk engine** (`risk_engine.py`): 50% ML probability, 25% ARIMA, 20% Kalman slope, 5% CHoCH → composite signal score; ATR-based stops, R:R targets and position sizing capped by account equity % and max-loss.
- **LLM analyst** (`llm_analyst.py`): after the signal score, an optional LLM explains the trade decision in plain language. Supports **Ollama**, **Hugging Face**, **OpenRouter** (open-source) and **OpenAI/ChatGPT**, **Anthropic/Claude** (paid). Config lives in `.env` (see `.env.example`); degrades gracefully when unconfigured.
- **Dhan integration** (`dhan_client.py`): intraday/daily OHLCV, symbol → security-id resolution (built-in map + live CSV cache), live LTP, and order placement.
- **Sandbox mode**: transparent **yfinance** fallback so demos work without credentials.

---

## 🗂 Project Structure

| File | Purpose |
|---|---|
| `app.py` | Gradio UI and pipeline orchestration |
| `config.py` | `Config` dataclass (broker + credentials, risk, ML params) + validation |
| `broker_base.py` | Common `BrokerClient` interface, registry + factory |
| `dhan_client.py` | Dhan API client (data, quotes, orders) + symbol resolution + yfinance fallback |
| `zerodha_client.py` | Zerodha (Kite Connect) client — data + orders |
| `binance_client.py` | Binance (python-binance) client — crypto data + orders |
| `tradingview_client.py` | TradingView webhook-forward client — alert orders + yfinance data |
| `features.py` | Technical feature engineering |
| `ts_engine.py` | ARIMA / GARCH / Kalman time-series engine |
| `ml_engine.py` | XGBoost / LightGBM direction classifier |
| `risk_engine.py` | Signal scoring + trade plan / position sizing |
| `llm_analyst.py` | LLM commentary via Ollama / Hugging Face / OpenRouter / OpenAI / Anthropic |
| `utils.py` | Data normalisation, plotting, markdown & numeric helpers |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+

### Installation

```bash
# create a virtual environment (optional but recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

### Run the app

```bash
python app.py
```

Open the printed Gradio URL in your browser.

By default **Sandbox mode** is enabled, so no broker credentials are needed — the pipeline fetches live 5-minute data via **yfinance** (`RELIANCE.NS`, `TCS.NS`, etc.).

### Go live with a broker

1. Pick a **Broker** from the dropdown (Dhan / Zerodha / Binance / TradingView).
2. Fill in that broker's credentials under **Broker credentials**.
3. Uncheck **Sandbox / demo mode**.
4. Adjust risk settings (account equity, % risk per trade, max loss, reward:risk).
5. Click **Run Pipeline ▶**.

#### Dhan
Enter **Client ID** + **Access Token** (from [Dhan Console](https://console.dhan.co/)).

#### Zerodha (Kite Connect)
Enter your **API key** and **Access token**. Generate them at the [Zerodha developer dashboard](https://developers.kite.trade/) and exchange the `request_token` for an access token after login. Symbols resolve automatically (`NSE:RELIANCE`).

#### Binance
Enter **API Key** + **API Secret** (create at [Binance API management](https://www.binance.com/en/my/settings/api-management)). Leave **testnet** checked for paper trading. Symbols use base+quote, e.g. `BTCUSDT`, `ETH/USDT`, or `BTC:USDT`.

#### TradingView
TradingView is a charting platform, not a REST broker. This integration forwards **alert payloads** to your own **webhook URL** when the pipeline generates a signal; set `TV_WEBHOOK_URL` in `.env` or the UI. Data for symbols is fetched via yfinance; use `TV_SYMBOL_MAP` (e.g. `NIFTY:^NSEI,BTCUSDT:BTC-USD`) to map TradingView symbols to yfinance tickers. Without a webhook URL, signals are returned as a **pending alert** (no order sent).

---

## ⚙️ Configuration

`Config` (`config.py`) fields:

| Field | Default | Notes |
|---|---|---|
| `symbol` / `exchange` | `RELIANCE` / `NSE` | NSE, BSE, MCX, NSE_FNO, CUR |
| `lookback_days` | `10` | History window for intraday fetch |
| `account_equity` | `100000` | Used for position sizing |
| `risk_per_trade` | `0.01` | 1% of equity per trade |
| `max_risk_per_trade` | `5000` | ₹ cap on per-trade risk |
| `rr_ratio` | `2.0` | Reward : risk for targets |
| `train_window` | `250` | ML training rows |
| `ema_span` / `rsi_period` / `atr_period` | `169 / 14 / 14` | Indicator windows |
| `sandbox` | `True` | Use yfinance data when `True` |

Security IDs are resolved automatically for a built-in set of popular instruments (RELIANCE, TCS, HDFCBANK, NIFTY, BANKNIFTY, …) or via the live Dhan scrip-master CSV (cached to `security_master.csv`).

---

## 🤖 LLM Analyst (`.env`)

After the signal score, the pipeline can ask an LLM to explain the trade in plain language. All credentials and the selected model are read from a local `.env` file (copy `.env.example` → `.env`; `.env` is git-ignored).

It is enabled **out of the box** with the free, local **Ollama** provider (`LLM_PROVIDER=ollama`, `LLM_MODEL=llama3`) — just install [Ollama](https://ollama.com) and run `ollama pull llama3`. The UI's **LLM Analyst provider** dropdown defaults to whatever is set in `.env`, and the **LLM model(s)** dropdown lists that provider's models (shown with respect to the selected provider).

| Variable | Purpose | Example |
|---|---|---|
| `LLM_PROVIDER` | `none`, `ollama`, `huggingface`, `openrouter`, `openai`, `anthropic` | `ollama` |
| `LLM_MODEL` | Model id for the selected provider | `llama3`, `gpt-4o-mini`, `claude-3-5-sonnet`, `mistralai/Mistral-7B-Instruct-v0.3` |
| `LLM_API_KEY` | API key for paid / hosted providers (not needed for local Ollama) | — |
| `LLM_BASE_URL` | Custom endpoint (Ollama defaults to `http://localhost:11434`) | `http://localhost:11434` |
| `LLM_TEMPERATURE` | Sampling temperature (default `0.3`) | `0.3` |
| `LLM_MAX_TOKENS` | Max output tokens (default `512`) | `512` |
| `LLM_TIMEOUT` | Request timeout in seconds (default `120`) | `120` |
| `LLM_SYSTEM_PROMPT` | Optional custom system prompt (defaults to a built-in analyst prompt) | — |

**Provider cheat-sheet**

- **Ollama (open-source, local):** set `LLM_PROVIDER=ollama`, `LLM_MODEL=llama3` (or any pulled model). No API key needed; requires Ollama running locally.
- **Hugging Face (open-source, hosted):** `LLM_PROVIDER=huggingface`, `LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3`, optional `HUGGINGFACE_API_KEY` for rate limits.
- **OpenRouter (open + closed):** `LLM_PROVIDER=openrouter`, `LLM_MODEL=meta-llama/llama-3.1-8b-instruct`, plus `OPENROUTER_API_KEY`.
- **OpenAI / ChatGPT (paid):** `LLM_PROVIDER=openai`, `LLM_MODEL=gpt-4o-mini`, plus `OPENAI_API_KEY`.
- **Anthropic / Claude (paid):** `LLM_PROVIDER=anthropic`, `LLM_MODEL=claude-3-5-sonnet-20241022`, plus `ANTHROPIC_API_KEY`.

The provider can also be switched at runtime from the **LLM Analyst provider** dropdown in the UI. When you pick a provider, the **LLM model(s)** dropdown shows the models available for that provider (you can select **one or more** at once — the pipeline queries each selected model and shows its commentary separately under the `── Model: … ──` heading). If a provider/model/key is missing or a call fails, the pipeline shows a short notice and the signal score + trade plan still work normally.

---

## 📚 Dependencies

See `requirements.txt`. Highlights:

- **Data / ML:** `pandas`, `numpy`, `scipy`, `scikit-learn`
- **Time-series:** `statsmodels`, `arch`, `pykalman`
- **Gradient boosting:** `xgboost`, `lightgbm`
- **Brokers:** `dhanhq`, `kiteconnect`, `python-binance`
- **Visualisation / UI:** `matplotlib`, `gradio`
- **Data fallback:** `yfinance`

---

## ⚠️ Disclaimer

This software is for **educational / research purposes only** and does **not** constitute financial advice. Algorithmic trading involves substantial risk of loss. Use at your own risk — always paper-trade in sandbox mode before considering live markets.