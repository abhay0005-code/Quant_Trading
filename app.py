"""
Gradio UI for the Quant Trading pipeline.

Pipeline:  Dhan 5-min OHLCV → Feature Engineering → Time-Series Engine
           → Quant ML Engine → Signal Score → Risk Engine → (Dhan order)
"""
from __future__ import annotations

import logging
import os

# Opt out of Gradio/HuggingFace telemetry BEFORE importing gradio, otherwise
# the analytics call at import/startup can hang on restricted hosts (e.g.
# Railway) and block the server from starting.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED_LOCAL", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "0")

import gradio as gr
from gradio import HTML

from config import Config
from broker_base import create_broker, BrokerClient
from dhan_client import DhanDataClient
from features import add_features
from ts_engine import run_time_series_engine
from ml_engine import train_and_predict
from risk_engine import signal_score, compute_trade_plan
from llm_analyst import (generate_analysis, check_llm_connection,
                         LLM_MODEL_OPTIONS)
from utils import plot_candlestick, plot_indicator_panel, dataframe_to_markdown

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("app")

INDICATORS = ["rsi", "volume_ratio", "volatility", "ema_slope", "ret_5"]

LLM_PROVIDERS = ["none", "ollama", "huggingface", "openrouter", "openai", "anthropic"]
_DEFAULT_LLM = os.environ.get("LLM_PROVIDER", "none").strip().lower()
if _DEFAULT_LLM not in LLM_PROVIDERS:
    _DEFAULT_LLM = "none"

BROKERS = ["dhan", "zerodha", "binance", "tradingview"]
_DEFAULT_BROKER = os.environ.get("BROKER", "dhan").strip().lower()
if _DEFAULT_BROKER not in BROKERS:
    _DEFAULT_BROKER = "dhan"

# Stores the latest pipeline context so the LLM tab can re-analyse on demand.
_latest_llm_ctx: dict = {}

# Strategy options for the LLM Analysis tab dropdown.
STRATEGY_OPTIONS = [
    "Swing High / Swing Low",
    "BOS — Break of Structure",
    "CHoCH — Change of Character",
    "Higher High / Higher Low",
    "Lower High / Lower Low",
    "Support/Resistance",
    "Previous Day High/Low",
    "Opening Range High/Low",
    "5-minute High/Low",
    "Candle breakout",
    "Retest",
    "Rejection candle",
    "Engulfing candle",
    "Pin bar",
]

# Indicator options for the LLM Analysis tab dropdown.
INDICATOR_OPTIONS = [
    "RSI",
    "MACD",
    "Stochastic RSI",
    "ROC",
    "ADX",
    "200 EMA",
    "VWAP",
    "Supertrend",
    "9 EMA",
    "21 EMA",
]


def _build_config(broker, symbol, exchange, client_id, access_token, sandbox,
                  kite_key, kite_token, binance_key, binance_secret,
                  binance_testnet, tv_webhook, tv_symmap,
                  equity, risk_pct, max_loss, rr, train_window):
    return Config(
        broker=(broker or "dhan").strip().lower(),
        symbol=str(symbol).strip().upper(), exchange=exchange,
        client_id=client_id.strip(), access_token=access_token.strip(),
        sandbox=bool(sandbox),
        kite_api_key=kite_key.strip(), kite_access_token=kite_token.strip(),
        binance_api_key=binance_key.strip(),
        binance_api_secret=binance_secret.strip(),
        binance_testnet=bool(binance_testnet),
        tv_webhook_url=tv_webhook.strip(), tv_symbol_map=tv_symmap.strip(),
        account_equity=float(equity),
        risk_per_trade=float(risk_pct) / 100.0, max_risk_per_trade=float(max_loss),
        rr_ratio=float(rr), train_window=int(train_window))


def _client_for(cfg: Config) -> BrokerClient:
    """Build the correct broker client for the config."""
    if cfg.broker == "zerodha":
        return create_broker("zerodha", api_key=cfg.kite_api_key,
                             access_token=cfg.kite_access_token,
                             sandbox=cfg.sandbox)
    if cfg.broker == "binance":
        return create_broker("binance", api_key=cfg.binance_api_key,
                             api_secret=cfg.binance_api_secret,
                             testnet=cfg.binance_testnet)
    if cfg.broker == "tradingview":
        return create_broker("tradingview", webhook_url=cfg.tv_webhook_url,
                             symbol_map=cfg.tv_symbol_map, sandbox=True)
    # Default: Dhan
    return DhanDataClient(cfg.client_id, cfg.access_token, sandbox=cfg.sandbox)


def _broker_label(broker: str, sandbox: bool, client=None) -> str:
    """Human-readable status line for the active broker."""
    b = (broker or "dhan").strip().lower()
    if b == "binance":
        mode = "TESTNET" if (client is None or client.testnet) else "LIVE"
        return f"BINANCE {mode}"
    if b == "tradingview":
        return "TRADINGVIEW"
    if b == "zerodha":
        return "ZERODHA LIVE" if not sandbox else "ZERODHA (SANDBOX/demo)"
    return "SANDBOX (yfinance)" if sandbox else "DHAN LIVE"


def update_broker_visibility(broker) -> list:
    """Show only the credential group matching the selected broker."""
    b = (broker or "dhan").strip().lower()
    return [
        gr.Group(visible=(b == "dhan")),
        gr.Group(visible=(b == "zerodha")),
        gr.Group(visible=(b == "binance")),
        gr.Group(visible=(b == "tradingview")),
    ]


def check_connection(broker, client_id, access_token, sandbox,
                     kite_key, kite_token, binance_key, binance_secret,
                     binance_testnet, tv_webhook, tv_symmap) -> str:
    """Return an HTML status pill for the active broker connection."""
    try:
        cfg = _build_config(broker, "RELIANCE", "NSE", client_id, access_token,
                            sandbox, kite_key, kite_token, binance_key,
                            binance_secret, binance_testnet, tv_webhook,
                            tv_symmap, 100000, 1.0, 5000, 2.0, 250)
        client = _client_for(cfg)
        ok = client.connect()
        mode = _broker_label(cfg.broker, cfg.sandbox, client)
        color = "green" if ok else "red"
        verb = "CONNECTED" if ok else "NOT CONNECTED"
        return f'<div style="display:inline-block;padding:4px 10px;border-radius:12px;' \
               f'background:{color};color:#fff;font-weight:bold;">● {mode} — {verb}</div>'
    except Exception as e:
        return f'<div style="display:inline-block;padding:4px 10px;border-radius:12px;' \
               f'background:red;color:#fff;font-weight:bold;">● NOT CONNECTED — {e}</div>'


def update_llm_models(provider) -> gr.Dropdown:
    """Return updated Dropdown with the model choices for the selected provider."""
    provider = (provider or "none").strip().lower()
    choices = list(LLM_MODEL_OPTIONS.get(provider, []))
    env_default = os.environ.get("LLM_MODEL", "").strip()

    if env_default and env_default not in choices:
        choices.insert(0, env_default)

    value = None
    if choices:
        value = [env_default] if (env_default and env_default in choices) else [choices[0]]
    log.info("update_llm_models: provider=%s choices=%d value=%s", provider, len(choices), value)
    return gr.Dropdown(
        choices=choices if choices else [],
        value=value,
        multiselect=True,
        allow_custom_value=True,
        interactive=True,
        visible=True,
        label="LLM model(s) (select one or more)",
    )


def update_tab_llm_models(provider) -> gr.Dropdown:
    """Return updated Dropdown with the model choices for the tab's model dropdown."""
    provider = (provider or "none").strip().lower()
    choices = list(LLM_MODEL_OPTIONS.get(provider, []))
    env_default = os.environ.get("LLM_MODEL", "").strip()

    if env_default and env_default not in choices:
        choices.insert(0, env_default)

    value = None
    if choices:
        value = env_default if (env_default and env_default in choices) else choices[0]
    log.info("update_tab_llm_models: provider=%s choices=%d value=%s", provider, len(choices), value)
    return gr.Dropdown(
        choices=choices if choices else [],
        value=value,
        multiselect=False,
        allow_custom_value=True,
        interactive=True,
        visible=True,
        label="Select LLM Model",
    )


def check_tab_llm_connection(provider: str, model: str) -> str:
    """Return an HTML status pill showing LLM provider/model connectivity."""
    check = check_llm_connection(provider or "", model or "")
    ok = check["ok"]
    color = "green" if ok else "red"
    verb = "CONNECTED" if ok else "ISSUE"
    return (f'<div style="display:inline-block;padding:4px 10px;border-radius:12px;'
            f'background:{color};color:#fff;font-weight:bold;">'
            f'● LLM {verb} — {check["status"]}</div>')


def run_tab_llm_analysis(provider: str, model: str, strategies: list, indicators: list) -> str:
    """Generate LLM analysis using the selected provider/model from the tab."""
    global _latest_llm_ctx

    provider = (provider or "none").strip().lower()
    model = str(model or "").strip()

    if provider in ("none", "", "off"):
        return ("_No LLM provider selected. Choose a provider from the "
                "dropdown above._")

    if not model:
        return (f"_No model selected for **{provider}**. "
                "Pick a model from the dropdown above._")

    # Pre-flight connection check before calling the model.
    precheck = check_llm_connection(provider=provider, model=model)
    if not precheck["ok"]:
        return (f"### 🤖 LLM Analysis\n**Provider:** `{provider}` · "
                f"**Model:** `{model}`\n\n"
                f"⚠️ **Connection issue:** {precheck['status']}")

    if not _latest_llm_ctx:
        return ("_No pipeline data available. **Run the pipeline first** "
                "from the left panel, then use this tab to analyse with "
                "different LLM providers/models._")

    ctx = dict(_latest_llm_ctx)
    if strategies:
        ctx["selected_strategies"] = strategies
    if indicators:
        ctx["selected_indicators"] = indicators

    result = generate_analysis(ctx, provider=provider, model=model)
    return f"### 🤖 LLM Analysis\n**Provider:** `{provider}` · **Model:** `{model}`\n\n{result}"


def run_pipeline(broker, symbol, exchange, client_id, access_token, sandbox,
                 kite_key, kite_token, binance_key, binance_secret,
                 binance_testnet, tv_webhook, tv_symmap,
                 equity, risk_pct, max_loss, rr, train_window, engine,
                 llm_provider, llm_models):
    """Full pipeline. Returns (status_md, trade_md, chart, indicators,
    data_df, llm_md)."""
    try:
        cfg = _build_config(broker, symbol, exchange, client_id, access_token,
                            sandbox, kite_key, kite_token, binance_key,
                            binance_secret, binance_testnet, tv_webhook,
                            tv_symmap, equity, risk_pct, max_loss, rr,
                            train_window)
        errors = cfg.validate()
        if errors:
            raise ValueError("; ".join(errors))

        client = _client_for(cfg)
        client.connect()
        df = client.fetch_intraday(
            cfg.symbol, cfg.exchange, days=cfg.lookback_days,
            interval_minutes=5, security_id=cfg.security_id,
            instrument_type=cfg.instrument_type)

        # 1. Feature engineering
        df = add_features(df, cfg)

        # 2. Time-series engine
        ts = run_time_series_engine(df)

        # 3. Quant ML engine
        ml = train_and_predict(df, engine=engine, train_window=cfg.train_window)

        # 4. Signal score
        last = df.iloc[-1]
        score = signal_score(ml["p_up"], ts["arima_return"],
                             ts["kalman_slope"], int(last.get("choch", 0)))

        # 5. Risk engine
        price = float(last["close"])
        plan = compute_trade_plan(price, float(last["atr"]) or 0.0, score,
                                  cfg.account_equity, cfg.risk_per_trade,
                                  cfg.max_risk_per_trade, cfg.rr_ratio)

        status = (
            f"### {cfg.symbol} · {cfg.exchange} · "
            f"{_broker_label(cfg.broker, cfg.sandbox, client)}\n"
            "| Component | Result |\n|---|---|\n"
            f"| Bars loaded | {len(df)} |\n"
            f"| ARIMA next-return | {ts['arima_return']:+.5f} |\n"
            f"| GARCH volatility | {ts['garch_vol']:.5f} |\n"
            f"| Kalman trend | {ts['kalman_trend']:.2f} (slope {ts['kalman_slope']:+.4f}) |\n"
            f"| P(UP) / P(DOWN) | {ml['p_up']:.2%} / {ml['p_down']:.2%} ({ml['model']}) |\n"
            f"| Signal | **{ml['signal']}** · score **{score:+.3f}** |\n")

        trade_md = ("**Risk Engine / Trade Plan**\n\n| Field | Value |\n|---|---|\n"
                    + "\n".join(f"| {k} | {v} |" for k, v in plan.items()))

        # 6. LLM analyst — runs over every selected model for the provider
        global _latest_llm_ctx
        last_row = df.iloc[-1]
        llm_ctx = {
            "symbol": cfg.symbol, "score": score, "signal": ml["signal"],
            "p_up": ml["p_up"], "p_down": ml["p_down"],
            "arima_return": ts["arima_return"], "garch_vol": ts["garch_vol"],
            "kalman_slope": ts["kalman_slope"], "plan": plan,
            "strategies": {
                "swing_high": int(last_row.get("swing_high", 0)),
                "swing_low": int(last_row.get("swing_low", 0)),
                "bos_up": int(last_row.get("bos_up", 0)),
                "bos_down": int(last_row.get("bos_down", 0)),
                "choch": int(last_row.get("choch", 0)),
                "higher_high": int(last_row.get("higher_high", 0)),
                "higher_low": int(last_row.get("higher_low", 0)),
                "lower_high": int(last_row.get("lower_high", 0)),
                "lower_low": int(last_row.get("lower_low", 0)),
                "support": round(float(last_row.get("support", 0)), 2),
                "resistance": round(float(last_row.get("resistance", 0)), 2),
                "pd_high": round(float(last_row.get("pd_high", 0)), 2),
                "pd_low": round(float(last_row.get("pd_low", 0)), 2),
                "or_high": round(float(last_row.get("or_high", 0)), 2),
                "or_low": round(float(last_row.get("or_low", 0)), 2),
                "breakout_up": int(last_row.get("breakout_up", 0)),
                "breakout_down": int(last_row.get("breakout_down", 0)),
                "retest_support": int(last_row.get("retest_support", 0)),
                "retest_resistance": int(last_row.get("retest_resistance", 0)),
                "bull_engulf": int(last_row.get("bull_engulf", 0)),
                "bear_engulf": int(last_row.get("bear_engulf", 0)),
                "rejection_up": int(last_row.get("rejection_up", 0)),
                "rejection_down": int(last_row.get("rejection_down", 0)),
                "pin_bar_bull": int(last_row.get("pin_bar_bull", 0)),
                "pin_bar_bear": int(last_row.get("pin_bar_bear", 0)),
                "close": round(float(last_row.get("close", 0)), 2),
            },
            "indicators": {
                "rsi": round(float(last_row.get("rsi", 0)), 2),
                "macd_line": round(float(last_row.get("macd_line", 0)), 4),
                "macd_signal": round(float(last_row.get("macd_signal", 0)), 4),
                "macd_hist": round(float(last_row.get("macd_hist", 0)), 4),
                "stoch_rsi_k": round(float(last_row.get("stoch_rsi_k", 0)), 2),
                "stoch_rsi_d": round(float(last_row.get("stoch_rsi_d", 0)), 2),
                "roc": round(float(last_row.get("roc", 0)), 4),
                "adx": round(float(last_row.get("adx", 0)), 2),
                "plus_di": round(float(last_row.get("plus_di", 0)), 2),
                "minus_di": round(float(last_row.get("minus_di", 0)), 2),
                "ema_200": round(float(last_row.get("ema_200", 0)), 2),
                "ema_21": round(float(last_row.get("ema_21", 0)), 2),
                "ema_9": round(float(last_row.get("ema_9", 0)), 2),
                "vwap": round(float(last_row.get("vwap", 0)), 2),
                "supertrend": round(float(last_row.get("supertrend", 0)), 2),
                "supertrend_dir": int(last_row.get("supertrend_dir", 0)),
            },
        }
        _latest_llm_ctx = llm_ctx

        active_provider = (llm_provider or "none").strip().lower()
        selected_models = [m for m in (llm_models or []) if str(m).strip()]
        if active_provider not in ("none", "", "off") and selected_models:
            parts = [f"**━━ Model: {m} ━━**\n\n"
                     f"{generate_analysis(llm_ctx, provider=active_provider, model=m)}"
                     for m in selected_models]
            llm_text = "\n\n".join(parts)
        else:
            llm_text = generate_analysis(llm_ctx, provider=active_provider)
        llm_md = f"**🤖 LLM Analysis**\n\n{llm_text}"

        chart = plot_candlestick(df.tail(120), title=f"{cfg.symbol} 5-min",
                                 ema_cols=["ema_169"], vwap_col="vwap")
        panel = plot_indicator_panel(df.tail(120), INDICATORS)
        data_md = dataframe_to_markdown(df[["close", "ema_169", "vwap", "rsi",
                                            "atr", "volume_ratio"]].tail(20))

        log.info("Pipeline complete for %s: %s", cfg.symbol, plan["action"])
        return status, trade_md, chart, panel, data_md, llm_md

    except Exception as e:
        log.exception("Pipeline failed")
        return f"**Error:** {e}", "—", None, None, "—", "—"


with gr.Blocks(title="Quant Trading — Multi-Broker 5-min Pipeline") as demo:
    gr.Markdown(
        "# 📈 Quant Trading System\n"
        "Multi-broker 5-min OHLCV (Dhan · Zerodha · Binance · TradingView) → "
        "Features (EMA169/ATR/VWAP/RSI/Vol ratio/Swings/BOS-CHoCH/Patterns/"
        "Momentum) → ARIMA + GARCH + Kalman → XGBoost/LightGBM P(UP) → "
        "Signal Score → Risk Engine → broker order")

    with gr.Row():
        with gr.Column(scale=1):
            broker = gr.Dropdown(BROKERS, value=_DEFAULT_BROKER, label="Broker")
            symbol = gr.Textbox(label="Symbol", value="RELIANCE")
            exchange = gr.Dropdown(["NSE", "BSE", "NSE_FNO", "MCX", "USDT"],
                                   value="NSE", label="Exchange",
                                   info="NSE/BSE/MCX for Dhan·Zerodha · USDT pair (e.g. BTCUSDT) for Binance")

            with gr.Accordion("Broker credentials", open=True):
                dhan_group = gr.Group(visible=(_DEFAULT_BROKER == "dhan"))
                with dhan_group:
                    gr.Markdown("**Dhan**")
                    with gr.Row():
                        client_id = gr.Textbox(label="Dhan Client ID", value="")
                        access_token = gr.Textbox(label="Dhan Access Token",
                                                  value="", type="password")
                zerodha_group = gr.Group(visible=(_DEFAULT_BROKER == "zerodha"))
                with zerodha_group:
                    gr.Markdown("**Zerodha (Kite Connect)**")
                    with gr.Row():
                        kite_key = gr.Textbox(label="Zerodha API Key", value="")
                        kite_token = gr.Textbox(label="Zerodha Access Token",
                                                value="", type="password")
                binance_group = gr.Group(visible=(_DEFAULT_BROKER == "binance"))
                with binance_group:
                    gr.Markdown("**Binance**")
                    with gr.Row():
                        binance_key = gr.Textbox(label="Binance API Key", value="")
                        binance_secret = gr.Textbox(label="Binance API Secret",
                                                    value="", type="password")
                    binance_testnet = gr.Checkbox(label="Binance testnet (paper)",
                                                  value=True)
                tv_group = gr.Group(visible=(_DEFAULT_BROKER == "tradingview"))
                with tv_group:
                    gr.Markdown("**TradingView**")
                    tv_webhook = gr.Textbox(
                        label="Webhook URL (order forward)",
                        value=os.environ.get("TV_WEBHOOK_URL", ""))
                    tv_symmap = gr.Textbox(
                        label="Symbol map (TV:YF, e.g. NIFTY:^NSEI)",
                        value=os.environ.get("TV_SYMBOL_MAP", ""))
            sandbox = gr.Checkbox(label="Sandbox / demo mode (use yfinance data)",
                                  value=True)
            conn_html = gr.HTML(
                '<div style="display:inline-block;padding:4px 10px;border-radius:12px;'
                'background:green;color:#fff;font-weight:bold;">● WAITING…</div>')
            engine = gr.Radio(["xgboost", "lightgbm"], value="xgboost",
                              label="ML Engine")
            llm_provider = gr.Dropdown(LLM_PROVIDERS, value=_DEFAULT_LLM,
                                       label="LLM Analyst provider")
            llm_models = gr.Dropdown(
                choices=LLM_MODEL_OPTIONS.get(_DEFAULT_LLM, []),
                value=([os.environ.get("LLM_MODEL", "").strip()]
                       if os.environ.get("LLM_MODEL", "").strip()
                       in LLM_MODEL_OPTIONS.get(_DEFAULT_LLM, [])
                       else ([LLM_MODEL_OPTIONS[_DEFAULT_LLM][0]]
                             if _DEFAULT_LLM in LLM_MODEL_OPTIONS
                             and LLM_MODEL_OPTIONS[_DEFAULT_LLM] else None)),
                multiselect=True,
                allow_custom_value=True,
                interactive=True,
                visible=True,
                label="LLM model(s) (select one or more)")
            with gr.Accordion("Risk settings", open=True):
                equity = gr.Number(label="Account Equity", value=100000)
                risk_pct = gr.Number(label="Risk per trade (%)", value=1.0)
                max_loss = gr.Number(label="Max loss per trade (₹)", value=5000)
                rr = gr.Number(label="Reward : Risk", value=2.0)
            train_window = gr.Slider(100, 1000, value=250, step=10,
                                     label="ML training window (bars)")
            with gr.Row():
                run_btn = gr.Button("Run Pipeline ▶", variant="primary")
                auto_run = gr.Checkbox(label="Auto-run every 5 min", value=False)
            auto_timer = gr.Timer(300, active=False)

        with gr.Column(scale=2):
            status_md = gr.Markdown("### Status\n_Press **Run Pipeline** to start._")
            trade_md = gr.Markdown("### Trade Plan\n—")
            with gr.Tab("Price chart"):
                chart = gr.Plot()
            with gr.Tab("Indicators"):
                panel = gr.Plot()
            with gr.Tab("Recent data"):
                data_md = gr.Markdown()
            with gr.Tab("LLM Analysis"):
                with gr.Row():
                    tab_llm_provider = gr.Dropdown(
                        LLM_PROVIDERS, value=_DEFAULT_LLM,
                        label="LLM Provider")
                    tab_llm_models = gr.Dropdown(
                        choices=LLM_MODEL_OPTIONS.get(_DEFAULT_LLM, []),
                        value=([os.environ.get("LLM_MODEL", "").strip()]
                               if os.environ.get("LLM_MODEL", "").strip()
                               in LLM_MODEL_OPTIONS.get(_DEFAULT_LLM, [])
                               else ([LLM_MODEL_OPTIONS[_DEFAULT_LLM][0]]
                                     if _DEFAULT_LLM in LLM_MODEL_OPTIONS
                                     and LLM_MODEL_OPTIONS[_DEFAULT_LLM] else None)),
                        multiselect=False,
                        allow_custom_value=True,
                        interactive=True,
                        visible=True,
                        label="Select LLM Model")
                tab_strategy = gr.Dropdown(
                    STRATEGY_OPTIONS,
                    value=None,
                    multiselect=True,
                    label="Select Strategies (for analysis focus)")
                tab_indicator = gr.Dropdown(
                    INDICATOR_OPTIONS,
                    value=None,
                    multiselect=True,
                    label="Select Indicators (for analysis focus)")
                tab_llm_status = gr.HTML(
                    '<div style="display:inline-block;padding:4px 10px;border-radius:12px;'
                    'background:gray;color:#fff;font-weight:bold;">● Checking LLM…</div>')
                tab_run_btn = gr.Button("Run LLM Analysis ▶", variant="primary")
                llm_md = gr.Markdown("_Select provider, model, strategies, and indicators, then click **Run LLM Analysis**._")

    _broker_inputs = [broker, symbol, exchange, client_id, access_token,
                      sandbox, kite_key, kite_token, binance_key, binance_secret,
                      binance_testnet, tv_webhook, tv_symmap]
    _run_inputs = (_broker_inputs
                   + [equity, risk_pct, max_loss, rr, train_window, engine,
                      llm_provider, llm_models])

    run_btn.click(
        run_pipeline,
        inputs=_run_inputs,
        outputs=[status_md, trade_md, chart, panel, data_md, llm_md])

    # Auto-run: enable/disable the 5-minute timer when checkbox changes.
    auto_run.change(
        lambda checked: gr.Timer(active=bool(checked)),
        inputs=auto_run,
        outputs=auto_timer)

    # Timer tick triggers the same pipeline run.
    auto_timer.tick(
        run_pipeline,
        inputs=_run_inputs,
        outputs=[status_md, trade_md, chart, panel, data_md, llm_md])

    # Show the models relevant to the selected LLM provider.
    llm_provider.change(
        update_llm_models,
        inputs=llm_provider,
        outputs=llm_models)

    # Tab LLM: update model choices when provider changes inside the tab.
    tab_llm_provider.change(
        update_tab_llm_models,
        inputs=tab_llm_provider,
        outputs=tab_llm_models)

    # Tab LLM: refresh connection status when provider/model changes.
    tab_llm_provider.change(
        check_tab_llm_connection,
        inputs=[tab_llm_provider, tab_llm_models],
        outputs=tab_llm_status)
    tab_llm_models.change(
        check_tab_llm_connection,
        inputs=[tab_llm_provider, tab_llm_models],
        outputs=tab_llm_status)

    # Tab LLM: run analysis with selected provider/model/strategies/indicators.
    tab_run_btn.click(
        run_tab_llm_analysis,
        inputs=[tab_llm_provider, tab_llm_models, tab_strategy, tab_indicator],
        outputs=llm_md)

    # Keep the connection status pill (green = connected, red = not) in sync.
    demo.load(
        check_connection,
        inputs=_broker_inputs,
        outputs=conn_html).then(
        check_tab_llm_connection,
        inputs=[tab_llm_provider, tab_llm_models],
        outputs=tab_llm_status)
    for _inp in (_broker_inputs
                 + [broker, kite_key, kite_token, binance_key, binance_secret,
                    binance_testnet, tv_webhook, tv_symmap]):
        _inp.change(
            check_connection,
            inputs=_broker_inputs,
            outputs=conn_html)

    # Refresh connection status when the broker selection changes too.
    broker.change(
        check_connection,
        inputs=_broker_inputs,
        outputs=conn_html)
    # Show only the selected broker's credentials.
    broker.change(
        update_broker_visibility,
        inputs=broker,
        outputs=[dhan_group, zerodha_group, binance_group, tv_group])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        quiet=True,
        show_error=True,
        inbrowser=False,
    )

