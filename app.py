"""
Gradio UI for the Quant Trading pipeline.

Pipeline:  Dhan 5-min OHLCV → Feature Engineering → Time-Series Engine
           → Quant ML Engine → Signal Score → Risk Engine → (Dhan order)
"""
from __future__ import annotations

import logging
import os

import gradio as gr
from gradio import HTML

from config import Config
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


def _build_config(symbol, exchange, client_id, access_token, sandbox,
                  equity, risk_pct, max_loss, rr, train_window):
    return Config(
        symbol=str(symbol).strip().upper(), exchange=exchange,
        client_id=client_id.strip(), access_token=access_token.strip(),
        sandbox=bool(sandbox), account_equity=float(equity),
        risk_per_trade=float(risk_pct) / 100.0, max_risk_per_trade=float(max_loss),
        rr_ratio=float(rr), train_window=int(train_window))


def check_connection(client_id, access_token, sandbox) -> str:
    """Return an HTML status pill: green dot when the Dhan broker is
    connected, red otherwise."""
    try:
        client = DhanDataClient(str(client_id).strip(),
                                str(access_token).strip(), sandbox=bool(sandbox))
        ok = client.connect()
        mode = "SANDBOX (yfinance)" if client.sandbox else "DHAN LIVE"
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


def run_pipeline(symbol, exchange, client_id, access_token, sandbox,
                 equity, risk_pct, max_loss, rr, train_window, engine,
                 llm_provider, llm_models):
    """Full pipeline. Returns (status_md, trade_md, chart, indicators,
    data_df, llm_md)."""
    try:
        cfg = _build_config(symbol, exchange, client_id, access_token, sandbox,
                            equity, risk_pct, max_loss, rr, train_window)
        errors = cfg.validate()
        if errors:
            raise ValueError("; ".join(errors))

        client = DhanDataClient(cfg.client_id, cfg.access_token, sandbox=cfg.sandbox)
        client.connect()
        df = client.fetch_intraday(
            cfg.symbol, cfg.exchange, cfg.security_id, cfg.instrument_type,
            days=cfg.lookback_days, interval_minutes=5)

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
            f"{'SANDBOX (yfinance)' if client.sandbox else 'LIVE DHAN'}\n"
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


with gr.Blocks(title="Quant Trading — Dhan 5-min Pipeline") as demo:
    gr.Markdown(
        "# 📈 Quant Trading System\n"
        "Dhan 5-min OHLCV → Features (EMA169/ATR/VWAP/RSI/Vol ratio/Swings/"
        "BOS-CHoCH/Patterns/Momentum) → ARIMA + GARCH + Kalman → "
        "XGBoost/LightGBM P(UP) → Signal Score → Risk Engine → Dhan order")

    with gr.Row():
        with gr.Column(scale=1):
            symbol = gr.Textbox(label="Symbol", value="RELIANCE")
            exchange = gr.Dropdown(["NSE", "BSE", "NSE_FNO", "MCX"],
                                   value="NSE", label="Exchange")
            with gr.Row():
                client_id = gr.Textbox(label="Dhan Client ID", value="")
                access_token = gr.Textbox(label="Dhan Access Token", value="",
                                          type="password")
            sandbox = gr.Checkbox(label="Sandbox mode (use yfinance data)",
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

    run_btn.click(
        run_pipeline,
        inputs=[symbol, exchange, client_id, access_token, sandbox,
                equity, risk_pct, max_loss, rr, train_window, engine,
                llm_provider, llm_models],
        outputs=[status_md, trade_md, chart, panel, data_md, llm_md])

    # Auto-run: enable/disable the 5-minute timer when checkbox changes.
    auto_run.change(
        lambda checked: gr.Timer(active=bool(checked)),
        inputs=auto_run,
        outputs=auto_timer)

    # Timer tick triggers the same pipeline run.
    auto_timer.tick(
        run_pipeline,
        inputs=[symbol, exchange, client_id, access_token, sandbox,
                equity, risk_pct, max_loss, rr, train_window, engine,
                llm_provider, llm_models],
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
        inputs=[client_id, access_token, sandbox],
        outputs=conn_html).then(
        check_tab_llm_connection,
        inputs=[tab_llm_provider, tab_llm_models],
        outputs=tab_llm_status)
    for _inp in (client_id, access_token, sandbox):
        _inp.change(
            check_connection,
            inputs=[client_id, access_token, sandbox],
            outputs=conn_html)

if __name__ == "__main__":
    demo.launch()

