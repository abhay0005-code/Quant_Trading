"""
LLM Analyst: post-processing commentary on the quant pipeline output.

Runs *after* the signal score / risk engine and asks a Large Language Model
to explain the trade decision in plain language.

Supported providers (configured via ``.env``):

  - ollama       (open-source, local)  e.g. llama3, mistral, qwen2
  - huggingface  (open-source, hosted) e.g. mistralai/Mistral-7B-Instruct-v0.3
  - openrouter   (many open + closed models) e.g. meta-llama/llama-3.1-8b
  - openai       (paid) e.g. gpt-4o, gpt-4o-mini
  - anthropic    (paid) e.g. claude-3-5-sonnet

Every provider degrades gracefully: if it is unconfigured, unreachable or
errors, ``generate_analysis`` returns a short fallback notice instead of
raising into the pipeline.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()  # reads .env in the working directory
except Exception:  # pragma: no cover - python-dotenv optional
    load_dotenv = None

log = logging.getLogger("llm_analyst")


# ──────────────────────────────────────────────────────────────────────
#  Env helpers
# ──────────────────────────────────────────────────────────────────────

def _get(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


# ──────────────────────────────────────────────────────────────────────
#  Prompt builder
# ──────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a senior quantitative trading analyst. Explain the following "
    "automated trading decision in concise, plain language for a retail "
    "trader. Give a clear bottom-line verdict, the key drivers considered, "
    "and the most important risks. Never promise profits and remind the "
    "user that markets carry risk."
)


def build_prompt(context: dict[str, Any]) -> str:
    """Turn a pipeline snapshot into a text prompt for the LLM."""
    sym = context.get("symbol", "?")
    score = context.get("score")
    signal = context.get("signal", "N/A")
    plan = context.get("plan", {})

    metrics = [
        f"P(UP)={context.get('p_up'):.1%}",
        f"P(DOWN)={context.get('p_down'):.1%}",
        f"ARIMA next-return={context.get('arima_return'):+.5f}",
        f"GARCH vol={context.get('garch_vol'):.5f}",
        f"Kalman slope={context.get('kalman_slope'):+.4f}",
        f"Composite signal score={score:+.3f}",
    ]

    plan_lines = "\n".join(f"  - {k}: {v}" for k, v in plan.items()) or "  - (no trade)"

    # Strategy data from the pipeline
    strategies = context.get("strategies", {})
    selected = context.get("selected_strategies", [])
    strat_lines = []

    if strategies:
        strat_lines.append("Strategy signals (current bar):")
        strat_map = {
            "Swing High / Swing Low": ("Swing High", strategies.get("swing_high", 0)),
            "BOS — Break of Structure": ("BOS Up/Down", f"{strategies.get('bos_up', 0)}/{strategies.get('bos_down', 0)}"),
            "CHoCH — Change of Character": ("CHoCH", strategies.get("choch", 0)),
            "Higher High / Higher Low": ("HH/HL", f"{strategies.get('higher_high', 0)}/{strategies.get('higher_low', 0)}"),
            "Lower High / Lower Low": ("LH/LL", f"{strategies.get('lower_high', 0)}/{strategies.get('lower_low', 0)}"),
            "Support/Resistance": ("S/R", f"Support={strategies.get('support', 'N/A')} Resistance={strategies.get('resistance', 'N/A')}"),
            "Previous Day High/Low": ("PD H/L", f"High={strategies.get('pd_high', 'N/A')} Low={strategies.get('pd_low', 'N/A')}"),
            "Opening Range High/Low": ("OR H/L", f"High={strategies.get('or_high', 'N/A')} Low={strategies.get('or_low', 'N/A')}"),
            "5-minute High/Low": ("5m H/L", f"High={strategies.get('bar_high', 'N/A')} Low={strategies.get('bar_low', 'N/A')}"),
            "Candle breakout": ("Breakout", f"Up={strategies.get('breakout_up', 0)} Down={strategies.get('breakout_down', 0)}"),
            "Retest": ("Retest", f"Support={strategies.get('retest_support', 0)} Resistance={strategies.get('retest_resistance', 0)}"),
            "Rejection candle": ("Rejection", f"Up={strategies.get('rejection_up', 0)} Down={strategies.get('rejection_down', 0)}"),
            "Engulfing candle": ("Engulfing", f"Bull={strategies.get('bull_engulf', 0)} Bear={strategies.get('bear_engulf', 0)}"),
            "Pin bar": ("Pin Bar", f"Bull={strategies.get('pin_bar_bull', 0)} Bear={strategies.get('pin_bar_bear', 0)}"),
        }

        if selected:
            strat_lines.append("  (User-selected focus strategies: " + ", ".join(selected) + ")")
            for s in selected:
                if s in strat_map:
                    label, val = strat_map[s]
                    strat_lines.append(f"  - {label}: {val}")
        else:
            for s, (label, val) in strat_map.items():
                strat_lines.append(f"  - {label}: {val}")

    strat_section = "\n".join(strat_lines) if strat_lines else "  - (no strategy data)"

    # Indicator data from the pipeline
    indicators = context.get("indicators", {})
    selected_ind = context.get("selected_indicators", [])
    ind_lines = []

    if indicators:
        ind_lines.append("Indicator values (current bar):")
        ind_map = {
            "RSI": ("RSI", indicators.get("rsi", "N/A")),
            "MACD": ("MACD", f"Line={indicators.get('macd_line', 'N/A')} Signal={indicators.get('macd_signal', 'N/A')} Hist={indicators.get('macd_hist', 'N/A')}"),
            "Stochastic RSI": ("Stoch RSI", f"K={indicators.get('stoch_rsi_k', 'N/A')} D={indicators.get('stoch_rsi_d', 'N/A')}"),
            "ROC": ("ROC", f"{indicators.get('roc', 'N/A')}%"),
            "ADX": ("ADX", f"{indicators.get('adx', 'N/A')} (+DI={indicators.get('plus_di', 'N/A')} -DI={indicators.get('minus_di', 'N/A')})"),
            "200 EMA": ("EMA 200", indicators.get("ema_200", "N/A")),
            "VWAP": ("VWAP", indicators.get("vwap", "N/A")),
            "Supertrend": ("Supertrend", f"{indicators.get('supertrend', 'N/A')} (Dir={'↑' if indicators.get('supertrend_dir', 0) == 1 else '↓'})"),
            "9 EMA": ("EMA 9", indicators.get("ema_9", "N/A")),
            "21 EMA": ("EMA 21", indicators.get("ema_21", "N/A")),
        }

        if selected_ind:
            ind_lines.append("  (User-selected focus indicators: " + ", ".join(selected_ind) + ")")
            for s in selected_ind:
                if s in ind_map:
                    label, val = ind_map[s]
                    ind_lines.append(f"  - {label}: {val}")
        else:
            for s, (label, val) in ind_map.items():
                ind_lines.append(f"  - {label}: {val}")

    ind_section = "\n".join(ind_lines) if ind_lines else "  - (no indicator data)"

    return (
        f"Trade decision for {sym} on a 5-minute timeframe.\n\n"
        f"Signal: {signal} · Composite score: {score:+.3f}\n\n"
        "Quant metrics:\n" + "\n".join(f"  - {m}" for m in metrics) +
        "\n\n" + strat_section +
        "\n\n" + ind_section +
        "\n\nRisk / trade plan:\n" + plan_lines +
        "\n\nPlease analyse this decision using the selected strategies and "
        "indicators, and give your final, user-facing recommendation and commentary."
    )

# ──────────────────────────────────────────────────────────────────────
#  Provider implementations (chat-completion style)
# ──────────────────────────────────────────────────────────────────────

def _ollama(model: str, system: str, prompt: str,
            cfg: dict[str, Any]) -> str:
    base = cfg.get("base_url", "http://localhost:11434").rstrip("/")
    options = {"temperature": cfg.get("temperature", 0.3),
               "num_predict": cfg.get("max_tokens", 512)}
    timeout = cfg.get("timeout", 120)

    # Preferred: /api/chat (required on Ollama >= 0.7 and for chat models
    # such as llama3). Falls back to /api/generate for older Ollama versions
    # or raw (non-chat) completion models.
    chat_payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
        "stream": False,
        "options": options,
    }
    r = requests.post(f"{base}/api/chat", json=chat_payload,
                      timeout=timeout)
    if r.status_code == 404:
        gen_payload = {
            "model": model, "system": system, "prompt": prompt,
            "stream": False, "options": options,
        }
        r = requests.post(f"{base}/api/generate", json=gen_payload,
                          timeout=timeout)
    r.raise_for_status()
    data = r.json()

    # /api/chat -> data["message"]["content"]; /api/generate -> data["response"]
    err = data.get("error")
    if err:
        raise RuntimeError(err)
    text = ""
    if isinstance(data, dict):
        msg = data.get("message")
        if isinstance(msg, dict):
            text = msg.get("content", "") or ""
        text = text or data.get("response", "") or ""
    return text.strip()


def _huggingface(model: str, system: str, prompt: str,
                 cfg: dict[str, Any]) -> str:
    api_key = cfg.get("api_key", "")
    url = cfg.get("base_url") or (
        f"https://api-inference.huggingface.co/models/{model}")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    payload = {
        "inputs": f"{system}\n\n{prompt}",
        "parameters": {"max_new_tokens": cfg.get("max_tokens", 512),
                       "temperature": cfg.get("temperature", 0.3)},
    }
    r = requests.post(url, json=payload, headers=headers,
                      timeout=cfg.get("timeout", 120))
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list) and data:
        text = data[0].get("generated_text", "")
    elif isinstance(data, dict):
        text = data.get("generated_text", data.get("error", ""))
    else:
        text = str(data)
    return text.strip()


def _openrouter(model: str, system: str, prompt: str,
                cfg: dict[str, Any]) -> str:
    headers = {"Authorization": f"Bearer {cfg['api_key']}",
               "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
        "temperature": cfg.get("temperature", 0.3),
        "max_tokens": cfg.get("max_tokens", 512),
    }
    r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                      json=payload, headers=headers,
                      timeout=cfg.get("timeout", 120))
    r.raise_for_status()
    return (r.json()["choices"][0]["message"]["content"] or "").strip()


def _openai(model: str, system: str, prompt: str,
            cfg: dict[str, Any]) -> str:
    headers = {"Authorization": f"Bearer {cfg['api_key']}",
               "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
        "temperature": cfg.get("temperature", 0.3),
        "max_tokens": cfg.get("max_tokens", 512),
    }
    r = requests.post("https://api.openai.com/v1/chat/completions",
                      json=payload, headers=headers,
                      timeout=cfg.get("timeout", 120))
    r.raise_for_status()
    return (r.json()["choices"][0]["message"]["content"] or "").strip()


def _anthropic(model: str, system: str, prompt: str,
               cfg: dict[str, Any]) -> str:
    headers = {
        "x-api-key": cfg["api_key"],
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "system": system,
        "max_tokens": cfg.get("max_tokens", 512),
        "temperature": cfg.get("temperature", 0.3),
        "messages": [{"role": "user", "content": prompt}],
    }
    r = requests.post("https://api.anthropic.com/v1/messages",
                      json=payload, headers=headers,
                      timeout=cfg.get("timeout", 120))
    r.raise_for_status()
    data = r.json()
    parts = [c.get("text", "") for c in data.get("content", [])
             if c.get("type") == "text"]
    return "".join(parts).strip()


_PROVIDERS = {
    "ollama": _ollama,
    "huggingface": _huggingface,
    "openrouter": _openrouter,
    "openai": _openai,
    "anthropic": _anthropic,
}


# Model suggestions per provider, shown in the UI's model selector. The
# list is not exhaustive — any valid model id can still be typed manually.
LLM_MODEL_OPTIONS: dict[str, list[str]] = {
    "ollama": [
        # Llama family
        "llama3", "llama3:70b", "llama3.1:8b", "llama3.1:70b",
        "llama3.2:1b", "llama3.2:3b", "llama2", "llama2:13b",
        # Mistral family
        "mistral", "mistral:7b", "mistral-nemo", "mixtral:8x7b",
        "codestral:22b", "dolphin-mistral",
        # Qwen family
        "qwen2:0.5b", "qwen2:1.5b", "qwen2:7b", "qwen2.5:7b", "qwen2.5-coder:7b",
        # Others
        "phi3:mini", "phi3:medium", "phi4-mini", "gemma:2b", "gemma:7b",
        "gemma2:2b", "gemma2:9b", "gemma2:27b", "gemma3:1b", "gemma3:4b",
        "llava:7b", "openhermes", "neural-chat", "starling-lm",
        "deepseek-coder:6.7b", "deepseek-r1:7b", "codegemma", "nomic-embed-text",
        "starcoder2:15b", "command-r", "aya", "bakllava",
    ],
    "huggingface": [
        # Mistral family
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Mistral-7B-v0.3",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "mistralai/Mixtral-8x22B-Instruct-v0.1",
        # Meta Llama family
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.1-405B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        # Google Gemma family
        "google/gemma-2-9b-it", "google/gemma-2-27b-it",
        "google/gemma-3-1b-it", "google/gemma-3-4b-it",
        # Qwen family
        "Qwen/Qwen2-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        # Microsoft Phi family
        "microsoft/phi-3-mini-4k-instruct",
        "microsoft/Phi-3-small-8k-instruct",
        "microsoft/Phi-3-medium-4k-instruct",
        "microsoft/Phi-4-instruct",
        # DeepSeek family
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        # Others
        "HuggingFaceH4/zephyr-7b-beta",
        "tiiuae/falcon-7b-instruct", "tiiuae/falcon-40b-instruct",
        "facebook/blenderbot-400M-distill",
        "NousResearch/Hermes-2-Pro-Mistral-7B",
        "NousResearch/Hermes-3-Llama-3.1-405B",
        "Intel/neural-chat-7b-v3-1",
        "Open-Orca/Mistral-7B-OpenOrca",
        "bigscience/bloomz-7b1",
        "CohereForAI/c4ai-command-r-plus",
    ],
    "openrouter": [
        # Meta Llama family
        "meta-llama/llama-3.1-8b-instruct",
        "meta-llama/llama-3.1-70b-instruct",
        "meta-llama/llama-3.1-405b-instruct",
        "meta-llama/llama-3.3-70b-instruct",
        # Mistral family
        "mistralai/mistral-7b-instruct",
        "mistralai/mistral-large",
        "mistralai/mistral-large-latest",
        "mistralai/mixtral-8x7b-instruct",
        "mistralai/mixtral-8x22b-instruct",
        # Google Gemini / Gemma
        "google/gemini-pro",
        "google/gemini-flash-1.5",
        "google/gemini-flash-2.0",
        "google/gemini-pro-1.5",
        "google/gemma-2-9b-it",
        "google/gemma-2-27b-it",
        # OpenAI (via OpenRouter)
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "openai/gpt-4-turbo",
        "openai/gpt-4.1",
        "openai/gpt-4.1-mini",
        "openai/o1",
        "openai/o1-mini",
        "openai/o3-mini",
        # Anthropic (via OpenRouter)
        "anthropic/claude-3.5-sonnet",
        "anthropic/claude-3.5-haiku",
        "anthropic/claude-3-opus",
        "anthropic/claude-3.7-sonnet",
        "anthropic/claude-3-haiku",
        # Qwen family
        "qwen/qwen-2.5-72b-instruct",
        "qwen/qwen-2.5-32b-instruct",
        "qwen/qwen-2.5-14b-instruct",
        # DeepSeek
        "deepseek/deepseek-chat",
        "deepseek/deepseek-r1",
        # Cohere
        "cohere/command-r-plus",
        "cohere/command-r",
        # Microsoft
        "microsoft/phi-3-medium-128k-instruct",
        "microsoft/phi-4",
        # Others
        "nousresearch/hermes-3-llama-3.1-405b",
        "nousresearch/hermes-3-llama-3.1-70b",
        "databricks/dbrx-instruct",
    ],
    "openai": [
        "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4",
        "gpt-3.5-turbo", "o1", "o1-mini", "o1-preview",
        "o3-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
        "chatgpt-4o-latest",
    ],
    "anthropic": [
        "claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20240620",
        "claude-3-5-haiku-20241022", "claude-3-opus-20240229",
        "claude-3-sonnet-20240229", "claude-3-haiku-20240307",
        "claude-3-7-sonnet-20250219",
    ],
}


# ──────────────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────────────

def check_llm_connection(provider: str | None = None, model: str | None = None) -> dict:
    """
    Pre-flight check before calling the LLM. Returns a dict:
      {"ok": bool, "status": str, "provider": str, "model": str}

    Verifies (in order):
      - provider is supported
      - model is selected
      - API key present (for key-required providers)
      - for Ollama: the Ollama server is reachable and has the model
      - for hosted providers: network reachability is not tested pre-call
        (config errors only surface at call time), so this returns ok=True
        once the key is present.
    """
    from urllib.parse import urljoin

    provider = (provider or _get("LLM_PROVIDER", "none")).strip().lower()
    model = (model or _get("LLM_MODEL", "")).strip()

    if provider in ("none", "off", "disabled", ""):
        return {"ok": False, "status": "No LLM provider selected.",
                "provider": provider, "model": model}

    if provider not in _PROVIDERS:
        return {"ok": False,
                "status": f"Unknown provider '{provider}'.",
                "provider": provider, "model": model}

    if not model:
        return {"ok": False,
                "status": f"No model selected for '{provider}'.",
                "provider": provider, "model": model}

    if provider == "ollama":
        base = _get("LLM_BASE_URL") or _get("OLLAMA_BASE_URL", "http://localhost:11434")
        base = base.rstrip("/")
        try:
            r = requests.get(f"{base}/api/tags", timeout=5)
            if r.status_code != 200:
                return {"ok": False,
                        "status": f"Ollama server not reachable at {base} (HTTP {r.status_code}).",
                        "provider": provider, "model": model}
            tags = r.json().get("models", [])
            names = [t.get("name", "") for t in tags]
            if not any(model in n or n in model for n in names):
                return {"ok": False,
                        "status": f"Model '{model}' not found in Ollama. "
                                  f"Pull it with: `ollama pull {model}`",
                        "provider": provider, "model": model}
            return {"ok": True, "status": "Ollama connected.",
                    "provider": provider, "model": model}
        except requests.RequestException as e:
            return {"ok": False,
                    "status": f"Ollama unreachable: {e}",
                    "provider": provider, "model": model}

    # Hosted providers: key check
    api_key = _get("LLM_API_KEY") or _get(provider.upper() + "_API_KEY")
    if provider in ("huggingface",) and not api_key:
        # HF works without a key on the free inference tier (rate-limited)
        return {"ok": True,
                "status": "HuggingFace (free tier, no API key set).",
                "provider": provider, "model": model}

    if provider in ("openrouter", "openai", "anthropic") and not api_key:
        return {"ok": False,
                "status": f"Missing API key for '{provider}'. Set "
                          f"`LLM_API_KEY` or `{provider.upper()}_API_KEY` in `.env`.",
                "provider": provider, "model": model}

    return {"ok": True,
            "status": f"Ready to call {provider}.",
            "provider": provider, "model": model}


def generate_analysis(context: dict[str, Any],
                      provider: str | None = None,
                      model: str | None = None) -> str:
    """
    Ask the configured LLM to analyse a pipeline snapshot.

    ``context`` should contain: symbol, score, signal, p_up, p_down,
    arima_return, garch_vol, kalman_slope and plan (dict).

    Provider + model come from ``.env`` unless overridden here. Returns:
      - the model's markdown/plain-text analysis, or
      - a concise fallback notice when no provider / key is configured.
    """
    provider = (provider or _get("LLM_PROVIDER", "none")).strip().lower()
    if provider in ("", "none", "off", "disabled"):
        return "_LLM commentary disabled (set `LLM_PROVIDER` in `.env`)._"

    fn = _PROVIDERS.get(provider)
    if fn is None:
        return (f"_Unknown LLM provider `{provider}`. Supported: "
                f"{', '.join(sorted(_PROVIDERS))}._")

    model = (model or _get("LLM_MODEL", "")).strip()
    if not model:
        return (f"_LLM provider `{provider}` is enabled but `LLM_MODEL` is "
                f"not set in `.env`._")

    api_key = _get("LLM_API_KEY") or _get(provider.upper() + "_API_KEY")
    if provider in ("openrouter", "openai", "anthropic") and not api_key:
        return (f"_LLM provider `{provider}` needs an API key in `.env` "
                f"(`LLM_API_KEY` or `{provider.upper()}_API_KEY`)._")

    cfg = {
        "api_key": api_key,
        "base_url": _get("LLM_BASE_URL") or _get(
            "OLLAMA_BASE_URL", "http://localhost:11434"),
        "temperature": _float_env("LLM_TEMPERATURE", 0.3),
        "max_tokens": int(_get("LLM_MAX_TOKENS", "512") or 512),
        "timeout": int(_get("LLM_TIMEOUT", "120") or 120),
    }

    try:
        system = _get("LLM_SYSTEM_PROMPT") or SYSTEM_PROMPT
        prompt = build_prompt(context)
        log.info("Calling LLM provider=%s model=%s", provider, model)
        return fn(model, system, prompt, cfg)
    except Exception as e:  # pragma: no cover - network/provider errors
        log.warning("LLM analysis failed (%s): %s", provider, e)
        return (f"_LLM analysis unavailable ({e}). Signal score and trade "
                f"plan above remain valid._")


def _float_env(key: str, default: float) -> float:
    raw = _get(key)
    try:
        return float(raw) if raw else default
    except ValueError:
        return default

