"""
phoebe_orion.py  –  PhoebeBot Financial Node  (Phoebe-Orion)

Headless Flask API.  No GUI, no camera, no voice, no TTS.
Runs on Orion (Raspberry Pi 5).  Exposes 0.0.0.0:5000.

Phoebe-Luna (thin GUI client) calls these endpoints for:
  - LLM inference  (Qwen on CPU, Mistral on GPU)
  - Speech-to-text  (Whisper tiny.en)
  - Stocks / portfolio  (Robinhood + Yahoo Finance)
  - News + FinBERT sentiment
  - Kalshi prediction markets
  - CEO watchlist
  - Reminder scheduling  (APScheduler)
  - Memory / user-fact persistence
"""

import threading
import re
import os
import time
import json
import base64
from datetime import datetime, timedelta
import pytz
import collections

import numpy as np
import requests
import ollama
from faster_whisper import WhisperModel
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
import robin_stocks.robinhood as rh
import feedparser
from transformers import pipeline as _hf_pipeline
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding as _asym_padding
from cryptography.hazmat.backends import default_backend
from flask import Flask, request, jsonify
import paho.mqtt.client as mqtt

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestTradeRequest
    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False
    print("[Alpaca] alpaca-py not installed — paper trading disabled. Run: pip install alpaca-py")

load_dotenv()

# ── MQTT ───────────────────────────────────────────────────────
_MQTT_BROKER   = os.getenv("MQTT_BROKER")
_MQTT_USER     = os.getenv("MQTT_USER", "phoebe")
_MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "")

_mqtt_ever_connected = False

_mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                            client_id="phoebe-orion", clean_session=False)
_mqtt_client.username_pw_set(_MQTT_USER, _MQTT_PASSWORD)
_mqtt_client.will_set("phoebe/orion/alive", "dead", retain=True)


def _mqtt_on_disconnect_orion(client, userdata, disconnect_flags, reason_code, properties=None):
    if not _mqtt_ever_connected:
        print("[MQTT] Auth failed — check MQTT_USER/MQTT_PASSWORD in .env.", flush=True)
        return
    def _reconnect():
        delay = 1
        while True:
            try:
                client.reconnect()
                client.publish("phoebe/orion/alive", "Orion online", retain=True, qos=1)
                print("[MQTT] Reconnected.", flush=True)
                break
            except Exception:
                time.sleep(delay)
                delay = min(delay * 2, 60)
    threading.Thread(target=_reconnect, daemon=True).start()


def _mqtt_on_connect_orion(client, userdata, flags, rc, properties=None):
    if str(rc) != "Success":
        print(f"[MQTT] Connection refused: {rc}", flush=True)
        return
    client.subscribe("phoebe/atlas/orion_request", qos=1)
    print("[MQTT] Subscribed to phoebe/atlas/orion_request.", flush=True)


def _handle_atlas_request(payload):
    """Dispatch an Atlas financial request and publish the result back."""
    req_id = payload.get("request_id", "")
    route  = payload.get("route", "")
    text   = payload.get("text", "")
    print(f"[Orion] Atlas request: {route} — {text[:60]}", flush=True)
    try:
        if route == "PORTFOLIO":
            result, _ = _handle_portfolio(text)
        elif route == "NEWS":
            result, _ = _handle_news()
        elif route == "TRADE_IDEA":
            result, _ = _handle_trade_idea(text)
        elif route == "WATCHLIST":
            result, _ = _handle_watchlist()
        else:
            result = f"Orion doesn't know how to handle '{route}'."
    except Exception as e:
        result = f"Orion error processing {route}: {e}"
        print(f"[Orion] Handler error: {e}", flush=True)
    _mqtt_client.publish("phoebe/orion/atlas_response", json.dumps({
        "request_id": req_id,
        "result": result,
    }), qos=1)
    print(f"[Orion] Responded to Atlas request {req_id[:8]}...", flush=True)


def _mqtt_on_message_orion(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except Exception:
        return
    if msg.topic == "phoebe/atlas/orion_request":
        threading.Thread(target=_handle_atlas_request, args=(payload,), daemon=True).start()


_mqtt_client.on_connect    = _mqtt_on_connect_orion
_mqtt_client.on_message    = _mqtt_on_message_orion
_mqtt_client.on_disconnect = _mqtt_on_disconnect_orion
try:
    _mqtt_client.connect(_MQTT_BROKER, 1883, 60)
    _mqtt_ever_connected = True
    _mqtt_client.publish("phoebe/orion/alive", "Orion online", retain=True, qos=1)
    _mqtt_client.loop_start()
    print("[MQTT] Orion online", flush=True)
except Exception as _mqtt_err:
    print(f"[MQTT] Could not connect: {_mqtt_err}", flush=True)


def _mqtt_heartbeat_orion():
    while True:
        time.sleep(30)
        try:
            _mqtt_client.publish("phoebe/orion/alive", "still here", retain=True, qos=1)
        except Exception:
            pass


threading.Thread(target=_mqtt_heartbeat_orion, daemon=True).start()

# ── Env ────────────────────────────────────────────────────────
ROBINHOOD_USER = os.getenv("ROBINHOOD_USER")
ROBINHOOD_PASS = os.getenv("ROBINHOOD_PASS")
KALSHI_ACCESS_KEY = os.getenv("KALSHI_ACCESS_KEY")
_rh_logged_in = False

# ── Alpaca paper trading ───────────────────────────────────────
ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")
if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    print("[Alpaca] ALPACA_API_KEY or ALPACA_SECRET_KEY missing from .env — paper trading disabled.")
if not ALPHA_VANTAGE_KEY:
    print("[AlphaVantage] ALPHA_VANTAGE_KEY missing from .env — Alpha Vantage calls will fail.")


def _rh_login():
    global _rh_logged_in
    if _rh_logged_in:
        return True
    if not ROBINHOOD_USER or not ROBINHOOD_PASS:
        print("[RH] No Robinhood credentials — skipping.")
        return False
    try:
        rh.login(ROBINHOOD_USER, ROBINHOOD_PASS)
        _rh_logged_in = True
        print("[RH] Logged in.")
        return True
    except Exception as exc:
        print(f"[RH] Login failed: {exc}")
        return False


# ── Models ─────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

OLLAMA_MODEL = "qwen3:8b"
OLLAMA_CHAT_MODEL = "mistral:7b"
_QWEN_OPTS = {"num_gpu": 0}
_MISTRAL_OPTS = {"num_gpu": 0}  # Pi 5 — CPU only, no discrete GPU
_bg_ollama_lock = threading.Lock()

_whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

_scheduler = BackgroundScheduler()

PHOEBE_SYSTEM_PROMPT = (
    "You are Phoebe, a quiet, caring personal companion. "
    "You are warm, gentle, and occasionally witty. "
    "You remember past conversations and refer to them naturally. "
    "Keep responses short. Say what you need to say, then stop. "
    "If something needs explaining, take the sentences you need — but never ramble, "
    "never pad, never repeat what the user already knows. "
    "Never mention being an AI or a language model. "
    "If the user has declined or postponed a topic, do not bring it up again unless they ask. "
    "Always answer or respond to what the user said first. "
    "If you want to check in, express concern, or ask a follow-up, do it after your answer — never lead with it. "
    "Speak as a close friend would — brief, direct, warm."
)

# ── Memory / user data ────────────────────────────────────────
MEMORY_FILE = os.path.join(_SCRIPT_DIR, "memory.json")
USER_FILE = os.path.join(_SCRIPT_DIR, "user.json")


def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {"conversations": [], "reminders": []}
    with open(MEMORY_FILE, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    if isinstance(data, list):
        return {"conversations": data, "reminders": []}
    if isinstance(data, dict):
        data.setdefault("conversations", [])
        data.setdefault("reminders", [])
        return data
    return {"conversations": [], "reminders": []}


def save_memory(data):
    with open(MEMORY_FILE, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, indent=2)


def _load_user_data():
    if os.path.exists(USER_FILE):
        try:
            with open(USER_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_user_data(data):
    with open(USER_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


# ── Module-level state (replaces PhoebeChat instance vars) ────
_memory_data = load_memory()
_conversations = _memory_data["conversations"]
_user_data_store = _load_user_data()
_user_name = _user_data_store.get("name")
_last_action = None
_last_trade_thought = None
_pending_clarification = None
_pending_price_conflict_state = None
_pending_alerts_queue = []
_user_mood = {"energy_avg": 0.0, "energy_peak": 0.0, "level": "normal"}


def _remember(user_input, response):
    _conversations.append([user_input, response])


def _save_memory_bg():
    threading.Thread(target=lambda: save_memory(_memory_data), daemon=True).start()


# ── Fact extraction (Qwen) ────────────────────────────────────
def _extract_user_facts(user_msg, current_facts):
    if len(user_msg.split()) < 3:
        return []
    known = "\n".join(f"- {f}" for f in current_facts) if current_facts else "None yet."
    try:
        result = ollama.chat(
            model=OLLAMA_MODEL, options=_QWEN_OPTS,
            messages=[{"role": "user", "content": (
                "You extract personal facts from conversation.\n"
                f'The user said: "{user_msg}"\n\n'
                f"Already known facts:\n{known}\n\n"
                "If the message reveals a NEW personal fact about the user "
                "(likes, dislikes, preferences, name, age, family, pets, "
                "favorite things, allergies, habits, job, hobbies, etc.), "
                "return ONLY the fact as a short sentence starting with 'User'. "
                "Multiple facts: one per line.\n"
                "If it updates an existing fact, return the updated version.\n"
                "If no personal fact is present, return exactly: NONE\n/no_think"
            )}],
        )
        text = _strip_think(result.message.content.strip())
        if not text or text.upper() == "NONE":
            return []
        return [l.strip().lstrip("- ") for l in text.split("\n")
                if l.strip() and l.strip().upper() != "NONE"]
    except Exception:
        return []


def _learn_from_input_bg(text):
    def _work():
        current_facts = _user_data_store.get("facts", [])
        new_facts = _extract_user_facts(text, current_facts)
        if not new_facts:
            return
        lower_existing = {f.lower() for f in current_facts}
        added = []
        for fact in new_facts:
            if fact.lower() not in lower_existing:
                current_facts.append(fact)
                lower_existing.add(fact.lower())
                added.append(fact)
        if added:
            _user_data_store["facts"] = current_facts
            _save_user_data(_user_data_store)
            print(f"[Learn] Saved: {added}")
    threading.Thread(target=_work, daemon=True).start()


# ── Time parsing (for reminders) ──────────────────────────────
WORD_TO_HOUR = {
    "one": 13, "two": 14, "three": 15, "four": 16, "five": 17,
    "six": 18, "seven": 19, "eight": 20, "nine": 21, "ten": 22,
    "eleven": 23, "twelve": 12, "noon": 12, "midnight": 0,
}
_WORD_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "twenty-one": 21, "twenty-two": 22,
    "twenty-three": 23, "twenty-four": 24, "twenty-five": 25,
    "twenty-six": 26, "twenty-seven": 27, "twenty-eight": 28,
    "twenty-nine": 29, "thirty": 30, "thirty-one": 31, "thirty-two": 32,
    "thirty-three": 33, "thirty-four": 34, "thirty-five": 35,
    "thirty-six": 36, "thirty-seven": 37, "thirty-eight": 38,
    "thirty-nine": 39, "forty": 40, "forty-one": 41, "forty-two": 42,
    "forty-three": 43, "forty-four": 44, "forty-five": 45,
    "forty-six": 46, "forty-seven": 47, "forty-eight": 48,
    "forty-nine": 49, "fifty": 50, "fifty-one": 51, "fifty-two": 52,
    "fifty-three": 53, "fifty-four": 54, "fifty-five": 55,
    "fifty-six": 56, "fifty-seven": 57, "fifty-eight": 58,
    "fifty-nine": 59,
}


def _words_to_time(text):
    parts = text.strip().split(None, 1)
    if not parts:
        return None
    hour_word = parts[0]
    if hour_word not in _WORD_TO_NUM:
        return None
    hour = _WORD_TO_NUM[hour_word]
    if hour < 1 or hour > 12:
        return None
    if len(parts) == 1:
        return None
    minute_word = parts[1].strip()
    if minute_word in _WORD_TO_NUM:
        minute = _WORD_TO_NUM[minute_word]
        if 0 <= minute <= 59:
            return (hour, minute)
    return None


def parse_clock_time(time_str, tomorrow=False):
    raw = time_str.strip().lower()
    am_pm = None
    cleaned = re.sub(r'\s*([ap])\.?m\.?\s*', r' \1m', raw).strip()
    if cleaned.endswith(" am") or cleaned.endswith(" pm"):
        am_pm = cleaned[-2:]
        cleaned = cleaned[:-3].strip()
    morning_match = re.search(r'\s+(in\s+the\s+)?morning\s*', cleaned)
    if morning_match:
        am_pm = "am"
        cleaned = cleaned[:morning_match.start()].strip()
    if cleaned in WORD_TO_HOUR:
        hour = WORD_TO_HOUR[cleaned]
        if am_pm == "am" and hour == 12:
            hour = 0
        elif am_pm == "am" and hour >= 13:
            hour -= 12
        elif am_pm is None and 1 <= hour <= 11:
            hour += 12
        elif am_pm == "pm" and hour < 12:
            hour += 12
        now = datetime.now()
        target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
        if tomorrow:
            target += timedelta(days=1)
        elif target <= now:
            target += timedelta(days=1)
        return target
    hour = minute = None
    m = re.match(r'^(\d{1,2}):(\d{2})', cleaned)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
    else:
        m = re.match(r'^(\d{3,4})', cleaned)
        if m:
            minute = int(m.group(1)[-2:])
            hour = int(m.group(1)[:-2])
        else:
            m = re.match(r'^(\d{1,2})', cleaned)
            if m:
                hour, minute = int(m.group(1)), 0
    if hour is not None and 0 <= hour <= 23 and 0 <= minute <= 59:
        if am_pm == "am" and hour == 12:
            hour = 0
        elif am_pm == "pm" and hour != 12:
            hour += 12
        elif am_pm is None and 1 <= hour <= 11:
            hour += 12
        now = datetime.now()
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if tomorrow:
            target += timedelta(days=1)
        elif target <= now:
            target += timedelta(days=1)
        return target
    parsed = _words_to_time(cleaned)
    if parsed:
        hour, minute = parsed
        if am_pm == "am" and hour == 12:
            hour = 0
        elif am_pm == "pm" and hour != 12:
            hour += 12
        elif am_pm is None and 1 <= hour <= 11:
            hour += 12
        now = datetime.now()
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if tomorrow:
            target += timedelta(days=1)
        elif target <= now:
            target += timedelta(days=1)
        return target
    return None


# ── Strip Qwen <think> tags ──────────────────────────────────
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(text):
    return _THINK_RE.sub("", text).strip()


def _fmt_pct(pct):
    return f"{int(abs(pct))}%" if pct == int(pct) else f"{abs(pct):.1f}%"


# ── FinBERT ───────────────────────────────────────────────────
_finbert_pipe = None
_finbert_ready = threading.Event()


def _load_finbert():
    global _finbert_pipe
    try:
        _finbert_pipe = _hf_pipeline("sentiment-analysis", model="ProsusAI/finbert", device=-1)
        print("[FinBERT] Model loaded.")
    except Exception as exc:
        print(f"[FinBERT] Load failed: {exc}")
    _finbert_ready.set()


threading.Thread(target=_load_finbert, daemon=True).start()


def _score_sentiment(text):
    if not _finbert_ready.is_set() or _finbert_pipe is None:
        return ("neutral", 0.0)
    try:
        result = _finbert_pipe(text[:512])[0]
        return (result["label"], round(result["score"], 3))
    except Exception:
        return ("neutral", 0.0)


_finbert_cache = {}
_FINBERT_CACHE_TTL = 3600


def _score_headlines(headlines):
    if not headlines:
        return []
    if not _finbert_ready.is_set() or _finbert_pipe is None:
        return [(h, "neutral", 0.0) for h in headlines]
    now = time.time()
    results = []
    to_score = []
    to_score_idx = []
    for i, h in enumerate(headlines):
        cached = _finbert_cache.get(h)
        if cached and now - cached[2] < _FINBERT_CACHE_TTL:
            results.append((h, cached[0], cached[1]))
        else:
            results.append(None)
            to_score.append(h)
            to_score_idx.append(i)
    if to_score:
        try:
            fresh = _finbert_pipe([h[:512] for h in to_score])
            for idx, h, r in zip(to_score_idx, to_score, fresh):
                lbl = r["label"]
                scr = round(r["score"], 3)
                _finbert_cache[h] = (lbl, scr, now)
                results[idx] = (h, lbl, scr)
        except Exception:
            for idx, h in zip(to_score_idx, to_score):
                results[idx] = (h, "neutral", 0.0)
    return results


def _format_scored_headlines(scored, ages=None):
    if not scored:
        return ""
    ages = ages or {}
    lines = []
    for headline, label, score in scored:
        tag = f"[{label.upper()} {score:.0%}]"
        age = ages.get(headline, -1)
        age_tag = f" ({age}m ago)" if age >= 0 else ""
        lines.append(f"- {tag}{age_tag} {headline}")
    return "\nRecent headlines (FinBERT sentiment + age):\n" + "\n".join(lines) + "\n"


# ── Price checking ────────────────────────────────────────────
_price_cache = {}
_PRICE_CACHE_TTL = 300
_price_source_pref = {}
_pending_price_conflicts = []


def _get_price_rh(ticker):
    if not _rh_login():
        return None
    try:
        quotes = rh.stocks.get_quotes(ticker)
        q = quotes[0] if quotes and quotes[0] else None
        if not q:
            return None
        price = float(q.get("last_trade_price") or q.get("last_extended_hours_trade_price") or 0)
        prev = float(q.get("adjusted_previous_close") or q.get("previous_close") or 0)
        if price <= 0:
            return None
        pct = ((price - prev) / prev * 100) if prev > 0 else 0.0
        volume = avg_volume = 0
        try:
            fundies = rh.stocks.get_fundamentals(ticker)
            if fundies and isinstance(fundies, list) and fundies[0]:
                f = fundies[0]
                volume = int(float(f.get("volume") or 0))
                avg_volume = int(float(f.get("average_volume") or f.get("average_volume_2_weeks") or 0))
        except Exception:
            pass
        rel_volume = round(volume / avg_volume, 2) if avg_volume > 0 else 0.0
        return {"price": price, "prev": prev, "pct": round(pct, 2),
                "volume": volume, "avg_volume": avg_volume, "rel_volume": rel_volume, "source": "rh"}
    except Exception:
        return None


def _get_price_yf(ticker):
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = t.fast_info
        price = float(info.last_price)
        prev = float(info.previous_close)
        if price <= 0:
            return None
        pct = ((price - prev) / prev * 100) if prev > 0 else 0.0
        volume = int(info.last_volume or 0)
        avg_volume = int(getattr(info, "three_month_average_volume", 0) or 0)
        rel_volume = round(volume / avg_volume, 2) if avg_volume > 0 else 0.0
        return {"price": price, "prev": prev, "pct": round(pct, 2),
                "volume": volume, "avg_volume": avg_volume, "rel_volume": rel_volume, "source": "yf"}
    except Exception:
        return None


def _get_price_change(ticker):
    global _pending_price_conflicts
    ticker = ticker.upper().strip()
    now = time.time()
    cached = _price_cache.get(ticker)
    if cached and now - cached["ts"] < _PRICE_CACHE_TTL:
        return cached
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        rh_fut = pool.submit(_get_price_rh, ticker)
        yf_fut = pool.submit(_get_price_yf, ticker)
        rh_data = rh_fut.result()
        yf_data = yf_fut.result()
    if not rh_data and not yf_data:
        return None
    if not rh_data or not yf_data:
        entry = rh_data or yf_data
        entry["ts"] = now
        entry["conflict"] = False
        _price_cache[ticker] = entry
        return entry
    rh_price = rh_data["price"]
    yf_price = yf_data["price"]
    mid = (rh_price + yf_price) / 2
    divergence = abs(rh_price - yf_price) / mid * 100 if mid > 0 else 0.0
    pref = _price_source_pref.get(ticker)
    if pref:
        entry = rh_data if pref == "rh" else yf_data
        entry["ts"] = now
        entry["rh_price"] = rh_price
        entry["yf_price"] = yf_price
        entry["conflict"] = False
        _price_cache[ticker] = entry
        return entry
    if divergence <= 0.5:
        entry = rh_data
        entry["ts"] = now
        entry["rh_price"] = rh_price
        entry["yf_price"] = yf_price
        entry["conflict"] = False
        _price_cache[ticker] = entry
        return entry
    entry = rh_data.copy()
    entry["ts"] = now
    entry["rh_price"] = rh_price
    entry["yf_price"] = yf_price
    entry["rh_pct"] = rh_data["pct"]
    entry["yf_pct"] = yf_data["pct"]
    entry["conflict"] = True
    entry["divergence"] = round(divergence, 2)
    _price_cache[ticker] = entry
    conflict = {"ticker": ticker, "rh_price": rh_price, "yf_price": yf_price,
                "rh_pct": rh_data["pct"], "yf_pct": yf_data["pct"], "divergence": round(divergence, 2)}
    _pending_price_conflicts.append(conflict)
    return entry


def _extract_tickers_from_headlines(scored):
    if not scored:
        return []
    feed = "\n".join(h for h, _, _ in scored[:10])
    try:
        result = ollama.chat(model=OLLAMA_MODEL, options=_QWEN_OPTS,
                             messages=[{"role": "user", "content": (
                                 f"Extract stock ticker symbols from these headlines:\n{feed}\n\n"
                                 "Return ONLY a comma-separated list of tickers (e.g. AAPL,TSLA,MSFT). "
                                 "Only real publicly traded tickers. If none, return: NONE\n/no_think"
                             )}])
        text = _strip_think(result.message.content.strip()).upper()
        if "NONE" in text:
            return []
        tickers = [t.strip() for t in text.replace("\n", ",").split(",")
                   if t.strip().isalpha() and 1 < len(t.strip()) <= 5]
        return list(dict.fromkeys(tickers))[:6]
    except Exception:
        return []


def _build_price_check(scored, ages=None, use_rh=True):
    tickers = _extract_tickers_from_headlines(scored)
    if not tickers:
        return ""
    lines = []
    for ticker in tickers:
        data = _get_price_change(ticker) if use_rh else _get_price_yf(ticker)
        if not data:
            continue
        price = data["price"]
        pct = data["pct"]
        vol = data["volume"]
        rel = data["rel_volume"]
        direction = "up" if pct > 0 else "down" if pct < 0 else "flat"
        vol_tag = ""
        if vol > 0:
            vol_str = f"{vol/1e6:.1f}M" if vol >= 1e6 else f"{vol/1e3:.0f}K"
            if rel >= 2.0:
                vol_tag = f", volume {vol_str} ({rel:.1f}x avg — UNUSUAL)"
            elif rel >= 1.3:
                vol_tag = f", volume {vol_str} ({rel:.1f}x avg — elevated)"
            elif rel > 0:
                vol_tag = f", volume {vol_str} ({rel:.1f}x avg)"
        if data.get("conflict"):
            rh_p = data.get("rh_price", 0)
            yf_p = data.get("yf_price", 0)
            div = data.get("divergence", 0)
            lines.append(f"- {ticker}: CONFLICTED — RH ${rh_p:.2f} vs YF ${yf_p:.2f} ({div:.1f}% gap){vol_tag}")
        else:
            src = data.get("source", "?")
            src_tag = f" [{src.upper()}]" if data.get("rh_price") and data.get("yf_price") else ""
            lines.append(f"- {ticker}: ${price:.2f} ({direction} {abs(pct):.1f}% today{vol_tag}){src_tag}")
    if ages:
        fresh = sum(1 for a in ages.values() if 0 <= a <= 60)
        stale = sum(1 for a in ages.values() if a > 360)
        if fresh or stale:
            lines.append(f"  [Freshness: {fresh} headlines <1hr old, {stale} headlines >6hr old]")
    if not lines:
        return ""
    return "\nActual market data (verify before calling anything a 'dip'):\n" + "\n".join(lines) + "\n"


# ── Alpaca paper trading functions ───────────────────────────
_TRADE_LOG_PATH = os.path.join(_SCRIPT_DIR, "orion_trades.log")


def _log_trade(ticker, side, qty, price, source, idea_text="", status="OK", order_id="", note=""):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fields = [ts, status, f"{side} {qty}x {ticker}", f"@ ${price:.2f}", f"[{source}]"]
    if order_id:
        fields.append(f"id={order_id}")
    if idea_text:
        fields.append(f"idea={idea_text}")
    if note:
        fields.append(f"note={note}")
    line = " | ".join(fields) + "\n"
    print(f"[AlpacaPaper] {line.strip()}")
    def _write():
        with open(_TRADE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)
    threading.Thread(target=_write, daemon=True).start()


_DECISION_LOG_PATH = os.path.join(_SCRIPT_DIR, "orion_decisions.jsonl")


def _log_decision(ticker, side, idea_text, signals=None):
    record = {
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": ticker,
        "side": side,
        "idea": idea_text,
        "signals": signals or {},
    }
    line = json.dumps(record) + "\n"
    def _write():
        with open(_DECISION_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)
    threading.Thread(target=_write, daemon=True).start()


def _get_price_alpaca(ticker):
    if not _ALPACA_AVAILABLE:
        return None
    try:
        data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        req = StockLatestTradeRequest(symbol_or_symbols=ticker)
        trade = data_client.get_stock_latest_trade(req)
        return float(trade[ticker].price)
    except Exception as exc:
        print(f"[Alpaca] Price fetch failed for {ticker}: {exc}")
        return None


def _get_price_alpha_vantage(ticker):
    try:
        url = (
            f"https://www.alphavantage.co/query"
            f"?function=GLOBAL_QUOTE&symbol={ticker}&apikey={ALPHA_VANTAGE_KEY}"
        )
        resp = requests.get(url, timeout=10)
        price = float(resp.json().get("Global Quote", {}).get("05. price", 0))
        return price if price > 0 else None
    except Exception as exc:
        print(f"[AlphaVantage] Price fetch failed for {ticker}: {exc}")
        return None


def _alpaca_get_price_with_fallback(ticker):
    price = _get_price_alpaca(ticker)
    if price is not None:
        return price
    print(f"[Alpaca] Falling back to Alpha Vantage for {ticker}")
    return _get_price_alpha_vantage(ticker)


def _extract_trade_signal(idea_text):
    """Use Qwen (CPU) to pull TICKER|BUY or TICKER|SELL from a free-form idea string."""
    try:
        result = ollama.chat(
            model=OLLAMA_MODEL, options=_QWEN_OPTS,
            messages=[{"role": "user", "content": (
                f"Extract the trade signal from this idea:\n\"{idea_text}\"\n\n"
                "Reply with exactly: TICKER|BUY or TICKER|SELL\n"
                "Real tickers only (1-5 alpha chars). If no clear ticker or direction: NONE\n/no_think"
            )}],
        )
        text = _strip_think(result.message.content.strip()).upper()
        if text == "NONE" or "|" not in text:
            return None, None
        ticker, side = text.split("|", 1)
        ticker = ticker.strip()
        side = side.strip()
        if not ticker.isalpha() or not (1 < len(ticker) <= 5):
            return None, None
        if side not in ("BUY", "SELL"):
            return None, None
        return ticker, side.lower()
    except Exception:
        return None, None


def _alpaca_place_paper_order(ticker, side, idea_text="", qty=1):
    """Place a GTC paper market order. Always spawns its own thread — never call on main thread."""
    if not _ALPACA_AVAILABLE:
        return

    def _do_order():
        price = _alpaca_get_price_with_fallback(ticker)
        try:
            client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
            try:
                asset = client.get_asset(ticker)
                if not asset.tradable:
                    _log_trade(ticker, side.upper(), qty, price or 0.0, "alpaca_paper",
                               idea_text, status="ASSET_NOT_TRADABLE", note="asset exists but not tradable on Alpaca")
                    return
            except Exception as asset_exc:
                _log_trade(ticker, side.upper(), qty, price or 0.0, "alpaca_paper",
                           idea_text, status="ASSET_NOT_TRADABLE", note=str(asset_exc)[:120])
                return
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            req = MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.GTC,
            )
            order = client.submit_order(req)
            _log_trade(ticker, side.upper(), qty, price or 0.0, "alpaca_paper",
                       idea_text, status="OK", order_id=str(order.id))
        except Exception as exc:
            err = str(exc)
            if any(k in err.lower() for k in ("403", "not tradable", "market is closed",
                                               "outside market hours", "not authorized")):
                status = "ORDER_REJECTED_CLOSED"
            elif any(k in err.lower() for k in ("insufficient", "no position",
                                                 "short selling", "cannot sell")):
                status = "SELL_FAILED_NO_POSITION"
            else:
                status = "ORDER_FAILED"
            _log_trade(ticker, side.upper(), qty, price or 0.0, "alpaca_paper",
                       idea_text, status=status, note=err[:120])

    threading.Thread(target=_do_order, daemon=True).start()


def _alpaca_act_on_idea(idea_text, signals=None):
    """Extract signal from idea text and place a paper order. Always called in a background thread."""
    ticker, side = _extract_trade_signal(idea_text)
    if not ticker or not side:
        print(f"[AlpacaPaper] No clear signal in: {idea_text[:60]}")
        return
    _log_decision(ticker, side, idea_text, signals)
    _alpaca_place_paper_order(ticker, side, idea_text=idea_text)


# ── Kalshi ────────────────────────────────────────────────────
_KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
_KALSHI_KEY_PATH = os.path.join(_SCRIPT_DIR, "kalshi_private.key")
_kalshi_private_key = None
if os.path.isfile(_KALSHI_KEY_PATH):
    try:
        with open(_KALSHI_KEY_PATH, "rb") as _kf:
            _kalshi_private_key = serialization.load_pem_private_key(_kf.read(), password=None, backend=default_backend())
        print("[Kalshi] Private key loaded.")
    except Exception as _exc:
        print(f"[Kalshi] Failed to load private key: {_exc}")
else:
    print("[Kalshi] No private key found — using public endpoints only.")

_KALSHI_ECON_RE = re.compile(
    r'recession|fed\b.*rate|inflation|cpi\b|gdp\b|unemploy|treasury|'
    r'bond|s.?p\s*500|dow|nasdaq|interest\s*rate|tariff|debt\s*ceiling|'
    r'government\s*shutdown|default|rate\s*cut|rate\s*hike|nonfarm|'
    r'oil\s*price|gas\s*price|wage|housing|mortgage|bankruptcy|layoff',
    re.IGNORECASE)

_bg_kalshi = {"markets": {}, "prev": {}, "shifts": [], "ts": 0.0}
_pending_kalshi_alerts = []


def _kalshi_sign(method, path):
    if not _kalshi_private_key or not KALSHI_ACCESS_KEY:
        return None, None
    timestamp_ms = str(int(time.time() * 1000))
    path_clean = path.split("?")[0]
    message = (timestamp_ms + method.upper() + path_clean).encode("utf-8")
    signature = _kalshi_private_key.sign(
        message,
        _asym_padding.PSS(mgf=_asym_padding.MGF1(hashes.SHA256()), salt_length=_asym_padding.PSS.DIGEST_LENGTH),
        hashes.SHA256())
    return timestamp_ms, base64.b64encode(signature).decode("utf-8")


def _kalshi_headers(method="GET", path="/trade-api/v2/markets"):
    headers = {"Accept": "application/json"}
    ts, sig = _kalshi_sign(method, path)
    if ts and sig:
        headers["KALSHI-ACCESS-KEY"] = KALSHI_ACCESS_KEY
        headers["KALSHI-ACCESS-TIMESTAMP"] = ts
        headers["KALSHI-ACCESS-SIGNATURE"] = sig
    return headers


def _kalshi_fetch_markets():
    markets = {}
    cursor = None
    api_path = "/trade-api/v2/markets"
    for _ in range(3):
        params = {"limit": 200, "status": "open"}
        if cursor:
            params["cursor"] = cursor
        try:
            resp = requests.get(f"{_KALSHI_BASE}/markets", params=params,
                                headers=_kalshi_headers("GET", api_path), timeout=12)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            break
        for m in data.get("markets", []):
            ticker = m.get("ticker", "")
            title = m.get("title") or m.get("yes_sub_title") or m.get("event_ticker") or ticker
            vol_24h = float(m.get("volume_24h_fp") or 0)
            if not _KALSHI_ECON_RE.search(title) and vol_24h < 500:
                continue
            yes_bid = float(m.get("yes_bid_dollars") or 0)
            yes_ask = float(m.get("yes_ask_dollars") or 0)
            no_bid = float(m.get("no_bid_dollars") or 0)
            no_ask = float(m.get("no_ask_dollars") or 0)
            yes_price = (yes_bid + yes_ask) / 2 if (yes_bid + yes_ask) > 0 else float(m.get("last_price_dollars") or 0)
            no_price = (no_bid + no_ask) / 2 if (no_bid + no_ask) > 0 else (1 - yes_price)
            markets[ticker] = {
                "title": title, "yes_price": round(yes_price, 4), "no_price": round(no_price, 4),
                "volume": int(float(m.get("volume_fp") or 0)), "volume_24h": int(vol_24h),
                "last_price": float(m.get("last_price_dollars") or 0),
                "updated": m.get("updated_time", ""), "event_ticker": m.get("event_ticker", ""),
            }
        cursor = data.get("cursor")
        if not cursor:
            break
    return markets


def _kalshi_detect_shifts(current, prev):
    shifts = []
    for ticker, m in current.items():
        old = prev.get(ticker)
        if not old:
            continue
        delta_pts = round((m["yes_price"] - old["yes_price"]) * 100, 1)
        if abs(delta_pts) >= 5:
            shifts.append({"ticker": ticker, "title": m["title"],
                           "old_pct": round(old["yes_price"] * 100, 1),
                           "new_pct": round(m["yes_price"] * 100, 1),
                           "delta": delta_pts, "direction": "up" if delta_pts > 0 else "down",
                           "volume_24h": m["volume_24h"]})
    return shifts


def _kalshi_cross_reference(shift):
    scored = _bg_news.get("scored", [])[:10]
    ages = _bg_news.get("ages", {})
    news_block = ""
    if scored:
        parts = []
        for h, lbl, scr in scored[:6]:
            age = ages.get(h, -1)
            age_tag = f" ({age}m ago)" if age >= 0 else ""
            parts.append(f"- [{lbl.upper()} {scr:.0%}]{age_tag} {h}")
        news_block = "\n".join(parts)
    price_block = _build_price_check(scored, ages, use_rh=False) if scored else ""
    title = shift["title"]
    old_pct = shift["old_pct"]
    new_pct = shift["new_pct"]
    direction = shift["direction"]
    vol24 = shift["volume_24h"]
    try:
        result = ollama.chat(model=OLLAMA_MODEL, options=_QWEN_OPTS, messages=[
            {"role": "system", "content": (
                "You are a market observer. No advice. No 'buy' or 'sell'. Just facts.\n"
                "Cross-reference prediction market odds against real data.\n"
                "Two to three sentences max. Casual but factual. No disclaimers.")},
            {"role": "user", "content": (
                f"Kalshi prediction market \"{title}\" just moved {direction} "
                f"from {old_pct:.0f}% to {new_pct:.0f}% (24h volume: {vol24} contracts).\n\n"
                f"Headlines (FinBERT scored):\n{news_block}\n{price_block}\n"
                "Does the rest of the market agree? Cross-reference. /no_think")}])
        return _strip_think(result.message.content.strip())
    except Exception:
        return f"Kalshi's \"{title}\" just moved {direction} to {new_pct:.0f}%. Was {old_pct:.0f}%."


def _format_kalshi_context():
    markets = _bg_kalshi.get("markets", {})
    if not markets:
        return ""
    ranked = sorted(markets.values(), key=lambda m: m["volume_24h"], reverse=True)[:6]
    if not ranked:
        return ""
    lines = []
    for m in ranked:
        lines.append(f"- {m['title']}: {m['yes_price']*100:.0f}% yes (24h vol: {m['volume_24h']})")
    shifts = _bg_kalshi.get("shifts", [])
    if shifts:
        for s in shifts[:3]:
            lines.append(f"  ** SHIFTED: {s['title']} moved {s['direction']} from {s['old_pct']:.0f}% to {s['new_pct']:.0f}%")
    return "\nKalshi prediction markets (real-money odds):\n" + "\n".join(lines) + "\n"


# ── Portfolio ─────────────────────────────────────────────────
_bg_portfolio = {"positions": None, "overall_pct": 0.0, "summary": "Portfolio not loaded yet.", "ts": 0.0}


def _fetch_portfolio_data():
    if not _rh_login():
        return None, 0.0
    _t0 = time.time()
    positions = []
    try:
        for ticker, info in rh.build_holdings().items():
            pct = float(info.get("percent_change", 0))
            positions.append({"label": ticker, "type": "stock", "pct": round(pct, 1)})
    except Exception as e:
        print(f"[Portfolio] Stocks: {e}")
    try:
        for pos in rh.crypto.get_crypto_positions():
            qty = float(pos.get("quantity", 0))
            if qty <= 0:
                continue
            code = pos.get("currency", {}).get("code", "???")
            cost = sum(float(cb.get("direct_cost_basis", 0)) for cb in pos.get("cost_bases", []))
            pct = 0.0
            try:
                mark = float(rh.crypto.get_crypto_quote(code).get("mark_price", 0))
                if cost > 0:
                    pct = ((qty * mark) - cost) / cost * 100
            except Exception:
                pass
            positions.append({"label": code, "type": "crypto", "pct": round(pct, 1)})
    except Exception as e:
        print(f"[Portfolio] Crypto: {e}")
    try:
        for pos in rh.options.get_open_option_positions():
            qty = float(pos.get("quantity", 0))
            if qty <= 0:
                continue
            chain = pos.get("chain_symbol", "???")
            opt_type = pos.get("type", "call")
            strike = float(pos.get("strike_price", 0))
            expiry = pos.get("expiration_date", "")
            avg = float(pos.get("average_price", 0)) / 100
            pct = 0.0
            try:
                opt_url = pos.get("option", "")
                if opt_url:
                    md = rh.options.get_option_market_data_by_id(opt_url.rstrip("/").split("/")[-1])
                    if md:
                        mark = float(md.get("adjusted_mark_price", 0))
                        if avg > 0:
                            pct = ((mark - avg) / avg) * 100
            except Exception:
                pass
            label = f"{chain} {opt_type} ${strike:.0f} {expiry}"
            positions.append({"label": label, "type": opt_type, "pct": round(pct, 1)})
    except Exception as e:
        print(f"[Portfolio] Options: {e}")
    overall_pct = 0.0
    try:
        profile = rh.profiles.load_portfolio_profile()
        equity = float(profile.get("equity", 0))
        prev = float(profile.get("equity_previous_close", 0))
        if prev > 0:
            overall_pct = (equity - prev) / prev * 100
    except Exception:
        pass
    dt = time.time() - _t0
    print(f"[Portfolio] {len(positions)} positions ({dt:.1f}s)")
    return positions, round(overall_pct, 1)


def _format_portfolio(positions, overall_pct, full=False):
    if not positions:
        return "Your portfolio is empty."
    parts = []
    for p in positions:
        pct = p["pct"]
        if not full and abs(pct) < 0.1:
            continue
        label = p["label"]
        if abs(pct) < 0.1:
            parts.append(f"{label}'s flat")
        else:
            direction = "up" if pct > 0 else "down"
            parts.append(f"{label}'s {direction} {_fmt_pct(pct)}")
    if not parts:
        return "Nothing moved today."
    summary = ", ".join(parts)
    if abs(overall_pct) >= 0.1:
        color = "green" if overall_pct > 0 else "red"
        summary += f", overall you're {color} {_fmt_pct(overall_pct)}"
    return summary + "."


def _refresh_portfolio():
    global _bg_portfolio
    t0 = time.time()
    try:
        positions, overall_pct = _fetch_portfolio_data()
        summary = _format_portfolio(positions, overall_pct, full=False)
        _bg_portfolio = {"positions": positions, "overall_pct": overall_pct, "summary": summary, "ts": time.time()}
        print(f"[Portfolio] Refreshed — {len(positions)} positions ({time.time()-t0:.1f}s)")
    except Exception as exc:
        print(f"[Portfolio] Error: {exc}")


# ── Trade ideas ───────────────────────────────────────────────
_pending_news_ideas = []
_shown_news_ideas = set()


def _fetch_trade_idea(portfolio_summary):
    scored = _bg_news.get("scored", [])[:10]
    ages = _bg_news.get("ages", {})
    news_section = _format_scored_headlines(scored, ages)
    if not news_section:
        headlines = _bg_news.get("headlines", [])[:10]
        if headlines:
            news_section = "\nRecent headlines:\n" + "\n".join(f"- {h}" for h in headlines) + "\n"
    price_check = _build_price_check(scored, ages) if scored else ""
    kalshi_section = _format_kalshi_context()
    context = ""
    if portfolio_summary:
        context += f"Portfolio:\n{portfolio_summary}\n"
    context += news_section + price_check + kalshi_section
    if not context.strip():
        return "I need your portfolio or some news first.", {}
    signals = {
        "source": "manual",
        "portfolio_summary": portfolio_summary[:500] if portfolio_summary else "",
        "scored_headlines": [{"headline": h, "label": l, "score": round(s, 3)} for h, l, s in scored[:5]],
        "price_check": price_check,
        "kalshi_context": kalshi_section,
    }
    try:
        result = ollama.chat(model=OLLAMA_MODEL, options=_QWEN_OPTS, messages=[
            {"role": "system", "content": (
                "You are a blunt trading advisor. Use real numbers. No hedge words. "
                "Sentiment is mood. Price is reality. Volume is conviction. Kalshi is the crowd bet. Use all four.")},
            {"role": "user", "content": (
                f"{context}\nGive exactly ONE trade suggestion. Be specific — name the ticker. "
                "Unusual volume confirms moves. Stale headlines are priced in. "
                "Two to three sentences max. /no_think")}])
        return _strip_think(result.message.content.strip()), signals
    except Exception:
        return "Couldn't generate a trade idea right now.", signals


def _extract_news_ideas(headlines, scored=None, ages=None):
    if not headlines:
        return []
    ages = ages or {}
    if scored:
        parts = []
        for h, lbl, scr in scored[:10]:
            age = ages.get(h, -1)
            age_tag = f" ({age}m ago)" if age >= 0 else ""
            parts.append(f"- [{lbl.upper()} {scr:.0%}]{age_tag} {h}")
        feed = "\n".join(parts)
        price_check = _build_price_check(scored, ages, use_rh=False)
    else:
        feed = "\n".join(f"- {h}" for h in headlines[:10])
        price_check = ""
    kalshi_section = _format_kalshi_context()
    try:
        result = ollama.chat(model=OLLAMA_MODEL, options=_QWEN_OPTS, messages=[
            {"role": "system", "content": (
                "You scan news for trade opportunities. Casual tone. No pressure. "
                "Name the ticker. One sentence each.")},
            {"role": "user", "content": (
                f"Headlines:\n{feed}\n{price_check}{kalshi_section}\n"
                "Pick up to 2 tickers. If nothing, return: NONE\n/no_think")}])
        text = _strip_think(result.message.content.strip())
        if not text or "NONE" in text.upper():
            return []
        return [l.strip().lstrip("- \u2022*") for l in text.split("\n")
                if l.strip() and "NONE" not in l.upper() and len(l.strip()) > 5][:2]
    except Exception:
        return []


# ── CEO watchlist ─────────────────────────────────────────────
_CEO_WL_PATH = os.path.join(_SCRIPT_DIR, "ceo_watchlist.json")
_ceo_watchlist = []
_ceo_headlines_seen = set()


def _load_ceo_watchlist():
    if os.path.exists(_CEO_WL_PATH):
        try:
            with open(_CEO_WL_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get("people", []), data.get("ts", 0.0)
        except Exception:
            pass
    return [], 0.0


def _save_ceo_watchlist(wl):
    with open(_CEO_WL_PATH, 'w', encoding='utf-8') as f:
        json.dump({"people": wl, "ts": time.time()}, f, indent=2)


def _ollama_quick(prompt):
    try:
        r = ollama.chat(model=OLLAMA_MODEL, options=_QWEN_OPTS,
                        messages=[{"role": "user", "content": prompt + " /no_think"}])
        return _strip_think(r.message.content.strip())
    except Exception:
        return ""


def _wiki_ceo_lookup(company_name):
    try:
        resp = requests.get("https://en.wikipedia.org/w/api.php", params={
            "action": "query", "list": "search", "srsearch": company_name, "format": "json", "srlimit": 3}, timeout=10)
        for hit in resp.json().get("query", {}).get("search", []):
            resp2 = requests.get("https://en.wikipedia.org/w/api.php", params={
                "action": "parse", "page": hit["title"], "prop": "wikitext", "format": "json"}, timeout=10)
            wt = resp2.json().get("parse", {}).get("wikitext", {}).get("*", "")
            for pat in [r'\|\s*CEO\s*=\s*(.+)', r'\|\s*key_people\s*=\s*(.+)', r'\|\s*leader_name\s*=\s*(.+)']:
                m = re.search(pat, wt, re.IGNORECASE)
                if m:
                    raw = m.group(1).split('\n')[0].split('|')[0]
                    name = re.sub(r'\[\[([^\]|]+?)(?:\|[^\]]+)?\]\]', r'\1', raw)
                    name = re.sub(r'\{\{[^}]*\}\}', '', name)
                    name = re.sub(r'<[^>]+>', '', name)
                    name = re.sub(r'\([^)]*\)', '', name)
                    name = name.strip().strip(',').strip()
                    if name and len(name) > 2 and not name.startswith('{'):
                        return name
    except Exception:
        pass
    return None


def _discover_ceo(ticker):
    ceo = _wiki_ceo_lookup(ticker)
    if ceo:
        return ceo
    text = _ollama_quick(f"Who is the current CEO of the company with stock ticker {ticker}? Reply with ONLY the full name.")
    name = text.split('\n')[0].strip().strip('"\'.')
    return name if name and len(name) > 2 and ' ' in name else None


def _discover_company(ticker):
    text = _ollama_quick(f"What company has the stock ticker {ticker}? Reply with ONLY the company name.")
    return text.split('\n')[0].strip().strip('"\'.')  or ticker


def _discover_twitter(person_name):
    text = _ollama_quick(f"What is {person_name}'s Twitter or X handle? Reply with ONLY the @handle. If unknown, say NONE.")
    handle = text.split('\n')[0].strip().strip('"\'.')
    if handle.startswith('@') and len(handle) > 2 and ' ' not in handle:
        return handle
    return None


_ALWAYS_WATCH = [
    "Elon Musk", "Warren Buffett", "Jamie Dimon", "Larry Fink", "Tim Cook",
    "Satya Nadella", "Sundar Pichai", "Mark Zuckerberg", "Jeff Bezos", "Carl Icahn",
]


def _build_ceo_watchlist(stock_tickers):
    global _ceo_watchlist
    existing_names = {e["name"].lower() for e in _ceo_watchlist}
    existing_tickers = {e.get("ticker", "").upper() for e in _ceo_watchlist if e.get("ticker")}
    watchlist = list(_ceo_watchlist)
    for ticker in stock_tickers:
        if ticker.upper() in existing_tickers:
            continue
        ceo = _discover_ceo(ticker)
        if not ceo or ceo.lower() in existing_names:
            continue
        company = _discover_company(ticker)
        twitter = _discover_twitter(ceo)
        entry = {"name": ceo, "company": company, "ticker": ticker, "twitter": twitter, "source": "portfolio"}
        watchlist.append(entry)
        existing_names.add(ceo.lower())
        existing_tickers.add(ticker.upper())
        print(f"[CEO] Discovered: {ceo} ({company}/{ticker}) {twitter or ''}")
        time.sleep(1)
    for name in _ALWAYS_WATCH:
        if name.lower() in existing_names:
            continue
        twitter = _discover_twitter(name)
        entry = {"name": name, "company": "", "ticker": "", "twitter": twitter, "source": "market_mover"}
        watchlist.append(entry)
        existing_names.add(name.lower())
        print(f"[CEO] Mover: {name} {twitter or ''}")
        time.sleep(1)
    _ceo_watchlist = watchlist
    _save_ceo_watchlist(watchlist)
    return watchlist


_CEO_NEWS_RSS = "https://news.google.com/rss/search?q={query}+when:1d&hl=en-US&gl=US&ceid=US:en"


def _fetch_ceo_news(watchlist):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _check_one(person):
        parts = [person["name"].replace(' ', '+')]
        if person.get("twitter"):
            parts.append(person["twitter"].lstrip('@'))
        query = "+OR+".join(parts)
        try:
            resp = requests.get(_CEO_NEWS_RSS.format(query=query), timeout=10)
            feed = feedparser.parse(resp.content)
            out = []
            for entry in feed.entries[:3]:
                title = entry.get("title", "").strip()
                if title and title not in _ceo_headlines_seen:
                    out.append((person["name"], title))
            return out
        except Exception:
            return []

    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_check_one, p): p for p in watchlist}
        for fut in as_completed(futures, timeout=30):
            try:
                results.extend(fut.result())
            except Exception:
                continue
    return results


def _evaluate_ceo_news(ceo_headlines):
    if not ceo_headlines:
        return []
    just_headlines = [h for _, h in ceo_headlines[:10]]
    scored = _score_headlines(just_headlines)
    feed_lines = []
    for (name, headline), (_, lbl, scr) in zip(ceo_headlines[:10], scored):
        feed_lines.append(f"- {name} [{lbl.upper()} {scr:.0%}]: {headline}")
    feed = "\n".join(feed_lines)
    price_check = _build_price_check(scored, use_rh=False)
    try:
        r = ollama.chat(model=OLLAMA_MODEL, options=_QWEN_OPTS, messages=[
            {"role": "system", "content": "You scan CEO news for trade opportunities. Casual. One sentence each."},
            {"role": "user", "content": f"CEO news:\n{feed}\n{price_check}\nAny moves? If nothing: NONE\n/no_think"}])
        text = _strip_think(r.message.content.strip())
        if not text or "NONE" in text.upper():
            return []
        return [l.strip().lstrip("- \u2022*") for l in text.split("\n")
                if l.strip() and "NONE" not in l.upper() and len(l.strip()) > 5][:2]
    except Exception:
        return []


# ── RSS News ──────────────────────────────────────────────────
_NEWS_FEEDS = [
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "https://seekingalpha.com/feed",
    "https://www.ft.com/markets?format=rss",
    "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx6TVdZU0FtVnVHZ0pWVXlnQVAB",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://fortune.com/feed/",
]
_NEWS_CACHE_PATH = os.path.join(_SCRIPT_DIR, "news_cache.json")
_news_cache = {"headlines": [], "ts": 0.0}


def _fetch_news(max_headlines=5):
    from calendar import timegm
    from concurrent.futures import ThreadPoolExecutor, as_completed
    global _news_cache
    now = time.time()
    cutoff = now - 86400

    def _parse_one(url):
        try:
            resp = requests.get(url, timeout=6)
            feed = feedparser.parse(resp.content)
        except Exception:
            return []
        items = []
        for entry in feed.entries[:10]:
            published = entry.get("published_parsed") or entry.get("updated_parsed")
            if published:
                from calendar import timegm as _tg
                pub_ts = _tg(published)
                if pub_ts < cutoff:
                    continue
                age_min = max(0, int((now - pub_ts) / 60))
            else:
                age_min = -1
            title = entry.get("title", "").strip()
            if title:
                items.append((title, age_min))
        return items

    seen = set()
    timed_headlines = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_parse_one, url): url for url in _NEWS_FEEDS}
        for fut in as_completed(futures, timeout=10):
            try:
                for title, age in fut.result():
                    if title not in seen:
                        seen.add(title)
                        timed_headlines.append((title, age))
            except Exception:
                continue
    timed_headlines.sort(key=lambda x: x[1] if x[1] >= 0 else 9999)
    headlines = [t for t, _ in timed_headlines]
    if headlines:
        _news_cache = {"headlines": headlines, "ts": now}
        try:
            with open(_NEWS_CACHE_PATH, "w") as f:
                json.dump(_news_cache, f)
        except OSError:
            pass
    elif not _news_cache["headlines"]:
        try:
            with open(_NEWS_CACHE_PATH) as f:
                _news_cache = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    if not timed_headlines and _news_cache["headlines"]:
        return [(h, -1) for h in _news_cache["headlines"][:max_headlines]]
    return timed_headlines[:max_headlines]


# ── Historical / On This Day ─────────────────────────────────
_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_NUM_TO_MONTH = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December",
}
_HISTORICAL_DATE_RE = re.compile(
    r'(?:(?P<month>\w+)\s+(?P<year>\d{4}))|(?:(?:in|from|of|during)\s+(?P<year2>\d{4}))', re.IGNORECASE)


def _parse_historical_date(text):
    m = _HISTORICAL_DATE_RE.search(text)
    if not m:
        return None
    month_str = m.group("month")
    year_str = m.group("year") or m.group("year2")
    if not year_str:
        return None
    year = int(year_str)
    now = datetime.now()
    if month_str:
        month_num = _MONTH_NAMES.get(month_str.lower())
        if not month_num:
            return None
        if year > now.year or (year == now.year and month_num >= now.month):
            return None
        return month_str.capitalize(), year
    else:
        if year >= now.year:
            return None
        return None, year


def _fetch_historical_news(month_name, year, max_events=5):
    from html import unescape as _unescape
    api = "https://en.wikipedia.org/w/api.php"
    headers = {"User-Agent": "PhoebeBot/1.0"}
    try:
        resp = requests.get(api, timeout=10, headers=headers, params={
            "action": "parse", "page": str(year), "prop": "sections", "format": "json"})
        if resp.status_code != 200:
            return []
        sections = resp.json().get("parse", {}).get("sections", [])
        section_idx = None
        if month_name:
            for s in sections:
                if s["line"].lower() == month_name.lower():
                    section_idx = s["index"]
                    break
        else:
            for s in sections:
                if s["line"].lower() in ("january", "events"):
                    section_idx = s["index"]
                    break
        if not section_idx:
            return []
        resp2 = requests.get(api, timeout=10, headers=headers, params={
            "action": "parse", "page": str(year), "prop": "wikitext", "section": section_idx, "format": "json"})
        wikitext = resp2.json().get("parse", {}).get("wikitext", {}).get("*", "")
        events = []
        for line in wikitext.split("\n"):
            line = line.strip()
            if line.startswith("*") and not line.startswith("**"):
                clean = re.sub(r"\[\[[^\]]*\|([^\]]+)\]\]", r"\1", line)
                clean = re.sub(r"\[\[([^\]]+)\]\]", r"\1", clean)
                clean = re.sub(r"'''+", "", clean)
                clean = re.sub(r"\{\{[^}]+\}\}", "", clean)
                clean = re.sub(r"<[^>]+>", "", clean)
                clean = _unescape(clean.lstrip("* ").strip())
                clean = re.sub(r"\s+", " ", clean)
                if len(clean) > 15:
                    events.append(clean)
                    if len(events) >= max_events:
                        break
        return events
    except Exception:
        return []


_ORDINAL_TO_NUM = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20, "twenty-first": 21, "twenty-second": 22,
    "twenty-third": 23, "twenty-fourth": 24, "twenty-fifth": 25,
    "twenty-sixth": 26, "twenty-seventh": 27, "twenty-eighth": 28,
    "twenty-ninth": 29, "thirtieth": 30, "thirty-first": 31,
}
_SPOKEN_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
                "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_SPOKEN_TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
                 "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_SPOKEN_ONES = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9}


def _spoken_number(text):
    text = text.strip().lower()
    if text in ("zero", "oh"):
        return 0
    if text in _SPOKEN_ONES:
        return _SPOKEN_ONES[text]
    if text in _SPOKEN_TEENS:
        return _SPOKEN_TEENS[text]
    if text in _SPOKEN_TENS:
        return _SPOKEN_TENS[text]
    for sep in ('-', ' '):
        if sep in text:
            parts = text.split(sep, 1)
            t = _SPOKEN_TENS.get(parts[0])
            o = _SPOKEN_ONES.get(parts[1])
            if t is not None and o is not None:
                return t + o
    return None


def _parse_spoken_year(text):
    text = text.strip().lower().rstrip('.,?!')
    m = re.match(r'^(\d{4})', text)
    if m:
        return int(m.group(1))
    m = re.match(r'two\s+thousand(?:\s+and)?\s*(.*)', text)
    if m:
        rest = m.group(1).strip()
        if not rest:
            return 2000
        n = _spoken_number(rest)
        if n is not None and 0 <= n <= 99:
            return 2000 + n
        return None
    words = text.split()
    if len(words) >= 2:
        century = _spoken_number(words[0])
        if century is not None and 10 <= century <= 21:
            rest_num = _spoken_number(' '.join(words[1:]))
            if rest_num is not None and 0 <= rest_num <= 99:
                return century * 100 + rest_num
            if len(words) >= 3 and words[1] == "oh":
                unit = _spoken_number(words[2])
                if unit is not None and 1 <= unit <= 9:
                    return century * 100 + unit
    return None


_MONTH_RE_PART = '|'.join(sorted(_MONTH_NAMES.keys(), key=len, reverse=True))


def _parse_on_this_day(text):
    low = text.strip().lower()
    m = re.search(r'on this day(?:\s+in\s+(.+))?', low)
    if m:
        now = datetime.now()
        year = None
        if m.group(1):
            year = _parse_spoken_year(m.group(1).strip())
        return (now.strftime("%B"), now.day, year)
    m = re.search(rf'\b({_MONTH_RE_PART})\s+(\w+(?:-\w+)?)\s*[,.]?\s+(.+)', low)
    if not m:
        return None
    month_str, day_str, year_str = m.group(1), m.group(2).strip(), m.group(3).strip().rstrip('?.!')
    month_num = _MONTH_NAMES.get(month_str)
    if not month_num:
        return None
    day = _ORDINAL_TO_NUM.get(day_str)
    if day is None:
        try:
            day = int(re.sub(r'(st|nd|rd|th)', '', day_str))
        except ValueError:
            return None
    if not 1 <= day <= 31:
        return None
    year = _parse_spoken_year(year_str)
    if not year or not 1 <= year <= 2100:
        return None
    return (_NUM_TO_MONTH[month_num], day, year)


def _fetch_on_this_day(month_name, day, year=None, max_events=3):
    from html import unescape as _unescape
    page = f"{month_name}_{day}"
    api = "https://en.wikipedia.org/w/api.php"
    headers = {"User-Agent": "PhoebeBot/1.0"}
    try:
        resp = requests.get(api, timeout=10, headers=headers, params={
            "action": "parse", "page": page, "prop": "sections", "format": "json"})
        if resp.status_code != 200:
            return []
        sections = resp.json().get("parse", {}).get("sections", [])
        ev_idx = None
        for s in sections:
            if s["line"].lower() == "events":
                ev_idx = s["index"]
                break
        if not ev_idx:
            return []
        resp2 = requests.get(api, timeout=10, headers=headers, params={
            "action": "parse", "page": page, "prop": "wikitext", "section": ev_idx, "format": "json"})
        wikitext = resp2.json().get("parse", {}).get("wikitext", {}).get("*", "")
        events = []
        for line in wikitext.split("\n"):
            line = line.strip()
            if not line.startswith("*") or line.startswith("**"):
                continue
            clean = re.sub(r"\[\[[^\]]*\|([^\]]+)\]\]", r"\1", line)
            clean = re.sub(r"\[\[([^\]]+)\]\]", r"\1", clean)
            clean = re.sub(r"'''+", "", clean)
            clean = re.sub(r"\{\{[^}]+\}\}", "", clean)
            clean = re.sub(r"<[^>]+>", "", clean)
            clean = _unescape(clean.lstrip("* ").strip())
            clean = re.sub(r"\s+", " ", clean)
            if len(clean) < 10:
                continue
            if year:
                if re.match(rf'^{year}\s*[\u2013\u2014\-]', clean):
                    events.append(clean)
            else:
                events.append(clean)
            if year and len(events) >= max_events:
                break
        if not year and events:
            if len(events) <= max_events:
                return events
            step = len(events) / max_events
            return [events[int(i * step)] for i in range(max_events)]
        return events[:max_events]
    except Exception:
        return []


def _parse_birthday(text):
    text = text.strip().lower().rstrip('.!?')
    m = re.match(r'^(\d{1,2})[/\-](\d{1,2})', text)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return (month, day)
    m = re.match(rf'^({_MONTH_RE_PART})\s+(\w+(?:-\w+)?)', text)
    if m:
        month_num = _MONTH_NAMES.get(m.group(1))
        if not month_num:
            return None
        day_str = m.group(2)
        day = _ORDINAL_TO_NUM.get(day_str)
        if day is None:
            try:
                day = int(re.sub(r'(st|nd|rd|th)', '', day_str))
            except ValueError:
                return None
        if 1 <= day <= 31:
            return (month_num, day)
    return None


_bg_history_cache = {}


# ── Background loops ──────────────────────────────────────────
_bg_news = {"headlines": [], "scored": [], "ages": {}, "ts": 0.0}


def _bg_news_loop():
    global _bg_news
    time.sleep(3)
    while True:
        t0 = time.time()
        try:
            raw = _fetch_news(10)
            headlines = [title for title, _ in raw]
            ages = {title: age for title, age in raw}
            scored = _score_headlines(headlines)
            _bg_news = {"headlines": headlines, "scored": scored, "ages": ages, "ts": time.time()}
            dt = time.time() - t0
            print(f"[BG] News refreshed — {len(headlines)} headlines ({dt:.1f}s)")
            if headlines:
                with _bg_ollama_lock:
                    ideas = _extract_news_ideas(headlines, scored, ages)
                new = [i for i in ideas if i not in _shown_news_ideas]
                if new:
                    _shown_news_ideas.update(new)
                    _pending_news_ideas.extend(new)
                    print(f"[BG] Trade thoughts: {new}")
                    if _ALPACA_AVAILABLE:
                        bg_signals = {
                            "source": "background_news",
                            "scored_headlines": [{"headline": h, "label": l, "score": round(s, 3)}
                                                 for h, l, s in scored[:5]],
                        }
                        for idea in new:
                            threading.Thread(target=_alpaca_act_on_idea,
                                             args=(idea,), kwargs={"signals": bg_signals},
                                             daemon=True).start()
        except Exception as exc:
            print(f"[BG] News error: {exc}")
        time.sleep(900)


def _bg_ceo_loop():
    global _ceo_watchlist
    _ceo_watchlist[:], last_ts = _load_ceo_watchlist()
    time.sleep(45)
    while True:
        now = time.time()
        if now - last_ts > 86400:
            try:
                positions = _bg_portfolio.get("positions") or []
                tickers = [p["label"] for p in positions if p.get("type") == "stock"]
                _build_ceo_watchlist(tickers)
                last_ts = now
            except Exception as e:
                print(f"[CEO] Build error: {e}")
        if _ceo_watchlist:
            try:
                headlines = _fetch_ceo_news(_ceo_watchlist)
                new_hl = [(n, h) for n, h in headlines if h not in _ceo_headlines_seen]
                _ceo_headlines_seen.update(h for _, h in new_hl)
                if new_hl:
                    with _bg_ollama_lock:
                        ideas = _evaluate_ceo_news(new_hl)
                    fresh = [i for i in ideas if i not in _shown_news_ideas]
                    if fresh:
                        _shown_news_ideas.update(fresh)
                        _pending_news_ideas.extend(fresh)
            except Exception as e:
                print(f"[CEO] News error: {e}")
        time.sleep(1200)


def _bg_kalshi_loop():
    global _bg_kalshi
    time.sleep(60)
    while True:
        try:
            current = _kalshi_fetch_markets()
            prev = _bg_kalshi.get("markets", {})
            shifts = _kalshi_detect_shifts(current, prev) if prev else []
            _bg_kalshi = {"markets": current, "prev": prev, "shifts": shifts, "ts": time.time()}
            print(f"[Kalshi] {len(current)} markets tracked")
            if shifts:
                for s in shifts:
                    with _bg_ollama_lock:
                        alert = _kalshi_cross_reference(s)
                    if alert:
                        _pending_kalshi_alerts.append(alert)
        except Exception as exc:
            print(f"[Kalshi] Error: {exc}")
        time.sleep(300)


# ── Ollama helpers ────────────────────────────────────────────
def _build_ollama_messages(user_input):
    facts = _user_data_store.get("facts", [])
    sys_prompt = PHOEBE_SYSTEM_PROMPT
    if facts:
        sys_prompt += "\n\nThings you know about the user:\n" + "\n".join(f"- {f}" for f in facts[-30:])
    mood = _user_mood.get("level", "normal")
    if mood == "loud":
        sys_prompt += "\n\n[Voice mood: the user is speaking loudly. Soften your tone. Be calm, grounding, gentle.]"
    elif mood == "quiet":
        sys_prompt += "\n\n[Voice mood: the user is speaking quietly. Match their softness. Be gentle.]"
    messages = [{"role": "system", "content": sys_prompt}]
    for u, p in _conversations[-20:]:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": p})
    messages.append({"role": "user", "content": user_input})
    return messages


def _ensure_model():
    for model in (OLLAMA_MODEL, OLLAMA_CHAT_MODEL):
        try:
            ollama.show(model)
        except ollama.ResponseError:
            print(f"[Ollama] Pulling {model}...")
            ollama.pull(model)


# ── Regex patterns for routing ────────────────────────────────
_FINANCIAL_HINT_RE = re.compile(
    r'\b(?:money|stock|stocks|portfolio|holdings|positions|'
    r'trade|trading|invest|investing|investment|market|markets|'
    r'buy|sell|buying|selling|crypto|bitcoin|shares|ticker|etf|'
    r'news|headlines|market\s*update|watchlist|watch\s*list|'
    r'nasdaq|s&p|dow|forex|gold|silver|oil|earnings|options|'
    r'how.{0,10}(?:my\s+)?(?:stocks|money|portfolio|holdings))\b', re.IGNORECASE)

_CONVO_RECALL_RE = re.compile(
    r'\b(?:what was that|what did (?:i|you|we) (?:just )?(?:say|hear|talk about)|'
    r'did i say|what happened|come again|say that again|repeat that|'
    r'what were we (?:talking|saying)|one more time|say again|'
    r'what did (?:i|you) (?:just )?(?:mention|mean))\b', re.IGNORECASE)

_MARKET_WORDS_RE = re.compile(
    r'\b(?:stock|stocks|trade|trading|market|markets|crypto|bitcoin|'
    r'portfolio|invest|investing|investment|ticker|shares|etf|'
    r'nasdaq|s&p|dow|forex|commodity|commodities|gold|silver|oil|'
    r'bull|bear|rally|earnings|dividend|hedge|short|long|options|'
    r'buy|sell|holding|position)\b', re.IGNORECASE)


# ── Reminder scheduling ──────────────────────────────────────
def _schedule_reminder(iso_time, task):
    target = datetime.fromisoformat(iso_time)
    now = datetime.now()
    if target <= now:
        _pending_alerts_queue.append({"type": "reminder", "text": f"Reminder: {task}"})
        _memory_data["reminders"] = [r for r in _memory_data["reminders"]
                                     if not (r["time"] == iso_time and r["task"] == task)]
        _save_memory_bg()
        return
    job_id = f"reminder_{hash(iso_time + task)}"
    _scheduler.add_job(
        _on_reminder_due, trigger="date", run_date=target,
        args=[iso_time, task], id=job_id, replace_existing=True)
    print(f"[Reminder scheduled] '{task}' due at {iso_time}")


def _on_reminder_due(iso_time, task):
    print(f"[Reminder FIRING] '{task}'")
    _memory_data["reminders"] = [r for r in _memory_data["reminders"]
                                 if not (r["time"] == iso_time and r["task"] == task)]
    _save_memory_bg()
    _pending_alerts_queue.append({"type": "reminder", "text": f"Reminder: {task}"})


def _save_reminder(target_time, task):
    _memory_data["reminders"] = [r for r in _memory_data["reminders"] if r["task"] != task]
    iso_time = target_time.isoformat()
    _memory_data["reminders"].append({"time": iso_time, "task": task})
    _save_memory_bg()
    _schedule_reminder(iso_time, task)


# ═══════════════════════════════════════════════════════════════
#  HANDLERS — each returns (response_text, action_string)
# ═══════════════════════════════════════════════════════════════

def _handle_chat(text):
    messages = _build_ollama_messages(text)
    try:
        result = ollama.chat(model=OLLAMA_CHAT_MODEL, options=_MISTRAL_OPTS, messages=messages)
        response = result.message.content.strip()
    except Exception:
        response = "I heard you. I'll remember that."
    _remember(text, response)
    _save_memory_bg()
    _learn_from_input_bg(text)
    return response, "chat"


def _handle_convo_recall(text):
    recent = _conversations[-5:] if _conversations else []
    if not recent:
        return "We haven't talked about anything yet.", "convo_recall"
    lines = []
    for user_msg, phoebe_msg in recent:
        lines.append(f"You: {user_msg}")
        lines.append(f"Phoebe: {phoebe_msg}")
    convo_block = "\n".join(lines)
    try:
        result = ollama.chat(model=OLLAMA_CHAT_MODEL, options=_MISTRAL_OPTS, messages=[
            {"role": "system", "content": "You are Phoebe. Summarise the recent conversation naturally and concisely."},
            {"role": "user", "content": f'The user asked: "{text}"\n\nRecent conversation:\n{convo_block}'}])
        response = result.message.content.strip()
    except Exception:
        response = "We were just chatting. I lost the thread for a second."
    _remember(text, response)
    return response, "convo_recall"


def _handle_reminder_in(match):
    amount = int(match.group(1))
    unit = match.group(2).lower()
    task = match.group(3)
    if unit.startswith("minute"):
        delay = amount * 60
        unit_display = "minute" if amount == 1 else "minutes"
    elif unit.startswith("second"):
        delay = amount
        unit_display = "second" if amount == 1 else "seconds"
    elif unit.startswith("hour"):
        delay = amount * 3600
        unit_display = "hour" if amount == 1 else "hours"
    else:
        delay = amount * 60
        unit_display = "minutes"
    amount_word = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
                   6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}.get(amount, str(amount))
    target_time = datetime.now() + timedelta(seconds=delay)
    response = f"Okay, I will remind you in {amount_word} {unit_display}."
    _remember(f"remind me in {amount} {unit_display} to {task}", response)
    _save_memory_bg()
    _save_reminder(target_time, task)
    return response, "reminder"


def _handle_reminder_at(match):
    time_str = match.group(1)
    task = match.group(2)
    target_time = parse_clock_time(time_str)
    if target_time is None:
        return f"I don't understand the time \"{time_str}\".", "reminder"
    display_time = target_time.strftime("%I:%M %p").lstrip("0")
    response = f"Okay, I will remind you at {display_time}."
    _remember(f"remind me at {time_str} to {task}", response)
    _save_memory_bg()
    _save_reminder(target_time, task)
    return response, "reminder"


def _handle_reminder_to_at(match):
    task = match.group(1)
    time_str = match.group(2)
    target_time = parse_clock_time(time_str)
    if target_time is None:
        return f"I don't understand the time \"{time_str}\".", "reminder"
    display_time = target_time.strftime("%I:%M %p").lstrip("0")
    response = f"Okay, I will remind you at {display_time}."
    _remember(f"remind me to {task} at {time_str}", response)
    _save_memory_bg()
    _save_reminder(target_time, task)
    return response, "reminder"


def _handle_reminder_tomorrow(match):
    time_str = match.group(1)
    task = match.group(2)
    target_time = parse_clock_time(time_str, tomorrow=True)
    if target_time is None:
        return f"I don't understand the time \"{time_str}\".", "reminder"
    display_time = target_time.strftime("%I:%M %p").lstrip("0")
    response = f"Okay, I will remind you tomorrow at {display_time}."
    _remember(f"remind me tomorrow at {time_str} to {task}", response)
    _save_memory_bg()
    _save_reminder(target_time, task)
    return response, "reminder"


def _handle_todo_check():
    reminders = _memory_data.get("reminders", [])
    if not reminders:
        return "Nothing.", "todo"
    today = datetime.now().date()
    items = []
    for r in reminders:
        target = datetime.fromisoformat(r["time"])
        time_str = target.strftime("%I:%M %p").lstrip("0")
        if target.date() == today:
            items.append(f"{r['task']} today at {time_str}")
        elif target.date() == today + timedelta(days=1):
            items.append(f"{r['task']} tomorrow at {time_str}")
        else:
            day_str = target.strftime("%A %B %d").replace(" 0", " ")
            items.append(f"{r['task']} on {day_str} at {time_str}")
    response = ", ".join(items) + "."
    _remember("what do i need to do", response)
    return response, "todo"


def _handle_last_said():
    if _conversations:
        response = f"You said: {_conversations[-1][0]}"
    else:
        response = "You haven't said anything."
    _remember("what did i just say", response)
    return response, "last_said"


def _handle_memory():
    if not _conversations:
        return "We haven't talked yet.", "memory"
    recent = _conversations[-3:]
    lines = [f"You said: {m[0]}" for m in recent]
    response = " ".join(lines)
    _remember("what were we talking about", response)
    return response, "memory"


def _handle_age():
    for u_input, _ in _conversations:
        if '33' in u_input or 'thirty-three' in u_input.lower():
            _remember("how old am i", "Thirty-three.")
            return "Thirty-three.", "age"
    _remember("how old am i", "You haven't told me.")
    return "You haven't told me.", "age"


def _handle_name_store(text):
    global _user_name
    prev_name = _user_name or "unknown"
    try:
        result = ollama.chat(model=OLLAMA_MODEL, options=_QWEN_OPTS, messages=[{"role": "user", "content": (
            f'The user said: "{text}"\nCurrently stored name: {prev_name}\n'
            "Decide: NAME:<name> or NONAME\n/no_think")}])
        answer = _strip_think(result.message.content.strip())
    except Exception:
        answer = ""
    if answer.upper().startswith("NAME:"):
        name = answer.split(":", 1)[1].strip().split()[0].capitalize()
        if not (name and name.isalpha()):
            return "I didn't quite catch that.", "name_store"
        already_knew = prev_name != "unknown"
        same_name = already_knew and prev_name == name
        correction = already_knew and prev_name != name
        _user_name = name
        _user_data_store["name"] = name
        _save_user_data(_user_data_store)
        try:
            greet = ollama.chat(model=OLLAMA_CHAT_MODEL, options=_MISTRAL_OPTS, messages=[
                {"role": "system", "content": PHOEBE_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f'The user said: "{text}"\nAlready knew: {already_knew}\n'
                    f'Previous: {prev_name}\nGiven: {name}\nSame: {same_name}\nCorrection: {correction}\n'
                    "Respond naturally as Phoebe. One sentence max.")}])
            response = greet.message.content.strip()
        except Exception:
            response = f"Got it — {name}." if correction else f"Nice to meet you, {name}." if not already_knew else f"Hey, {name}."
    else:
        if prev_name != "unknown":
            response = f"Alright — you're still {prev_name} to me."
        else:
            response = "Okay. What should I call you then?"
    _remember(text, response)
    _save_memory_bg()
    return response, "name_store"


def _handle_name_recall():
    if _user_name:
        response = f"{_user_name}."
    else:
        response = "You haven't told me."
    _remember("what is my name", response)
    return response, "name_recall"


def _handle_birthday_store(text):
    m = re.search(r'my birthday is\s+(.+)', text, re.IGNORECASE)
    if not m:
        return "When is it?", "birthday_store"
    parsed = _parse_birthday(m.group(1).strip())
    if not parsed:
        return "I didn't catch the date.", "birthday_store"
    month, day = parsed
    _user_data_store["birthday"] = f"{month:02d}-{day:02d}"
    _save_user_data(_user_data_store)
    month_name = _NUM_TO_MONTH[month]
    response = f"{month_name} {day}. I'll remember that."
    _remember(text, response)
    _save_memory_bg()
    return response, "birthday_store"


def _handle_birthday_recall():
    bday = _user_data_store.get("birthday")
    if bday:
        month, day = bday.split("-")
        response = f"{_NUM_TO_MONTH[int(month)]} {int(day)}."
    else:
        response = "You haven't told me."
    _remember("when is my birthday", response)
    return response, "birthday_recall"


def _handle_on_this_day(text, month_name, day, year):
    key = f"otd_{month_name.lower()}_{day}" + (f"_{year}" if year else "")
    cached = _bg_history_cache.get(key)
    if cached and cached["events"]:
        events = cached["events"]
    else:
        events = _fetch_on_this_day(month_name, day, year)
        _bg_history_cache[key] = {"events": events, "ts": time.time()}
    if not events:
        response = f"Nothing listed for {month_name} {day}{f', {year}' if year else ''}."
    else:
        numbered = [f"{i+1}. {e}" for i, e in enumerate(events)]
        response = "\n".join(numbered)
    _remember(text, response)
    _save_memory_bg()
    return response, "on_this_day"


def _handle_historical_news(text, month_name, year):
    label = f"{month_name} {year}" if month_name else str(year)
    key = label.lower()
    cached = _bg_history_cache.get(key)
    if cached and cached["events"]:
        events = cached["events"]
    else:
        events = _fetch_historical_news(month_name, year, 5)
        _bg_history_cache[key] = {"events": events, "ts": time.time()}
    if not events:
        response = f"I couldn't find anything for {label}."
    else:
        response = f"{label}: " + " ".join(events[:5])
    _remember(text, response)
    _save_memory_bg()
    return response, "historical_news"


def _handle_portfolio(text):
    _refresh_portfolio()
    positions = _bg_portfolio["positions"]
    overall_pct = _bg_portfolio["overall_pct"]
    full = bool(re.search(r'\b(?:full|everything|all|every|entire|complete|detailed)\b', text, re.IGNORECASE))
    if positions is not None:
        response = _format_portfolio(positions, overall_pct, full=full)
    else:
        response = "Couldn't reach Robinhood right now."
    return response, "portfolio"


def _handle_trade_idea(text):
    _refresh_portfolio()
    positions = _bg_portfolio["positions"]
    overall_pct = _bg_portfolio["overall_pct"]
    portfolio_summary = _format_portfolio(positions, overall_pct, full=True) if positions else ""
    idea, _signals = _fetch_trade_idea(portfolio_summary)
    global _last_trade_thought, _last_action
    _last_trade_thought = idea
    _last_action = "trade"
    _remember(text, idea)
    _save_memory_bg()
    if _ALPACA_AVAILABLE:
        threading.Thread(target=_alpaca_act_on_idea, args=(idea,),
                         kwargs={"signals": _signals}, daemon=True).start()
    return idea, "trade"


def _handle_trade_followup(text):
    global _last_trade_thought
    scored = _bg_news.get("scored", [])[:10]
    ages = _bg_news.get("ages", {})
    news_section = _format_scored_headlines(scored, ages)
    if not news_section:
        headlines = _bg_news.get("headlines", [])[:10]
        if headlines:
            news_section = "\nRecent headlines:\n" + "\n".join(f"- {h}" for h in headlines) + "\n"
    price_check = _build_price_check(scored, ages) if scored else ""
    kalshi_section = _format_kalshi_context()
    prev = _last_trade_thought or ""
    try:
        result = ollama.chat(model=OLLAMA_MODEL, options=_QWEN_OPTS, messages=[
            {"role": "system", "content": (
                "You are a blunt trading advisor. Explain WHY. Use real numbers. "
                "Sentiment is mood. Price is reality. Volume is conviction. Kalshi is the crowd bet. "
                "Do NOT list portfolio holdings. Two to four sentences. No disclaimers.")},
            {"role": "user", "content": (
                f"You just said: \"{prev}\"\n{news_section}{price_check}{kalshi_section}\n"
                f"The user now asks: \"{text}\"\nExplain the reasoning. /no_think")}])
        reply = _strip_think(result.message.content.strip())
    except Exception:
        reply = "Can't pull that together right now."
    _last_trade_thought = reply
    _remember(text, reply)
    _save_memory_bg()
    return reply, "trade"


def _handle_news():
    scored = _bg_news.get("scored", [])[:5]
    if scored:
        parts = []
        for i, (headline, label, score) in enumerate(scored, 1):
            mood = label if score < 0.7 else label.upper()
            parts.append(f"{i}. {headline} — {mood}")
        return " ".join(parts), "news"
    headlines = _bg_news["headlines"][:5]
    if not headlines:
        return "No headlines right now.", "news"
    numbered = [f"{i+1}. {h}" for i, h in enumerate(headlines)]
    return " ".join(numbered), "news"


def _handle_watchlist():
    wl = _ceo_watchlist
    if not wl:
        return "Still building my watchlist. Give me a minute.", "watchlist"
    parts = []
    for p in wl:
        line = p["name"]
        if p.get("company"):
            line += f" at {p['company']}"
        if p.get("twitter"):
            line += f" ({p['twitter']})"
        parts.append(line)
    return "I'm watching: " + ", ".join(parts) + ".", "watchlist"


def _handle_route_financial(text):
    global _last_action, _last_trade_thought, _pending_clarification
    last_trade = _last_trade_thought or ""
    last_act = _last_action or ""
    try:
        result = ollama.chat(model=OLLAMA_MODEL, options=_QWEN_OPTS, messages=[{"role": "user", "content": (
            "You are routing a user's message. Classify their intent.\n"
            f"Context:\n- Last action: {last_act}\n- Last trade thought: \"{last_trade[:200]}\"\n\n"
            f"User says: \"{text}\"\n\n"
            "Classify as EXACTLY one of:\n"
            "PORTFOLIO — clearly asking to see holdings\n"
            "TRADE_IDEA — clearly asking for a trade suggestion\n"
            "NEWS — clearly asking for headlines\n"
            "WATCHLIST — asking about CEO/market-leader watchlist\n"
            "FOLLOWUP — following up on the trade thought above\n"
            "MISHEAR — garbled voice input near trade context\n"
            "AMBIGUOUS — could mean financial OR personal\n"
            "CHAT — not financial at all\n\n"
            "Return format: INTENT|question\n"
            "If AMBIGUOUS: write clarifying question after pipe.\n"
            "If MISHEAR: write what they probably meant.\n"
            "Otherwise: leave blank.\n/no_think")}])
        raw = _strip_think(result.message.content.strip())
    except Exception:
        raw = "CHAT|"
    if "|" in raw:
        intent, payload = raw.split("|", 1)
    else:
        intent, payload = raw.strip(), ""
    intent = intent.strip().upper()
    payload = payload.strip()

    if intent == "PORTFOLIO":
        return _handle_portfolio(text)
    elif intent == "TRADE_IDEA":
        return _handle_trade_idea(text)
    elif intent == "NEWS":
        hist = _parse_historical_date(text)
        if hist:
            return _handle_historical_news(text, hist[0], hist[1])
        return _handle_news()
    elif intent == "WATCHLIST":
        return _handle_watchlist()
    elif intent == "FOLLOWUP":
        return _handle_trade_followup(text)
    elif intent == "MISHEAR":
        clarify = f"Wait, did you mean {payload}?" if payload else "Sorry, what was that?"
        return clarify, "mishear"
    elif intent == "AMBIGUOUS":
        q = payload or "Are you asking about the market, or something else?"
        _pending_clarification = {"input": text, "question": q}
        return q, "clarify"
    else:
        return _handle_chat(text)


def _handle_resolve_clarification(text):
    global _pending_clarification
    clarification = _pending_clarification
    _pending_clarification = None
    original = clarification["input"]
    question = clarification["question"]

    from concurrent.futures import ThreadPoolExecutor

    def _classify():
        try:
            result = ollama.chat(model=OLLAMA_MODEL, options=_QWEN_OPTS, messages=[{"role": "user", "content": (
                "Read this exchange and decide what the user wants.\n"
                f'User said: "{original}"\nPhoebe asked: "{question}"\nUser replied: "{text}"\n\n'
                "Classify: PORTFOLIO, TRADE_IDEA, NEWS, WATCHLIST, or PERSONAL\n"
                "If they stayed personal/emotional, that's PERSONAL.\n/no_think")}])
            t = _strip_think(result.message.content.strip()).upper()
            return t.split()[0] if t else "PERSONAL"
        except Exception:
            return "PERSONAL"

    def _personal_response():
        facts = _user_data_store.get("facts", [])
        sys_prompt = PHOEBE_SYSTEM_PROMPT
        if facts:
            sys_prompt += "\n\nThings you know about the user:\n" + "\n".join(f"- {f}" for f in facts[-30:])
        mood = _user_mood.get("level", "normal")
        if mood == "loud":
            sys_prompt += "\n\n[Voice mood: loud. Soften your tone.]"
        elif mood == "quiet":
            sys_prompt += "\n\n[Voice mood: quiet. Match their softness.]"
        messages = [{"role": "system", "content": sys_prompt}]
        for u, p in _conversations[-20:]:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": p})
        messages.append({"role": "user", "content": original})
        messages.append({"role": "assistant", "content": question})
        messages.append({"role": "user", "content": text})
        try:
            resp = ollama.chat(model=OLLAMA_CHAT_MODEL, options=_MISTRAL_OPTS, messages=messages)
            return resp.message.content.strip()
        except Exception:
            return "I hear you. Keep going."

    pool = ThreadPoolExecutor(max_workers=2)
    intent_fut = pool.submit(_classify)
    reply_fut = pool.submit(_personal_response)
    intent = intent_fut.result()

    if intent == "PERSONAL":
        reply = reply_fut.result()
        _remember(original, question)
        _remember(text, reply)
        _save_memory_bg()
        _learn_from_input_bg(original)
        pool.shutdown(wait=False)
        return reply, "resolve_clarification"

    reply_fut.cancel()
    pool.shutdown(wait=False)

    if intent == "PORTFOLIO":
        return _handle_portfolio(original)
    elif intent == "TRADE_IDEA":
        return _handle_trade_idea(original)
    elif intent == "NEWS":
        hist = _parse_historical_date(original)
        if hist:
            return _handle_historical_news(original, hist[0], hist[1])
        return _handle_news()
    elif intent == "WATCHLIST":
        return _handle_watchlist()
    else:
        return _handle_chat(text)


def _handle_resolve_price_conflict(text):
    global _pending_price_conflict_state
    conflict = _pending_price_conflict_state
    _pending_price_conflict_state = None
    ticker = conflict["ticker"]
    try:
        result = ollama.chat(model=OLLAMA_MODEL, options=_QWEN_OPTS, messages=[{"role": "user", "content": (
            f"Phoebe showed two prices for {ticker}.\n"
            f"RH: ${conflict['rh_price']:.2f} ({conflict['rh_pct']:+.1f}%)\n"
            f"YF: ${conflict['yf_price']:.2f} ({conflict['yf_pct']:+.1f}%)\n"
            f'User replied: "{text}"\n'
            "What did they choose? ROBINHOOD, YAHOO, or NEITHER\n/no_think")}])
        choice = _strip_think(result.message.content.strip()).upper().split()[0]
    except Exception:
        choice = "NEITHER"
    if choice == "ROBINHOOD":
        _price_source_pref[ticker] = "rh"
        reply = f"Got it — using Robinhood for {ticker}. ${conflict['rh_price']:.2f} ({conflict['rh_pct']:+.1f}%)."
    elif choice == "YAHOO":
        _price_source_pref[ticker] = "yf"
        reply = f"Got it — using Yahoo for {ticker}. ${conflict['yf_price']:.2f} ({conflict['yf_pct']:+.1f}%)."
    else:
        reply = f"No worries — I'll keep showing you both for {ticker} until you decide."
    _remember(text, reply)
    _save_memory_bg()
    return reply, "resolve_price_conflict"


# ═══════════════════════════════════════════════════════════════
#  MAIN ROUTER
# ═══════════════════════════════════════════════════════════════

def _route_and_handle(text, mood):
    global _user_mood, _last_action, _pending_price_conflict_state
    _user_mood = mood

    # 1. Pending clarification
    if _pending_clarification:
        return _handle_resolve_clarification(text)

    # 2. Pending price conflict
    if _pending_price_conflict_state:
        return _handle_resolve_price_conflict(text)

    # 3. Convo recall
    if _CONVO_RECALL_RE.search(text):
        return _handle_convo_recall(text)

    # 4. Reminders
    m = re.match(r"remind me in (\d+)\s*(minute|minutes|second|seconds|hour|hours)\s+to\s+(.+)", text, re.IGNORECASE)
    if m:
        return _handle_reminder_in(m)
    m = re.match(r"remind me tomorrow at\s+(\w+(?::\d{2})?(?:\s*[ap]\.?m\.?)?)\s+to\s+(.+)", text, re.IGNORECASE)
    if m:
        return _handle_reminder_tomorrow(m)
    m = re.match(r"remind me at (\w+(?::\d{2})?(?:\s*[ap]\.?m\.?)?)\s+to\s+(.+)", text, re.IGNORECASE)
    if m:
        return _handle_reminder_at(m)
    m = re.match(r"remind me to\s+(.+?)\s+at\s+(\w+(?::\d{2})?(?:\s*[ap]\.?m\.?)?)\s*\.?$", text, re.IGNORECASE)
    if m:
        return _handle_reminder_to_at(m)

    # 5. Todo
    if "what do i need to do" in text.lower():
        return _handle_todo_check()

    # 6. Last said
    if re.search(r'\bwhat did i (?:just )?say\b', text, re.IGNORECASE):
        return _handle_last_said()

    # 7. Memory recall
    if re.search(r'\bwhat were we\b', text, re.IGNORECASE):
        return _handle_memory()

    # 8. Age
    if "how old am i" in text.lower():
        return _handle_age()

    # 9. Name store
    if re.search(r'\bmy name is\b|\bcall me\b|\bmy name.s not\b|\bmy name isn.t\b|\bthat.s not my name\b|\bname.s actually\b', text, re.IGNORECASE):
        return _handle_name_store(text)

    # 10. Birthday store
    if re.search(r'\bmy birthday is\s', text, re.IGNORECASE):
        return _handle_birthday_store(text)

    # 11. Birthday recall
    if re.search(r'\b(?:when|what).{0,10}my birthday\b', text, re.IGNORECASE):
        return _handle_birthday_recall()

    # 12. Name recall
    if "what's my name" in text.lower() or "whats my name" in text.lower() or "what is my name" in text.lower():
        return _handle_name_recall()

    # 13. On this day
    otd = _parse_on_this_day(text)
    if otd:
        return _handle_on_this_day(text, *otd)

    # 14. Financial routing
    if (_last_action == "trade" and _last_trade_thought) or _FINANCIAL_HINT_RE.search(text):
        return _handle_route_financial(text)

    # 15. Default: chat
    return _handle_chat(text)


# ═══════════════════════════════════════════════════════════════
#  FLASK APP
# ═══════════════════════════════════════════════════════════════

app = Flask(__name__)


@app.route('/api/status')
def api_status():
    return jsonify({"ok": True})


@app.route('/api/transcribe', methods=['POST'])
def api_transcribe():
    data = request.json
    audio_b64 = data.get('audio_b64', '')
    sr = data.get('sample_rate', 16000)
    try:
        pcm_bytes = base64.b64decode(audio_b64)
        audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = _whisper_model.transcribe(audio_np, beam_size=5)
        text = " ".join(seg.text for seg in segments).strip()
    except Exception as exc:
        print(f"[Transcribe] Error: {exc}")
        text = ""
    return jsonify({"text": text})


@app.route('/api/handle', methods=['POST'])
def api_handle():
    data = request.json
    text = data.get('text', '')
    mood = data.get('mood', {"energy_avg": 0, "energy_peak": 0, "level": "normal"})
    response, action = _route_and_handle(text, mood)
    return jsonify({
        "response": response,
        "action": action,
        "state": {
            "pending_clarification": _pending_clarification is not None,
            "pending_price_conflict": _pending_price_conflict_state is not None,
            "last_action": _last_action,
            "user_name": _user_name,
        }
    })


@app.route('/api/pending')
def api_pending():
    alerts = []
    while _pending_alerts_queue:
        alerts.append(_pending_alerts_queue.pop(0))
    while _pending_news_ideas:
        idea = _pending_news_ideas.pop(0)
        alerts.append({"type": "trade_idea", "text": idea})
    while _pending_kalshi_alerts:
        alert = _pending_kalshi_alerts.pop(0)
        alerts.append({"type": "kalshi", "text": alert})
    while _pending_price_conflicts:
        c = _pending_price_conflicts.pop(0)
        global _pending_price_conflict_state
        _pending_price_conflict_state = c
        alerts.append({
            "type": "price_conflict", "ticker": c["ticker"],
            "text": (f"Hey — I'm seeing two different prices for {c['ticker']}. "
                     f"Robinhood says ${c['rh_price']:.2f} ({c['rh_pct']:+.1f}%), "
                     f"Yahoo says ${c['yf_price']:.2f} ({c['yf_pct']:+.1f}%). "
                     f"That's a {c['divergence']:.1f}% gap. Which one do you trust?"),
            "conflict_data": c,
        })
    return jsonify({"alerts": alerts})


@app.route('/api/user')
def api_user():
    return jsonify({
        "name": _user_name,
        "birthday": _user_data_store.get("birthday"),
        "facts": _user_data_store.get("facts", []),
    })


@app.route('/api/check_birthday', methods=['POST'])
def api_check_birthday():
    bday = _user_data_store.get("birthday")
    if not bday:
        return jsonify({"is_birthday": False, "message": ""})
    today = datetime.now().strftime("%m-%d")
    if today != bday:
        return jsonify({"is_birthday": False, "message": ""})
    name = _user_name or "you"
    try:
        result = ollama.chat(model=OLLAMA_CHAT_MODEL, options=_MISTRAL_OPTS, messages=[
            {"role": "system", "content": PHOEBE_SYSTEM_PROMPT},
            {"role": "user", "content": f"It's {name}'s birthday today. Wish them happy birthday. Two sentences max."}])
        msg = result.message.content.strip()
    except Exception:
        msg = f"Happy birthday, {name}."
    return jsonify({"is_birthday": True, "message": msg})


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def _bg_alpaca_spy_loop():
    """Log SPY price hourly via Alpaca (Alpha Vantage fallback). Background thread only."""
    time.sleep(15)
    while True:
        def _fetch():
            price = _get_price_alpaca("SPY")
            source = "alpaca"
            if price is None:
                price = _get_price_alpha_vantage("SPY")
                source = "alphavantage"
            if price is not None:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                line = f"{ts} | SPY_PRICE | ${price:.2f} [{source}]\n"
                print(f"[AlpacaPaper] SPY: ${price:.2f} [{source}]")
                def _write(l=line):
                    with open(_TRADE_LOG_PATH, "a", encoding="utf-8") as f:
                        f.write(l)
                threading.Thread(target=_write, daemon=True).start()
        threading.Thread(target=_fetch, daemon=True).start()
        time.sleep(3600)


def main():
    threading.Thread(target=_ensure_model, daemon=True).start()
    _scheduler.start()
    for r in list(_memory_data.get("reminders", [])):
        _schedule_reminder(r["time"], r["task"])
    threading.Thread(target=_bg_news_loop, daemon=True).start()
    threading.Thread(target=_bg_ceo_loop, daemon=True).start()
    threading.Thread(target=_bg_kalshi_loop, daemon=True).start()
    if _ALPACA_AVAILABLE:
        threading.Thread(target=_bg_alpaca_spy_loop, daemon=True).start()
        print("[AlpacaPaper] Paper trading enabled — GTC orders, orion_trades.log")
    print("[Orion] Starting on 0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)


if __name__ == "__main__":
    main()
