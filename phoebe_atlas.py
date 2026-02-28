"""
phoebe_atlas.py  —  PhoebeBot Core Brain  (Atlas)

Always-on headless brain running on Jetson Orin Nano.
Handles: Ollama inference, Whisper transcription, routing, SQLite memory,
         MQTT coordination, and health reporting.

Nodes it talks to:
  Luna  → sends audio/text requests, receives spoken responses
  Orion → sends financial data, receives analysis/routing
  PC    → opportunistic GPU donor (not required)
  Banshee → watchdog (monitors Atlas via ICMP, not MQTT)

MQTT topics (Atlas owns):
  Subscribe:
    phoebe/luna/request    — JSON {request_id, type, data}
    phoebe/orion/request   — JSON {request_id, type, data}
    phoebe/pc/gpu_state    — "FREE" or "GAMING"
  Publish:
    phoebe/atlas/response       — JSON {request_id, type, result}
    phoebe/atlas/alive          — heartbeat (retained, 30s)
    phoebe/atlas/health         — JSON {cpu, mem, disk, uptime, last_task}
    phoebe/atlas/script_preview — JSON {script_id, description, script, timestamp, expires_in}
    phoebe/atlas/script_result  — JSON {script_id, success, output, error}
  Subscribe (added):
    phoebe/atlas/script_approve — JSON {script_id, action: "approve"/"reject"}

Request types (from Luna):
  transcribe     — data: {audio_b64: "<base64 WAV>"}
  chat           — data: {text: "...", history: [...]}
  route          — data: {text: "...", history: [...]}
  user_get       — data: {}  → result: {name: "..."}
  birthday_check — data: {}  → result: {is_birthday: bool, message: "..."}
  health         — data: {}  → result: {cpu_pct, mem_pct, mem_used_gb, mem_total_gb, disk_pct, disk_free_gb, uptime, last_task, finbert_ready, whisper_ready}

Install on Atlas:
  pip install faster-whisper ollama paho-mqtt python-dotenv psutil requests

Run:
  python3 phoebe_atlas.py
"""

import os
import sys
import re
import ast
import json
import time
import base64
import sqlite3
import tempfile
import threading
import subprocess
import collections
from datetime import datetime

import uuid
import requests
import psutil
import ollama
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────
_MQTT_USER     = os.getenv("MQTT_USER", "phoebe")
_MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "")

# Single Brain on Atlas — Jetson 8GB unified memory can't hold both 8B models simultaneously.
# Qwen handles routing AND chat. Mistral remains available on PC (phoebe_backup.py) as overflow
# when PC voice node is built. Revisit dual-model if smaller quantizations become available.
OLLAMA_MODEL      = "qwen3:8b"
OLLAMA_CHAT_MODEL = "qwen3:8b"
OLLAMA_CODE_MODEL = "qwen2.5-coder:1.5b"   # lightweight code model — fast CPU inference for script generation
_QWEN_OPTS    = {"num_gpu": 6}   # 6 GPU layers on Jetson — default num_ctx (2048) fits in unified memory
_MISTRAL_OPTS = {"num_gpu": 6}   # same model as Qwen on Atlas
_CODE_OPTS        = {"num_gpu": 0}                                     # 1.5B model runs fine on CPU — keeps GPU free for routing/chat
_ollama_lock      = threading.Lock()        # prevents concurrent model calls (warmup + request = OOM)
_ollama_code_client = ollama.Client(timeout=180)  # 3-min timeout — CPU generation is slow but shouldn't hang

WHISPER_MODEL = "tiny.en"

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "atlas_memory.db")

_START_TIME = time.time()
_last_task  = "idle"

# ── Home Assistant config ──────────────────────────────────────
_HA_IP    = os.getenv("HA_IP")
_HA_TOKEN = os.getenv("HA_TOKEN", "")

# ── Self-scripting config ──────────────────────────────────────
_SCRIPT_SANDBOX_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phoebe_scripts")
_SCRIPT_RUN_TIMEOUT     = 30    # seconds a script is allowed to run
_SCRIPT_APPROVE_TIMEOUT = 600   # 10 minutes — wait for explicit approve/reject, no countdown pressure
_SCRIPT_MAX_LINES       = 150   # reject scripts longer than this

_script_pending  = {}   # script_id → threading.Event (approval gate)
_script_approved = {}   # script_id → bool (True=approve, False=reject)
_last_script_id  = None # most recently previewed script — used to resolve "approve"/"reject" by voice

# ── SQLite memory ─────────────────────────────────────────────
def _db_connect():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            role      TEXT NOT NULL,
            content   TEXT NOT NULL,
            ts        TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_facts (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            ts    TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS episodes (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            summary TEXT NOT NULL,
            tone    TEXT NOT NULL,
            ts      TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn

_db = _db_connect()
_db_lock = threading.Lock()


def _save_memory(role, content):
    ts = datetime.utcnow().isoformat()
    with _db_lock:
        _db.execute("INSERT INTO memories (role, content, ts) VALUES (?,?,?)",
                    (role, content, ts))
        _db.commit()


def _load_history(limit=20):
    with _db_lock:
        rows = _db.execute(
            "SELECT role, content FROM memories ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]


def _save_fact(key, value):
    ts = datetime.utcnow().isoformat()
    with _db_lock:
        _db.execute(
            "INSERT OR REPLACE INTO user_facts (key, value, ts) VALUES (?,?,?)",
            (key, value, ts)
        )
        _db.commit()


def _load_facts():
    with _db_lock:
        rows = _db.execute("SELECT key, value FROM user_facts").fetchall()
    return {k: v for k, v in rows}


# ── Episodic memory ────────────────────────────────────────────
_EPISODE_TAG_SYSTEM = (
    "Tag this conversation exchange. Reply with JSON only — no explanation.\n"
    '{"tone": "<one word: happy/sad/worried/curious/frustrated/grateful/excited/neutral>", '
    '"summary": "<one sentence describing what was discussed>"}\n/no_think'
)


def _save_episode(user_msg: str, reply: str):
    """Background thread: ask Qwen to tag tone + summary, save to episodes table."""
    prompt = f'User: "{user_msg}"\nPhoebe: "{reply}"'
    raw = _qwen(prompt, system=_EPISODE_TAG_SYSTEM)
    match = re.search(r'\{[^{}]+\}', raw)
    if not match:
        return
    try:
        data = json.loads(match.group())
    except Exception:
        return
    tone    = str(data.get("tone", "neutral"))[:50]
    summary = str(data.get("summary", ""))[:300]
    if not summary:
        return
    ts = datetime.utcnow().isoformat()
    with _db_lock:
        _db.execute(
            "INSERT INTO episodes (summary, tone, ts) VALUES (?,?,?)",
            (summary, tone, ts)
        )
        _db.commit()
    print(f"[Episode] Saved: [{tone}] {summary[:70]}", flush=True)


def _load_episodes(limit=5):
    with _db_lock:
        rows = _db.execute(
            "SELECT tone, summary, ts FROM episodes ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [{"tone": t, "summary": s, "ts": ts} for t, s, ts in reversed(rows)]


# ── Whisper (lazy load) ───────────────────────────────────────
_whisper_model = None
_whisper_ready = threading.Event()


def _load_whisper():
    global _whisper_model
    try:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        print(f"[Whisper] {WHISPER_MODEL} ready.", flush=True)
    except Exception as e:
        print(f"[Whisper] Failed to load: {e}", flush=True)
    _whisper_ready.set()


threading.Thread(target=_load_whisper, daemon=True).start()


def _transcribe_audio(audio_b64: str) -> str:
    """Decode base64 WAV, run Whisper, return transcript."""
    _whisper_ready.wait(timeout=60)
    if _whisper_model is None:
        return ""
    try:
        wav_bytes = base64.b64decode(audio_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp_path = f.name
        segments, _ = _whisper_model.transcribe(
            tmp_path,
            language="en",
            initial_prompt=(
                "turn on, turn off, lights, light, couch, couch light, couch lights, "
                "nightstand, Apple TV, television, TV, living room, bedroom, "
                "dim, brightness, volume, play, pause, mute"
            ),
        )
        transcript = " ".join(s.text.strip() for s in segments).strip()
        os.unlink(tmp_path)
        return transcript
    except Exception as e:
        print(f"[Whisper] Transcription error: {e}", flush=True)
        return ""


# ── FinBERT sentiment ─────────────────────────────────────────
_finbert_pipeline = None
_finbert_ready    = threading.Event()


def _load_finbert():
    global _finbert_pipeline
    try:
        from transformers import pipeline as hf_pipeline
        _finbert_pipeline = hf_pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            top_k=None,
        )
        print("[FinBERT] Ready.", flush=True)
    except Exception as e:
        print(f"[FinBERT] Failed to load: {e}", flush=True)
    _finbert_ready.set()


threading.Thread(target=_load_finbert, daemon=True).start()


def _score_sentiment(texts: list) -> list:
    """Score a list of strings with FinBERT.
    Returns list of {label, score} dicts (highest-scoring label per text).
    Falls back to [] if FinBERT not ready or errored.
    """
    if not _finbert_ready.wait(timeout=5) or _finbert_pipeline is None:
        return []
    try:
        results = _finbert_pipeline(texts, truncation=True, max_length=512)
        return [max(r, key=lambda x: x["score"]) for r in results]
    except Exception as e:
        print(f"[FinBERT] Scoring error: {e}", flush=True)
        return []


# ── Ollama helpers ────────────────────────────────────────────
def _qwen(prompt: str, system: str = "") -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    with _ollama_lock:
        try:
            resp = ollama.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                options=_QWEN_OPTS,
                think=False,
                keep_alive=-1,
            )
            return resp["message"]["content"].strip()
        except Exception as e:
            print(f"[Qwen] Error: {e}", flush=True)
            return ""


def _mistral(messages: list) -> str:
    with _ollama_lock:
        try:
            resp = ollama.chat(
                model=OLLAMA_CHAT_MODEL,
                messages=messages,
                options=_MISTRAL_OPTS,
                think=False,
                keep_alive=-1,
            )
            return resp["message"]["content"].strip()
        except Exception as e:
            print(f"[Mistral] Error: {e}", flush=True)
            return ""


# ── Routing ───────────────────────────────────────────────────
# Fast pre-filter — skip Mistral entirely for obvious home control commands
_HOME_CONTROL_RE = re.compile(
    r'\b(turn\s+(on|off)|switch\s+(on|off)|dim|brighten|set\s+(brightness|color|volume)|'
    r'play|pause|mute|unmute|volume\s+(up|down)|next\s+track|previous\s+track)\b'
    r'.*\b(light|lights|lamp|bulb|tv|television|appletv|apple\s+tv|nightstand|couch|'
    r'bedroom|living\s+room|kitchen|bathroom|vanity|mirror|thermostat|switch)\b'
    r'|\b(light|lights|lamp|tv|television|appletv|nightstand|couch|kitchen|bathroom|vanity|mirror)\b'
    r'.*\b(turn\s+(on|off)|switch\s+(on|off)|on|off)\b',
    re.IGNORECASE,
)

# Fast pre-filter — if none of these keywords are present, it's definitely CHAT
_FINANCIAL_HINT_RE = re.compile(
    r'\b(stock|stocks|ticker|tickers|portfolio|trade|trades|trading|market|markets|'
    r'invest|investment|investing|buy|sell|short|long|option|options|futures|'
    r'earnings|dividend|dividends|watchlist|watch\s+list|'
    r'price|prices|priced|share|shares|equity|equities|'
    r'nasdaq|nyse|s&p|dow|russell|etf|index|indices|'
    r'crypto|bitcoin|btc|ethereum|eth|'
    r'news|headline|headlines|'
    r'money|cash|account|balance|'
    r'kalshi|prediction\s+market|alpaca|robinhood|finbert|sentiment)\b',
    re.IGNORECASE,
)

_ROUTE_SYSTEM = """
You are a routing classifier. Given a user message, classify intent as exactly one of:
CHAT, PORTFOLIO, TRADE_IDEA, NEWS, WATCHLIST, HOME_CONTROL, SCRIPT_REQUEST, AMBIGUOUS

Rules:
- CHAT: personal conversation, general questions, small talk
- PORTFOLIO: questions about their stock holdings, account balance
- TRADE_IDEA: requests for stock/options trade suggestions
- NEWS: market news, earnings, company news requests
- WATCHLIST: adding/removing/viewing watch list tickers
- HOME_CONTROL: controlling lights, thermostat, smart devices
- SCRIPT_REQUEST: user wants to write, create, or automate a script or program
- AMBIGUOUS: unclear — could be personal or financial

Reply with ONLY the single classification word. No explanation.
""".strip()


def _route(text: str, history: list) -> str:
    # Fast path: if deterministic HA parser succeeds, it's HOME_CONTROL — no LLM needed
    if _parse_ha_intent(text) is not None:
        print(f"[Route] HOME_CONTROL (fast parse path): {text!r}", flush=True)
        return "HOME_CONTROL"
    # Regex fast path for commands the parser won't catch (media, ambiguous phrasing)
    if _HOME_CONTROL_RE.search(text):
        print(f"[Route] HOME_CONTROL (regex path): {text!r}", flush=True)
        return "HOME_CONTROL"
    # Fast path for script requests
    if _SCRIPT_RE.search(text):
        print(f"[Route] SCRIPT_REQUEST (regex path): {text!r}", flush=True)
        return "SCRIPT_REQUEST"
    # Fast path: no financial keywords → definitely CHAT, skip Mistral entirely
    if not _FINANCIAL_HINT_RE.search(text):
        print(f"[Route] CHAT (fast path — no financial keywords): {text!r}", flush=True)
        return "CHAT"
    # Use Mistral (GPU) for routing on Atlas — Qwen CPU is too slow on Jetson
    context = "\n".join(f"{m['role']}: {m['content']}" for m in history[-4:])
    msgs = [
        {"role": "system", "content": _ROUTE_SYSTEM},
        {"role": "user",   "content": f"Recent context:\n{context}\n\nNew message: {text}"},
    ]
    result = _mistral(msgs)
    for label in ("CHAT", "PORTFOLIO", "TRADE_IDEA", "NEWS",
                  "WATCHLIST", "HOME_CONTROL", "SCRIPT_REQUEST", "AMBIGUOUS"):
        if label in result.upper():
            return label
    return "CHAT"


# ── Handlers ──────────────────────────────────────────────────
_NAME_RE = re.compile(
    r"(?:my name is|i(?:'m| am)|call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    re.IGNORECASE,
)


def _maybe_extract_name(text: str):
    """Extract and persist the user's name if they mention it."""
    m = _NAME_RE.search(text)
    if m:
        name = m.group(1).strip().title()
        _save_fact("name", name)
        print(f"[Atlas] Saved name: {name}", flush=True)


def _build_chat_messages(text: str, history: list) -> list:
    facts = _load_facts()
    fact_str = "\n".join(f"- {k}: {v}" for k, v in facts.items())
    system = (
        "You are Phoebe, a warm and direct personal AI. "
        "You remember context and speak naturally, not like a robot. "
        "If you do not know the user's name, ask them — never assume or invent one.\n"
    )
    if fact_str:
        system += f"\nWhat you know about the user:\n{fact_str}"
    episodes = _load_episodes(limit=5)
    if episodes:
        ep_str = "\n".join(f"- [{e['tone']}] {e['summary']}" for e in episodes)
        system += f"\n\nRecent memory (past conversations):\n{ep_str}"
    # Inject FinBERT sentiment if ready and score is non-neutral
    scores = _score_sentiment([text])
    if scores:
        top = scores[0]
        if top["label"] != "neutral" and top["score"] > 0.6:
            system += (
                f"\n\nFinBERT sentiment on user's message: {top['label']} "
                f"({top['score']:.0%} confidence). "
                "Factor this into your tone if the topic is financial or market-related."
            )
    msgs = [{"role": "system", "content": system}]
    msgs.extend(history[-10:])
    msgs.append({"role": "user", "content": text})
    return msgs


def handle_chat(text: str, history: list) -> str:
    global _last_task
    _last_task = "chat"
    threading.Thread(target=_maybe_extract_name, args=(text,), daemon=True).start()
    msgs = _build_chat_messages(text, history)
    reply = _mistral(msgs)
    threading.Thread(target=_save_memory, args=("user", text), daemon=True).start()
    threading.Thread(target=_save_memory, args=("assistant", reply), daemon=True).start()
    threading.Thread(target=_save_episode, args=(text, reply), daemon=True).start()
    return reply


def handle_route_and_respond(text: str, history: list) -> dict:
    """Route the text and return {route, reply}."""
    global _last_task

    # Intercept approve/reject before routing — resolves a pending script approval
    if _APPROVE_RE.match(text) and _last_script_id and _last_script_id in _script_pending:
        _script_approved[_last_script_id] = True
        _script_pending[_last_script_id].set()
        return {"route": "SCRIPT_APPROVE", "reply": "Approved. Running the script now."}
    if _REJECT_RE.match(text) and _last_script_id and _last_script_id in _script_pending:
        _script_approved[_last_script_id] = False
        _script_pending[_last_script_id].set()
        return {"route": "SCRIPT_REJECT", "reply": "Script discarded."}

    route = _route(text, history)
    _last_task = f"route:{route}"

    if route == "CHAT":
        reply = handle_chat(text, history)
    elif route == "HOME_CONTROL":
        reply = handle_home_control(text)
    elif route == "SCRIPT_REQUEST":
        reply = handle_script_request(text)
    elif route == "AMBIGUOUS":
        reply = "Could you clarify — are you asking about something personal, or something financial?"
    else:
        # Financial routes — forward to Orion and wait for its response
        reply = _orion_request(route, text)
    return {"route": route, "reply": reply}


# ── Home Assistant ─────────────────────────────────────────────
_HA_ENTITY_NAMES = {
    "light.couch":                         "couch lights",
    "light.couch_left_kauf_bulb":          "left couch light",
    "light.couch_right_kauf_bulb":         "right couch light",
    "media_player.living_room_appletv":    "living room Apple TV",
    "media_player.living_room_television": "living room TV",
    "media_player.bedroom_appletv":        "bedroom Apple TV",
    "light.nightstand":                    "nightstand light",
    "light.kitchen_fixture":               "kitchen light",
    "light.bathroom_vanity":               "bathroom vanity",
    "light.right_mirror_kauf_bulb":        "right mirror",
    "light.mirror_left_kauf_bulb":         "left mirror",
}

_HA_ACTION_PHRASES = {
    "turn_on":              "turned on",
    "turn_off":             "turned off",
    "media_play":           "playing",
    "media_pause":          "paused",
    "volume_up":            "volume up",
    "volume_down":          "volume down",
    "volume_mute":          "muted",
    "media_next_track":     "next track",
    "media_previous_track": "previous track",
}

_HA_CONTROL_PROMPT = """
You are a home control parser. Extract the intent from the user's command and reply with JSON only.

Known entities:
- light.couch — "couch lights", "living room lights", "the lights"
- light.couch_left_kauf_bulb — "left couch light", "couch left"
- light.couch_right_kauf_bulb — "right couch light", "couch right"
- media_player.living_room_appletv — "living room Apple TV", "Apple TV"
- media_player.living_room_television — "living room television", "LG TV", "the TV"
- media_player.bedroom_appletv — "bedroom Apple TV", "bedroom TV"
- light.nightstand — "nightstand", "nightstand light", "bedroom light"
- light.kitchen_fixture — "kitchen light", "kitchen lights", "kitchen"
- light.bathroom_vanity — "bathroom vanity", "bathroom mirror", "bathroom lights", "vanity"
- light.right_mirror_kauf_bulb — "right mirror", "mirror right"
- light.mirror_left_kauf_bulb — "left mirror", "mirror left"

Service map:
- turn on / switch on → {"domain": "homeassistant", "service": "turn_on"}
- turn off / switch off → {"domain": "homeassistant", "service": "turn_off"}
- dim to X% / set brightness X% → {"domain": "light", "service": "turn_on", "params": {"brightness_pct": X}}
- set color to blue / change to red / make it green / light blue / warm white:
    → {"domain": "light", "service": "turn_on", "params": {"color_name": "<css_color>"}}
  Use exact CSS color names. Examples: red, green, blue, white, yellow, purple, orange, pink,
  cyan, magenta, coral, teal, lime, violet, indigo, turquoise, lavender, lightblue, lightyellow,
  lightgreen, hotpink, gold, crimson, navy, royalblue, skyblue, deepskyblue, dodgerblue, steelblue
- warm / cozy / candlelight → {"domain": "light", "service": "turn_on", "params": {"color_temp": 450}}
- cool white / daylight / bright white → {"domain": "light", "service": "turn_on", "params": {"color_temp": 175}}
- play / resume → {"domain": "media_player", "service": "media_play"}
- pause → {"domain": "media_player", "service": "media_pause"}
- volume up → {"domain": "media_player", "service": "volume_up"}
- volume down → {"domain": "media_player", "service": "volume_down"}
- mute → {"domain": "media_player", "service": "volume_mute", "params": {"is_volume_muted": true}}
- unmute → {"domain": "media_player", "service": "volume_mute", "params": {"is_volume_muted": false}}
- next → {"domain": "media_player", "service": "media_next_track"}
- previous → {"domain": "media_player", "service": "media_previous_track"}

Reply with JSON only — no explanation, no markdown fences:
{"entity_id": "...", "domain": "...", "service": "...", "params": {}}

If the device or action is unclear, reply: {"error": "unclear"}
""".strip()


def _ha_call(domain, service, entity_id, params=None):
    """POST to HA REST API. Returns True on success."""
    if not _HA_TOKEN:
        print("[HA] HA_TOKEN not set.", flush=True)
        return False
    url = f"http://{_HA_IP}:8123/api/services/{domain}/{service}"
    headers = {
        "Authorization": f"Bearer {_HA_TOKEN}",
        "Content-Type":  "application/json",
    }
    data = {"entity_id": entity_id}
    if params:
        data.update(params)
    try:
        print(f"[HA] Calling {url} entity={entity_id} params={params}", flush=True)
        resp = requests.post(url, headers=headers, json=data, timeout=10)
        print(f"[HA] Response {resp.status_code}", flush=True)
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"[HA] {domain}/{service} {entity_id} failed: {e}", flush=True)
        return False


# Deterministic HA intent parser — no LLM needed for standard commands
_HA_ENTITY_MAP = [
    (re.compile(r'\bleft\s+couch|couch\s+left\b', re.I),           "light.couch_left_kauf_bulb"),
    (re.compile(r'\bright\s+couch|couch\s+right\b', re.I),         "light.couch_right_kauf_bulb"),
    (re.compile(r'\bright\s+mirror|mirror\s+right\b', re.I),       "light.right_mirror_kauf_bulb"),
    (re.compile(r'\bleft\s+mirror|mirror\s+left\b', re.I),         "light.mirror_left_kauf_bulb"),
    (re.compile(r'\bbedroom\s+(apple\s*tv|tv)\b', re.I),           "media_player.bedroom_appletv"),
    (re.compile(r'\bliving\s+room\s+apple\s*tv\b', re.I),          "media_player.living_room_appletv"),
    (re.compile(r'\bapple\s*tv\b', re.I),                          "media_player.living_room_appletv"),
    (re.compile(r'\b(tv|television|lg)\b', re.I),                  "media_player.living_room_television"),
    (re.compile(r'\bnightstand\b', re.I),                          "light.nightstand"),
    (re.compile(r'\bcouch\b', re.I),                               "light.couch"),
    (re.compile(r'\bkitchen\b', re.I),                             "light.kitchen_fixture"),
    (re.compile(r'\b(bathroom|vanity|mirror)\b', re.I),            "light.bathroom_vanity"),
    (re.compile(r'\b(light|lights|lamp|bulb)\b', re.I),            "light.couch"),
]
_HA_COLORS_RE = re.compile(
    r'\b(red|green|blue|white|yellow|purple|orange|pink|cyan|magenta|coral|teal|lime|'
    r'violet|indigo|turquoise|lavender|lightblue|lightyellow|lightgreen|hotpink|gold|'
    r'crimson|navy|royalblue|skyblue|deepskyblue|dodgerblue|steelblue)\b', re.I,
)


def _parse_ha_intent(text: str):
    """Deterministic parser. Returns intent dict or None if parsing fails."""
    t = text.lower()
    entity_id = None
    for pattern, eid in _HA_ENTITY_MAP:
        if pattern.search(text):
            entity_id = eid
            break
    if not entity_id:
        return None
    is_light = entity_id.startswith("light.")
    is_media = entity_id.startswith("media_player.")

    # Brightness — match "50%", "50 percent", "50 per cent"
    m = re.search(r'(\d+)\s*(?:%|per\s*cent|percent)', t)
    if m and is_light:
        return {"entity_id": entity_id, "domain": "light", "service": "turn_on",
                "params": {"brightness_pct": int(m.group(1))}}
    # Dim / brighten without a specific percentage
    if re.search(r'\bdim\b', t) and is_light:
        return {"entity_id": entity_id, "domain": "light", "service": "turn_on",
                "params": {"brightness_pct": 25}}
    if re.search(r'\b(brighten|brighter|full|max)\b', t) and is_light:
        return {"entity_id": entity_id, "domain": "light", "service": "turn_on",
                "params": {"brightness_pct": 100}}
    # Color temp
    if re.search(r'\b(warm|cozy|candlelight)\b', t) and is_light:
        return {"entity_id": entity_id, "domain": "light", "service": "turn_on",
                "params": {"color_temp": 450}}
    if re.search(r'\b(cool|daylight|bright\s+white)\b', t) and is_light:
        return {"entity_id": entity_id, "domain": "light", "service": "turn_on",
                "params": {"color_temp": 175}}
    # Color name
    cm = _HA_COLORS_RE.search(t)
    if cm and is_light:
        return {"entity_id": entity_id, "domain": "light", "service": "turn_on",
                "params": {"color_name": cm.group(1)}}
    # Media controls
    if is_media:
        if re.search(r'\b(play|resume)\b', t):
            return {"entity_id": entity_id, "domain": "media_player", "service": "media_play", "params": {}}
        if re.search(r'\bpause\b', t):
            return {"entity_id": entity_id, "domain": "media_player", "service": "media_pause", "params": {}}
        if re.search(r'\bvolume\s+up\b', t):
            return {"entity_id": entity_id, "domain": "media_player", "service": "volume_up", "params": {}}
        if re.search(r'\bvolume\s+down\b', t):
            return {"entity_id": entity_id, "domain": "media_player", "service": "volume_down", "params": {}}
        if re.search(r'\bunmute\b', t):
            return {"entity_id": entity_id, "domain": "media_player", "service": "volume_mute",
                    "params": {"is_volume_muted": False}}
        if re.search(r'\bmute\b', t):
            return {"entity_id": entity_id, "domain": "media_player", "service": "volume_mute",
                    "params": {"is_volume_muted": True}}
        if re.search(r'\bnext\b', t):
            return {"entity_id": entity_id, "domain": "media_player", "service": "media_next_track", "params": {}}
        if re.search(r'\b(previous|prev|back)\b', t):
            return {"entity_id": entity_id, "domain": "media_player", "service": "media_previous_track", "params": {}}
    # On / off
    if re.search(r'\b(turn\s+on|switch\s+on)\b|\bon\b', t):
        domain = "light" if is_light else "homeassistant"
        return {"entity_id": entity_id, "domain": domain, "service": "turn_on", "params": {}}
    if re.search(r'\b(turn\s+off|switch\s+off)\b|\boff\b', t):
        domain = "light" if is_light else "homeassistant"
        return {"entity_id": entity_id, "domain": domain, "service": "turn_off", "params": {}}
    return None


def handle_home_control(text: str) -> str:
    global _last_task
    _last_task = "home_control"
    # Fast deterministic parse — no LLM for standard commands
    intent = _parse_ha_intent(text)
    if intent:
        print(f"[HA] Fast parse: {intent}", flush=True)
    else:
        # Fall back to Qwen for unusual phrasing
        raw = _qwen(text, system=_HA_CONTROL_PROMPT)
        print(f"[HA] Qwen raw: {raw!r}", flush=True)
        try:
            intent = json.loads(raw.strip())
        except Exception:
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if not m:
                return "I couldn't parse that as a home control command."
            try:
                intent = json.loads(m.group(0))
            except Exception:
                return "I couldn't parse that as a home control command."
    entity_id = intent.get("entity_id")
    domain    = intent.get("domain")
    service   = intent.get("service")
    if intent.get("error") or not entity_id or not domain or not service:
        return "I'm not sure which device you mean. Can you be more specific?"
    params = intent.get("params") or {}
    ok = _ha_call(domain, service, entity_id, params)
    if ok:
        action = _HA_ACTION_PHRASES.get(service, service.replace("_", " "))
        device = _HA_ENTITY_NAMES.get(entity_id, entity_id.split(".")[-1].replace("_", " "))
        return f"{action.capitalize()} — {device}."
    return "I couldn't reach Home Assistant right now."


# ── Orion request-response ────────────────────────────────────
_orion_pending = {}   # request_id → threading.Event
_orion_results  = {}  # request_id → result string


def _orion_request(route: str, text: str, timeout: int = 60) -> str:
    """Forward a financial request to Orion via MQTT and wait for the response."""
    req_id = str(uuid.uuid4())
    event  = threading.Event()
    _orion_pending[req_id] = event
    _mqtt_client.publish("phoebe/atlas/orion_request", json.dumps({
        "request_id": req_id,
        "route":      route,
        "text":       text,
    }), qos=1)
    print(f"[Atlas] Forwarded {route} to Orion (req={req_id[:8]}...)", flush=True)
    if event.wait(timeout=timeout):
        result = _orion_results.pop(req_id, "")
        _orion_pending.pop(req_id, None)
        return result or f"Orion returned an empty response."
    _orion_pending.pop(req_id, None)
    print(f"[Atlas] Orion timed out for {route}", flush=True)
    return "Orion didn't respond in time. Try again in a moment."


# ── Self-scripting ────────────────────────────────────────────
_SCRIPT_GEN_SYSTEM = """
You are a Python script generator running on a Linux Jetson Orin Nano (Atlas).
Write a Python 3 script that accomplishes exactly what the user describes.

Rules:
- Self-contained. Runs with plain python3, no venv.
- No sudo. No destructive file operations. No rm -rf. No deleting system files.
- Print a clear result or confirmation to stdout when done.
- Must complete within 30 seconds.
- Keep it minimal — do exactly what was asked, nothing more.
- Available: standard library, requests, paho-mqtt, psutil, sqlite3.

Reply with ONLY the Python script. No explanation. No markdown fences.
""".strip()

_SCRIPT_RE = re.compile(
    r'\b(write|create|make|build)\s+(a\s+|me\s+a\s+|me\s+)?(python\s+)?(script|program|automation)\b'
    r'|\bautomate\b',
    re.IGNORECASE,
)
_APPROVE_RE = re.compile(r'^\s*approve\s*$', re.IGNORECASE)
_REJECT_RE  = re.compile(r'^\s*(reject|cancel|no|discard)\s*$', re.IGNORECASE)


def _code(prompt: str, system: str = "") -> str:
    """Call the lightweight code model (qwen2.5-coder:1.5b) on CPU."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    with _ollama_lock:
        try:
            resp = _ollama_code_client.chat(
                model=OLLAMA_CODE_MODEL,
                messages=messages,
                options=_CODE_OPTS,
                keep_alive=-1,
            )
            return resp["message"]["content"].strip()
        except Exception as e:
            print(f"[Code] Error: {e} — falling back to Qwen.", flush=True)
            return _qwen(prompt, system=system)


def _generate_script(description: str) -> str:
    raw = _code(description, system=_SCRIPT_GEN_SYSTEM).strip()
    # Strip markdown fences if model wrapped output (```python ... ```)
    raw = re.sub(r'^```[a-zA-Z]*\n?', '', raw)
    raw = re.sub(r'\n?```$', '', raw)
    return raw.strip()


_SANDBOX_BLOCKED_IMPORTS = {
    "socket", "subprocess", "ftplib", "smtplib", "telnetlib",
    "paramiko", "asyncio", "multiprocessing", "ctypes", "cffi",
}
_SANDBOX_BLOCKED_CALLS = {"eval", "exec", "compile", "__import__"}
_SANDBOX_BLOCKED_OS_ATTRS = {
    "system", "popen", "execv", "execve", "execvp", "execvpe",
    "execl", "execle", "execlp", "execlpe",
    "spawnl", "spawnle", "spawnlp", "spawnlpe",
    "spawnv", "spawnve", "spawnvp", "spawnvpe",
}


def _script_static_analysis(script: str):
    """AST-walk the script and reject anything dangerous. Returns (ok, reason)."""
    if len(script.splitlines()) > _SCRIPT_MAX_LINES:
        return False, f"script too long ({len(script.splitlines())} lines, max {_SCRIPT_MAX_LINES})"
    if "sudo" in script:
        return False, "script contains 'sudo'"
    try:
        tree = ast.parse(script)
    except SyntaxError as e:
        return False, f"syntax error: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name.split(".")[0]
                if mod in _SANDBOX_BLOCKED_IMPORTS:
                    return False, f"blocked import: {mod}"
        if isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod in _SANDBOX_BLOCKED_IMPORTS:
                return False, f"blocked import: {mod}"
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _SANDBOX_BLOCKED_CALLS:
                return False, f"blocked call: {func.id}()"
            if isinstance(func, ast.Attribute):
                if func.attr in _SANDBOX_BLOCKED_CALLS:
                    return False, f"blocked call: .{func.attr}()"
                if func.attr in _SANDBOX_BLOCKED_OS_ATTRS:
                    return False, f"blocked call: os.{func.attr}()"
    return True, ""


def _run_script_sandboxed(script: str, script_id: str) -> dict:
    """Write script to sandbox dir, run with timeout, return {success, output, error}."""
    ok, reason = _script_static_analysis(script)
    if not ok:
        print(f"[Script] {script_id[:8]} — blocked by static analysis: {reason}", flush=True)
        return {"success": False, "output": "", "error": f"Sandbox blocked: {reason}"}
    os.makedirs(_SCRIPT_SANDBOX_DIR, exist_ok=True)
    script_path = os.path.join(_SCRIPT_SANDBOX_DIR, f"script_{script_id[:8]}.py")
    sandbox_env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": _SCRIPT_SANDBOX_DIR,
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    try:
        with open(script_path, "w") as f:
            f.write(script)
        proc = subprocess.run(
            ["python3", script_path],
            capture_output=True,
            text=True,
            timeout=_SCRIPT_RUN_TIMEOUT,
            cwd=_SCRIPT_SANDBOX_DIR,
            env=sandbox_env,
        )
        return {
            "success": proc.returncode == 0,
            "output":  proc.stdout.strip(),
            "error":   proc.stderr.strip(),
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "output": "", "error": "Script timed out."}
    except Exception as e:
        return {"success": False, "output": "", "error": str(e)}
    finally:
        try:
            os.unlink(script_path)
        except Exception:
            pass


def _script_approval_wait(script_id: str, script: str):
    """Background thread: wait for approve/reject, then run or discard."""
    event = _script_pending.get(script_id)
    if not event:
        return
    approved = event.wait(timeout=_SCRIPT_APPROVE_TIMEOUT)
    _script_pending.pop(script_id, None)

    if not approved:
        print(f"[Script] {script_id[:8]} — timed out, discarded.", flush=True)
        _mqtt_client.publish("phoebe/atlas/script_result", json.dumps({
            "script_id": script_id, "success": False,
            "output": "", "error": "Timed out — discarded.",
        }), qos=1)
        return

    action = _script_approved.pop(script_id, False)
    if not action:
        print(f"[Script] {script_id[:8]} — rejected.", flush=True)
        _mqtt_client.publish("phoebe/atlas/script_result", json.dumps({
            "script_id": script_id, "success": False,
            "output": "", "error": "Rejected.",
        }), qos=1)
        return

    print(f"[Script] {script_id[:8]} — approved, running...", flush=True)
    result = _run_script_sandboxed(script, script_id)
    _mqtt_client.publish("phoebe/atlas/script_result", json.dumps({
        "script_id": script_id, **result,
    }), qos=1)
    status = "success" if result["success"] else "failed"
    print(f"[Script] {script_id[:8]} — {status}: {(result['output'] or result['error'])[:100]}", flush=True)


def handle_script_request(description: str) -> str:
    """Generate script, publish preview, start async approval wait. Returns immediately."""
    global _last_task, _last_script_id
    _last_task = "script_request"

    script_id = str(uuid.uuid4())
    print(f"[Script] Generating for: {description!r}", flush=True)
    script = _generate_script(description)
    if not script.strip():
        return "I couldn't generate a script for that. Try being more specific."

    _last_script_id = script_id
    _mqtt_client.publish("phoebe/atlas/script_preview", json.dumps({
        "script_id":   script_id,
        "description": description,
        "script":      script,
        "timestamp":   datetime.utcnow().isoformat(),
        "expires_in":  _SCRIPT_APPROVE_TIMEOUT,
    }), qos=1)
    print(f"[Script] Preview published (id={script_id[:8]}...)", flush=True)

    event = threading.Event()
    _script_pending[script_id] = event
    threading.Thread(target=_script_approval_wait, args=(script_id, script), daemon=True).start()

    return (
        "Script ready. Take a look at the preview on phoebe/atlas/script_preview. "
        "Say 'approve' to run it or 'reject' to discard it — no rush, I'll wait."
    )


# ── MQTT ──────────────────────────────────────────────────────
_mqtt_ever_connected = False


def _handle_request(payload: dict):
    """Dispatch a request from Luna or Orion in a background thread."""
    req_id   = payload.get("request_id", "")
    req_type = payload.get("type", "")
    data     = payload.get("data", {})
    result   = {}

    try:
        if req_type == "transcribe":
            transcript = _transcribe_audio(data.get("audio_b64", ""))
            result = {"text": transcript}

        elif req_type == "chat":
            text    = data.get("text", "")
            history = data.get("history", [])
            reply   = handle_chat(text, history)
            result  = {"reply": reply}

        elif req_type == "route":
            text    = data.get("text", "")
            history = data.get("history", [])
            result  = handle_route_and_respond(text, history)

        elif req_type == "user_get":
            facts  = _load_facts()
            result = {"name": facts.get("name", "")}

        elif req_type == "birthday_check":
            facts       = _load_facts()
            birthday    = facts.get("birthday", "")
            is_birthday = False
            message     = ""
            if birthday:
                today = datetime.utcnow()
                parts = birthday.split("-")
                try:
                    if len(parts) == 2:
                        b_month, b_day = int(parts[0]), int(parts[1])
                    elif len(parts) == 3:
                        b_month, b_day = int(parts[1]), int(parts[2])
                    else:
                        b_month, b_day = 0, 0
                    if b_month == today.month and b_day == today.day:
                        is_birthday = True
                        name    = facts.get("name", "")
                        message = f"Happy birthday{', ' + name if name else ''}! I hope today is wonderful."
                except (ValueError, IndexError):
                    pass
            result = {"is_birthday": is_birthday, "message": message}

        elif req_type == "health":
            cpu  = psutil.cpu_percent(interval=1)
            mem  = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            uptime_secs = int(time.time() - _START_TIME)
            h, rem = divmod(uptime_secs, 3600)
            m, s   = divmod(rem, 60)
            result = {
                "cpu_pct":    cpu,
                "mem_pct":    mem.percent,
                "mem_used_gb": round(mem.used / 1e9, 1),
                "mem_total_gb": round(mem.total / 1e9, 1),
                "disk_pct":   disk.percent,
                "disk_free_gb": round(disk.free / 1e9, 1),
                "uptime":     f"{h}h {m}m {s}s",
                "last_task":  _last_task,
                "finbert_ready": _finbert_ready.is_set() and _finbert_pipeline is not None,
                "whisper_ready": _whisper_model is not None,
            }

        else:
            result = {"error": f"Unknown request type: {req_type}"}

    except Exception as e:
        print(f"[Atlas] Handler error ({req_type}): {e}", flush=True)
        result = {"error": str(e)}

    response = json.dumps({
        "request_id": req_id,
        "type": req_type,
        "result": result,
    })
    _mqtt_client.publish("phoebe/atlas/response", response, qos=1)


def _mqtt_on_connect(client, userdata, flags, rc, properties=None):
    global _mqtt_ever_connected
    if str(rc) != "Success":
        print(f"[MQTT] Connection refused: {rc}", flush=True)
        return
    _mqtt_ever_connected = True
    print("[MQTT] Atlas connected to broker.", flush=True)
    client.subscribe("phoebe/luna/request",         qos=1)
    client.subscribe("phoebe/orion/request",        qos=1)
    client.subscribe("phoebe/orion/atlas_response", qos=1)
    client.subscribe("phoebe/pc/gpu_state",         qos=1)
    client.subscribe("phoebe/atlas/script_approve", qos=1)
    client.publish("phoebe/atlas/alive", "Atlas online", retain=True, qos=1)
    print("[MQTT] Subscribed and published alive.", flush=True)


def _mqtt_on_message(client, userdata, msg):
    # Plain-string topics — handle before JSON parse
    if msg.topic == "phoebe/pc/gpu_state":
        state = msg.payload.decode("utf-8", errors="ignore").strip()
        print(f"[MQTT] PC gpu_state → {state}", flush=True)
        return

    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except Exception:
        print(f"[MQTT] Bad JSON on {msg.topic}", flush=True)
        return

    # Script approval from any node (manual MQTT or Luna voice)
    if msg.topic == "phoebe/atlas/script_approve":
        script_id = payload.get("script_id", "")
        action    = payload.get("action", "").lower()
        if script_id in _script_pending:
            _script_approved[script_id] = (action == "approve")
            _script_pending[script_id].set()
            print(f"[Script] {script_id[:8]} — {action} received via MQTT.", flush=True)
        return

    # Orion financial response — resolve pending _orion_request() call
    if msg.topic == "phoebe/orion/atlas_response":
        req_id = payload.get("request_id", "")
        result = payload.get("result", "")
        if req_id in _orion_pending:
            _orion_results[req_id] = result
            _orion_pending[req_id].set()
        return

    print(f"[MQTT] {msg.topic} → type={payload.get('type','?')}", flush=True)
    threading.Thread(target=_handle_request, args=(payload,), daemon=True).start()


def _mqtt_on_disconnect(client, userdata, disconnect_flags, reason_code, properties=None):
    if not _mqtt_ever_connected:
        print("[MQTT] Auth failed — check MQTT_USER/MQTT_PASSWORD in .env.", flush=True)
        return
    print("[MQTT] Disconnected — reconnecting...", flush=True)
    def _reconnect():
        delay = 1
        while True:
            try:
                client.reconnect()
                client.publish("phoebe/atlas/alive", "Atlas online", retain=True, qos=1)
                print("[MQTT] Reconnected.", flush=True)
                break
            except Exception:
                time.sleep(delay)
                delay = min(delay * 2, 60)
    threading.Thread(target=_reconnect, daemon=True).start()


_mqtt_client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2,
    client_id="phoebe-atlas",
    clean_session=False,
)
_mqtt_client.username_pw_set(_MQTT_USER, _MQTT_PASSWORD)
_mqtt_client.will_set("phoebe/atlas/alive", "dead", retain=True)
_mqtt_client.on_connect  = _mqtt_on_connect
_mqtt_client.on_message  = _mqtt_on_message
_mqtt_client.on_disconnect = _mqtt_on_disconnect

try:
    _mqtt_client.connect("localhost", 1883, 60)
    _mqtt_client.loop_start()
except Exception as e:
    print(f"[MQTT] Could not connect: {e}", flush=True)


# ── Heartbeat ─────────────────────────────────────────────────
def _heartbeat_loop():
    while True:
        time.sleep(30)
        try:
            _mqtt_client.publish("phoebe/atlas/alive", "still here", retain=True, qos=1)
            uptime = int(time.time() - _START_TIME)
            health = {
                "cpu":       psutil.cpu_percent(interval=1),
                "mem":       psutil.virtual_memory().percent,
                "disk":      psutil.disk_usage("/").percent,
                "uptime_s":  uptime,
                "last_task": _last_task,
            }
            _mqtt_client.publish("phoebe/atlas/health", json.dumps(health),
                                 retain=True, qos=1)
        except Exception:
            pass


threading.Thread(target=_heartbeat_loop, daemon=True).start()


# ── Main ──────────────────────────────────────────────────────
def _warmup():
    """Pre-warm Qwen and the code model at startup."""
    print("[Atlas] Warming up Qwen...", flush=True)
    _qwen("hello", system="Reply with one word: ready")
    print("[Atlas] Qwen warm.", flush=True)
    print("[Atlas] Warming up code model...", flush=True)
    _code("print('hello')", system="Reply with one word: ready")
    print("[Atlas] Code model warm.", flush=True)


if __name__ == "__main__":
    print("[Atlas] Core brain starting...", flush=True)
    print(f"[Atlas] DB: {DB_PATH}", flush=True)
    print(f"[Atlas] Single model: {OLLAMA_MODEL} (CPU) — handles routing + chat", flush=True)
    threading.Thread(target=_warmup, daemon=True).start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Atlas] Shutdown.", flush=True)
        _mqtt_client.publish("phoebe/atlas/alive", "offline", retain=True, qos=1)
        _mqtt_client.disconnect()
