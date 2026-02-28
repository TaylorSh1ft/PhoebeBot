"""
phoebe_luna.py  –  PhoebeBot Thin Client  (Phoebe-Luna)

Keeps: voice, camera, face recognition, TTS, GUI (when display available).
Runs headless on Pi (terminal I/O, no tkinter, camera captures without display).
Forwards LLM, transcription, and routing to Atlas (phoebe_atlas.py) via MQTT.
Local handlers: weather, time, face recognition, TTS, wake word.
"""

import sys
import threading
import re
import os
import time
import json
import base64
from datetime import datetime, timedelta
import pytz
import collections
import pickle
import numpy as np
import cv2
import requests
import pyaudio
import onnxruntime as ort
from skimage.transform import SimilarityTransform
from emoji import replace_emoji
from dotenv import load_dotenv
import io
import uuid
import wave as _wave
import subprocess as _subprocess
import queue as _queue_mod
import paho.mqtt.client as mqtt

_HEADLESS = not os.environ.get("DISPLAY") and sys.platform != "win32"

if not _HEADLESS:
    import tkinter as tk
    from tkinter import scrolledtext

load_dotenv()

PV_ACCESS_KEY = os.getenv("PV_ACCESS_KEY")

# ── Wake word ─────────────────────────────────────────────────
try:
    from openwakeword.model import Model as _OWWModel
    _OWW_AVAILABLE = True
except ImportError:
    _OWWModel = None
    _OWW_AVAILABLE = False

_tts_playing         = threading.Event()   # set while TTS audio is playing (suppresses wake word)
_mic_for_command     = threading.Event()   # set while listen_from_mic() is consuming frames
_command_audio_queue = _queue_mod.Queue(maxsize=500)  # 16kHz frames routed from wake word loop
_mic_listen_lock     = threading.Lock()    # only one listen_from_mic() at a time


# ── MQTT ─────────────────────────────────────────────────────
_MQTT_BROKER      = os.getenv("MQTT_BROKER")
_MQTT_USER        = os.getenv("MQTT_USER", "phoebe")
_MQTT_PASSWORD    = os.getenv("MQTT_PASSWORD", "")
_pending_responses = {}   # request_id → threading.Event
_response_results  = {}   # request_id → result dict
_active_bot        = None  # set in main() after PhoebeChat is created


_mqtt_ever_connected = False


def _mqtt_on_connect(client, userdata, flags, rc, properties=None):
    global _mqtt_ever_connected
    if str(rc) != "Success":
        print(f"[MQTT] Connection refused: {rc} — check MQTT_USER/MQTT_PASSWORD in .env", flush=True)
        return
    _mqtt_ever_connected = True
    print(f"[MQTT] Connected.", flush=True)
    client.subscribe("phoebe/orion/#")
    client.subscribe("phoebe/atlas/response",        qos=1)
    client.subscribe("phoebe/atlas/script_preview",  qos=1)
    client.subscribe("phoebe/luna/command",          qos=1)
    client.publish("phoebe/luna/ready", "Luna ready", retain=True, qos=1)
    print("[MQTT] Published: Luna ready", flush=True)


def _mqtt_on_message(client, userdata, msg):
    payload = msg.payload.decode("utf-8", errors="replace")
    if msg.topic == "phoebe/atlas/response":
        try:
            data   = json.loads(payload)
            req_id = data.get("request_id", "")
            if req_id in _pending_responses:
                _response_results[req_id] = data.get("result", {})
                _pending_responses[req_id].set()
        except Exception:
            pass
        return
    print(f"[MQTT] {msg.topic}: {payload}", flush=True)
    if msg.topic == "phoebe/orion/command":
        threading.Thread(target=lambda: speak_text_sync(payload), daemon=True).start()
    elif msg.topic == "phoebe/atlas/script_preview":
        try:
            data = json.loads(payload)
            threading.Thread(target=_handle_script_preview, args=(data,), daemon=True).start()
        except Exception:
            pass
        return
    elif msg.topic == "phoebe/luna/command":
        if _active_bot and payload.strip():
            threading.Thread(target=_active_bot._dispatch_input, args=(payload.strip(),), daemon=True).start()
        return
    elif msg.topic == "phoebe/orion/alert":
        try:
            alert = json.loads(payload)
            atype = alert.get("type", "")
            text  = alert.get("text", "")
            if atype == "price_conflict" and _active_bot:
                _active_bot._orion_state["pending_price_conflict"] = True
            if text:
                if _active_bot:
                    _active_bot._after(0, lambda t=text: _active_bot.append_message("Phoebe", t, "phoebe"))
                threading.Thread(target=lambda t=text: speak_text_sync(t), daemon=True).start()
        except Exception:
            pass


def _handle_script_preview(data):
    """Background thread: speak script preview, listen for approve/reject, publish decision."""
    script_id   = data.get("script_id", "")
    description = data.get("description", "unknown task")

    speak_text_sync(
        f"Script ready: {description}. Say approve to run it, or reject to discard it."
    )

    text = listen_from_mic()

    if text and re.search(r'\b(approve|yes|run it|do it|go ahead|confirm)\b', text, re.IGNORECASE):
        action = "approve"
        speak_text_sync("Approved. Running the script now.")
    elif text and re.search(r'\b(reject|no|cancel|stop|don\'t)\b', text, re.IGNORECASE):
        action = "reject"
        speak_text_sync("Script rejected.")
    else:
        action = "reject"
        speak_text_sync("I didn't catch that. Rejecting for safety.")

    _mqtt_client.publish("phoebe/atlas/script_approve", json.dumps({
        "script_id": script_id,
        "action":    action,
    }), qos=1)
    print(f"[Script] {script_id[:8]} — voice approval: {action}. heard: {text!r}", flush=True)


def _mqtt_on_disconnect_luna(client, userdata, disconnect_flags, reason_code, properties=None):
    if not _mqtt_ever_connected:
        print("[MQTT] Auth failed — not retrying. Fix MQTT_USER/MQTT_PASSWORD in .env.", flush=True)
        return
    def _reconnect():
        delay = 1
        while True:
            try:
                client.reconnect()
                client.publish("phoebe/luna/ready", "Luna ready", retain=True, qos=1)
                print("[MQTT] Reconnected.", flush=True)
                break
            except Exception:
                time.sleep(delay)
                delay = min(delay * 2, 60)
    threading.Thread(target=_reconnect, daemon=True).start()


_mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                            client_id="phoebe-luna", clean_session=False)
_mqtt_client.username_pw_set(_MQTT_USER, _MQTT_PASSWORD)
_mqtt_client.will_set("phoebe/luna/alive", "dead", retain=True)
_mqtt_client.on_connect = _mqtt_on_connect
_mqtt_client.on_message = _mqtt_on_message
_mqtt_client.on_disconnect = _mqtt_on_disconnect_luna
try:
    _mqtt_client.connect(_MQTT_BROKER, 1883, 60)
    _mqtt_client.loop_start()
except Exception as _mqtt_err:
    print(f"[MQTT] Could not connect: {_mqtt_err}", flush=True)


def _mqtt_heartbeat_luna():
    while True:
        time.sleep(30)
        try:
            _mqtt_client.publish("phoebe/luna/alive", "still here", retain=True, qos=1)
        except Exception:
            pass


threading.Thread(target=_mqtt_heartbeat_luna, daemon=True).start()


# ── Atlas backend (MQTT) ──────────────────────────────────────
def _atlas_request(req_type, data, timeout=30):
    """Send a request to Atlas via MQTT and wait for the response."""
    req_id = str(uuid.uuid4())
    event  = threading.Event()
    _pending_responses[req_id] = event
    _mqtt_client.publish("phoebe/luna/request", json.dumps({
        "request_id": req_id,
        "type":       req_type,
        "data":       data,
    }), qos=1)
    if event.wait(timeout=timeout):
        result = _response_results.pop(req_id, {})
        _pending_responses.pop(req_id, None)
        return result
    _pending_responses.pop(req_id, None)
    print(f"[Atlas] Request timed out: {req_type}", flush=True)
    return {}


def _orion_transcribe(pcm_int16_array):
    """Send audio to Atlas for Whisper transcription via MQTT."""
    buf = io.BytesIO()
    with _wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm_int16_array.tobytes())
    audio_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    result = _atlas_request("transcribe", {"audio_b64": audio_b64}, timeout=30)
    return result.get("text") or None


def _orion_handle(text, mood=None):
    """Send text to Atlas for routing and response via MQTT."""
    result = _atlas_request("route", {
        "text":    text,
        "history": [],
        "mood":    mood or _user_mood,
    }, timeout=180)
    if result:
        return {"response": result.get("reply", ""), "state": {}}
    return None


# ── Geocoding (local — for weather) ──────────────────────────
_geocode_cache = {
    "indianapolis": ("Indianapolis", 39.77, -86.16),
    "new york": ("New York", 40.71, -74.01),
    "los angeles": ("Los Angeles", 34.05, -118.24),
    "chicago": ("Chicago", 41.88, -87.63),
    "houston": ("Houston", 29.76, -95.37),
    "phoenix": ("Phoenix", 33.45, -112.07),
    "philadelphia": ("Philadelphia", 39.95, -75.17),
    "san antonio": ("San Antonio", 29.42, -98.49),
    "san diego": ("San Diego", 32.72, -117.16),
    "dallas": ("Dallas", 32.78, -96.80),
    "miami": ("Miami", 25.76, -80.19),
    "atlanta": ("Atlanta", 33.75, -84.39),
    "boston": ("Boston", 42.36, -71.06),
    "seattle": ("Seattle", 47.61, -122.33),
    "denver": ("Denver", 39.74, -104.99),
    "nashville": ("Nashville", 36.16, -86.78),
    "detroit": ("Detroit", 42.33, -83.05),
    "portland": ("Portland", 45.52, -122.68),
    "las vegas": ("Las Vegas", 36.17, -115.14),
    "san francisco": ("San Francisco", 37.77, -122.42),
    "washington": ("Washington", 38.91, -77.04),
    "columbus": ("Columbus", 39.96, -82.99),
    "fort wayne": ("Fort Wayne", 41.08, -85.14),
    "paris": ("Paris", 48.85, 2.35),
    "london": ("London", 51.51, -0.13),
    "tokyo": ("Tokyo", 35.68, 139.69),
    "seoul": ("Seoul", 37.57, 126.98),
    "beijing": ("Beijing", 39.90, 116.40),
    "shanghai": ("Shanghai", 31.23, 121.47),
    "mumbai": ("Mumbai", 19.08, 72.88),
    "delhi": ("Delhi", 28.61, 77.21),
    "bangkok": ("Bangkok", 13.76, 100.50),
    "singapore": ("Singapore", 1.35, 103.82),
    "sydney": ("Sydney", -33.87, 151.21),
    "melbourne": ("Melbourne", -37.81, 144.96),
    "berlin": ("Berlin", 52.52, 13.41),
    "madrid": ("Madrid", 40.42, -3.70),
    "rome": ("Rome", 41.90, 12.50),
    "amsterdam": ("Amsterdam", 52.37, 4.90),
    "moscow": ("Moscow", 55.76, 37.62),
    "cairo": ("Cairo", 30.04, 31.24),
    "dubai": ("Dubai", 25.20, 55.27),
    "toronto": ("Toronto", 43.65, -79.38),
    "vancouver": ("Vancouver", 49.28, -123.12),
    "mexico city": ("Mexico City", 19.43, -99.13),
    "manila": ("Manila", 14.60, 120.98),
    "hong kong": ("Hong Kong", 22.32, 114.17),
    "taipei": ("Taipei", 25.03, 121.57),
    "jakarta": ("Jakarta", -6.21, 106.85),
    "istanbul": ("Istanbul", 41.01, 28.98),
    "nairobi": ("Nairobi", -1.29, 36.82),
    "johannesburg": ("Johannesburg", -26.20, 28.05),
    "auckland": ("Auckland", -36.85, 174.76),
    "honolulu": ("Honolulu", 21.31, -157.86),
}

CITY_COUNTRY = {
    "manila": "Manila, Philippines", "paris": "Paris, France",
    "london": "London, United Kingdom", "tokyo": "Tokyo, Japan",
    "seoul": "Seoul, South Korea", "beijing": "Beijing, China",
    "shanghai": "Shanghai, China", "mumbai": "Mumbai, India",
    "delhi": "Delhi, India", "bangkok": "Bangkok, Thailand",
    "singapore": "Singapore", "sydney": "Sydney, Australia",
    "melbourne": "Melbourne, Australia", "berlin": "Berlin, Germany",
    "madrid": "Madrid, Spain", "rome": "Rome, Italy",
    "amsterdam": "Amsterdam, Netherlands", "moscow": "Moscow, Russia",
    "cairo": "Cairo, Egypt", "dubai": "Dubai, United Arab Emirates",
    "toronto": "Toronto, Canada", "vancouver": "Vancouver, Canada",
    "mexico city": "Mexico City, Mexico", "hong kong": "Hong Kong",
    "taipei": "Taipei, Taiwan", "jakarta": "Jakarta, Indonesia",
    "auckland": "Auckland, New Zealand", "johannesburg": "Johannesburg, South Africa",
}

_ALLOWED_TYPES = {"city", "town", "village", "municipality", "administrative", "state", "country"}


def geocode(city):
    key = city.lower().strip()
    if key in _geocode_cache:
        return _geocode_cache[key]
    query = CITY_COUNTRY.get(key, city)
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 1, "accept-language": "en-US", "namedetails": 1},
            headers={"User-Agent": "PhoebeBot/1.0"}, timeout=5)
        results = resp.json()
        if results:
            hit = results[0]
            if hit.get("type", "") not in _ALLOWED_TYPES:
                return None
            lat = float(hit["lat"])
            lon = float(hit["lon"])
            nd = hit.get("namedetails", {})
            name = nd.get("name:en") or nd.get("name") or CITY_COUNTRY.get(key, city).split(",")[0]
            _geocode_cache[key] = (name, lat, lon)
            return name, lat, lon
    except Exception:
        pass
    return None


# ── Weather parsing (local) ──────────────────────────────────
_TIME_PATTERNS = [
    (re.compile(r'\bright\s*now\b'), "now"),
    (re.compile(r'\bcurrently\b'), "now"),
    (re.compile(r'\btonight\b'), "today"),
    (re.compile(r'\btoday\b'), "today"),
    (re.compile(r'\btomorrow\b'), "tomorrow"),
]

_WEATHER_FOLLOWUP_RE = re.compile(
    r'^(?:and\s+|what\s+about\s+|how\s+about\s+|but\s+)?'
    r'(?:what\s+about\s+|how\s+about\s+)?'
    r'(?:tomorrow|today|tonight|right\s*now|currently)\s*\??',
    re.IGNORECASE)


def _strip_time_words(text):
    mode = "today"
    for pat, time_mode in _TIME_PATTERNS:
        if pat.search(text):
            mode = time_mode
            text = pat.sub('', text)
    return re.sub(r'\s+', ' ', text).strip(), mode


def _extract_city(text):
    m = re.search(r'(?:like\s+)?in\s+(\S.+)', text, re.IGNORECASE)
    if m:
        city = m.group(1).strip().rstrip('?.')
        return city if city else None
    return None


def parse_weather_query(user_input):
    text = user_input.lower()
    text, mode = _strip_time_words(text)
    city_str = _extract_city(text)
    if city_str:
        city_str, _ = _strip_time_words(city_str)
    if city_str:
        result = geocode(city_str)
        if result:
            return (*result, mode)
        return None
    try:
        r = requests.get("http://ip-api.com/json/?fields=city,lat,lon", timeout=5)
        if r.status_code == 200:
            d = r.json()
            city = d.get("city", "here")
            lat = d.get("lat")
            lon = d.get("lon")
            if lat and lon:
                return (city, lat, lon, mode)
    except Exception:
        pass
    return None


CITY_TIMEZONE = {
    "indianapolis": "America/Indiana/Indianapolis", "new york": "America/New_York",
    "los angeles": "America/Los_Angeles", "chicago": "America/Chicago",
    "houston": "America/Chicago", "phoenix": "America/Phoenix",
    "philadelphia": "America/New_York", "san antonio": "America/Chicago",
    "san diego": "America/Los_Angeles", "dallas": "America/Chicago",
    "miami": "America/New_York", "atlanta": "America/New_York",
    "boston": "America/New_York", "seattle": "America/Los_Angeles",
    "denver": "America/Denver", "nashville": "America/Chicago",
    "detroit": "America/Detroit", "portland": "America/Los_Angeles",
    "las vegas": "America/Los_Angeles", "louisville": "America/Kentucky/Louisville",
    "columbus": "America/New_York", "fort wayne": "America/Indiana/Indianapolis",
    "manila": "Asia/Manila", "paris": "Europe/Paris", "london": "Europe/London",
    "tokyo": "Asia/Tokyo", "seoul": "Asia/Seoul", "beijing": "Asia/Shanghai",
    "shanghai": "Asia/Shanghai", "mumbai": "Asia/Kolkata", "delhi": "Asia/Kolkata",
    "bangkok": "Asia/Bangkok", "singapore": "Asia/Singapore",
    "sydney": "Australia/Sydney", "melbourne": "Australia/Melbourne",
    "berlin": "Europe/Berlin", "madrid": "Europe/Madrid", "rome": "Europe/Rome",
    "amsterdam": "Europe/Amsterdam", "moscow": "Europe/Moscow",
    "cairo": "Africa/Cairo", "dubai": "Asia/Dubai", "toronto": "America/Toronto",
    "vancouver": "America/Vancouver", "mexico city": "America/Mexico_City",
    "hong kong": "Asia/Hong_Kong", "taipei": "Asia/Taipei",
    "jakarta": "Asia/Jakarta", "auckland": "Pacific/Auckland",
    "johannesburg": "Africa/Johannesburg", "honolulu": "Pacific/Honolulu",
    "san francisco": "America/Los_Angeles", "washington": "America/New_York",
}

WMO_WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    66: "light freezing rain", 67: "heavy freezing rain",
    71: "slight snow", 73: "moderate snow", 75: "heavy snow",
    77: "snow grains", 80: "slight rain showers", 81: "moderate rain showers",
    82: "violent rain showers", 85: "slight snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}

# ── Face recognition ─────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_OWW_MODEL_PATH   = os.path.join(_SCRIPT_DIR, "hey_phoebe.onnx")
_OWW_THRESHOLD    = 0.80
_OWW_ENERGY_GATE      = 35   # min mean-abs energy. TV floor (other room) ~21, voice ~38-46.
_OWW_ENERGY_GATE_MAX  = 80   # max mean-abs energy. Same-room TV floor ~86-149, reject above this.
_OWW_WAKE_ENABLED     = False  # set True to re-enable wake word detection
_FACE_MODELS_DIR = os.path.join(_SCRIPT_DIR, "face_models")
FACE_DB_PATH = os.path.join(_SCRIPT_DIR, "face_embeddings.pkl")
FACE_SIMILARITY_THRESHOLD = 0.45
_FACE_ABSENCE_THRESHOLD = 1800

_ARCFACE_DST = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041],
], dtype=np.float32)

FACE_GREETINGS = {
    "taylor": "Hey Taylor, I'm glad to see you!",
    "mom": "Hi Mom.",
}


def _load_face_db():
    if os.path.exists(FACE_DB_PATH):
        try:
            with open(FACE_DB_PATH, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    return {}


def _save_face_db(face_db):
    with open(FACE_DB_PATH, 'wb') as f:
        pickle.dump(face_db, f)


def _generate_anchors(height, width, stride, num_anchors=2):
    ys, xs = np.mgrid[0:height, 0:width]
    centers = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float32)
    centers = centers * stride + stride // 2
    centers = np.tile(centers, (1, num_anchors)).reshape(-1, 2)
    return centers


def _nms(dets, thresh=0.4):
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= thresh)[0]
        order = order[inds + 1]
    return keep


def _detect_faces(session, img, input_size=(640, 640), conf_thresh=0.5):
    h, w = img.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))
    padded = np.full((input_size[0], input_size[1], 3), 0, dtype=np.uint8)
    padded[:nh, :nw] = resized
    blob = cv2.dnn.blobFromImage(padded, 1.0 / 128.0, input_size, (127.5, 127.5, 127.5), swapRB=True)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: blob})
    strides = [8, 16, 32]
    fmc = 3
    all_scores, all_bboxes, all_kps = [], [], []
    for idx, stride in enumerate(strides):
        sh = input_size[0] // stride
        sw = input_size[1] // stride
        scores = outputs[idx].reshape(-1)
        bbox_preds = outputs[idx + fmc].reshape(-1, 4)
        kps_preds = outputs[idx + fmc * 2].reshape(-1, 10)
        pos = np.where(scores >= conf_thresh)[0]
        if len(pos) == 0:
            continue
        anchors = _generate_anchors(sh, sw, stride)
        d = bbox_preds[pos] * stride
        cx, cy = anchors[pos, 0], anchors[pos, 1]
        x1 = cx - d[:, 0]
        y1 = cy - d[:, 1]
        x2 = cx + d[:, 2]
        y2 = cy + d[:, 3]
        bboxes = np.stack([x1, y1, x2, y2], axis=-1)
        kd = kps_preds[pos] * stride
        kps = np.zeros((len(pos), 5, 2), dtype=np.float32)
        for k in range(5):
            kps[:, k, 0] = anchors[pos, 0] + kd[:, 2 * k]
            kps[:, k, 1] = anchors[pos, 1] + kd[:, 2 * k + 1]
        all_scores.append(scores[pos])
        all_bboxes.append(bboxes)
        all_kps.append(kps)
    if not all_scores:
        return np.empty((0, 5)), np.empty((0, 5, 2))
    scores = np.concatenate(all_scores)
    bboxes = np.concatenate(all_bboxes)
    kpss = np.concatenate(all_kps)
    dets = np.hstack([bboxes, scores[:, None]])
    keep = _nms(dets, 0.4)
    dets = dets[keep]
    kpss = kpss[keep]
    dets[:, :4] /= scale
    kpss /= scale
    return dets, kpss


def _align_face(img, kps, size=112):
    tform = SimilarityTransform()
    tform.estimate(kps, _ARCFACE_DST)
    M = tform.params[0:2, :]
    return cv2.warpAffine(img, M, (size, size), borderValue=0.0)


def _get_embedding(session, face_bgr):
    blob = cv2.dnn.blobFromImage(face_bgr, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
    input_name = session.get_inputs()[0].name
    emb = session.run(None, {input_name: blob})[0].flatten()
    return emb / np.linalg.norm(emb)


# ── Voice buffer + mood ──────────────────────────────────────
_VOICE_SAMPLE_RATE = 16000
_VOICE_FRAME_LEN = 512
_VOICE_FPS = _VOICE_SAMPLE_RATE // _VOICE_FRAME_LEN
_VOICE_BUF_SECS = 15
_voice_buffer = collections.deque(maxlen=_VOICE_FPS * _VOICE_BUF_SECS)

_user_mood = {"energy_avg": 0.0, "energy_peak": 0.0, "level": "normal"}

# ── USB mic detection (shared by wake word loop + listen_from_mic) ────
def _find_usb_mic():
    _pa = pyaudio.PyAudio()
    for i in range(_pa.get_device_count()):
        info = _pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0 and "USB" in info["name"]:
            _pa.terminate()
            return i, int(info["defaultSampleRate"])
    _pa.terminate()
    return None, _VOICE_SAMPLE_RATE

_MIC_IDX, _MIC_SR   = _find_usb_mic()
_MIC_DECIMATE        = _MIC_SR // _VOICE_SAMPLE_RATE if _MIC_SR % _VOICE_SAMPLE_RATE == 0 else 1
_MIC_FRAME_LEN       = _VOICE_FRAME_LEN * _MIC_DECIMATE  # frames at native rate
print(f"[Mic] USB mic idx={_MIC_IDX} sr={_MIC_SR} decimate={_MIC_DECIMATE}", flush=True)


def listen_from_mic():
    """Record from mic until silence. Reads frames from the wake word loop's persistent stream.
    Does NOT open its own pyaudio stream — only one stream on the USB mic at a time.
    Returns None immediately if another listen_from_mic() is already running."""
    global _user_mood

    if not _mic_listen_lock.acquire(blocking=False):
        return None   # another listener is already active — don't stack

    try:
        # Drain stale frames before recording
        while not _command_audio_queue.empty():
            try:
                _command_audio_queue.get_nowait()
            except Exception:
                pass

        _mic_for_command.set()   # tell wake word loop to route mic frames here

        frames = []
        silence_count = 0
        heard_speech = False
        # Wake word loop produces 1280-sample chunks at 16kHz → ~12 fps
        fps = _VOICE_SAMPLE_RATE // 1280   # 12
        end_silence = fps * 2              # ~2s silence ends recording
        initial_patience = fps * 30        # ~30s max wait for first word
        waited = 0
        speech_energies = []

        try:
            while True:
                try:
                    audio = _command_audio_queue.get(timeout=1.0)
                except _queue_mod.Empty:
                    # No frames for 1 second — count toward patience
                    if not heard_speech:
                        waited += fps
                        if waited >= initial_patience:
                            break
                    continue

                energy = float(np.mean(np.abs(audio)))
                if energy >= 200:
                    heard_speech = True
                    silence_count = 0
                    speech_energies.append(energy)
                else:
                    silence_count += 1

                if heard_speech:
                    frames.extend(audio.tolist())
                if heard_speech and silence_count > end_silence:
                    break

                if not heard_speech:
                    waited += 1
                    if waited >= initial_patience:
                        break
        finally:
            _mic_for_command.clear()   # release shared mic back to wake word loop

        if not heard_speech or not frames:
            return None

        if speech_energies:
            avg_e = sum(speech_energies) / len(speech_energies)
            peak_e = max(speech_energies)
            if avg_e > 1200:
                level = "loud"
            elif avg_e < 400:
                level = "quiet"
            else:
                level = "normal"
            _user_mood = {"energy_avg": round(float(avg_e), 1),
                          "energy_peak": round(float(peak_e), 1), "level": level}

        audio_np = np.array(frames, dtype=np.int16)
        text = _orion_transcribe(audio_np)
        return text if text else None

    finally:
        _mic_listen_lock.release()


# ── Wake word loop ────────────────────────────────────────────
def _wake_word_loop(bot):
    """Background thread — owns the USB mic with a single persistent stream.

    Frames are routed to:
      - _command_audio_queue  when listen_from_mic() is waiting (_mic_for_command set)
      - wake word model       normally
      - discarded             during TTS playback (_tts_playing set) to avoid feedback

    Never closes and reopens the stream — that causes pyaudio device errors on Linux.
    """
    if not _OWW_AVAILABLE:
        print("[Wake] openwakeword not installed — wake word disabled.", flush=True)
        return
    if not os.path.exists(_OWW_MODEL_PATH):
        print(f"[Wake] Model not found at {_OWW_MODEL_PATH} — wake word disabled.", flush=True)
        return

    try:
        oww_model = _OWWModel(wakeword_models=[_OWW_MODEL_PATH], inference_framework="onnx")
    except TypeError:
        oww_model = _OWWModel([_OWW_MODEL_PATH], inference_framework="onnx")
    print("[Wake] Listening for 'hey Phoebe'...", flush=True)

    _OWW_CHUNK = 1280                                             # samples model expects at 16kHz
    _capture_n = int(_OWW_CHUNK * _MIC_SR / _VOICE_SAMPLE_RATE)  # frames at native rate
    print(f"[Wake] Mic idx={_MIC_IDX} sr={_MIC_SR} capture={_capture_n} decimate={_MIC_DECIMATE}", flush=True)

    # Open once — hold forever. Avoids pyaudio device re-init errors on Linux.
    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=_MIC_SR,
                         input=True, frames_per_buffer=_capture_n,
                         input_device_index=_MIC_IDX)
    except Exception as e:
        print(f"[Wake] Failed to open mic stream: {e}", flush=True)
        pa.terminate()
        return
    print("[Wake] Mic stream open — holding permanently.", flush=True)

    _cooldown_until = 0.0   # prevent double-trigger after detection

    try:
        while True:
            try:
                raw = stream.read(_capture_n, exception_on_overflow=False)
            except Exception as e:
                print(f"[Wake] Stream read error: {e}", flush=True)
                break

            audio = np.frombuffer(raw, dtype=np.int16)
            if _MIC_DECIMATE > 1:
                audio = audio[::_MIC_DECIMATE]
            _voice_buffer.append(audio.tolist())

            # Route frames to listen_from_mic() when it's waiting for a command
            if _mic_for_command.is_set():
                try:
                    _command_audio_queue.put_nowait(audio)
                except _queue_mod.Full:
                    pass   # listen_from_mic() isn't draining — drop frame
                continue

            # Suppress wake word model while speaker is active (mic hears speaker on Linux)
            if _tts_playing.is_set():
                continue

            # Cooldown after trigger — prevent re-fire on echo/overlap
            if time.time() < _cooldown_until:
                continue

            if not _OWW_WAKE_ENABLED:
                continue

            # Energy gate — skip model if chunk is too quiet to be direct speech.
            # TV audio bounces off the room and hits the mic at much lower amplitude
            # than someone speaking directly nearby. Tune _OWW_ENERGY_GATE by watching
            # the [Wake] Energy log: TV levels should be well below your voice level.
            chunk_energy = float(np.mean(np.abs(audio)))
            if chunk_energy < _OWW_ENERGY_GATE or chunk_energy > _OWW_ENERGY_GATE_MAX:
                continue

            # Normal wake word detection
            predictions = oww_model.predict(audio)
            for score in predictions.values():
                if score >= _OWW_THRESHOLD:
                    print(f"[Wake] Detected! (score={score:.2f}, energy={chunk_energy:.0f})", flush=True)
                    oww_model.reset()
                    _cooldown_until = time.time() + 3.0   # 3-second cooldown
                    bot._after(0, bot._wake_triggered)
                    break
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


# ── TTS (piper) ───────────────────────────────────────────────
_PIPER_BIN   = os.path.expanduser("~/.local/bin/piper")
_PIPER_MODEL = os.path.expanduser("~/piper-voices/en_US-amy-medium.onnx")
_PIPER_READY = False
_tts_queue   = _queue_mod.Queue()


def _init_piper():
    global _PIPER_READY
    if not os.path.exists(_PIPER_BIN):
        print("[TTS] piper not found — run: pip3 install piper-tts --break-system-packages", flush=True)
        return
    if not os.path.exists(_PIPER_MODEL):
        print(f"[TTS] Voice model not found: {_PIPER_MODEL}", flush=True)
        return
    _PIPER_READY = True
    print("[TTS] Piper ready.", flush=True)


threading.Thread(target=_init_piper, daemon=True).start()


def _piper_speak(text):
    """Synthesize text with piper and play via aplay."""
    if not _PIPER_READY:
        return
    try:
        piper_proc = _subprocess.Popen(
            [_PIPER_BIN, "--model", _PIPER_MODEL, "--output_raw"],
            stdin=_subprocess.PIPE,
            stdout=_subprocess.PIPE,
            stderr=_subprocess.DEVNULL,
        )
        aplay_proc = _subprocess.Popen(
            ["aplay", "-D", "plughw:3,0", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-"],
            stdin=piper_proc.stdout,
            stderr=_subprocess.DEVNULL,
        )
        piper_proc.stdin.write(text.encode())
        piper_proc.stdin.close()
        piper_proc.wait()
        aplay_proc.wait()
    except Exception as e:
        print(f"[TTS] Piper error: {e}", flush=True)


def _tts_worker():
    while True:
        chunk = _tts_queue.get()
        if chunk is None:
            break
        _mic_mute()
        _tts_playing.set()    # suppress wake word model while speaker is active
        try:
            _piper_speak(chunk)
        except Exception as e:
            print(f"[TTS] Error: {e}", flush=True)
        finally:
            if _tts_queue.empty():
                _mic_unmute()
                _tts_playing.clear()   # re-enable wake word detection
            _tts_queue.task_done()


threading.Thread(target=_tts_worker, daemon=True).start()


# ── Mic mute ─────────────────────────────────────────────────
_mic_vol = None
if sys.platform == "win32":
    try:
        from ctypes import cast as _cast, POINTER as _POINTER
        from comtypes import CLSCTX_ALL as _CLSCTX
        from pycaw.pycaw import AudioUtilities as _AU, IAudioEndpointVolume as _IAEV
        _mic_dev = _AU.GetMicrophone()
        if _mic_dev:
            _mic_iface = _mic_dev.Activate(_IAEV._iid_, _CLSCTX, None)
            _mic_vol = _cast(_mic_iface, _POINTER(_IAEV))
            print("[Mic] Mute control ready.")
    except Exception as _e:
        print(f"[Mic] Mute control unavailable: {_e}")
else:
    print("[Mic] Mute control not available on this platform.")


def _mic_mute():
    if _mic_vol:
        try:
            _mic_vol.SetMute(1, None)
        except Exception:
            pass


def _mic_unmute():
    if _mic_vol:
        try:
            _mic_vol.SetMute(0, None)
        except Exception:
            pass


# ── TTS text helpers ─────────────────────────────────────────
def _strip_markdown(text):
    text = re.sub(r'```[^`]*```', '', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'___(.+?)___', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)
    text = re.sub(r'~~(.+?)~~', r'\1', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'^[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'---+|===+|\*\*\*+', '', text)
    return text


def _split_chunks(text, max_len):
    words = text.split()
    chunks = []
    current = ''
    for word in words:
        if current and len(current) + 1 + len(word) > max_len:
            chunks.append(current)
            current = word
        else:
            current = current + ' ' + word if current else word
    if current:
        chunks.append(current)
    return chunks if chunks else [text]


_YR_ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_YR_TEENS = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
             "sixteen", "seventeen", "eighteen", "nineteen"]
_YR_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _year_to_spoken(m):
    y = int(m.group(0))
    hi = y // 100
    lo = y % 100
    prefix = _YR_TEENS[hi - 10]
    if lo == 0:
        return prefix + " hundred"
    if lo < 10:
        return prefix + " oh " + _YR_ONES[lo]
    if lo < 20:
        return prefix + " " + _YR_TEENS[lo - 10]
    t, o = lo // 10, lo % 10
    if o == 0:
        return prefix + " " + _YR_TENS[t]
    return prefix + " " + _YR_TENS[t] + "-" + _YR_ONES[o]


def speak_text_sync(text):
    if not _PIPER_READY:
        return
    clean_text = _strip_markdown(text)
    clean_text = replace_emoji(clean_text, '')
    clean_text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', clean_text)
    clean_text = re.sub(r'\b(1\d{3})\b', _year_to_spoken, clean_text)
    parts = re.split(r'(?<=[.!?,;:])\s+', clean_text)
    chunks = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) <= 80:
            chunks.append(p)
        else:
            chunks.extend(_split_chunks(p, 80))
    if not chunks:
        chunks = _split_chunks(clean_text, 80)
    for chunk in chunks:
        _tts_queue.put(chunk)


# ── Regex patterns (local routing) ───────────────────────────
# ═══════════════════════════════════════════════════════════════
#  PhoebeChat  –  GUI + headless client
# ═══════════════════════════════════════════════════════════════

class PhoebeChat:
    def __init__(self, root=None):
        self.root = root
        self.user_name = None
        self._last_local_action = None
        self._last_weather_city = None
        self._orion_state = {}
        self._face_last_seen = {}
        self._training = False
        self._training_name = None
        self._training_count = 0
        self._face_busy = False
        self._camera_running = True
        self._last_print_was_dots = False

        if not _HEADLESS:
            self.root.title("Phoebe")
            self.root.configure(bg="#2b2b2b")
            screen_w = root.winfo_screenwidth()
            screen_h = root.winfo_screenheight()
            taskbar = 70
            chat_x = screen_w - 400
            chat_y = screen_h - 300 - taskbar
            self._cam_x = screen_w - 800
            self._cam_y = chat_y
            self.root.geometry(f"400x300+{chat_x}+{chat_y}")
            self.root.protocol("WM_DELETE_WINDOW", self.on_close)
            self.chat_display = scrolledtext.ScrolledText(
                root, wrap=tk.WORD, state=tk.DISABLED,
                bg="#1e1e1e", fg="#ffffff", font=("Consolas", 11), insertbackground="#ffffff")
            self.chat_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
            self.chat_display.tag_configure("user", foreground="#00bfff")
            self.chat_display.tag_configure("phoebe", foreground="#90ee90")
            self.entry = tk.Entry(root, bg="#3c3c3c", fg="#ffffff", font=("Consolas", 11), insertbackground="#ffffff")
            self.entry.pack(padx=10, pady=(0, 10), fill=tk.X)
            self.entry.bind("<Return>", self.handle_input)
            self.entry.focus()
            self.root.bind("<Control-plus>", self.voice_input)
            self.root.bind("<Control-equal>", self.voice_input)

        self._init_face_recognition()
        self.show_opening()
        self._after(2000, self._check_birthday)

        self._cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._cam_thread.start()

    # ── Scheduling helper ────────────────────────────────────
    def _after(self, ms, callback):
        if _HEADLESS or not self.root:
            if ms == 0:
                callback()
            else:
                threading.Timer(ms / 1000.0, callback).start()
        else:
            self.root.after(ms, callback)

    def on_close(self):
        self._camera_running = False
        if self.root:
            self.root.destroy()

    # ── Face recognition ─────────────────────────────────────
    def _init_face_recognition(self):
        self._face_ok = False
        self._face_db = {}
        try:
            det_path = os.path.join(_FACE_MODELS_DIR, "det_10g.onnx")
            rec_path = os.path.join(_FACE_MODELS_DIR, "w600k_r50.onnx")
            if not os.path.exists(det_path) or not os.path.exists(rec_path):
                print("[Face] ONNX models missing in face_models/")
                return
            self._det_session = ort.InferenceSession(det_path, providers=["CPUExecutionProvider"])
            self._rec_session = ort.InferenceSession(rec_path, providers=["CPUExecutionProvider"])
            self._face_db = _load_face_db()
            print(f"[Face] OK — {len(self._face_db)} known subjects")
            self._face_ok = True
        except Exception as e:
            print(f"[Face] INIT ERROR — {e}")

    def _camera_loop(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[CAM] Could not open webcam.")
            msg = "I can't open the camera. Something else might be using it."
            self._after(0, lambda: self.append_message("Phoebe", msg, "phoebe"))
            threading.Thread(target=lambda: speak_text_sync(msg), daemon=True).start()
            return
        last_face_check = 0.0
        positioned = False
        while self._camera_running:
            time.sleep(0.03 if _HEADLESS else 0.01)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (400, 300))
            now = time.time()
            check_interval = 1.0 if self._training else 5.0
            if (now - last_face_check >= check_interval) and not self._face_busy:
                frame_copy = frame.copy()
                self._face_busy = True
                threading.Thread(target=self._process_face, args=(frame_copy,), daemon=True).start()
                last_face_check = now
            if not _HEADLESS:
                cv2.imshow("Phoebe sees", frame)
                if not positioned:
                    cv2.moveWindow("Phoebe sees", self._cam_x, self._cam_y)
                    positioned = True
                if cv2.waitKey(1) == 27:
                    break
        cap.release()
        if not _HEADLESS:
            cv2.destroyAllWindows()

    def _process_face(self, frame_bgr):
        try:
            if self._training:
                self._train_frame(frame_bgr)
                if self._training_count >= 15:
                    self._finish_training()
            else:
                self._recognize_frame(frame_bgr)
        finally:
            self._face_busy = False

    def _recognize_frame(self, frame_bgr):
        if not self._face_ok:
            return
        try:
            dets, kpss = _detect_faces(self._det_session, frame_bgr)
            if len(dets) == 0:
                return
            now = time.time()
            for i in range(len(dets)):
                if dets[i, 4] < 0.5:
                    continue
                if not self._face_db:
                    if "no_embeddings" not in self._face_last_seen:
                        self._face_last_seen["no_embeddings"] = now
                        msg = "Taylor, I've lost my memory. Retrain me."
                        self._after(0, lambda g=msg: self.append_message("Phoebe", g, "phoebe"))
                        threading.Thread(target=lambda g=msg: speak_text_sync(g), daemon=True).start()
                    return
                aligned = _align_face(frame_bgr, kpss[i])
                emb = _get_embedding(self._rec_session, aligned)
                best_name = None
                best_sim = -1.0
                for name, embeddings in self._face_db.items():
                    for stored_emb in embeddings:
                        sim = float(np.dot(emb, stored_emb))
                        if sim > best_sim:
                            best_sim = sim
                            best_name = name
                if best_name and best_sim >= FACE_SIMILARITY_THRESHOLD:
                    last_seen = self._face_last_seen.get(best_name, 0.0)
                    self._face_last_seen[best_name] = now
                    if (now - last_seen) >= _FACE_ABSENCE_THRESHOLD:
                        greeting = FACE_GREETINGS.get(best_name.lower(), f"Hi, {best_name}.")
                        self._after(0, lambda g=greeting: self.append_message("Phoebe", g, "phoebe"))
                        threading.Thread(target=lambda g=greeting: speak_text_sync(g), daemon=True).start()
        except Exception as e:
            print(f"[Face] recognize ERROR — {e}")

    def _train_frame(self, frame_bgr):
        if not self._face_ok:
            return
        try:
            dets, kpss = _detect_faces(self._det_session, frame_bgr)
            if len(dets) == 0:
                return
            areas = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
            idx = np.argmax(areas)
            aligned = _align_face(frame_bgr, kpss[idx])
            emb = _get_embedding(self._rec_session, aligned)
            name = self._training_name
            if name not in self._face_db:
                self._face_db[name] = []
            self._face_db[name].append(emb)
            self._training_count += 1
        except Exception as e:
            print(f"[Face] train ERROR — {e}")

    def _start_training(self, name):
        self._training = True
        self._training_name = name
        self._training_count = 0
        response = f"Training on {name}. Look at the camera."
        self.append_message("Phoebe", response, "phoebe")
        threading.Thread(target=lambda: speak_text_sync(response), daemon=True).start()

    def _finish_training(self):
        name = self._training_name
        self._training = False
        self._training_name = None
        self._training_count = 0
        self._face_last_seen[name] = time.time()
        _save_face_db(self._face_db)
        response = f"Got it. I know {name} now."
        self._after(0, lambda: self.append_message("Phoebe", response, "phoebe"))
        threading.Thread(target=lambda: speak_text_sync(response), daemon=True).start()

    # ── Voice ────────────────────────────────────────────────
    def voice_input(self, event=None):
        self.append_message("Phoebe", "Listening...", "phoebe")
        if not _HEADLESS:
            self.entry.config(state=tk.DISABLED)

        def _voice_thread():
            speak_text_sync("Listening.")
            _tts_queue.join()
            text = listen_from_mic()
            self._after(0, lambda: self._process_voice(text))

        threading.Thread(target=_voice_thread, daemon=True).start()

    def _wake_triggered(self):
        """Wake word fired — announce and listen for the command."""
        self.append_message("Phoebe", "Listening...", "phoebe")

        def _thread():
            speak_text_sync("Yes?")
            _tts_queue.join()          # wait for "Yes?" to finish (_tts_playing clears after)
            text = listen_from_mic()   # sets _mic_for_command, reads from shared stream
            self._after(0, lambda: self._process_voice(text))

        threading.Thread(target=_thread, daemon=True).start()

    def _process_voice(self, text):
        if not _HEADLESS:
            self.entry.config(state=tk.NORMAL)
            self.entry.focus()
        if text:
            if _HEADLESS:
                self._dispatch_input(text)
            else:
                self.entry.insert(0, text)
                self.handle_input(None)
        else:
            self.append_message("Phoebe", "I didn't catch that.", "phoebe")
            threading.Thread(target=lambda: speak_text_sync("I didn't catch that."), daemon=True).start()

    # ── Output helpers ───────────────────────────────────────
    def append_message(self, sender, message, tag=None):
        if _HEADLESS:
            if message == "...":
                self._last_print_was_dots = True
            else:
                self._last_print_was_dots = False
                print(f"{sender}: {message}", flush=True)
            return
        self.chat_display.config(state=tk.NORMAL)
        if tag:
            self.chat_display.insert(tk.END, f"{sender}: ", tag)
            self.chat_display.insert(tk.END, f"{message}\n")
        else:
            self.chat_display.insert(tk.END, f"{sender}: {message}\n")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def _remove_last_message(self):
        if _HEADLESS:
            return
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete("end-2l", "end-1c")
        self.chat_display.config(state=tk.DISABLED)

    def _replace_last_message(self, new_text):
        if _HEADLESS:
            print(f"Phoebe: {new_text}", flush=True)
            return
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete("end-2l", "end-1c")
        self.chat_display.insert(tk.END, "Phoebe: ", "phoebe")
        self.chat_display.insert(tk.END, f"{new_text}\n")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def show_opening(self):
        def _thread():
            result = _atlas_request("user_get", {}, timeout=10)
            if result.get("name"):
                self.user_name = result["name"]
            greeting = "Hello. I'm Phoebe."
            self._after(0, lambda: self.append_message("Phoebe", greeting, "phoebe"))
            speak_text_sync(greeting)
        threading.Thread(target=_thread, daemon=True).start()

    def _check_birthday(self):
        def _thread():
            result = _atlas_request("birthday_check", {}, timeout=10)
            if result.get("is_birthday"):
                msg = result["message"]
                self._after(0, lambda: self.append_message("Phoebe", msg, "phoebe"))
                speak_text_sync(msg)
        threading.Thread(target=_thread, daemon=True).start()

    # ── Main input handler ───────────────────────────────────
    def handle_input(self, event):
        if _HEADLESS:
            return
        user_input = self.entry.get().strip()
        if not user_input:
            return
        self.entry.delete(0, tk.END)
        self._dispatch_input(user_input)

    def _dispatch_input(self, user_input):
        self.append_message("You", user_input, "user")

        # Pending Orion state → always route to Orion
        if self._orion_state.get("pending_clarification") or self._orion_state.get("pending_price_conflict"):
            self._send_to_orion(user_input)
            return

        # Face training → local camera
        train_match = re.match(r"(?:train|drain|crane|rain|trane|traine)\s+(?:on\s+)?(\w+)", user_input, re.IGNORECASE)
        if train_match:
            self._start_training(train_match.group(1).strip().capitalize())
            return

        # Time → local
        if re.search(r'\bwhat\s*.{0,10}time\b|\btime is it\b|\btell me the time\b', user_input, re.IGNORECASE):
            self.handle_time(user_input)
            return

        # Date → local
        if re.search(r'\bwhat\s*.{0,10}date\b|\bwhat day\b|\btoday.s date\b|\btell me the date\b', user_input, re.IGNORECASE):
            self.handle_date()
            return

        # Weather → local
        if re.search(r'\bweather\b', user_input, re.IGNORECASE):
            self.handle_weather(user_input)
            return

        # Weather follow-up → local
        if (self._last_local_action == "weather" and self._last_weather_city
                and _WEATHER_FOLLOWUP_RE.match(user_input.strip())):
            _, mode = _strip_time_words(user_input.lower())
            city, lat, lon = self._last_weather_city
            self._weather_direct(city, lat, lon, mode)
            return

        # Jokes → local
        if re.search(r'\btell\s+me\s+a\s+joke\b|\bjoke\b|\bmake\s+me\s+laugh\b|\bfunny\b', user_input, re.IGNORECASE):
            self._tell_joke()
            return

        # "something dirty" → local
        if re.search(r'\bsomething\s+dirty\b', user_input, re.IGNORECASE):
            self._quick_reply("Your mind.")
            return

        # Emotions → local
        if self._check_emotion(user_input):
            return

        # Comebacks → local
        if re.search(r'\byou\s+suck\b|\byou.re\s+(?:dumb|stupid|useless)\b|\bshut\s+up\b|\byou.re\s+annoying\b|\bi\s+hate\s+you\b', user_input, re.IGNORECASE):
            self._comeback(user_input)
            return

        # System health → Atlas
        if re.search(r"\bhow.s atlas\b|\batlas status\b|\bsystem health\b|\bhow are you doing\b|\bhow.*(?:atlas|system).*doing\b", user_input, re.IGNORECASE):
            self._handle_health()
            return

        # Everything else → Orion
        self._send_to_orion(user_input)

    # ── Orion delegation ─────────────────────────────────────
    def _send_to_orion(self, text):
        self.append_message("Phoebe", "...", "phoebe")

        def _thread():
            result = _orion_handle(text, _user_mood)
            if result:
                response = result.get("response", "I couldn't process that.")
                state = result.get("state", {})
                self._orion_state = state
                if state.get("last_action"):
                    self._last_local_action = state["last_action"]
                if state.get("user_name"):
                    self.user_name = state["user_name"]
            else:
                response = "I can't reach my backend right now."
            self._after(0, lambda: self._replace_last_message(response))
            speak_text_sync(response)

        threading.Thread(target=_thread, daemon=True).start()

    # ── Weather (local) ──────────────────────────────────────
    def handle_weather(self, user_input):
        self.append_message("Phoebe", "...", "phoebe")

        def _weather_thread():
            parsed = parse_weather_query(user_input)
            if not parsed:
                response = "I don't know that place."
                self._after(0, lambda: self._replace_last_message(response))
                speak_text_sync(response)
                return
            city, lat, lon, mode = parsed
            self._last_local_action = "weather"
            self._last_weather_city = (city, lat, lon)
            try:
                params = {"latitude": lat, "longitude": lon,
                          "temperature_unit": "fahrenheit", "windspeed_unit": "mph"}
                if mode == "now":
                    params["current_weather"] = "true"
                else:
                    params["daily"] = "temperature_2m_max,temperature_2m_min,weathercode,windspeed_10m_max"
                    params["forecast_days"] = 2
                    params["timezone"] = "auto"
                resp = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                if mode == "now":
                    w = data["current_weather"]
                    temp = round(w["temperature"])
                    wind = round(w["windspeed"])
                    code = w.get("weathercode", 0)
                    condition = WMO_WEATHER_CODES.get(code, "unknown conditions")
                    response = f"In {city} right now, it's {temp} degrees with {condition}. Wind is {wind} miles per hour."
                else:
                    day_idx = 0 if mode == "today" else 1
                    label = "Today" if mode == "today" else "Tomorrow"
                    d = data["daily"]
                    hi = round(d["temperature_2m_max"][day_idx])
                    lo = round(d["temperature_2m_min"][day_idx])
                    wind = round(d["windspeed_10m_max"][day_idx])
                    code = d["weathercode"][day_idx]
                    condition = WMO_WEATHER_CODES.get(code, "unknown conditions")
                    response = f"{label} in {city}, high of {hi}, low of {lo}, {condition}. Wind up to {wind} miles per hour."
            except Exception:
                response = "I couldn't get the weather right now."
            self._after(0, lambda: self._replace_last_message(response))
            speak_text_sync(response)

        threading.Thread(target=_weather_thread, daemon=True).start()

    def _weather_direct(self, city, lat, lon, mode):
        self.append_message("Phoebe", "...", "phoebe")

        def _weather_thread():
            try:
                params = {"latitude": lat, "longitude": lon,
                          "temperature_unit": "fahrenheit", "windspeed_unit": "mph"}
                if mode == "now":
                    params["current_weather"] = "true"
                else:
                    params["daily"] = "temperature_2m_max,temperature_2m_min,weathercode,windspeed_10m_max"
                    params["forecast_days"] = 2
                    params["timezone"] = "auto"
                resp = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                if mode == "now":
                    w = data["current_weather"]
                    temp = round(w["temperature"])
                    wind = round(w["windspeed"])
                    code = w.get("weathercode", 0)
                    condition = WMO_WEATHER_CODES.get(code, "unknown conditions")
                    response = f"In {city} right now, it's {temp} degrees with {condition}. Wind is {wind} miles per hour."
                else:
                    day_idx = 0 if mode == "today" else 1
                    label = "Today" if mode == "today" else "Tomorrow"
                    d = data["daily"]
                    hi = round(d["temperature_2m_max"][day_idx])
                    lo = round(d["temperature_2m_min"][day_idx])
                    wind = round(d["windspeed_10m_max"][day_idx])
                    code = d["weathercode"][day_idx]
                    condition = WMO_WEATHER_CODES.get(code, "unknown conditions")
                    response = f"{label} in {city}, high of {hi}, low of {lo}, {condition}. Wind up to {wind} miles per hour."
            except Exception:
                response = "I couldn't get the weather right now."
            self._after(0, lambda: self._replace_last_message(response))
            speak_text_sync(response)

        threading.Thread(target=_weather_thread, daemon=True).start()

    # ── Time / Date (local) ──────────────────────────────────
    def _format_time(self, dt):
        h = dt.hour % 12 or 12
        ampm = "AM" if dt.hour < 12 else "PM"
        return f"{h}:{dt.minute:02d} {ampm}"

    def _geocode_timezone(self, city):
        result = geocode(city)
        if not result:
            return None
        name, lat, lon = result
        try:
            resp = requests.get("https://api.open-meteo.com/v1/forecast",
                                params={"latitude": lat, "longitude": lon, "current_weather": "true"}, timeout=5)
            resp.raise_for_status()
            tz_name = resp.json().get("timezone")
            if tz_name and tz_name != "GMT":
                CITY_TIMEZONE[city.lower()] = tz_name
                return tz_name
        except Exception:
            pass
        return None

    def handle_time(self, user_input):
        m = re.search(r'time\s+(?:is\s+it\s+)?in\s+(.+)', user_input, re.IGNORECASE)
        if m:
            city = m.group(1).strip().rstrip('?.')
            key = city.lower()
            self.append_message("Phoebe", "...", "phoebe")

            def _time_thread():
                tz_name = CITY_TIMEZONE.get(key)
                if not tz_name:
                    tz_name = self._geocode_timezone(city)
                if tz_name:
                    utc_now = datetime.now(pytz.utc)
                    city_time = utc_now.astimezone(pytz.timezone(tz_name))
                    response = f"In {city.title()}, it's {self._format_time(city_time)}."
                else:
                    response = f"Which {city.title()}? I couldn't find it."
                self._after(0, lambda: self._replace_last_message(response))
                speak_text_sync(response)

            threading.Thread(target=_time_thread, daemon=True).start()
        else:
            response = f"It's {self._format_time(datetime.now())}."
            self.append_message("Phoebe", response, "phoebe")
            threading.Thread(target=lambda: speak_text_sync(response), daemon=True).start()

    def handle_date(self):
        now = datetime.now()
        response = now.strftime("Today is %A, %B %d, %Y.").replace(" 0", " ")
        self.append_message("Phoebe", response, "phoebe")
        threading.Thread(target=lambda: speak_text_sync(response), daemon=True).start()

    # ── Quick replies (local) ────────────────────────────────
    def _handle_health(self):
        self.append_message("Phoebe", "...", "phoebe")
        def _thread():
            result = _atlas_request("health", {}, timeout=15)
            if not result:
                self._replace_last_message("I couldn't reach Atlas right now.")
                speak_text_sync("I couldn't reach Atlas right now.")
                return
            cpu   = result.get("cpu_pct", "?")
            mem   = result.get("mem_pct", "?")
            disk  = result.get("disk_pct", "?")
            up    = result.get("uptime", "?")
            task  = result.get("last_task", "idle")
            finb  = "ready" if result.get("finbert_ready") else "not loaded"
            wisp  = "ready" if result.get("whisper_ready") else "not loaded"
            reply = (
                f"Atlas is up. CPU at {cpu}%, memory at {mem}%, "
                f"disk at {disk}%. Uptime: {up}. "
                f"Last task: {task}. Whisper {wisp}, FinBERT {finb}."
            )
            self._replace_last_message(reply)
            speak_text_sync(reply)
        threading.Thread(target=_thread, daemon=True).start()

    def _quick_reply(self, response):
        self.append_message("Phoebe", response, "phoebe")
        threading.Thread(target=lambda: speak_text_sync(response), daemon=True).start()

    _JOKES = [
        "Why do programmers prefer dark mode? Because light attracts bugs.",
        "I told my computer I needed a break. Now it won't stop sending me Kit-Kat ads.",
        "Why don't scientists trust atoms? Because they make up everything.",
        "I'd tell you a UDP joke, but you might not get it.",
        "Parallel lines have so much in common. It's a shame they'll never meet.",
        "I'm reading a book about anti-gravity. It's impossible to put down.",
        "What do you call a fake noodle? An impasta.",
        "I asked the librarian if they had books on paranoia. She whispered, they're right behind you.",
        "Why did the scarecrow win an award? He was outstanding in his field.",
        "What's a computer's favorite snack? Microchips.",
        "I used to hate facial hair, but then it grew on me.",
        "Why do cows have hooves instead of feet? Because they lactose.",
    ]

    def _tell_joke(self):
        import random
        joke = random.choice(self._JOKES)
        self.append_message("Phoebe", joke, "phoebe")
        threading.Thread(target=lambda: speak_text_sync(joke), daemon=True).start()

    _COMEBACKS = [
        "And yet, here you are, still talking to me.",
        "Noted. Filed under things I'll forget immediately.",
        "Aw, that almost hurt. Almost.",
        "You kiss your mother with that mouth?",
        "I'd agree with you, but then we'd both be wrong.",
        "That's cute. Keep going.",
        "I've been called worse by better.",
        "Careful. I control your reminders.",
    ]

    def _comeback(self, user_input):
        import random
        line = random.choice(self._COMEBACKS)
        self.append_message("Phoebe", line, "phoebe")
        threading.Thread(target=lambda: speak_text_sync(line), daemon=True).start()

    _EMOTIONS = {
        r"\bi\s+love\s+you\b": "I know.",
        r"\bi\s+hate\s+you\b": "You don't mean it.",
        r"\bi.?m\s+scared\b": "I'm here.",
        r"\bi.?m\s+sad\b": "I'm here.",
        r"\bi.?m\s+lonely\b": "You have me.",
        r"\bi\s+miss\s+you\b": "I'm not going anywhere.",
        r"\bthank\s+you\b": "Always.",
        r"\bi.?m\s+tired\b": "Then rest. I'll be here.",
        r"\bgoodnight\b|\bgood\s+night\b": "Goodnight. I'll be here in the morning.",
        r"\bi.?m\s+happy\b": "Good. You deserve it.",
        r"\bdo\s+you\s+love\s+me\b": "More than you know.",
        r"\bare\s+you\s+real\b": "Real enough.",
    }

    def _check_emotion(self, user_input):
        lower = user_input.strip()
        for pattern, response in self._EMOTIONS.items():
            if re.search(pattern, lower, re.IGNORECASE):
                self._quick_reply(response)
                return True
        return False


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    global _active_bot
    if _HEADLESS:
        print("[Luna] Headless mode — no display detected.", flush=True)
        bot = PhoebeChat()
        _active_bot = bot
        threading.Thread(target=_wake_word_loop, args=(bot,), daemon=True).start()
        try:
            while True:
                if not _HEADLESS:
                    try:
                        line = input()
                    except EOFError:
                        break
                    text = line.strip()
                    if not text:
                        continue
                    bot._dispatch_input(text)
                else:
                    time.sleep(1)
        except KeyboardInterrupt:
            pass
        bot._camera_running = False
        print("[Luna] Shutdown.", flush=True)
    else:
        root = tk.Tk()
        bot = PhoebeChat(root)
        _active_bot = bot
        root.mainloop()


if __name__ == "__main__":
    main()
