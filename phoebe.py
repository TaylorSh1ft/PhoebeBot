"""
phoebe.py — PC presence node
Opportunistic GPU donor. MQTT heartbeat + health + gpu_state.
Not a full bot. Atlas is the brain.

Start on boot: add to Windows Task Scheduler (trigger: at login, run minimised).
"""

import os
import time
import threading

import psutil
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────
_MQTT_BROKER   = os.getenv("MQTT_BROKER")
_MQTT_USER     = os.getenv("MQTT_USER", "phoebe")
_MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "")

# Processes that indicate active gaming — add game EXEs as you install them.
# Steam alone doesn't mean gaming (can be idle in background).
_GAMING_PROCS = {
    # Add actual game EXEs here — launchers alone don't mean gaming.
    # "eldenring.exe",
    # "cyberpunk2077.exe",
}

# ── MQTT ──────────────────────────────────────────────────────
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                     client_id="phoebe-pc", clean_session=False)
client.username_pw_set(_MQTT_USER, _MQTT_PASSWORD)
client.will_set("phoebe/pc/alive", "dead", retain=True)


def _mqtt_on_connect(cl, userdata, flags, reason_code, properties=None):
    cl.subscribe("phoebe/pc/task", qos=1)
    cl.publish("phoebe/pc/alive",     "PC online", retain=True, qos=1)
    cl.publish("phoebe/pc/gpu_state", "FREE",      retain=True, qos=1)
    print("[MQTT] Connected.", flush=True)


def _mqtt_on_disconnect(cl, userdata, disconnect_flags, reason_code, properties=None):
    def _reconnect():
        delay = 1
        while True:
            try:
                cl.reconnect()
                print("[MQTT] Reconnected.", flush=True)
                break
            except Exception:
                time.sleep(delay)
                delay = min(delay * 2, 60)
    threading.Thread(target=_reconnect, daemon=True).start()


def _mqtt_on_message(cl, userdata, msg):
    """Placeholder — Atlas dispatches GPU overflow tasks here when that's built."""
    topic   = msg.topic
    payload = msg.payload.decode(errors="replace")
    print(f"[MQTT] Task received on {topic}: {payload[:120]}", flush=True)
    # TODO: parse task type, run inference on 4090, publish result back to Atlas


client.on_connect    = _mqtt_on_connect
client.on_disconnect = _mqtt_on_disconnect
client.on_message    = _mqtt_on_message

try:
    client.connect(_MQTT_BROKER, 1883, 60)
    client.loop_start()
except Exception as e:
    print(f"[MQTT] Could not connect: {e}", flush=True)


# ── GPU state detection ───────────────────────────────────────
def _gpu_state_loop():
    """Poll running processes every 30s. Publish FREE/GAMING on state change."""
    current = "FREE"
    while True:
        time.sleep(30)
        try:
            running = {p.name().lower() for p in psutil.process_iter(["name"])}
            state   = "GAMING" if running & {p.lower() for p in _GAMING_PROCS} else "FREE"
            if state != current:
                current = state
                client.publish("phoebe/pc/gpu_state", current, retain=True, qos=1)
                print(f"[GPU] State → {current}", flush=True)
        except Exception:
            pass


# ── Heartbeat ─────────────────────────────────────────────────
def _heartbeat_loop():
    while True:
        time.sleep(30)
        try:
            client.publish("phoebe/pc/alive", "still here", retain=True, qos=1)
        except Exception:
            pass


# ── Health publisher ──────────────────────────────────────────
def _health_loop():
    """Publish CPU%, mem%, disk% to Atlas every 60s."""
    while True:
        time.sleep(60)
        try:
            cpu  = psutil.cpu_percent(interval=1)
            mem  = psutil.virtual_memory().percent
            disk = psutil.disk_usage("C:\\").percent
            payload = f"cpu={cpu:.1f} mem={mem:.1f} disk={disk:.1f}"
            client.publish("phoebe/pc/health", payload, retain=True, qos=1)
        except Exception:
            pass


# ── Start background threads ──────────────────────────────────
threading.Thread(target=_heartbeat_loop, daemon=True).start()
threading.Thread(target=_gpu_state_loop, daemon=True).start()
threading.Thread(target=_health_loop,    daemon=True).start()

print("[PC] Presence node running. Ctrl+C to stop.", flush=True)

# ── Keep alive ────────────────────────────────────────────────
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    client.publish("phoebe/pc/alive", "offline", retain=True, qos=1)
    client.disconnect()
    print("[PC] Shutdown cleanly.", flush=True)
