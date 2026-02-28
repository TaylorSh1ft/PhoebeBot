"""
Self-scripting test tool.
Run this to send a script request to Atlas and approve/reject it.

Usage:
    python test_scripting.py
"""

import json
import time
import threading
import paho.mqtt.client as mqtt
from dotenv import load_dotenv
import os

load_dotenv()

BROKER       = os.getenv("MQTT_BROKER")
MQTT_USER    = os.getenv("MQTT_USER", "phoebe")
MQTT_PASS    = os.getenv("MQTT_PASSWORD", "")

received_preview = {}
received_result  = {}
preview_event    = threading.Event()
result_event     = threading.Event()


def on_connect(client, userdata, flags, rc, properties=None):
    print(f"[MQTT] Connected: {rc}", flush=True)
    client.subscribe("phoebe/atlas/script_preview", qos=1)
    client.subscribe("phoebe/atlas/script_result",  qos=1)
    print("[MQTT] Subscribed to script_preview + script_result", flush=True)


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
    except Exception:
        return

    if msg.topic == "phoebe/atlas/script_preview":
        received_preview.update(payload)
        print("\n" + "="*60)
        print(f"[PREVIEW] Script ID : {payload.get('script_id','')[:8]}...")
        print(f"[PREVIEW] Description: {payload.get('description','')}")
        print(f"[PREVIEW] Expires in : {payload.get('expires_in')}s")
        print("-"*60)
        print(payload.get("script", ""))
        print("="*60)
        preview_event.set()

    elif msg.topic == "phoebe/atlas/script_result":
        received_result.update(payload)
        print("\n" + "="*60)
        print(f"[RESULT] Script ID : {payload.get('script_id','')[:8]}...")
        print(f"[RESULT] Success   : {payload.get('success')}")
        if payload.get("output"):
            print(f"[RESULT] Output    :\n{payload['output']}")
        if payload.get("error"):
            print(f"[RESULT] Error     : {payload['error']}")
        print("="*60)
        result_event.set()


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="phoebe-test")
client.username_pw_set(MQTT_USER, MQTT_PASS)
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, 1883, 60)
client.loop_start()

time.sleep(2)  # wait for connection + subscriptions

# ── Send a script request ──────────────────────────────────────
import sys
if len(sys.argv) > 1:
    description = " ".join(sys.argv[1:])
else:
    try:
        description = input("\nWhat should Phoebe script? > ").strip()
    except EOFError:
        description = ""
if not description:
    description = "write a script that prints the current date and time and the Atlas system uptime"

print(f"\n[TEST] Sending route request: {description!r}")
request = {
    "request_id": "test-001",
    "type": "route",
    "data": {
        "text": description,
        "history": []
    }
}
client.publish("phoebe/luna/request", json.dumps(request), qos=1)

# ── Wait for preview ───────────────────────────────────────────
print("\n[TEST] Waiting for script preview from Atlas (up to 10 min — CPU generation is slow)...")
if not preview_event.wait(timeout=600):
    print("[TEST] No preview received within 600s. Check Atlas logs.")
    client.loop_stop()
    exit(1)

# ── Approve or reject ──────────────────────────────────────────
script_id = received_preview.get("script_id", "")
try:
    answer = input("\nApprove and run this script? [approve/reject] > ").strip().lower()
except EOFError:
    answer = "approve"  # auto-approve when running non-interactively
if answer != "approve":
    answer = "reject"

print(f"\n[TEST] Sending {answer} for script {script_id[:8]}...")
client.publish("phoebe/atlas/script_approve", json.dumps({
    "script_id": script_id,
    "action":    answer,
}), qos=1)

if answer == "approve":
    print("[TEST] Waiting for result...")
    if not result_event.wait(timeout=60):
        print("[TEST] No result received within 60s.")
    else:
        print("[TEST] Done.")
else:
    print("[TEST] Script rejected.")

client.loop_stop()
