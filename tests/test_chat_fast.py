"""Quick test for chat fast-path (no financial keywords → skip Mistral routing)."""
import json, threading
import paho.mqtt.client as mqtt
from dotenv import load_dotenv
import os

load_dotenv()
BROKER    = os.getenv("MQTT_BROKER")
MQTT_USER = os.getenv("MQTT_USER", "phoebe")
MQTT_PASS = os.getenv("MQTT_PASSWORD", "")

result_event = threading.Event()

def on_connect(client, userdata, flags, rc, properties=None):
    client.subscribe("phoebe/atlas/response", qos=1)
    print("[MQTT] Connected + subscribed.")
    client.publish("phoebe/luna/request", json.dumps({
        "request_id": "chat-fast-001",
        "type": "route",
        "data": {"text": "how are you today", "history": []}
    }), qos=1)
    print("[TEST] Chat message sent: 'how are you today'")

def on_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())
    if payload.get("request_id") != "chat-fast-001":
        return
    result = payload.get("result", {})
    print("\n" + "="*50)
    reply = result.get('reply', result)
    print(f"  Reply: {reply}".encode('utf-8', errors='replace').decode('utf-8', errors='replace'))
    print("="*50)
    result_event.set()

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="phoebe-chat-fast-test")
client.username_pw_set(MQTT_USER, MQTT_PASS)
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, 1883, 60)
client.loop_start()

if not result_event.wait(timeout=180):
    print("[TEST] No response within 180s.")
client.loop_stop()
