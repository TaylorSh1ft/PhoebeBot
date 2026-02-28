"""Quick test for Atlas health request."""
import json, time, threading
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
        "request_id": "health-test-001",
        "type": "health",
        "data": {}
    }), qos=1)
    print("[TEST] Health request sent.")

def on_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())
    if payload.get("request_id") != "health-test-001":
        return
    result = payload.get("result", {})
    print("\n" + "="*50)
    print(f"  CPU:      {result.get('cpu_pct')}%")
    print(f"  Memory:   {result.get('mem_pct')}% ({result.get('mem_used_gb')}GB / {result.get('mem_total_gb')}GB)")
    print(f"  Disk:     {result.get('disk_pct')}% free: {result.get('disk_free_gb')}GB")
    print(f"  Uptime:   {result.get('uptime')}")
    print(f"  Last task:{result.get('last_task')}")
    print(f"  Whisper:  {result.get('whisper_ready')}")
    print(f"  FinBERT:  {result.get('finbert_ready')}")
    print("="*50)
    result_event.set()

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="phoebe-health-test")
client.username_pw_set(MQTT_USER, MQTT_PASS)
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, 1883, 60)
client.loop_start()

if not result_event.wait(timeout=15):
    print("[TEST] No response within 15s.")
client.loop_stop()
