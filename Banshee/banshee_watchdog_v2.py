#!/usr/bin/env python3
# banshee_watchdog_v2.py — Phoebe Watchdog
# Banshee (Pi Zero 2 W) → monitors all nodes. Silent alert on failure.
#
# ── Cron every 30 seconds ─────────────────────────────────────────────────────
#   crontab -e → add both lines:
#   * * * * * /usr/bin/python3 /home/<NODE_USER>/banshee_watchdog_v2.py
#   * * * * * sleep 30 && /usr/bin/python3 /home/<NODE_USER>/banshee_watchdog_v2.py
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import subprocess
import time
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

NODE_USER   = "phoebe"          # Linux username on all nodes — set to match your system
ATLAS_IP    = os.environ.get("ATLAS_IP", "YOUR_ATLAS_IP")
LUNA_IP     = os.environ.get("LUNA_IP",  "YOUR_LUNA_IP")
ORION_IP    = os.environ.get("ORION_IP", "YOUR_ORION_IP")
SSH_KEY     = f"/home/{NODE_USER}/.ssh/id_ed25519"
SSH_OPTS    = ["-o", "StrictHostKeyChecking=no",
               "-o", "ConnectTimeout=6",
               "-o", "BatchMode=yes"]
SSH_TIMEOUT = 10
DEAD_AFTER  = 30      # seconds dead before alert + restart (raised from 5)
STATE_FILE  = Path("/tmp/banshee_state.json")
NTFY_TOPIC  = "phoebe-banshee"

VENV        = f"/home/{NODE_USER}/phoebe-env/bin/activate"

# Atlas static IP — NEVER use mDNS here; broker lives on Atlas, .local fails when Atlas is down
ATLAS_RELAY_PIN = 17    # BCM GPIO pin wired to Atlas relay — verify before first use

# MQTT auth for recovery publish — must match Atlas broker credentials
MQTT_USER     = "phoebe"
MQTT_PASSWORD = ""      # set via env var in production; placeholder here


def _py(script):
    """Restart command: activate phoebe-env, launch detached."""
    return (
        f"bash -c 'source {VENV} && "
        f"nohup python {script} </dev/null >/dev/null 2>&1 &'"
    )


SERVICES = {
    "luna": {
        "host":      LUNA_IP,
        "user":      NODE_USER,
        "check":     "pgrep -f phoebe_luna.py",
        "alive":     lambda rc, out: rc == 0,
        "restart":   _py(f"/home/{NODE_USER}/phoebe_luna.py"),
        "label":     "Phoebe Luna",
        "two_stage": False,
    },
    "orion": {
        "host":      ORION_IP,
        "user":      NODE_USER,
        "check":     "pgrep -f phoebe_orion.py",
        "alive":     lambda rc, out: rc == 0,
        "restart":   _py(f"/home/{NODE_USER}/phoebe_orion.py"),
        "label":     "Phoebe Orion",
        "two_stage": False,
    },
    "pc": {
        "host":      "phoebe-pc.local",     # Windows — set static IP in router if mDNS unreliable
        "user":      NODE_USER,
        "check":     'tasklist /FI "IMAGENAME eq python.exe" /NH',
        "alive":     lambda rc, out: "python.exe" in out.lower(),
        "restart":   (
            "powershell -Command "
            "\"Start-Process python "
            "-ArgumentList 'C:/PhoebeLocal/PhoebeBot/phoebe.py' "
            "-WindowStyle Hidden\""
        ),
        "label":     "Phoebe PC",
        "two_stage": False,
    },
    "atlas": {
        "host":      ATLAS_IP,              # static IP — never .local
        "user":      NODE_USER,
        "check":     None,                  # ICMP ping only — SSH not used for Atlas health
        "alive":     None,                  # handled via _ping() in _watch()
        "restart":   None,                  # two-stage — handled in _atlas_restart()
        "label":     "Atlas",
        "two_stage": True,
    },
}


# ── State ─────────────────────────────────────────────────────────────────────

def _load():
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def _save(state):
    try:
        STATE_FILE.write_text(json.dumps(state))
    except Exception:
        pass


# ── Ping ──────────────────────────────────────────────────────────────────────

def _ping(host):
    """ICMP ping. Returns True if host responds."""
    try:
        r = subprocess.run(
            ["ping", "-c", "1", "-W", "3", host],
            capture_output=True, timeout=5,
        )
        return r.returncode == 0
    except Exception:
        return False


# ── SSH ───────────────────────────────────────────────────────────────────────

def _ssh(svc, cmd):
    try:
        r = subprocess.run(
            ["ssh", "-i", SSH_KEY, *SSH_OPTS,
             f"{svc['user']}@{svc['host']}", cmd],
            capture_output=True, text=True, timeout=SSH_TIMEOUT,
        )
        return r.returncode, r.stdout
    except Exception:
        return 1, ""


# ── Notify ────────────────────────────────────────────────────────────────────

def _notify(msg):
    try:
        subprocess.run(
            ["curl", "-s", "-d", msg, f"https://ntfy.sh/{NTFY_TOPIC}"],
            capture_output=True, timeout=10,
        )
    except Exception:
        pass


# ── Atlas two-stage restart ───────────────────────────────────────────────────

def _atlas_restart(svc):
    """Two-stage Atlas restart: SSH graceful shutdown → 60s → GPIO relay hard cut."""
    # Stage 1: graceful shutdown via SSH
    _notify("Atlas down — graceful shutdown signal sent. Hard power cut in 60s.")
    _ssh(svc, "sudo systemctl stop phoebe-atlas.service 2>/dev/null || true")
    time.sleep(60)

    # Stage 2: GPIO relay hard power cut + restore
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(ATLAS_RELAY_PIN, GPIO.OUT)
        GPIO.output(ATLAS_RELAY_PIN, GPIO.LOW)   # cut power
        time.sleep(5)
        GPIO.output(ATLAS_RELAY_PIN, GPIO.HIGH)  # restore power
        GPIO.cleanup()
        _notify("Atlas: hard power cut fired. Waiting for recovery.")
    except ImportError:
        _notify("Atlas: RPi.GPIO not available — hard cut skipped. Manual intervention required.")
        return
    except Exception as e:
        _notify(f"Atlas: GPIO relay error — {e}")
        return

    # Wait for Atlas to recover, then publish restart event
    for _ in range(24):   # up to 2 minutes
        time.sleep(5)
        if _ping(ATLAS_IP):
            _notify("Atlas recovered after hard restart.")
            try:
                import paho.mqtt.publish as _pub
                payload = json.dumps({"ts": time.time(), "reason": "watchdog_relay"})
                _pub.single(
                    "banshee/event/restart", payload=payload,
                    hostname=ATLAS_IP, port=1883,
                    auth={"username": MQTT_USER, "password": MQTT_PASSWORD},
                    retain=True, qos=1,
                )
            except Exception:
                pass   # broker may not be ready yet — ntfy alert already sent
            return

    _notify("Atlas: no recovery detected after hard restart. Manual check required.")


# ── Watch ─────────────────────────────────────────────────────────────────────

def _watch(name, svc, state):
    now = time.time()
    s   = state.get(name, {"down_since": None, "notified": False, "restarted": False})

    # Atlas: ICMP ping. All others: SSH + process check.
    if svc.get("two_stage"):
        alive = _ping(svc["host"])
    else:
        alive = svc["alive"](*_ssh(svc, svc["check"]))

    if alive:
        state[name] = {"down_since": None, "notified": False, "restarted": False}
        return

    if s["down_since"] is None:
        s = {"down_since": now, "notified": False, "restarted": False}

    if (now - s["down_since"]) >= DEAD_AFTER and not s["notified"]:
        _notify(f"{svc['label']} down — restarting.")
        s["notified"] = True

    if s["notified"]:
        if svc.get("two_stage"):
            # Atlas: fire once only — two-stage kill is not idempotent
            if not s.get("restarted"):
                s["restarted"] = True
                state[name] = s
                _save(state)
                _atlas_restart(svc)
        else:
            # Luna/Orion/PC: restart every cycle — nohup launch is idempotent
            _ssh(svc, svc["restart"])

    state[name] = s


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    state = _load()
    for name, svc in SERVICES.items():
        _watch(name, svc, state)
    _save(state)


if __name__ == "__main__":
    main()
