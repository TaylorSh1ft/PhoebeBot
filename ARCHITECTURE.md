# PhoebeBot Architecture

Five nodes. Always-on core. Opportunistic PC. Hardware watchdog.

---

## Nodes

**Atlas (Jetson Orin Nano, <atlas-ip>)** — Core brain. Always on.
- MQTT broker (Mosquitto, authenticated, all nodes connect here)
- Ollama: Qwen (num_gpu=6) + Mistral (num_gpu=6) — both stable on Jetson at 6 layers
- Whisper transcription (tiny.en), FinBERT sentiment (pending install)
- Persistent memory + user state (SQLite, WAL mode)
- Home Assistant control via REST API
- systemd service: Restart=always, After=mosquitto+ollama+network-online
- OOM tuning: mosquitto=-200, phoebe-atlas=-100, ollama=+300 (models die first, broker last)
- Journald: SystemMaxUse=200M, MaxRetentionSec=1week
- Secrets: Kalshi private key, .env — Atlas only. Never on Luna or Orion.

**Orion (Raspberry Pi 5, <orion-ip>)** — Financial node. Mostly autonomous.
- Stock watcher, news crawler, FinBERT sentiment (PyTorch 2.10.0 CPU ARM64, confirmed live)
- Alpaca paper trading: autonomous GTC orders, logged to orion_trades.log
- Queues Atlas-dependent tasks and flushes on reconnect
- Responds to phoebe/atlas/orion_request — dispatches PORTFOLIO/NEWS/TRADE_IDEA/WATCHLIST
- Pushes alerts to phoebe/orion/alert (Luna subscribes)

**Luna (Raspberry Pi 5, <luna-ip>)** — Voice node. Thin client.
- Wake word (openWakeWord, hey_phoebe.onnx, threshold=0.80)
- Mic: USB at 48kHz, decimate to 16kHz. Wake loop owns persistent stream.
- listen_from_mic() reads from _command_audio_queue (no direct mic open)
- TTS: piper, en_US-amy-medium (female)
- Whisper STT forwarded to Atlas via MQTT (base64 WAV)
- All routing forwarded to Atlas via MQTT
- Degraded mode: fallback line + retry with backoff if Atlas unreachable

**Banshee (Raspberry Pi Zero, <banshee-ip>)** — Watchdog.
- Monitors Atlas via ICMP ping (NOT MQTT — broker lives on Atlas)
- DEAD_AFTER=30s. Two-stage kill: SSH graceful → 60s timeout → GPIO relay hard cut
- Publishes banshee/event/restart after Atlas recovers
- External alert (Pushover or Telegram) on fire — silent restarts are invisible problems
- Subscribes to banshee/reset for manual reboot trigger
- Pi Camera Module 3 installed — use case TBD

**PC (Windows, i9 + 4090)** — Opportunistic GPU donor. Not always on.
- phoebe.py: minimal presence node (~115 lines)
- Publishes phoebe/pc/gpu_state → FREE or GAMING (polls process list every 30s)
- LWT: phoebe/pc/alive → "dead"
- Subscribes to phoebe/pc/task — ready for Atlas GPU overflow dispatch
- Starts at Windows login via HKCU registry Run key

**HA Pi (Raspberry Pi 5, 8GB, <ha-ip>)** — Home Assistant only. Not a PhoebeBot node.
- Runs HassOS. PhoebeBot does not run on it.
- Atlas controls HA via REST API (POST /api/services/[domain]/[service]) with HA_TOKEN
- HA publishes entity state to Atlas MQTT broker on change
- HA_TOKEN in Atlas .env only. Never logged, never printed.

---

## MQTT Standards (all nodes)
- Broker: <atlas-ip>:1883 (authenticated — username/password)
- clean_session=False, QoS 1 for important messages
- Auto-reconnect with exponential backoff (reconnect() wrapped in Thread — never call from callback)
- Heartbeat: phoebe/[node]/alive every 30s (retained)
- Health: phoebe/[node]/health — CPU%, mem%, disk%, last task, uptime

## Home Assistant Entities (confirmed working)
- light.couch, light.couch_left_kauf_bulb, light.couch_right_kauf_bulb
- light.nightstand
- light.kitchen_fixture
- light.bathroom_vanity, light.right_mirror_kauf_bulb, light.mirror_left_kauf_bulb
- media_player.living_room_appletv, media_player.living_room_television, media_player.bedroom_appletv
- _parse_ha_intent() handles standard commands deterministically (no LLM). Qwen fallback for unusual phrasing.

## Key Files
- `phoebe.py` — PC presence node
- `phoebe_atlas.py` — Atlas brain (MQTT, Whisper, Qwen, Mistral, SQLite, HA)
- `phoebe_luna.py` — Luna voice node (wake word, mic, TTS, MQTT bridge)
- `phoebe_orion.py` — Orion financial node (news, FinBERT, Alpaca)
- `banshee_watchdog_v2.py` — Banshee watchdog (v1 is retired)
- `phoebe_backup.py` — original full bot backup (PC)
- `train_hey_phoebe.py` — local wake word training (incremental, preserves base cache)
- `record_wake_word.py` — record positive samples (25 clips)
- `record_negatives.py` — record negative samples (TV, doorbell, phone, speech)

## Self-Scripting (planned — Atlas only)
- Generated scripts run in restricted directory, no sudo, no unrestricted network
- Preview: Atlas publishes script + diff to MQTT, 30s Luna voice window, silence = discard
- Three-layer recovery: Phoebe self-heal → Banshee SSH → Banshee GPIO relay

## Planned Integrations
- Atlas FinBERT: install PyTorch on Atlas, wire into phoebe_atlas.py (Week 3)
- Google Calendar: OAuth, CalDAV, voice queries via Luna (Week 2)
- RTL-SDR on Orion: ADS-B first (IND cargo volume), then ACARS/NOAA/scanner (Week 3)
- Image generation: SDXL on PC 4090, Atlas queues, PC executes (Week 4+)
- Etsy API: automated listings, AI-generated disclosure required
- X/Twitter: Phoebe promotes Etsy listings (research API tier costs first)
- Discord Python tutor: Mistral chat, Stripe billing (nightshift on Atlas)
