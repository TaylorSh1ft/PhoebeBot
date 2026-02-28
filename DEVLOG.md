# PhoebeBot Development Log

Short entries. Facts and state. No essays.

---

### 2026-02-27 — Episodic Memory + Chat Performance (think=False) + Fast-Path Routing

**Episodic Memory (confirmed working):**
- Added `episodes` table to SQLite (`id`, `summary`, `tone`, `ts`).
- `_save_episode(user_msg, reply)`: calls Qwen to tag tone + one-sentence summary, saves to DB in background thread after each chat exchange. Never blocks response.
- `_load_episodes(limit=5)`: fetches 5 most recent episodes, ordered oldest-first for narrative context.
- `_build_chat_messages()` now injects recent episodes into Mistral system prompt.
- Confirmed: episodes table populating correctly, full summaries stored (log truncates at 70 chars — data intact).
- Example saved: `[neutral] Phoebe explains that she doesn't experience emotions like humans but is ready to help`.

**Chat Routing Fast-Path:**
- Added `_FINANCIAL_HINT_RE` regex. If no financial keywords present, `_route()` returns CHAT immediately — no Ollama call at all.
- Confirmed: `[Route] CHAT (fast path — no financial keywords)` fires instantly for plain chat messages.
- Mistral routing call now only happens when message contains financial keywords.

**Qwen3 think=False — 26x speedup:**
- Root cause of 6.5-minute chat responses: `think=False` was placed inside `options={}` dict. Ollama ignores unknown option keys silently — Qwen was running full chain-of-thought reasoning on every response.
- Fix: `think=False` is a top-level keyword arg to `ollama.chat()`, confirmed via `inspect.signature(ollama.chat)`.
- Also removed `/no_think` text injection from messages (redundant with proper `think=False`).
- Result: chat response time dropped from ~6.5 minutes → **15 seconds**. Confirmed via timed MQTT test.
- Also removed `num_ctx: 4096` — caused Ollama model runner OOM (status 500) due to oversized KV cache with all components loaded. Default (2048) is correct for this Jetson configuration.

**Bad name fact cleared:**
- `user_facts` table had `name=Doing Is` — a phantom extraction from an earlier test. Deleted via Python sqlite3 one-liner over SSH.
- `_NAME_RE` regex didn't match "how are you today" — origin unclear, likely from a previous session test message.

**State after this session:**
- Chat: fast-path routing + think=False = 15s responses. Episodic memory saves after each exchange.
- All Atlas components loading cleanly: Whisper, FinBERT, Qwen, code model. Memory is tight (91MB free with all loaded) — no headroom for num_ctx increases or GPU layer bumps.
- Episodic memory working end-to-end.

---

### 2026-02-27 — TTS Fix + Wake Word Investigation + Self-Scripting v2 Voice Approval

**TTS Fixed (two bugs):**
- Bug 1: `_PIPER_MODEL` was built with `_SCRIPT_DIR` as base — pointed to `~/PhoebeBot/piper-voices/` which doesn't exist. Model lives at `~/piper-voices/`. Fixed to `os.path.expanduser("~/piper-voices/en_US-amy-medium.onnx")`.
- Bug 2: systemd service runs with restricted PATH — `piper` binary at `~/.local/bin/piper` not visible. Fixed with `_PIPER_BIN = os.path.expanduser("~/.local/bin/piper")` and using it in `_piper_speak()` instead of bare `"piper"`.
- TTS confirmed working: Phoebe speaks audibly through plughw:3,0 (USB audio card 3).

**Wake Word — Same-Room TV Problem (unresolved, parked):**
- Energy gate (min=35) successfully blocks distant TV (other room, energy ~21–28).
- Same-room TV at normal volume produces energy 51–186 at mic — louder than user voice from couch (38–46).
- Retrained hey_phoebe.onnx with 30 new couch-position TV negatives. Final loss improved 0.1006 → 0.0480. Made no practical difference — TV still false-triggers at 0.82–0.95 confidence.
- Added max energy gate (`_OWW_ENERGY_GATE_MAX = 80`) to reject high-energy TV frames. Untested with user voice + TV simultaneously — may block real triggers if user speaks over TV.
- Root cause: same-room TV is physically louder at the mic than the user's voice. Energy gate and model training cannot reliably separate them in this scenario.
- **Wake word disabled** via `_OWW_WAKE_ENABLED = False`. Mic stream stays open — `listen_from_mic()` still works for directed listening (scripting approval etc). Re-enable with one flag flip.

**Suggested paths forward for wake word (no timeline):**
- **Button trigger (recommended):** Cheap Zigbee button through HA → publish to `phoebe/luna/wake_trigger`. Luna subscribes and activates listen_from_mic(). 100% reliable, zero false triggers, natural UX. Needs small Luna handler + HA automation.
- **Whisper double-check:** After wake word fires, immediately transcribe buffered audio and confirm it contains "hey Phoebe" before proceeding. Adds ~1–2s latency, very accurate. More code.
- **More negatives:** Diminishing returns — 30 new clips didn't move the needle. Would need 100+ same-room clips of same content at same volume. Low confidence this solves it.

**Health Voice Command (confirmed working end-to-end):**
- Fixed two bugs in `_handle_health()`: `self._replace_last_message("Phoebe", reply, "phoebe")` → `self._replace_last_message(reply)` (wrong arg count), and `self.speak_text(reply)` → `speak_text_sync(reply)` (method doesn't exist).
- Added `phoebe/luna/command` MQTT topic — any text published here dispatches to `_dispatch_input()`. Useful for testing without wake word and is the foundation for future button trigger.
- Confirmed working: MQTT command → health regex → `_handle_health()` → Atlas health request → spoken CPU/mem/disk/uptime/last_task/Whisper/FinBERT reply.

**Self-Scripting v2 — Luna Voice Approval (confirmed working end-to-end):**
- Luna now subscribes to `phoebe/atlas/script_preview`.
- `_handle_script_preview()`: speaks description, calls `listen_from_mic()`, classifies approve/reject via regex, publishes `{script_id, action}` to `phoebe/atlas/script_approve`.
- Default on no-hear: reject (safe failure).
- Mic stream confirmed still active with wake word disabled — `listen_from_mic()` works.
- Atlas approval gate was already wired — any node publishing to `script_approve` resolves the pending event.
- Full loop confirmed: route → generate (11s) → preview published → Luna speaks it → Luna listens → transcribe → approve via MQTT → script runs → result returned.
- Issue: first two test runs timed out. Root cause 1: test description "print the current date and time on Atlas" didn't match `_SCRIPT_RE` regex — fell through to Mistral routing, got classified as CHAT, response sent to `phoebe/atlas/response` (not `script_preview`). Fix: use descriptions with "write/create/make/build + script/program". Root cause 2: `_code()` had no timeout on `ollama.chat()` — first test's Mistral call held `_ollama_lock` for 14+ minutes while `_code()` blocked waiting. Fix: replaced module-level `ollama.chat()` with `ollama.Client(timeout=180)` dedicated client — timeout exception releases lock cleanly via `with` block.
- Issue: `_ollama_lock` serializes all model calls including CPU-only qwen2.5-coder. If a GPU model (Mistral/Qwen) is mid-generation, code generation waits. Not a bug — just a known latency characteristic. Acceptable for rare script requests.
- Script generation timing: 11 seconds on Jetson CPU after lock acquired (fast once unblocked).

---

### 2026-02-26 — Wake Word Retrain + Mic Repositioning + TTS Fix
- Mic cable arrived. Fifine K669 moved to couch end table.
- Recorded 25 new positive clips via record_wake_word.py on Luna (device 1, 48kHz→16kHz decimation)
- Replaced old positives (bad mic position) with new ones, cleared positive_features.npy cache, retrained
- Training completed: final loss 0.1006, 500 augmented positives vs 4249 negatives
- Deployed new hey_phoebe.onnx to Luna
- Fixed TTS audio: aplay was defaulting to HDMI (nothing connected). Routed to plughw:3,0 (USB audio card 3). Phoebe now speaks audibly.
- Issue: record_wake_word.py was hardcoded to 16kHz — Fifine K669 only supports 48kHz natively. Fixed: record at MIC_RATE=48000, decimate by 3 to TARGET_RATE=16000 using numpy before saving WAV.
- Issue: train_hey_phoebe.py had Unicode arrow characters (→) in print statements — UnicodeEncodeError on Windows cp1252 terminal. Fixed: replaced with ASCII ->.
- Issue: record_env venv required but not obvious — training script docstring had the invocation but system python was used first. record_env/Scripts/python is the correct runner on PC.
- Issue: TTS still not audible despite aplay routing to plughw:3,0. speaker-test on card 3 produced audible tone, so the device is valid. Root cause unknown — piper stderr is suppressed (DEVNULL) so failures are silent. Next session: temporarily enable piper/aplay stderr, test piper manually on Luna to isolate whether the issue is piper, aplay, or the device mapping changing on restart.
- Pending: wake word still false-triggering on TV speech. _OWW_ENERGY_GATE=0 (disabled). Need to compare user voice energy vs TV energy from couch position to set gate threshold. Luna stopped for the night.
- Pending: energy gate tuning session — tail logs while saying "hey Phoebe" from couch, note energy values, set gate above TV floor but below voice floor.
- Pending: debug TTS audio pipeline — enable stderr on piper/aplay, test manually via SSH before restarting service.

### 2026-02-26 — Atlas Sandbox Hardening
- Added AST static analysis in _script_static_analysis() — runs before every script execution
- Blocked imports: socket, subprocess, ftplib, smtplib, telnetlib, paramiko, asyncio, multiprocessing, ctypes, cffi
- Blocked calls: eval, exec, compile, __import__, os.system, os.popen, os.exec*, os.spawn*
- String check: rejects any script containing "sudo"
- Line cap: _SCRIPT_MAX_LINES=150 — rejects oversized scripts
- Restricted subprocess env: scripts inherit only PATH/HOME/PYTHONDONTWRITEBYTECODE — no API keys or secrets
- Added import ast to phoebe_atlas.py
- Issue: Atlas does not have passwordless sudo over SSH (unlike Orion) — systemctl restart required manual login. Fix: SSH in and restart manually. Long-term: add NOPASSWD for systemctl restart phoebe-atlas specifically.

### 2026-02-26 — Orion Decision Logging
- Added orion_decisions.jsonl — structured JSONL trade decision log for calibration review
- Each record: timestamp, ticker, side, full idea text, signals snapshot
- Signals captured: source (manual/background_news), portfolio summary, top-5 FinBERT-scored headlines (label + confidence), price check text, Kalshi context
- Removed [:80] truncation from _log_trade — full idea text now in orion_trades.log
- _fetch_trade_idea() now returns (idea, signals) tuple — callers updated
- _alpaca_act_on_idea() accepts signals= kwarg, calls _log_decision() before order
- Background news loop passes its own scored_headlines snapshot as signals
- Issue: Alpaca paper account replaced with new $500 account — old keys invalidated. Updated ~/.env on Orion via SSH sed, restarted service. Issue: stale Python process was holding port 5000 after rapid restart cycles — resolved with fuser -k 5000/tcp.

### 2026-02-26 — Orion OTC Ticker Fix
- Added pre-flight asset validation in _alpaca_place_paper_order
- Before submitting any order: calls client.get_asset(ticker) and checks asset.tradable
- If asset not found or not tradable: logs ASSET_NOT_TRADABLE, skips order — no wasted attempt
- Fixes recurring ORDER_FAILED on OTC/foreign tickers (CONST, KZT, HELUS, KOSL, RMRT)
- No new imports needed — TradingClient.get_asset() already available
- Issue: root cause was Qwen extracting valid-looking but Alpaca-unsupported tickers from headlines (OTC/foreign stocks). Fixed at execution layer rather than generation layer — LLM output is untrusted, validation gate is the right place.

### 2026-02-25 — Health Command + Housekeeping
- Atlas: added health request type — returns CPU%, mem%, disk%, uptime, last_task, whisper_ready, finbert_ready
- Atlas: fixed phoebe/pc/gpu_state bad JSON warning — plain string handled before JSON parse
- Luna: added _handle_health() — asks Atlas, formats spoken reply, runs in background thread
- Luna: added health regex fast-path in _dispatch_input() ("how's Atlas", "system health", etc.)
- Deleted phoebe-neural.py, phoebe-good.py (old prototypes), Banshee/banshee_watchdog.py (v1)
- Pending: test health command via voice after mic cable arrives tomorrow
- Issue [reconstructed]: phoebe/pc/gpu_state was publishing a plain string ("FREE"/"GAMING") but Atlas was attempting JSON.loads() on it — logged a warning on every heartbeat. Fixed by checking for plain string before JSON parse.

### 2026-02-25 — Atlas FinBERT Live
- Installed transformers 5.2.0, upgraded Pillow 12.1.1 + scipy 1.15.3 (system versions too old for numpy 2.x)
- torch 2.10.0+cpu was already installed in ~/.local — confirmed working
- FinBERT (ProsusAI/finbert) loads and scores correctly: "Apple stock surged 5%" → positive 94%
- Added lazy-load pattern to phoebe_atlas.py: _load_finbert() background thread, _finbert_ready Event
- Added _score_sentiment(texts) helper — returns highest-scoring label per text, 5s timeout, graceful fallback
- Wired into _build_chat_messages(): non-neutral scores (>60% confidence) injected into Mistral system prompt
- Atlas now has Whisper + FinBERT + Qwen + code model all loading at startup
- Pending: test sentiment injection end-to-end via Luna voice input on a market-related chat
- Issue [reconstructed]: system-level Pillow and scipy were too old for numpy 2.x — transformers install failed until both were upgraded via pip3. torch was already present in ~/.local from a prior install and did not need reinstalling.

### 2026-02-25 — systemd for Luna + Orion
- Created /etc/systemd/system/phoebe-luna.service on Luna (<luna-ip>)
- Created /etc/systemd/system/phoebe-orion.service on Orion (<orion-ip>)
- Both: Restart=always, RestartSec=10, EnvironmentFile=~/.env, OOMScoreAdjust=-100
- Both enabled on boot — no more tmux sessions needed on any node
- All three bots (Atlas, Luna, Orion) now fully systemd-managed
- Luna confirmed running: [Wake] Mic stream open — holding permanently
- Orion confirmed running: MQTT connected to Atlas broker

### 2026-02-25 — GitHub Backup + Auto-Push
- Initialized git repo in C:\PhoebeLocal\PhoebeBot
- Created .gitignore (excludes .env, kalshi_private.key, wake_word_training/, audio files, face data)
- Created .env.example (safe template documenting all required keys)
- Initialized private GitHub backup repo for PhoebeBot
- Created backup_push.ps1 — commits changes and pushes to GitHub
- Added `backup` alias to PowerShell profile and Git Bash .bashrc
- Added Backup Rule to CLAUDE.md — Claude reminds user to run backup after milestones

### 2026-02-25 — Self-Scripting Foundation Complete
- Built full self-scripting loop on Atlas: MQTT preview + approval gate + sandboxed execution
- Added qwen2.5-coder:1.5b as dedicated code model (_code() with fallback to _qwen())
- MQTT topics: phoebe/atlas/script_preview, phoebe/atlas/script_approve, phoebe/atlas/script_result
- Approval window: 600s (10 min) — indefinite wait, no countdown pressure
- Fixed markdown fence stripping: qwen2.5-coder was wrapping output in ```python blocks — stripped in _generate_script() with regex
- Atlas crashed mid-session (kernel panic/watchdog reboot) — came back up on its own via systemd Restart=always
- Enabled persistent journald logging: /var/log/journal created — future crashes will be capturable
- Full loop confirmed working: route → generate → preview → approve → sandbox run → result
- Pending: Luna voice approval wire-up (subscribe to script_preview, speak prompt, listen, publish approve/reject)
- Pending: sandbox hardening (timeout already in, add restricted dir + no-network flag)
- Created test_scripting.py for CLI testing of the full loop
- Issue [reconstructed]: qwen2.5-coder:1.5b consistently wrapped generated scripts in ```python markdown fences despite being prompted not to. Fixed by stripping fences with regex in _generate_script() — LLM instruction alone was not reliable enough.
- Issue [reconstructed]: Atlas suffered a kernel panic / watchdog reboot mid-session. No data lost. systemd Restart=always brought it back automatically. Persistent journald enabled afterward so future crashes are capturable.

### 2026-02-25 — Wake Word Mic Repositioning + Energy Gate
- Recorded 50 negative clips through Luna's mic via remote arecord loop (TV audio, Comedy Central)
- Fixed train_hey_phoebe.py: os.listdir → os.walk (recursive) so subdirectory negatives are picked up
- Retrained hey_phoebe.onnx with 66 custom negatives (51 new TV clips + 15 existing)
- Still false-triggering at 0.98–0.99 — model cannot distinguish TV speech from "hey Phoebe" in embedding space
- Root cause identified: Fifine K669 mic sitting next to soundbar — TV audio louder at mic than user's voice
- Fix: ordered 15ft USB-B to USB-A cable (arriving tomorrow). Fifine moves to couch end table.
- Added _OWW_ENERGY_GATE = 0 constant to phoebe_luna.py — gate code in place, disabled pending mic repositioning
- Banshee Camera Module 3: assigned to desk presence detection, publishes to MQTT. Keeps watchdog role clean.
- Luna stopped for the day. Will retrain + tune energy gate after mic repositioning tomorrow.
- Pending: retrain with mic in new position, tune energy gate, test couch + desk coverage
- Pending: plug Razer wide-angle USB webcam into Luna — face recognition code already in phoebe_luna.py, just needs a camera. Test webcam mic as secondary audio device. Watch first startup log for USB bandwidth issues (webcam stream + mic simultaneously).
- Issue [reconstructed]: train_hey_phoebe.py used os.listdir() which only read top-level files — custom negatives organized into subdirectories were silently ignored during training. Fixed to os.walk() so all nested clips are picked up.
- Issue [reconstructed]: After retraining with 66 negatives, model still false-triggered at 0.98–0.99 confidence on TV speech. More negatives alone couldn't fix it — the mic was physically too close to the soundbar, making TV audio the dominant signal at the mic. Hardware repositioning is the real fix, not more training data.

---

### 2026-02-17 — Luna/Pi Migration
- Migrated phoebe_luna.py from Windows to Raspberry Pi 5 headless
- Removed winsound, pvporcupine, pvrecorder (imports, vars, functions, threads)
- Replaced PvRecorder with pyaudio in listen_from_mic() — same energy/mood logic preserved
- Added headless mode: _HEADLESS flag, _after() helper, _dispatch_input() unified routing
- Guarded comtypes/pycaw mic mute behind sys.platform == "win32"
- Removed voice recall (_play_voice_recall, winsound dependency) — _voice_buffer preserved
- Removed wake word loop (pvporcupine) — placeholder left for open wake word
- Added MQTT (paho-mqtt v2): phoebe.py publishes phoebe/pc/alive, Luna publishes phoebe/luna/ready
- Issue [reconstructed]: original Luna code used Windows-only libraries throughout (winsound, pvporcupine, pvrecorder, comtypes, pycaw) — none available on Linux ARM. Required full audio stack swap to pyaudio and systematic platform guards before the Pi would run the file at all.

### 2026-02-20 — Atlas Architecture Design
- Added Jetson Orin Nano ("Atlas") to the system as always-on core brain
- Designed five-node architecture: Atlas, Orion, Luna, Banshee, PC
- MQTT broker moves to Atlas. Atlas takes over Ollama. PC becomes GPU donor.
- Banshee: two-stage kill (graceful → relay), OOB heartbeat via ICMP/TCP not MQTT
- All nodes: clean_session=False, QoS 1, exponential backoff reconnect, static IPs, NTP

### 2026-02-21 — Open Wake Word Integration (Luna)
- Wrote record_wake_word.py, train_hey_phoebe.py (fully local, no Colab)
- Added openWakeWord integration to phoebe_luna.py: _wake_word_loop(), _wake_triggered()
- hey_phoebe.onnx generated and deployed to Luna
- Issue [reconstructed]: the official openWakeWord training pipeline on GitHub required Google Colab and cloud dependencies — not usable for a fully local, offline-capable system. Rather than compromise on that, wrote a custom training pipeline from scratch: record_wake_word.py to capture positive clips via the local mic, and train_hey_phoebe.py to handle feature extraction, embedding generation, incremental negative caching, and ONNX model export — all running locally on the PC with no external services. This is one of the more significant original contributions in the project.

### 2026-02-21 — MQTT Hardening + Atlas Bringup
- Atlas first boot: JP 6.2.1 (JP 5.1.3 was unstable)
- Installed Mosquitto (allow_anonymous false), Ollama, pulled qwen3:8b + mistral:7b
- Mistral GPU layer testing: 8 = cudaMalloc OOM, 6 = stable — locked to num_gpu:6
- MQTT hardening complete across all three bot files (audit items 1-5, 8, 9, 10, 22)
- Issue [reconstructed]: JetPack 5.1.3 was unstable on the Orin Nano — system would not reliably boot. Reflashed to JP 6.2.1, which has been stable ever since.
- Issue [reconstructed]: Mistral at num_gpu=8 hit cudaMalloc OOM on the Jetson's shared memory. Tested down to 6 layers — stable at 572MB GPU footprint. Locked there permanently.

### 2026-02-22 — HTTP Removal, Banshee Atlas Integration, QoS Hardening
- Removed all HTTP Orion backend from phoebe_luna.py
- Migrated user_get + birthday_check to _atlas_request MQTT calls
- Replaced _poll_orion_pending (HTTP poll) with phoebe/orion/alert push subscription
- Banshee v2 hardened: ICMP ping, DEAD_AFTER 5s → 30s, two-stage kill, recovery publish
- QoS 0 → QoS 1 on all heartbeat/status publish() calls (all four files)
- Issue [reconstructed]: Banshee's DEAD_AFTER was set to 5s — too aggressive, caused false-trigger restarts on brief network hiccups. Raised to 30s to match realistic Atlas recovery time.

### 2026-02-22 — Home Assistant Integration
- HA_TOKEN added to Atlas .env
- Added _ha_call(), handle_home_control() to phoebe_atlas.py
- HOME_CONTROL wired into handle_route_and_respond()
- Entities: lights (couch, couch L/R, nightstand), media_player (living room TV, AppleTVs)

### 2026-02-22 — Deployment + Node Bringup
- All nodes SCPd and running. Static IPs confirmed and locked.
- No venvs on any node — system Python3, pip3 only

### 2026-02-22 — Atlas systemd + MQTT Reconnect Fix
- Fixed _mqtt_on_disconnect threading bug in phoebe_atlas.py
- Created + enabled /etc/systemd/system/phoebe-atlas.service (Restart=always)
- OOM tuning: mosquitto OOMScoreAdjust=-200, phoebe-atlas=-100, ollama=300
- Journald capped: SystemMaxUse=200M, MaxRetentionSec=1week
- Atlas fully hardened — survives reboots, auto-recovers, disk-safe, OOM-safe
- Issue [reconstructed]: MQTT on_disconnect callback was calling reconnect() directly from the callback thread — paho-mqtt does not allow this and it caused a deadlock. Fixed by wrapping reconnect() in a new Thread so the callback returns immediately.

### 2026-02-22 — Orion MQTT Fix + FinBERT Live
- Root cause of Orion reconnect loop: empty MQTT_PASSWORD in .env
- PyTorch 2.10.0 installed on Orion (torch only, CPU ARM64)
- FinBERT (ProsusAI/finbert) live on Orion — confirmed "Model loaded"
- bert.embeddings.position_ids UNEXPECTED warning is benign (known HF artifact)
- alpaca-py installed, paper trading re-enabled, GTC orders, trades logged to orion_trades.log
- Kalshi: public endpoints working. RSA-PSS method appears deprecated — revisit against current API docs.
- Issue [reconstructed]: Orion was stuck in a reconnect loop on startup. Root cause was MQTT_PASSWORD left blank in .env — broker rejected auth silently and the client retried indefinitely. Fixed by populating the password. Easy fix, subtle symptom.

### 2026-02-21 (evening) — Luna↔Atlas Loop Complete
- phoebe_atlas.py built and deployed: MQTT, Whisper (tiny.en), Qwen routing, Mistral chat, SQLite WAL
- Fixed training bug: class weights inverted. Fixed to weight positives by imbalance ratio. Retrained 30 epochs.
- Fixed mic device: both wake word loop and listen_from_mic() share _MIC_IDX/_MIC_SR/_MIC_DECIMATE
- Full loop confirmed: [Wake] → MQTT transcribe → MQTT route → response back to Luna
- Issue [reconstructed]: wake word model was training with inverted class weights — positives were being down-weighted instead of up-weighted against the imbalanced negative set. Model was effectively learning to ignore "hey Phoebe". Fixed weight calculation, retrained 30 epochs.

### 2026-02-22 — Luna Mic Architecture Fix + TTS + Voice Model
- Fixed OSError: [Errno -9998]: wake loop and listen_from_mic() both tried to open mic simultaneously
  - Fix: wake word loop owns one persistent stream forever; listen_from_mic() reads from _command_audio_queue
  - _mic_for_command event signals wake loop to route frames to command queue
- Replaced _wake_active with _tts_playing event — no stream lifecycle changes on TTS
- Added 3-second cooldown after wake word trigger
- Fixed MQTT reconnect deadlock in _mqtt_on_disconnect_luna — wrapped in Thread
- TTS voice: en_US-amy-medium (female)
- Issue [reconstructed]: OSError -9998 (invalid number of channels / device busy) on Luna — two threads were both calling pyaudio.open() on the same USB mic device. PyAudio does not allow concurrent opens. Fixed with a single-owner architecture: wake word loop holds the stream permanently, listen_from_mic() pulls frames from a shared queue instead of opening its own stream.
- Issue [reconstructed]: MQTT reconnect deadlock on Luna mirrored the Atlas bug — same root cause, same fix. Both files needed the Thread wrapper pattern.

### 2026-02-22 — Atlas↔Orion + HA Color + Mic Lock
- Atlas: _orion_request() — publishes to phoebe/atlas/orion_request, waits Event up to 60s
- Atlas: PORTFOLIO/NEWS/TRADE_IDEA/WATCHLIST now call _orion_request() (no longer stubs)
- Orion: _handle_atlas_request() dispatches to correct handler
- HA color support: color_name (CSS names) + color_temp (warm=450, cool=175 mireds)
- Fixed broken JSON extraction: strip("```json") bug → replaced with re.search
- Luna mic lock: _mic_listen_lock prevents double-trigger queue corruption
- Issue [reconstructed]: JSON extraction used .strip("```json") — which strips individual characters, not the substring, corrupting valid JSON. Replaced with re.search() to find the first valid JSON object in the response regardless of surrounding text.
- Issue [reconstructed]: rapid double wake-word triggers caused two threads to write to the command audio queue simultaneously, corrupting the audio frame order. Fixed with _mic_listen_lock mutex.

### 2026-02-23 — HA Light Control Fixed (End-to-End Working)
- Fixed Whisper vocab: added initial_prompt with home control terms — eliminated "light" → "lay" errors
- Added _HOME_CONTROL_RE fast-path regex in _route() — skips Mistral for obvious HA commands
- Replaced Qwen call in handle_home_control() with deterministic _parse_ha_intent()
- Fixed JSON extraction bug: r'\{[^{}]*\}' → json.loads() + greedy regex fallback
- Raised Luna route timeout: 120s → 180s
- Added kitchen/bathroom entities: light.kitchen_fixture, light.bathroom_vanity, L/R mirror bulbs
- Qwen GPU layers: 0 → 6 (confirmed stable, 572MB available post-warmup)
- All HA commands complete in <1 second end-to-end
- Issue [reconstructed]: Whisper tiny.en was transcribing "turn on the light" as "turn on the lay" — common STT failure on short domain-specific words. Fixed by passing initial_prompt with home control vocabulary so Whisper biases toward known entity names.
- Issue [reconstructed]: HA commands were going through the full Qwen routing + Mistral chat pipeline — adding ~3 minutes of latency for something deterministic. Added _HOME_CONTROL_RE fast-path to skip LLM entirely for obvious commands, and replaced the Qwen JSON extraction in handle_home_control() with a deterministic parser. Latency dropped to <1s.

### 2026-02-24 — Wake Word Retrain + Architecture + Roadmap
- Recorded 25 positive clips (1-8 quiet, 9-25 TV on), 45 custom negative clips
- train_hey_phoebe.py: incremental custom negative embedding, preserves base cache
- record_negatives.py written for capturing TV/doorbell/phone/speech negatives
- Tested against State of the Union — still triggering (genuinely adversarial audio)
- Root cause: SotU is worst-case audio. Normal TV likely fine.
- Pending: test normal TV. If bad → retrain session with 50+ diverse TV speech negatives.
- Issue [reconstructed]: State of the Union address used as adversarial test audio — model triggered repeatedly. Determined this is an extreme edge case (dense, broadcast-quality human speech at high volume) rather than a model failure. Normal TV content is a more realistic baseline for tuning.

### 2026-02-23 — phoebe.py Slimmed to PC Presence Node
- Original full bot backed up → phoebe_backup.py
- phoebe.py rewritten as minimal PC presence node (~115 lines):
  MQTT, LWT, gpu_state FREE/GAMING, 30s heartbeat, 60s health, phoebe/pc/task stub
- Added phoebe.py to Windows startup via HKCU registry Run key

---

### Pre-Atlas Code Audit (2026-02-20, all items complete)

All audit items done:
1-5 CRITICAL: broker IP, LWT, clean_session, backoff, gpu_state ✓
6-8 HIGH: Banshee v2, Luna HTTP removal, heartbeat 30s ✓
9-11 MEDIUM: Orion key security, MQTT auth, QoS 1 ✓
15-19 HA INTEGRATION: token, broker, handler, route class, wire ✓
22 LOW: Orion docstring ✓

Remaining LOW housekeeping:
- [ ] Delete phoebe-neural.py and phoebe-good.py (old prototypes)
- [ ] Retire banshee_watchdog.py v1 (v2 is current)
