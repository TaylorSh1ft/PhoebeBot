# PhoebeBot

A personal AI assistant built from scratch on distributed embedded hardware. Five nodes. Local LLMs. Voice interface. Home automation. Autonomous paper trading. A hardware watchdog that keeps everything alive.

Built in under four weeks by one person — self-taught, no CS degree — using Python, Claude, and Grok as development partners.

---

## What It Does

PhoebeBot runs continuously in my home as a voice-activated AI assistant. It controls my lights and media, monitors the stock market, executes paper trades autonomously, transcribes and responds to voice commands, generates and runs its own Python scripts with my approval, and restarts itself if anything goes wrong.

It is not a cloud service. Every model, every process, every decision runs locally on hardware I own.

---

## Architecture

Five nodes communicate over an authenticated MQTT broker with QoS guarantees, retained heartbeats, and exponential backoff reconnect on every node.

**Atlas — Jetson Orin Nano** — The brain. Always on. Runs the MQTT broker, hosts two models via Ollama (Qwen3:8b for routing and conversation, qwen2.5-coder:1.5b for code generation — GPU layers tuned by trial to coexist within 8GB unified memory), handles Whisper STT, manages SQLite persistent memory and episodic memory, loads FinBERT for sentiment injection into chat, and controls Home Assistant via REST API. OOM priority tuned so the broker survives memory pressure — models die first, broker last.

**Luna — Raspberry Pi 5** — The voice. Thin client by design. Owns the USB mic stream permanently, decimates 48kHz to 16kHz for Whisper, detects a custom wake word, forwards audio to Atlas via MQTT, and speaks responses back via piper TTS. Degrades gracefully if Atlas is unreachable.

**Orion — Raspberry Pi 5** — The hunter. Financial node. Crawls news, scores headlines with FinBERT sentiment analysis, monitors watchlists, and places autonomous paper trades on Alpaca with structured JSONL decision logging. Pre-flight asset validation prevents wasted orders on untradable tickers.

**Banshee — Raspberry Pi Zero 2W** — The watchdog. Monitors Atlas via ICMP ping — not MQTT, because the broker lives on the node being watched. Two-stage recovery: SSH graceful restart first, GPIO relay hard cut after 60 seconds. Sends an external alert on every restart event so silent failures don't go unnoticed.

**PC — Windows, i9 + RTX 4090** — Opportunistic GPU donor. Publishes its availability state to MQTT every 30 seconds — FREE or GAMING. Atlas routes heavy compute jobs to the 4090 when it's available and falls back to local inference when it isn't. Starts automatically at login, exits cleanly on shutdown with LWT signaling.

---

## Key Technical Details

- **Local LLM inference** — Qwen3:8b (routing + conversation) and qwen2.5-coder:1.5b (code generation) on Jetson Orin Nano via Ollama. GPU layer allocation tuned by trial to stay within shared memory limits. Resolved a 26x latency regression caused by a silently-ignored API option that left chain-of-thought reasoning enabled on every response.

- **Custom wake word pipeline** — The official openWakeWord training pipeline required Google Colab. I wrote my own: `record_wake_word.py` captures positive samples through the local mic, `train_hey_phoebe.py` handles feature extraction, embedding generation, incremental negative caching, and ONNX export. Fully offline. No cloud dependency.

- **Self-scripting with voice approval** — Atlas generates Python scripts via a dedicated code model, publishes a preview over MQTT, and waits for voice approval from Luna before executing. Luna speaks the description, listens for approve or reject, and publishes the result. Scripts run in a sandboxed environment with AST static analysis blocking dangerous imports and calls. Silence defaults to reject.

- **Voice pipeline** — USB mic → 48kHz→16kHz decimation → openWakeWord detection → Whisper STT → MQTT to Atlas → Qwen3 routing and response → MQTT back to Luna → piper TTS spoken reply. The wake loop owns the mic stream permanently; `listen_from_mic()` reads from a shared queue to avoid device conflicts.

- **Episodic memory** — After each conversation, Qwen tags the exchange with an emotional tone and a one-sentence summary, saved to SQLite. The five most recent episodes are injected into the system prompt on every subsequent conversation so Phoebe retains context across sessions.

- **Fault tolerance** — systemd `Restart=always` on every node. Banshee two-stage hardware recovery. Luna degraded mode with retry backoff. Atlas journald persistence for post-crash log capture. PC LWT dead-man signaling.

- **FinBERT sentiment** — Runs on both Orion and Atlas (PyTorch CPU ARM64 on both). Orion uses it to score financial headlines before making trade decisions. Atlas loads it independently to inject market sentiment into Qwen's chat system prompt, contextualizing responses when financial topics come up in conversation.

---

## Stack

Python — MQTT (Mosquitto + paho-mqtt) — Ollama — Qwen3 — Whisper — piper — openWakeWord — FinBERT — PyTorch — SQLite — Home Assistant — Alpaca — systemd — Linux (Raspberry Pi OS + Ubuntu) — Windows

---

## Status

Actively in development. Core systems are live and running. Paper trading active. Self-scripting loop with voice approval confirmed working end-to-end. Episodic memory live. Wake word tuning in progress. RTL-SDR integration, morning briefing, and image generation on the near-term roadmap.

Demo video in progress.

---

## The Name

Phoebe and Banshee were my dogs. I lost them two weeks apart in December 2025. Banshee spent Phoebe's final weeks watching over her. When she was gone, she developed a heart condition and had to be put down shortly after. She died of a broken heart.

Phoebe is the system. Banshee watches over her. It felt right.

---

*Built with Claude and Grok as development partners. All architecture decisions, debugging, and system design directed by me.*
