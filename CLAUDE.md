# PhoebeBot Rules

## Threading Rule
Every feature — old or new — runs in its own thread.
No main-thread blocking. No stutter. No exceptions.

- Ollama calls: always in a background thread
- TTS (speak_text_sync): always in a background thread
- File I/O (save_memory, _save_user_data): always in a background thread
- Network I/O (requests.get/post): always in a background thread
- If a routing decision needs Ollama, show "..." immediately, decide in a thread, dispatch the handler on main thread via root.after(0, ...)

## Dual Brain Rule — Both on Atlas Jetson
Two models run via Ollama on Atlas. Auto-switch based on task.
- **Qwen (qwen3:8b)** = OLLAMA_MODEL — code, stocks, logic, classification, routing, fact extraction.
  `_QWEN_OPTS = {"num_gpu": 6}` — 6 layers on Atlas Jetson (confirmed stable, 572MB post-warmup).
  Used for: _route_financial, _resolve_clarification (classification), _resolve_price_conflict,
  _extract_tickers, _fetch_trade_idea, _extract_news_ideas, _evaluate_ceo_news, _evaluate_portfolio_news,
  _kalshi_cross_reference, handle_trade_followup, _is_trade_followup, _clarify_mishear,
  _extract_user_facts, _ollama_quick, handle_name_store (name extraction).
- **Mistral (mistral:7b)** = OLLAMA_CHAT_MODEL — chat, personal conversation, emotional responses.
  `_MISTRAL_OPTS = {"num_gpu": 6}` — 6 layers on Atlas Jetson (8 = cudaMalloc OOM, 6 = stable).
  Used for: handle_chat, _resolve_clarification (PERSONAL response), handle_name_store (greeting),
  birthday greeting.
- No throttling. No heat monitoring. No CPU affinity caps. No GPU gate. Clean execution.
- `_ensure_model` pulls both models at startup if not present.
- `/no_think` is a Qwen directive — do NOT send it to Mistral.

## Voice Recall Rule — Rolling 30-Second Buffer, Raw Playback
A rolling audio buffer captures the last 30 seconds of mic input at all times.
- `_voice_buffer`: deque of PCM frames from both wake word loop and listen_from_mic.
- Hint regex `_RECALL_HINT_RE` catches natural phrases: "what was that?", "repeat?", "play back", etc.
- Ollama (Qwen) confirms intent — classifies REPLAY vs CHAT. No fixed command.
- If REPLAY: `_save_voice_buffer_wav()` flushes buffer to WAV, `_play_voice_recall()` plays it via winsound.
- Mic is muted during playback to avoid feedback. Unmuted after.
- Phoebe says "Playing back the last N seconds." before playing the raw audio.
- WAV file is cleaned up after playback.
- If not a replay request, falls through to normal chat. No interruption.

## Mood Tracker Rule — VAD + Volume, No Model
Simple energy-based mood detection from voice input. No ML model. Just numbers.
- `listen_from_mic()` computes average and peak energy across speech frames.
- Thresholds: avg > 1200 = LOUD, avg < 400 = QUIET, else NORMAL.
- Stored in `_user_mood = {"energy_avg", "energy_peak", "level"}`.
- Injected into Mistral (chat) system prompt via `_build_ollama_messages`:
  - LOUD → "Soften your tone. Be calm, grounding, gentle."
  - QUIET → "Match their softness. Be gentle, unhurried."
  - NORMAL → no modifier.
- Also injected into PERSONAL branch of `_resolve_clarification`.
- Resets every time the user speaks via voice. Typed input keeps previous mood.

## Sentiment Rule
All headlines and news are scored with FinBERT (ProsusAI/finbert) before reaching Ollama.
- Scores are real financial sentiment — positive, negative, neutral with confidence
- Trade ideas, news scanning, and CEO news all receive scored headlines
- Ollama is told to trust the scores — they reflect trained financial mood, not keyword matching
- FinBERT loads lazily in a background thread at startup; falls back to raw headlines if not ready

## Price Verification Rule — Dual Source, No Blind Trust
All trade ideas are cross-checked against real market data from BOTH Robinhood and Yahoo Finance.
- _get_price_change always fetches from both _get_price_rh AND _get_price_yf.
- If prices agree (within 0.5%), Robinhood is used as default. Source is tagged.
- If prices diverge >0.5%, Phoebe ASKS the user which source to trust. No quiet fallback.
- _pending_price_conflicts queues conflicts; _check_news_ideas surfaces them with top priority.
- _resolve_price_conflict uses Ollama to read the user's answer (ROBINHOOD, YAHOO, NEITHER).
- User preference is stored in _price_source_pref per ticker — remembered for future lookups.
- _build_price_check shows CONFLICTED status with both prices when unresolved.
- Unusual volume (>2x avg) = confirmed move. Normal volume = noise. Low volume = ignore.
- Fresh headlines (<1hr) are actionable. Stale headlines (>6hr) are likely priced in.
- NEVER call it a "dip" unless price data confirms the drop. Negative sentiment alone is fear, not a dip.
- Four signals must align: sentiment (mood) + price (reality) + volume (conviction) + Kalshi (crowd bet)

## Routing Rule — No Hardcoded Assumptions
Ollama decides context for all financial-adjacent inputs. Nothing is hardcoded to a handler.
- Words like "money", "stock", "portfolio" are AMBIGUOUS — Phoebe asks before acting.
- _route_financial sends the input + context to Ollama, which classifies:
  PORTFOLIO, TRADE_IDEA, NEWS, WATCHLIST, FOLLOWUP, MISHEAR, AMBIGUOUS, HOME_CONTROL, or CHAT
- AMBIGUOUS → Phoebe asks a clarifying question (e.g. "Do you mean your cash, or the market?")
- Only after user confirms → Phoebe acts. No assumptions. No auto-dump.
- _FINANCIAL_HINT_RE is just a speed hint — it does NOT decide the action. Ollama does.
- _HOME_CONTROL_RE + _parse_ha_intent() are fast-path detectors — skip LLM for obvious HA commands.
- Trade follow-ups, mishears, and all financial routing go through the same unified router.

## Clarification Flow Rule — Read the Flow, Not the Words
When Phoebe asks for clarification, _resolve_clarification reads the EMOTIONAL FLOW of the exchange.
- If the user stays personal ("bills are killing me" after "money trouble"), that's PERSONAL.
  Phoebe stays with them — talks money, not stocks. Full context preserved.
- If the user confirms financial ("the market", "my portfolio"), dispatch to the right handler.
- PERSONAL responses are generated in-thread with the full conversation arc:
  original message → clarification question → user's answer, all injected into Ollama context.
- Memory saves the full arc so the thread is never lost.
- Ollama classifies as: PORTFOLIO, TRADE_IDEA, NEWS, WATCHLIST, or PERSONAL.
- Default on error is PERSONAL — when in doubt, stay with the human.

## Kalshi Rule
Kalshi prediction markets are pulled every 5 minutes via _bg_kalshi_loop.
- Auth: RSA-PSS signing. Private key loaded from `kalshi_private.key` (PEM). NEVER log it. NEVER print it.
- KALSHI_ACCESS_KEY (short ID) is in .env. Also NEVER log or print.
- _kalshi_sign builds: timestamp_ms + METHOD + path → RSA-PSS SHA256 → base64 signature.
- Headers: KALSHI-ACCESS-KEY, KALSHI-ACCESS-TIMESTAMP, KALSHI-ACCESS-SIGNATURE.
- Falls back to public (unsigned) requests if key file is missing.
- _kalshi_fetch_markets pulls /markets endpoint, filters for economic relevance + high volume.
- _kalshi_detect_shifts compares snapshots — any >5 pct point move gets flagged.
- _kalshi_cross_reference cross-checks the shift against FinBERT sentiment, real prices, volume, headlines.
- Ollama generates a fact-based observation — no advice, no "buy", no "sell". Just the fact and whether the data agrees.
- If prediction odds and real data disagree: "Might be noise. Might be early."
- Kalshi data is also injected into trade idea prompts as _format_kalshi_context.
- Four signals: sentiment (mood) + price (reality) + volume (conviction) + Kalshi (crowd bet).

## File Rule
The original nul was renamed to _ul (Windows reserves "nul" as a device name).
- Do NOT create nul — ever. Windows will mangle it.
- All code goes into _ul. No duplicates.
- If you need to reference or modify this file, it is _ul.

---

## Architecture Quick Reference

**Nodes**
- Atlas (Jetson Orin Nano): MQTT broker, Ollama, Whisper, SQLite, HA control
- Orion (Pi 5): finance, news, FinBERT, Alpaca paper trades
- Luna (Pi 5): voice (wake word + mic + TTS), thin MQTT client
- Banshee (Pi Zero): watchdog, two-stage kill (SSH → GPIO relay)
- HA Pi (Pi 5): HassOS only, not a PhoebeBot node
- PC (Windows): presence node, GPU donor when free

Node IPs are set via environment variables (see .env.example). Never hardcode them.

**MQTT**: broker on Atlas, port 1883, auth required, clean_session=False, QoS 1, exponential backoff.
Reconnect always in a Thread — never call reconnect() from MQTT callback thread directly.

**Current pending**
- Wake word: test normal TV false-trigger rate. If bad → 50+ diverse speech negatives, retrain.
- Atlas FinBERT: install PyTorch, wire into phoebe_atlas.py (Week 3 on roadmap).
- systemd for Luna and Orion.
- Housekeeping: delete phoebe-neural.py + phoebe-good.py, retire banshee_watchdog.py v1.

See `ARCHITECTURE.md` for full node details and HA entities.
See `DEVLOG.md` for full history.
See `ROADMAP.md` for week-by-week plan and milestones.

---

## Backup Rule
After every completed milestone or major session, remind the user to run `backup`.
- `backup` works from any terminal (PowerShell or Git Bash) — runs backup_push.ps1, commits changes, pushes to GitHub
- Remind at: end of roadmap milestones, after major features land, before long breaks
- Phrase: "Don't forget to run `backup` before you close up."

## Development Log Rule

Claude keeps a running log of all features built, migrations, and decisions.
- Every session that touches code or architecture gets a dated entry in `DEVLOG.md`
- Entries are short: what was done, what changed, what is next
- No essays. Just facts and state.
- When a pending item is completed, mark it done and remove from pending list
