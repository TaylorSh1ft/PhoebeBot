# PhoebeBot Roadmap

Set 2026-02-24. Updated 2026-02-25. Claude owns this — remind user at each milestone.

---

## Week 1 (ends 2026-03-01) — COMPLETE EARLY

- [x] Wake word retrain — recorded 66 custom negatives, retrained hey_phoebe.onnx
- [x] Mic repositioning — 15ft USB-B cable ordered, arrives 2026-02-26. Energy gate in place.
- [x] Self-scripting foundation — MQTT preview + approval gate + sandboxed execution. Full loop confirmed.
- [x] Auto-backup — GitHub private repo live, `backup` alias, backup_push.ps1
- [x] systemd for Luna + Orion — both nodes auto-start and restart on crash
- [x] Atlas FinBERT — lazy load, _score_sentiment(), injected into chat system prompt
- [ ] Wake word retrain with mic in new position — pending cable arrival tomorrow

---

## Week 2 (ends 2026-03-08) — PULLING FORWARD

- [ ] **Wed 2026-02-26** — Mic repositioning + wake word retrain in new position. Tune energy gate. Test couch + desk coverage. Plug in Razer webcam.
- [ ] **Thu 2026-02-27** — Self-scripting v2: restricted sandbox dir, Luna voice approval wire-up (subscribe to script_preview, speak prompt, listen, publish approve/reject).
- [ ] **Fri 2026-02-28** — System health voice command ("how's Atlas doing?" → CPU, mem, disk, uptime spoken back via Luna).
- [ ] **Weekend** — Google Calendar sync: OAuth setup, token refresh, voice queries via Luna.
- [ ] **MILESTONE: Start job hunt parallel track** — Build GitHub profile README. PhoebeBot is the centerpiece. Don't wait for "done." ← REMIND USER

---

## Week 3 (ends 2026-03-15)

- [ ] **Mon–Tue** — RTL-SDR: ADS-B only first. One signal stable before adding ACARS/NOAA.
- [ ] **Wed–Thu** — Episodic memory: SQLite episodes table on Atlas, Qwen tags emotional tone, Mistral receives recent episodes in system prompt.
- [ ] **Fri** — Morning briefing: first voice activity of the day triggers 60s rundown (weather, Orion alerts, Kalshi shifts, calendar, episodic note).

---

## Week 4 (ends 2026-03-22)

- [ ] **Mon–Tue** — Image generation: test Atlas memory headroom first. Route to PC 4090 if needed.
- [ ] **Wed** — Demo video: 5 min, natural, don't over-edit. ← REMIND USER WHEN HERE
- [ ] **Thu** — GitHub repo public: clean README, push all files. ← REMIND USER WHEN HERE
- [ ] **Fri** — Share it: parents, a friend. Low stakes first.
- [ ] **MILESTONE: Actively applying to jobs** — Jr. Systems Engineer, IoT Engineer, Jr. DevOps, Home Automation Integrator, Technical Support Engineer. PhoebeBot is the portfolio. ← REMIND USER

---

## Week 5+ (after job hunt is active)

- [ ] **Confidence ratings on financial answers** — four-signal alignment score stated on every market response. Orion already computes it, just needs surfacing.
- [ ] **Behavioral pattern learning** — passive logging ~2 weeks → Phoebe anticipates patterns. Never intrusive, always overridable.
- [ ] **Self-maintaining Phoebe** — routine updates via voice approval, not coding sessions.
- [ ] **Self-debugging Phoebe** — catches exceptions, diagnoses via Qwen, auto-fixes or surfaces patch for approval. Three-layer recovery: self-heal → Banshee SSH → Banshee relay.
- [ ] **Atlas Nightshift Mode** — lightweight queue when PC offline. Heavy jobs queued for morning.
- [ ] **PC morning GPU donor pipeline** — image gen jobs flush to 4090 on connect each morning.
- [ ] **Image generation + Etsy shop** — SDXL on 4090, 3-5 images daily. AI-generated disclosure required.
- [ ] **X/Twitter promotional account** — Phoebe promotes Etsy listings. Check API tier pricing first. ← REMIND USER
- [ ] **Phoebe Discord bot** — public Python tutor. Mistral chat, structured lessons, Stripe billing.

---

## On the shelf (no timeline)

- DoorDash — no public API, needs creative approach
- Car/medical/law RAG
- Elden Ring assistant
- Amazon Product Advertising API

---

## Trading Graduation Track (parallel, time-gated)

- [x] **Phase 1 (now)** — Orion paper trades autonomously on Alpaca. Logging everything.
- [ ] **Phase 2 (~2 months paper performance)** — migrate to real Robinhood equities. Voice confirmation required. User retains full veto. ← REMIND USER when approaching
- [ ] **Phase 3 (after proven Phase 2)** — options only. Four-signal alignment mandatory. Explicit voice approval every single time. No autonomous options execution ever.
