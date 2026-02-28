"""
record_negatives.py
Records negative audio samples for wake word retraining.
Run on the PC with your SteelSeries Alias mic.

These teach the model what to IGNORE:
  - TV speech / dialogue
  - Doorbell sounds
  - Phone ringing / notification sounds
  - General household speech (anything that's NOT "hey Phoebe")

Output: wake_word_samples/negatives/neg_CATEGORY_001.wav ...
Format: 16kHz, 16-bit, mono WAV
"""

import os
import wave
import pyaudio

SAMPLE_RATE  = 16000
CHANNELS     = 1
CHUNK        = 1024
FORMAT       = pyaudio.paInt16
RECORD_SECS  = 3   # 3s clips for negatives — slightly longer than positives

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, "wake_word_samples", "negatives")
os.makedirs(OUT_DIR, exist_ok=True)

CATEGORIES = [
    ("tv",       "TV audio — point mic at TV while dialogue or commercials play"),
    ("doorbell", "Doorbell sounds — ring your actual doorbell or play a YouTube clip"),
    ("phone",    "Phone sounds — ringtones, notification sounds, phone speech"),
    ("speech",   "General speech — talk normally about anything EXCEPT 'hey Phoebe'"),
]

pa = pyaudio.PyAudio()

print("\nAvailable input devices:")
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    if info["maxInputChannels"] > 0:
        print(f"  [{i}] {info['name']}")

choice = input("\nEnter device index (or Enter for system default): ").strip()
device_index = int(choice) if choice.isdigit() else None
if device_index is not None:
    print(f"Using: {pa.get_device_info_by_index(device_index)['name']}")
else:
    print("Using system default.")


def _record_category(cat_name, description, n_clips):
    print(f"\n{'='*55}")
    print(f"Category: {cat_name.upper()}")
    print(f"  {description}")
    print(f"  Recording {n_clips} clips of {RECORD_SECS}s each.")
    print(f"{'='*55}")

    # Find next available number for this category
    existing = [f for f in os.listdir(OUT_DIR)
                if f.startswith(f"neg_{cat_name}_") and f.endswith(".wav")]
    start_idx = len(existing) + 1

    for i in range(start_idx, start_idx + n_clips):
        input(f"\n  [{i:02d}/{start_idx + n_clips - 1}] Press Enter to record...")

        kwargs = {"input_device_index": device_index} if device_index is not None else {}
        stream = pa.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE,
                         input=True, frames_per_buffer=CHUNK, **kwargs)

        print("  Recording... ", end="", flush=True)
        frames = []
        for _ in range(int(SAMPLE_RATE / CHUNK * RECORD_SECS)):
            frames.append(stream.read(CHUNK, exception_on_overflow=False))
        stream.stop_stream()
        stream.close()
        print("Done.")

        fname = os.path.join(OUT_DIR, f"neg_{cat_name}_{i:03d}.wav")
        with wave.open(fname, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(pa.get_sample_size(FORMAT))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(frames))
        print(f"  Saved: {os.path.basename(fname)}")


# ── How many clips per category ───────────────────────────────
# TV speech is the biggest false-trigger — give it the most weight.
CLIPS_PER_CATEGORY = {
    "tv":       15,
    "doorbell": 10,
    "phone":    10,
    "speech":   10,
}

print(f"\nThis will record {sum(CLIPS_PER_CATEGORY.values())} total negative clips.")
print("You can skip any category by pressing Ctrl+C and re-running for the next one.")
print("Tip: have the TV on, a doorbell video queued up, and be ready to talk.\n")

for cat_name, description in CATEGORIES:
    n = CLIPS_PER_CATEGORY[cat_name]
    try:
        _record_category(cat_name, description, n)
    except KeyboardInterrupt:
        print(f"\n  Skipped remaining {cat_name} clips.")
        continue

pa.terminate()

total = len([f for f in os.listdir(OUT_DIR) if f.endswith(".wav")])
print(f"\nDone. {total} negative clips saved to:")
print(f"  {OUT_DIR}")
print("\nNext: run train_hey_phoebe.py")
