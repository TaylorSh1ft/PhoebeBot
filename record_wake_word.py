"""
record_wake_word.py
Records voice samples for openWakeWord custom wake word training.
Run this on the PC with your good mic (SteelSeries Alias) plugged in.

Output: wake_word_samples/hey_phoebe/hey_phoebe_001.wav ... hey_phoebe_015.wav
Format: 16kHz, 16-bit, mono WAV — exactly what openWakeWord expects.

After recording, upload the folder to the openWakeWord Colab training notebook:
https://github.com/dscripka/openWakeWord  →  notebooks/training_models.ipynb
Download the resulting hey_phoebe.onnx and place it next to phoebe_luna.py on Luna.
"""

import os
import sys
import wave
import pyaudio
import argparse
import numpy as np

# ── Config ────────────────────────────────────────────────────
PHRASE        = "hey_phoebe"
NUM_SAMPLES   = 25
RECORD_SECS   = 2       # 2s per clip — plenty for a 2-word phrase
MIC_RATE      = 48000   # Fifine K669 native rate
TARGET_RATE   = 16000   # openWakeWord expects 16kHz
DECIMATE      = MIC_RATE // TARGET_RATE  # 3
CHANNELS      = 1
CHUNK         = 1024
FORMAT        = pyaudio.paInt16

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "wake_word_samples", PHRASE)

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=None, help="Input device index (skip picker)")
args = parser.parse_args()

# ── List available mics so user can pick the right one ────────
pa = pyaudio.PyAudio()

if args.device is not None:
    device_index = args.device
    print(f"Using device {device_index}: {pa.get_device_info_by_index(device_index)['name']}")
else:
    print("\nAvailable input devices:")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            print(f"  [{i}] {info['name']}")

    device_index = None
    choice = input("\nEnter device index (or press Enter to use system default): ").strip()
    if choice.isdigit():
        device_index = int(choice)
        print(f"Using: {pa.get_device_info_by_index(device_index)['name']}")
    else:
        print("Using system default input device.")

# ── Record ────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

print(f"\nRecording {NUM_SAMPLES} samples of '{PHRASE.replace('_', ' ')}'")
print(f"Each clip is {RECORD_SECS} seconds.")
print(f"Say the phrase naturally — the way you'd actually say it to Phoebe.")
print(f"Vary your pace and tone slightly across samples. Don't be robotic.")
print(f"Saving to: {OUT_DIR}\n")

for i in range(1, NUM_SAMPLES + 1):
    input(f"[{i:02d}/{NUM_SAMPLES}] Press Enter, then say 'hey Phoebe' clearly...")

    kwargs = {"input_device_index": device_index} if device_index is not None else {}
    stream = pa.open(format=FORMAT, channels=CHANNELS, rate=MIC_RATE,
                     input=True, frames_per_buffer=CHUNK, **kwargs)

    print("  Recording... ", end="", flush=True)
    frames = []
    for _ in range(int(MIC_RATE / CHUNK * RECORD_SECS)):
        frames.append(stream.read(CHUNK, exception_on_overflow=False))
    stream.stop_stream()
    stream.close()
    print("Done.")

    # Decimate 48kHz → 16kHz
    audio = np.frombuffer(b"".join(frames), dtype=np.int16)
    audio = audio[::DECIMATE].astype(np.int16)

    fname = os.path.join(OUT_DIR, f"{PHRASE}_{i:03d}.wav")
    with wave.open(fname, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pa.get_sample_size(FORMAT))
        wf.setframerate(TARGET_RATE)
        wf.writeframes(audio.tobytes())
    print(f"  Saved: {os.path.basename(fname)}")

pa.terminate()

print(f"\nAll {NUM_SAMPLES} samples saved to:")
print(f"  {OUT_DIR}")
print("\nNext steps:")
print("  1. Open the openWakeWord training notebook in Google Colab")
print("     https://github.com/dscripka/openWakeWord")
print("     notebooks/training_models.ipynb → 'Open in Colab'")
print("  2. Upload your wake_word_samples/hey_phoebe/ folder when prompted")
print("  3. Download the resulting hey_phoebe.onnx")
print("  4. Copy hey_phoebe.onnx to the same directory as phoebe_luna.py on Luna")
