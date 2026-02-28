"""
train_hey_phoebe.py
Trains a custom "hey Phoebe" wake word model using openWakeWord.
Run after record_wake_word.py has captured your samples.

Steps:
  1. Downloads negative audio datasets (music + sound effects)
  2. Embeds all audio using openWakeWord's shared feature extractor
  3. Trains a small neural net classifier on top
  4. Exports hey_phoebe.onnx — drop this next to phoebe_luna.py on Luna

Usage:
  C:/PhoebeLocal/PhoebeBot/record_env/Scripts/python train_hey_phoebe.py
"""

import os
import sys
import wave
import random
import collections
import urllib.request
import zipfile
import numpy as np
import scipy.io.wavfile
import torch
from torch import nn
from tqdm import tqdm

# ── Compatibility patch (must be before any speechbrain/openwakeword import) ──
# speechbrain 1.x calls torchaudio.list_audio_backends() removed in torchaudio 2.x
import torchaudio as _ta
if not hasattr(_ta, "list_audio_backends"):
    _ta.list_audio_backends = lambda: ["soundfile"]

import openwakeword.utils

# ── Config ────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
POSITIVE_DIR = os.path.join(SCRIPT_DIR, "wake_word_samples", "hey_phoebe")
TRAIN_DIR    = os.path.join(SCRIPT_DIR, "wake_word_training")
OUTPUT_MODEL = os.path.join(SCRIPT_DIR, "hey_phoebe.onnx")

NEGATIVE_URLS = [
    ("fma_sample",    "https://f002.backblazeb2.com/file/openwakeword-resources/data/fma_sample.zip"),
    ("fsd50k_sample", "https://f002.backblazeb2.com/file/openwakeword-resources/data/fsd50k_sample.zip"),
]

# Custom negatives recorded by user (TV, doorbell, phone, speech).
# These are included alongside the downloaded datasets.
CUSTOM_NEG_DIR = os.path.join(SCRIPT_DIR, "wake_word_samples", "negatives")

N_AUGMENTATIONS = 20    # repeat each positive clip N times with different backgrounds
CLIP_SECS       = 3     # seconds per training window
SR              = 16000
CLIP_FRAMES     = SR * CLIP_SECS
N_EPOCHS        = 30
BATCH_SIZE      = 512
LR              = 0.001
LAYER_DIM       = 32

os.makedirs(TRAIN_DIR, exist_ok=True)

# ── Verify positive samples ───────────────────────────────────
if not os.path.isdir(POSITIVE_DIR):
    print(f"ERROR: {POSITIVE_DIR} not found. Run record_wake_word.py first.")
    sys.exit(1)
pos_wavs = [os.path.join(POSITIVE_DIR, f)
            for f in os.listdir(POSITIVE_DIR) if f.lower().endswith(".wav")]
if not pos_wavs:
    print(f"ERROR: No .wav files in {POSITIVE_DIR}")
    sys.exit(1)
print(f"\nFound {len(pos_wavs)} positive recordings.")

# ── Data helpers ──────────────────────────────────────────────
def _load_wav(path):
    """Load WAV as 16kHz mono int16 numpy array."""
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="int16", always_2d=False)
        if data.ndim > 1:
            data = data[:, 0]
        if sr != SR:
            from scipy.signal import resample
            data = resample(data.astype(np.float32),
                            int(len(data) * SR / sr)).astype(np.int16)
        return data
    except Exception:
        # Fallback to scipy
        try:
            rate, data = scipy.io.wavfile.read(path)
            if data.ndim > 1:
                data = data[:, 0]
            if data.dtype != np.int16:
                data = (data / np.abs(data).max() * 32767).astype(np.int16)
            if rate != SR:
                from scipy.signal import resample
                data = resample(data.astype(np.float32),
                                int(len(data) * SR / rate)).astype(np.int16)
            return data
        except Exception:
            return np.zeros(CLIP_FRAMES, dtype=np.int16)


def _wav_duration(path):
    """Return duration in seconds from WAV header (fast, no decode)."""
    try:
        with wave.open(path, "r") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0


def _find_wavs(dirs, min_secs=1.0, max_secs=3600):
    """Recursively find WAV files within duration range."""
    paths, durations = [], []
    for d in dirs:
        for root, _, files in os.walk(d):
            for f in files:
                if not f.lower().endswith(".wav"):
                    continue
                p = os.path.join(root, f)
                dur = _wav_duration(p)
                if min_secs <= dur <= max_secs:
                    paths.append(p)
                    durations.append(dur)
    return paths, durations


def _chunk_audio(data, chunk_frames):
    """Split audio into fixed-size chunks, dropping the last partial chunk."""
    n = len(data) // chunk_frames
    return [data[i * chunk_frames:(i + 1) * chunk_frames] for i in range(n)]


def _mix_fg_bg(fg, bg_path, start, snr_db, volume_gain):
    """Place fg in a CLIP_FRAMES window over bg audio, mixed at snr_db."""
    bg = _load_wav(bg_path).astype(np.float32)
    if len(bg) < CLIP_FRAMES:
        bg = np.tile(bg, CLIP_FRAMES // len(bg) + 1)
    bg_offset = random.randint(0, max(0, len(bg) - CLIP_FRAMES))
    window = bg[bg_offset:bg_offset + CLIP_FRAMES].copy()

    fg_f = fg.astype(np.float32)
    start = max(0, min(start, CLIP_FRAMES - len(fg_f)))
    end   = min(start + len(fg_f), CLIP_FRAMES)
    seg   = fg_f[:end - start]

    bg_seg = window[start:end]
    bg_rms = np.sqrt(np.mean(bg_seg ** 2)) if np.any(bg_seg) else 1.0
    fg_rms = np.sqrt(np.mean(seg ** 2)) if np.any(seg) else 1.0
    if fg_rms > 0 and bg_rms > 0:
        gain = bg_rms / (fg_rms * 10 ** (snr_db / 20.0))
        seg  = seg * gain

    window[start:end] += seg
    window *= volume_gain
    max_v = np.abs(window).max()
    if max_v > 0:
        window = window / max_v * 32767
    return window.astype(np.int16)

# ── Step 1: Download negative datasets ───────────────────────
print("\n[1/6] Downloading negative audio datasets...")
neg_dirs = []
for name, url in NEGATIVE_URLS:
    dest = os.path.join(TRAIN_DIR, name)
    if os.path.isdir(dest) and any(True for _ in os.scandir(dest)):
        print(f"  [skip] {name} already present.")
        neg_dirs.append(dest)
        continue
    zip_path = dest + ".zip"
    print(f"  Downloading {name}...")
    try:
        urllib.request.urlretrieve(url, zip_path)
        print(f"  Extracting {name}...")
        os.makedirs(dest, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
        os.remove(zip_path)
        neg_dirs.append(dest)
    except Exception as e:
        print(f"  WARNING: Could not get {name}: {e}")

_all_custom_wavs = [
    os.path.join(root, f)
    for root, _, files in os.walk(CUSTOM_NEG_DIR)
    for f in files if f.endswith(".wav")
] if os.path.isdir(CUSTOM_NEG_DIR) else []
if _all_custom_wavs:
    neg_dirs.append(CUSTOM_NEG_DIR)
    n_custom = len(_all_custom_wavs)
    print(f"  [+] Custom negatives: {n_custom} clips from {CUSTOM_NEG_DIR} (recursive)")
else:
    print("  [!] No custom negatives found. Run record_negatives.py for better results.")

if not neg_dirs:
    print("ERROR: No negative audio data. Check internet connection.")
    sys.exit(1)

# ── Step 2: Load feature extractor ────────────────────────────
print("\n[2/6] Loading openWakeWord audio feature extractor...")
F = openwakeword.utils.AudioFeatures()
emb_shape = F.get_embedding_shape(CLIP_SECS)   # (28, 96)
print(f"  Embedding shape per clip: {emb_shape}")

# ── Step 3: Negative embeddings ───────────────────────────────
neg_feat_path        = os.path.join(TRAIN_DIR, "negative_features.npy")
custom_feat_path     = os.path.join(TRAIN_DIR, "custom_negative_features.npy")
custom_manifest_path = os.path.join(TRAIN_DIR, "custom_negative_manifest.txt")

# Load already-embedded custom clips manifest (so we only embed NEW ones)
already_embedded = set()
if os.path.exists(custom_manifest_path):
    with open(custom_manifest_path) as f:
        already_embedded = set(line.strip() for line in f if line.strip())

if os.path.isdir(CUSTOM_NEG_DIR):
    new_custom_wavs = sorted([
        os.path.join(root, f)
        for root, _, files in os.walk(CUSTOM_NEG_DIR)
        for f in files
        if f.endswith(".wav") and os.path.join(root, f) not in already_embedded
    ])
else:
    new_custom_wavs = []

if new_custom_wavs:
    print(f"\n[3/6] Embedding {len(new_custom_wavs)} new custom negative clips...")
    new_feats = []
    for path in tqdm(new_custom_wavs, desc="  Embedding custom negatives"):
        audio  = _load_wav(path)
        chunks = _chunk_audio(audio, CLIP_FRAMES)
        if not chunks:
            continue
        batch = np.array(chunks, dtype=np.int16)
        embs  = F.embed_clips(batch, batch_size=64)
        new_feats.append(embs)
    if new_feats:
        new_arr = np.vstack(new_feats)
        # Merge with existing custom embeddings if present
        if os.path.exists(custom_feat_path):
            existing = np.load(custom_feat_path)
            new_arr  = np.vstack([existing, new_arr])
        np.save(custom_feat_path, new_arr)
        with open(custom_manifest_path, "a") as f:
            for p in new_custom_wavs:
                f.write(p + "\n")
        print(f"  Custom negative embeddings total: {len(new_arr)}")
else:
    print("\n[3/6] No new custom negatives to embed.")

if not os.path.exists(neg_feat_path):
    print("  Computing base negative embeddings (fma + fsd50k)...")
    base_dirs = [d for d in neg_dirs if d != CUSTOM_NEG_DIR]
    neg_paths, _ = _find_wavs(base_dirs, min_secs=CLIP_SECS)
    print(f"  {len(neg_paths)} base clips ≥{CLIP_SECS}s found")
    neg_features = []
    for path in tqdm(neg_paths, desc="  Embedding base negatives"):
        audio  = _load_wav(path)
        chunks = _chunk_audio(audio, CLIP_FRAMES)
        if not chunks:
            continue
        batch = np.array(chunks, dtype=np.int16)
        embs  = F.embed_clips(batch, batch_size=64)
        neg_features.append(embs)
    neg_arr = np.vstack(neg_features)
    np.save(neg_feat_path, neg_arr)
    print(f"  Saved {len(neg_arr)} base negative embeddings -> {neg_feat_path}")
else:
    print("  [skip] Base negative embeddings already cached.")

# Merge base + custom into final negatives for training
neg_arr = np.load(neg_feat_path)
if os.path.exists(custom_feat_path):
    custom_arr = np.load(custom_feat_path)
    neg_arr    = np.vstack([neg_arr, custom_arr])
    print(f"  Combined negatives: {len(neg_arr)} ({len(np.load(neg_feat_path))} base + {len(custom_arr)} custom)")

# ── Step 4: Positive embeddings (with augmentation) ───────────
pos_feat_path = os.path.join(TRAIN_DIR, "positive_features.npy")

if os.path.exists(pos_feat_path):
    print("\n[4/6] [skip] positive_features.npy already exists.")
else:
    print("\n[4/6] Computing positive embeddings (with augmentation)...")

    # Find all background paths for mixing
    bg_paths, _ = _find_wavs(neg_dirs, min_secs=1.0)
    print(f"  {len(pos_wavs)} recordings × {N_AUGMENTATIONS} augmentations"
          f" = {len(pos_wavs) * N_AUGMENTATIONS} positive examples")

    pos_features = []
    for wav_path in tqdm(pos_wavs, desc="  Augmenting positives"):
        fg = _load_wav(wav_path)
        dur_frames = len(fg)

        for _ in range(N_AUGMENTATIONS):
            # Place fg near end of window with random jitter
            jitter = int(random.uniform(0, 0.2) * SR)
            start  = max(0, CLIP_FRAMES - dur_frames - jitter)
            snr    = random.uniform(5, 15)
            vol    = random.uniform(0.7, 1.3)

            mixed = _mix_fg_bg(fg, random.choice(bg_paths), start, snr, vol)
            emb   = F.embed_clips(mixed[None], batch_size=1)
            pos_features.append(emb[0])

    pos_arr = np.array(pos_features)
    np.save(pos_feat_path, pos_arr)
    print(f"  Saved {len(pos_arr)} positive embeddings -> {pos_feat_path}")

# ── Step 5: Train ─────────────────────────────────────────────
print("\n[5/6] Training classifier...")

neg_feats = neg_arr
pos_feats = np.load(pos_feat_path)

print(f"  Negatives: {len(neg_feats)}   Positives: {len(pos_feats)}")

X = np.vstack((neg_feats, pos_feats)).astype(np.float32)
y = np.array([0] * len(neg_feats) + [1] * len(pos_feats),
             dtype=np.float32)[..., None]

loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
    batch_size=BATCH_SIZE, shuffle=True,
)

fcn = nn.Sequential(
    nn.Flatten(),
    nn.Linear(X.shape[1] * X.shape[2], LAYER_DIM),
    nn.LayerNorm(LAYER_DIM),
    nn.ReLU(),
    nn.Linear(LAYER_DIM, LAYER_DIM),
    nn.LayerNorm(LAYER_DIM),
    nn.ReLU(),
    nn.Linear(LAYER_DIM, 1),
    nn.Sigmoid(),
)

optimizer = torch.optim.Adam(fcn.parameters(), lr=LR)
loss_fn   = torch.nn.functional.binary_cross_entropy
history   = collections.defaultdict(list)

n_neg_total = len(neg_feats)
n_pos_total = len(pos_feats)
pos_weight  = n_neg_total / n_pos_total   # e.g. 3000/300 = 10 — boost positives

for epoch in tqdm(range(N_EPOCHS), desc="  Epochs"):
    for batch in loader:
        x_b, y_b = batch[0], batch[1]
        # Weight positives up to compensate for class imbalance
        weights = torch.ones(y_b.shape[0])
        weights[y_b.flatten() == 1] = pos_weight
        optimizer.zero_grad()
        preds = fcn(x_b)
        loss  = loss_fn(preds, y_b, weights[..., None])
        loss.backward()
        optimizer.step()
        history["loss"].append(float(loss.detach()))

print(f"  Final loss: {history['loss'][-1]:.4f}")

# ── Step 6: Export ONNX ───────────────────────────────────────
print("\n[6/6] Exporting to ONNX...")
fcn.eval()
dummy = torch.randn(1, X.shape[1], X.shape[2])
# Use legacy export path — avoids onnxscript requirement in torch 2.x
torch.onnx.export(
    fcn, dummy, OUTPUT_MODEL,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=12,
    dynamo=False,
)

print(f"\nDone! Model saved to:")
print(f"  {OUTPUT_MODEL}")
print("\nNext: copy hey_phoebe.onnx to the same directory as phoebe_luna.py on Luna.")
print("      pip install openwakeword on Luna, then start phoebe_luna.py.")
