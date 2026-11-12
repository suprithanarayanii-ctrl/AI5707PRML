import re
import subprocess
from pathlib import Path
from collections import defaultdict

import numpy as np
import librosa
import soundfile as sf

# Filename -> digit label
DIGIT_MAP = {"z": "zero", "o": "oh", "1": "one", "2": "two", "3": "three",
             "4": "four", "5": "five", "6": "six", "7": "seven",
             "8": "eight", "9": "nine"}

# Single digit + production letter only, e.g. "1a.wav", "zb.wav"
SINGLE_DIGIT_RE = re.compile(r"^([zo1-9])([ab])\.wav$", re.IGNORECASE)


def load_audio(path, target_sr=None):
    """Read a TIDIGITS file (NIST SPHERE). Falls back to sph2pipe if needed."""
    try:
        y, sr = sf.read(str(path), dtype="float32")
    except Exception:
        # Shorten-compressed SPHERE: needs sph2pipe installed and on PATH
        raw = subprocess.run(["sph2pipe", "-f", "wav", str(path)],
                             capture_output=True, check=True).stdout
        import io
        y, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if target_sr and sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return y, sr


def extract_mfcc(y, sr, n_mfcc=13, win_ms=25, hop_ms=10, n_mels=26,
                 use_deltas=True, cmvn=True, preemph=0.97):
    """Return (n_frames, n_features) MFCC matrix."""
    y = librosa.effects.preemphasis(y, coef=preemph)
    win = int(sr * win_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    n_fft = 1 << (win - 1).bit_length()          # next power of two

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft,
                                win_length=win, hop_length=hop,
                                n_mels=n_mels, window="hamming")
    feats = [mfcc]
    if use_deltas:
        feats.append(librosa.feature.delta(mfcc, order=1))
        feats.append(librosa.feature.delta(mfcc, order=2))
    feats = np.vstack(feats).T                    # (frames, features)

    if cmvn:                                      # per-utterance mean/var norm
        feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-8)
    return feats.astype(np.float32)


DIGIT_MAP = {"z": "0", "1": "1", "2": "2", "3": "3", "4": "4",
             "5": "5", "6": "6", "7": "7", "8": "8", "9": "9"}


def process_corpus(root, out_dir=".", **kwargs):
    """Save MFCCs for single-digit utterances as <out_dir>/<digit>.npz"""
    root, out_dir = Path(root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_digit = defaultdict(list)

    for path in sorted(root.rglob("*.wav")):
        m = SINGLE_DIGIT_RE.match(path.name)
        if not m:
            continue                      # skip multi-digit strings

        code = m.group(1).lower()
        if code not in DIGIT_MAP:
            continue                      # skip "oh"

        y, sr = load_audio(path)
        by_digit[DIGIT_MAP[code]].append(extract_mfcc(y, sr, **kwargs))

    for digit, mfccs in sorted(by_digit.items()):
        # saved as arr_0, arr_1, ... (one (frames, features) array per utterance)
        np.savez_compressed(out_dir / f"{digit}.npz", *mfccs)
        print(f"{digit}.npz: {len(mfccs)} utterances")



if __name__ == "__main__":
    records = process_corpus("Data/tidigits/", out_dir="Data/ExtractedData")
    print(records[0]["label"], records[0]["feats"].shape)