import hashlib
from collections import defaultdict
from typing import Dict, List, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np


Peak = Tuple[int, int]  # (time_bin, freq_bin)
HashAtTime = Tuple[str, int]  # (hash, anchor_time_bin)

DEFAULT_PIPELINE = {
    "sr": 11025,
    "n_fft": 2048,
    "hop_length": 512,
    "top_k_per_frame": 8,
    "min_db": -55.0,
    "fan_value": 12,
    "min_delta_t": 1,
    "max_delta_t": 80,
}


def normalized_pipeline(pipeline: dict | None = None) -> dict:
    cfg = dict(DEFAULT_PIPELINE)
    if pipeline:
        cfg.update(pipeline)
    return cfg


def load_audio(audio_path: str, sr: int = 11025) -> np.ndarray:
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    return y


def compute_log_spectrogram(y: np.ndarray, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    stft = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(stft)
    return librosa.amplitude_to_db(mag, ref=np.max)


def pick_peaks(spec_db: np.ndarray, top_k_per_frame: int = 8, min_db: float = -55.0) -> List[Peak]:
    peaks: List[Peak] = []
    _, n_times = spec_db.shape

    for t in range(n_times):
        frame = spec_db[:, t]
        idx = np.argpartition(frame, -top_k_per_frame)[-top_k_per_frame:]
        idx = idx[np.argsort(frame[idx])[::-1]]

        for f in idx:
            if frame[f] >= min_db:
                peaks.append((t, int(f)))

    return peaks


def _hash_pair(f1: int, f2: int, delta_t: int, digest_len: int = 20) -> str:
    payload = f"{f1}|{f2}|{delta_t}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:digest_len]


def build_hashes(
    peaks: List[Peak],
    fan_value: int = 12,
    min_delta_t: int = 1,
    max_delta_t: int = 80,
) -> List[HashAtTime]:
    hashes: List[HashAtTime] = []

    for i, (t1, f1) in enumerate(peaks):
        for j in range(1, fan_value + 1):
            k = i + j
            if k >= len(peaks):
                break

            t2, f2 = peaks[k]
            delta_t = t2 - t1
            if delta_t < min_delta_t or delta_t > max_delta_t:
                continue

            h = _hash_pair(f1, f2, delta_t)
            hashes.append((h, t1))

    return hashes


def analyze_signal(y: np.ndarray, pipeline: dict | None = None):
    cfg = normalized_pipeline(pipeline)
    if y.size == 0:
        return np.empty((0, 0), dtype=np.float32), []

    spec_db = compute_log_spectrogram(y, n_fft=cfg["n_fft"], hop_length=cfg["hop_length"])
    peaks = pick_peaks(spec_db, top_k_per_frame=cfg["top_k_per_frame"], min_db=cfg["min_db"])
    return spec_db, peaks


def fingerprint_signal(y: np.ndarray, pipeline: dict | None = None) -> List[HashAtTime]:
    cfg = normalized_pipeline(pipeline)
    if y.size == 0:
        return []

    spec_db, peaks = analyze_signal(y, cfg)
    return build_hashes(
        peaks,
        fan_value=cfg["fan_value"],
        min_delta_t=cfg["min_delta_t"],
        max_delta_t=cfg["max_delta_t"],
    )


def fingerprint_audio(audio_path: str, pipeline: dict | None = None) -> List[HashAtTime]:
    cfg = normalized_pipeline(pipeline)
    y = load_audio(audio_path, sr=cfg["sr"])
    return fingerprint_signal(y, cfg)


def analyze_audio(audio_path: str, pipeline: dict | None = None):
    cfg = normalized_pipeline(pipeline)
    y = load_audio(audio_path, sr=cfg["sr"])
    spec_db, peaks = analyze_signal(y, cfg)
    return y, spec_db, peaks, cfg


def plot_audio_analysis(
    y: np.ndarray,
    spec_db: np.ndarray,
    peaks: List[Peak],
    pipeline: dict,
    title: str,
    out_path: str | None = None,
) -> None:
    cfg = normalized_pipeline(pipeline)
    time_axis = np.arange(y.size) / cfg["sr"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)
    fig.suptitle(title)

    axes[0].plot(time_axis, y, linewidth=0.8)
    axes[0].set_title("Waveform")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")

    img = axes[1].imshow(spec_db, origin="lower", aspect="auto", cmap="magma")
    axes[1].set_title("Spectrogram (dB)")
    axes[1].set_xlabel("Time frame")
    axes[1].set_ylabel("Frequency bin")
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")

    axes[2].imshow(spec_db, origin="lower", aspect="auto", cmap="magma")
    if peaks:
        t = [p[0] for p in peaks]
        f = [p[1] for p in peaks]
        axes[2].scatter(t, f, s=6, c="cyan", alpha=0.7)
    axes[2].set_title("Detected Peaks")
    axes[2].set_xlabel("Time frame")
    axes[2].set_ylabel("Frequency bin")

    if out_path:
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot: {out_path}")
    else:
        plt.show()

    plt.close(fig)


def build_inverted_index(song_hashes: Dict[int, List[HashAtTime]]) -> Dict[str, List[Tuple[int, int]]]:
    index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for song_id, hashes in song_hashes.items():
        for h, t in hashes:
            index[h].append((song_id, t))
    return dict(index)
