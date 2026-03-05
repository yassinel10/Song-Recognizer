# Mini Shazam-Style Music Recognition (Python)

This project identifies a song from a short audio clip by comparing fingerprints against a local dataset.

## Core rule
Training songs and query audio always use the same feature-extraction pipeline:
- same sample rate (`sr`)
- same STFT (`n_fft`, `hop_length`)
- same peak picking (`top_k_per_frame`, `min_db`)
- same hash pairing (`fan_value`, `min_delta_t`, `max_delta_t`)

The pipeline is saved inside the DB file under `pipeline` and reused during recognition.

## Setup
```bash
python -m pip install -r requirements.txt
```

## Quick use with your reference songs
The script below uses the fixed list in `src/reference_match.py` (`REFERENCE_SONGS`).

Build/refresh DB + analyze a recorded clip:

```bash
python src/reference_match.py --query path_to_recorded_clip.wav --rebuild
```

Then normal use (no rebuild):

```bash
python src/reference_match.py --query path_to_recorded_clip.wav
```

Important:
- `reference_match.py` always requires `--query`.
- If you edit `REFERENCE_SONGS`, keep a comma after every entry in the list.

## Live microphone recognition (Shazam-style)
First run (rebuild DB):

```bash
python src/live_match.py --rebuild
```

Normal run:

```bash
python src/live_match.py
```

Recommended accurate + clean demo config:

```bash
python src/live_match.py --min-votes 22 --min-vote-ratio 0.022 --min-margin 5 --stable-hits 3 --history 5
```

Current live output is concise and showcase-friendly:

```text
the guessed song is: stereo_love
```

Useful live options:
- `--window` analysis window in seconds (default `8.0`)
- `--hop` refresh interval in seconds (default `1.5`)
- `--min-votes` minimum raw vote score
- `--min-vote-ratio` minimum `best_votes / query_hashes`
- `--min-margin` minimum gap vs second-ranked song
- `--stable-hits` consecutive detections required before printing
- `--history` short smoothing window for stability
- `--verbose` prints listening/no-match states

## Visualization while testing
Plot waveform, spectrogram, and detected peaks with the exact DB pipeline:

```bash
python src/reference_match.py --query path_to_recorded_clip.wav --plot --plot-out query_plot.png
```

Or use visualizer directly:

```bash
python src/visualize_audio.py --audio path_to_recorded_clip.wav --db reference_db.json --out analysis.png
```

## Advanced (custom dataset)
Build custom fingerprint DB:

```bash
python src/build_database.py --dataset dataset --out fingerprints_db.json
```

Recognize from custom DB:

```bash
python src/recognize_song.py --query query.wav --db fingerprints_db.json
```

## Notes
- WAV input gives best consistency.
- 5-15 second clean clips usually match best.
- For live recognition, keep speaker close to mic and reduce background noise.
- This is a mini Shazam-style baseline, not production-grade recognition.
