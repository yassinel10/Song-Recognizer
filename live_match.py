import argparse
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

from fingerprinting import DEFAULT_PIPELINE, fingerprint_signal
from recognize_song import load_db, match_query_hashes, pipeline_from_db
from reference_match import REFERENCE_SONGS, build_reference_db


def format_result(best_song: str | None, best_votes: int, min_votes: int) -> str:
    if best_song is None or best_votes < min_votes:
        return "No confident match"
    return f"Match: {best_song} (votes={best_votes})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Live microphone song recognition")
    parser.add_argument("--db", type=str, default="reference_db.json", help="Reference DB JSON file")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild reference DB before listening")
    parser.add_argument("--window", type=float, default=6.0, help="Analysis window in seconds")
    parser.add_argument("--hop", type=float, default=2.0, help="Refresh interval in seconds")
    parser.add_argument("--min-votes", type=int, default=40, help="Minimum votes to accept a match")
    parser.add_argument("--verbose", action="store_true", help="Print non-match states too")
    args = parser.parse_args()

    db_path = Path(args.db)
    if args.rebuild or not db_path.exists():
        build_reference_db(db_path, REFERENCE_SONGS, pipeline=DEFAULT_PIPELINE)

    db = load_db(str(db_path))
    pipeline = pipeline_from_db(db)

    sr = int(pipeline["sr"])
    window_samples = int(args.window * sr)
    hop_samples = int(args.hop * sr)
    if window_samples <= 0 or hop_samples <= 0:
        raise ValueError("window and hop must be > 0")

    print("Live recognition started. Press Ctrl+C to stop.")
    print(f"window={args.window:.1f}s hop={args.hop:.1f}s min_votes={args.min_votes}")

    stream = sd.InputStream(channels=1, samplerate=sr, dtype="float32", blocksize=hop_samples)
    rolling = np.zeros(0, dtype=np.float32)
    last_state: str | None = None

    with stream:
        try:
            while True:
                chunk, _ = stream.read(hop_samples)
                chunk = chunk[:, 0]

                rolling = np.concatenate([rolling, chunk])
                if rolling.size > window_samples:
                    rolling = rolling[-window_samples:]

                if rolling.size < window_samples:
                    continue

                hashes = fingerprint_signal(rolling, pipeline=pipeline)
                best_song, best_votes, ranking = match_query_hashes(hashes, db)
                is_match = best_song is not None and best_votes >= args.min_votes
                state = f"match:{best_song}" if is_match else "no_match"

                if state != last_state:
                    if is_match:
                        print(f"the guessed song is: {best_song}")
                    elif args.verbose:
                        print(f"[{time.strftime('%H:%M:%S')}] No confident match")
                    last_state = state
        except KeyboardInterrupt:
            print("Stopped.")


if __name__ == "__main__":
    main()
