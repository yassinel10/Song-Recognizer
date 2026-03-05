import argparse
import json
from pathlib import Path
from typing import List

from fingerprinting import (
    DEFAULT_PIPELINE,
    analyze_audio,
    build_inverted_index,
    fingerprint_audio,
    normalized_pipeline,
    plot_audio_analysis,
)
from recognize_song import load_db, pipeline_from_db, recognize_song

REFERENCE_SONGS = [
    r"C:\Users\Yassine\Downloads\The Weeknd - Starboy.wav",
    r"C:\Users\Yassine\Downloads\Ellie Goulding - Outside.wav",
    r"C:\Users\Yassine\Downloads\INNA - Caliente.wav",
    r"C:\Users\Yassine\Downloads\INNA - Love.wav",
    r"C:\Users\Yassine\Downloads\Saxobeat.wav",
    r"C:\Users\Yassine\Downloads\Stereo Love.wav",
    r"C:\Users\Yassine\Downloads\Travis Scott - Trance.wav",
    r"C:\Users\Yassine\Downloads\Morad - Sigue ft.(benyjr).wav",
    r"C:\Users\Yassine\Downloads\INNA - Amazing.wav",
]


def build_reference_db(out_path: Path, songs: List[str], pipeline: dict | None = None) -> None:
    cfg = normalized_pipeline(pipeline)
    song_hashes = {}
    song_meta = []

    for song_id, song_str in enumerate(songs):
        song_path = Path(song_str)
        if not song_path.exists():
            raise FileNotFoundError(f"Reference song not found: {song_path}")
        if song_path.suffix.lower() != ".wav":
            raise ValueError(f"Only WAV files are supported: {song_path}")

        print(f"Indexing reference: {song_path.name}")
        hashes = fingerprint_audio(str(song_path), pipeline=cfg)
        song_hashes[song_id] = hashes
        song_meta.append(
            {
                "id": song_id,
                "name": song_path.stem,
                "file": song_path.name,
                "path": str(song_path),
                "num_hashes": len(hashes),
            }
        )

    db = {
        "num_songs": len(song_meta),
        "songs": song_meta,
        "hashes": build_inverted_index(song_hashes),
        "pipeline": cfg,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(db, f)

    print(f"Saved reference DB: {out_path}")
    print(f"Pipeline: {cfg}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a recorded clip and find the closest match in your reference songs"
    )
    parser.add_argument("--query", type=str, required=True, help="Path to recorded WAV clip")
    parser.add_argument("--db", type=str, default="reference_db.json", help="Reference DB JSON file")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild reference DB from the fixed song list before matching",
    )
    parser.add_argument("--plot", action="store_true", help="Plot waveform, spectrogram, and detected peaks")
    parser.add_argument("--plot-out", type=str, default="", help="Optional path to save plot image")
    args = parser.parse_args()

    db_path = Path(args.db)
    query_path = Path(args.query)

    if not query_path.exists():
        raise FileNotFoundError(f"Query audio not found: {query_path}")

    if args.rebuild or not db_path.exists():
        build_reference_db(db_path, REFERENCE_SONGS, pipeline=DEFAULT_PIPELINE)

    best_song, best_votes, ranking = recognize_song(str(query_path), str(db_path))

    if best_song is None:
        print("No confident match found")
    else:
        print(f"Best match: {best_song}")
        print(f"Confidence score (votes): {best_votes}")
        print("Top matches:")
        for name, votes in ranking[:5]:
            print(f"  {name:25s} {votes}")

    if args.plot:
        db = load_db(str(db_path))
        cfg = pipeline_from_db(db)
        y, spec_db, peaks, _ = analyze_audio(str(query_path), pipeline=cfg)
        out_path = args.plot_out if args.plot_out else None
        plot_audio_analysis(
            y,
            spec_db,
            peaks,
            pipeline=cfg,
            title=f"Query Analysis: {query_path.name}",
            out_path=out_path,
        )


if __name__ == "__main__":
    main()

