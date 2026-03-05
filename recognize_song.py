import argparse
import json
from collections import Counter, defaultdict
from typing import List, Tuple

from fingerprinting import DEFAULT_PIPELINE, HashAtTime, fingerprint_audio, normalized_pipeline


def load_db(db_path: str):
    with open(db_path, "r", encoding="utf-8") as f:
        return json.load(f)


def match_query_hashes(query_hashes: List[HashAtTime], db) -> Tuple[str | None, int, List[Tuple[str, int]]]:
    if not query_hashes:
        return None, 0, []

    hashes_index = db["hashes"]
    offset_votes = Counter()

    # Vote by (song_id, time_offset)
    for h, t_query in query_hashes:
        for song_id, t_song in hashes_index.get(h, []):
            offset_votes[(int(song_id), int(t_song) - int(t_query))] += 1

    if not offset_votes:
        return None, 0, []

    best_per_song = defaultdict(int)
    for (song_id, _offset), votes in offset_votes.items():
        if votes > best_per_song[song_id]:
            best_per_song[song_id] = votes

    ranked = sorted(best_per_song.items(), key=lambda x: x[1], reverse=True)
    id_to_name = {int(song["id"]): song["name"] for song in db["songs"]}
    ranking = [(id_to_name[sid], votes) for sid, votes in ranked]

    best_song, best_votes = ranking[0]
    return best_song, best_votes, ranking


def pipeline_from_db(db) -> dict:
    return normalized_pipeline(db.get("pipeline", DEFAULT_PIPELINE))


def recognize_song(query_path: str, db_path: str) -> Tuple[str | None, int, List[Tuple[str, int]]]:
    db = load_db(db_path)
    pipeline = pipeline_from_db(db)
    query_hashes = fingerprint_audio(query_path, pipeline=pipeline)
    return match_query_hashes(query_hashes, db)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recognize song from a short clip (Shazam-style)")
    parser.add_argument("--query", type=str, required=True, help="Path to query WAV clip")
    parser.add_argument("--db", type=str, default="fingerprints_db.json", help="Path to DB JSON")
    args = parser.parse_args()

    best_song, best_votes, ranking = recognize_song(args.query, args.db)

    if best_song is None:
        print("No confident match found")
        return

    print(f"Matched song: {best_song}")
    print(f"Vote score: {best_votes}")
    print("\nTop matches:")
    for song_name, votes in ranking[:5]:
        print(f"  {song_name:25s} {votes}")


if __name__ == "__main__":
    main()
