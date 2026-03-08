#!/usr/bin/env python3

import json
import sys
import urllib.request
from pathlib import Path


API_URL = (
    "https://datasets-server.huggingface.co/first-rows"
    "?dataset=istupakov%2Frussian_librispeech&config=default&split=test"
)


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print(
            "Usage: prepare_russian_librispeech_subset.py <output_dir> [limit]",
            file=sys.stderr,
        )
        return 1

    output_dir = Path(sys.argv[1]).resolve()
    limit = int(sys.argv[2]) if len(sys.argv) == 3 else 10
    if limit <= 0:
        raise ValueError("limit must be > 0")

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    with urllib.request.urlopen(API_URL) as response:
        payload = json.load(response)

    rows = payload.get("rows", [])
    if not rows:
        raise RuntimeError("No rows returned from russian_librispeech test split")

    manifest_path = output_dir / "manifest.tsv"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for idx, item in enumerate(rows[:limit], start=1):
            row = item["row"]
            audio_src = row["audio"][0]["src"]
            reference = row["text"]
            filename = f"sample_{idx:03d}.wav"
            filepath = audio_dir / filename

            print(f"Downloading {filename}")
            urllib.request.urlretrieve(audio_src, filepath)
            manifest.write(f"audio/{filename}\t{reference}\n")

    print(f"Prepared {min(limit, len(rows))} files in {output_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
