#!/usr/bin/env python3
"""
Download FineMath-4.

FineMath-4+ schema (from HuggingFace dataset card):
    text, url, fetch_time, content_mime_type, warc_filename,
    warc_record_offset, warc_record_length, token_count, char_count,
    metadata, score, int_score, crawl, snapshot_type, language, language_score

For continued pre-training we only need "text". The WARC provenance fields,
scores, and crawl metadata are irrelevant for language modelling.

The "finemath-4plus" config is the quality-filtered subset used in the paper
(int_score >= 4), so no additional score filtering is needed on our end.

Usage:
    python download_finemath.py --out_dir data/raw/finemath --streaming --max_rows 1000
    python download_finemath.py --out_dir data/raw/finemath
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", default="HuggingFaceTB/finemath")
    # finemath-4plus is the subset used in the paper (~34B tokens, int_score >= 4)
    # finemath-3plus is a larger, noisier superset
    p.add_argument("--config", default="finemath-4plus")
    p.add_argument("--split", default="train")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--streaming", action="store_true")
    p.add_argument("--max_rows", type=int, default=None)
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize(example: dict) -> dict:
    # Only "text" is needed for CPT. URL is kept for deduplication purposes
    # if you ever want to cross-reference with other web datasets.
    return {
        "text": example["text"],
        "url": example["url"],
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    print(f"[finemath] {args.dataset_name} / {args.config} [{args.split}]")

    ds = load_dataset(
        args.dataset_name,
        args.config,
        split=args.split,
        cache_dir=args.cache_dir,
        streaming=args.streaming,
    )

    out_file = out_dir / "data.jsonl"
    count = 0

    with open(out_file, "w", encoding="utf-8") as f:
        for example in ds:
            if args.max_rows is not None and count >= args.max_rows:
                break
            f.write(json.dumps(normalize(example), ensure_ascii=False) + "\n")
            count += 1

    stats = {"dataset": args.dataset_name, "config": args.config, "num_rows": count}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"  → {count} rows saved to {out_file}")


if __name__ == "__main__":
    main()
