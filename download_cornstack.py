#!/usr/bin/env python3
"""
Download CornStack Python.

CornStack schema (from HuggingFace dataset card):
    query, document, metadata, negatives, negative_scores,
    document_score, document_rank

For continued pre-training we only need "document" — the rest
(negatives, scores, ranks) is contrastive retrieval data that
is irrelevant for language modelling.

Usage:
    python download_cornstack.py --out_dir data/raw/cornstack --streaming --max_rows 1000
    python download_cornstack.py --out_dir data/raw/cornstack
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", default="nomic-ai/cornstack-python-v1")
    p.add_argument("--split", default="train")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--streaming", action="store_true")
    p.add_argument("--max_rows", type=int, default=None)
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize(example: dict) -> dict:
    # "document" is the Python source file content.
    # "query" is a natural language description used for retrieval — skip it.
    # Everything else (negatives, scores, ranks) is retrieval scaffolding — skip it.
    return {"text": example["document"]}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    print(f"[cornstack] {args.dataset_name} [{args.split}]")

    ds = load_dataset(
        args.dataset_name,
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

    stats = {"dataset": args.dataset_name, "num_rows": count}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"  → {count} rows saved to {out_file}")


if __name__ == "__main__":
    main()
