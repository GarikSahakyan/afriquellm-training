#!/usr/bin/env python3
"""
Download CornStack Python for the AfriqueLLM reproduction project.

Dataset:
    nomic-ai/cornstack-python-v1
Subset:
    default
Split:
    train

Normalized output schema:
{
    "text": "...",
    "language": "code",
    "source": "cornstack",
    "doc_id": null,
    "url": null,
    "metadata": {...}
}

Examples:
    python download_cornstack.py --out_dir data/raw/cornstack --streaming --max_rows 1000
    python download_cornstack.py --out_dir data/raw/cornstack --max_rows 10000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

from datasets import load_dataset


LANG_CODE = "code"
LANG_NAME = "Python code"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="nomic-ai/cornstack-python-v1",
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Dataset config/subset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional Hugging Face cache dir",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optional row cap for testing",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=4,
        help="Processes for non-streaming map",
    )
    return parser.parse_args()


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def guess_column_name(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cols = set(columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def detect_schema(columns: Iterable[str]) -> dict:
    text_col = guess_column_name(columns, ["document", "text", "content"])
    metadata_col = guess_column_name(columns, ["metadata"])
    id_col = guess_column_name(columns, ["id", "doc_id", "query"])

    if text_col is None:
        raise ValueError(f"Could not find text/document column. Available columns: {list(columns)}")

    return {
        "text": text_col,
        "metadata": metadata_col,
        "id": id_col,
    }


def normalize_example(example: dict, schema: dict) -> dict:
    return {
        "text": example.get(schema["text"]),
        "language": LANG_CODE,
        "source": "cornstack",
        "doc_id": example.get(schema["id"]) if schema["id"] else None,
        "url": None,
        "metadata": example.get(schema["metadata"]) if schema["metadata"] else None,
    }


def save_metadata(out_dir: Path, dataset_name: str, config: str, split: str) -> None:
    metadata = {
        "dataset": dataset_name,
        "config": config,
        "split": split,
        "language_code": LANG_CODE,
        "language_name": LANG_NAME,
        "description": "CornStack Python subset for AfriqueLLM reproduction",
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def process_streaming(args: argparse.Namespace) -> None:
    print(f"[STREAMING] Loading {args.dataset_name} / {args.config} [{args.split}]")

    ds = load_dataset(
        args.dataset_name,
        args.config,
        split=args.split,
        cache_dir=args.cache_dir,
        streaming=True,
    )

    first_row = next(iter(ds))
    schema = detect_schema(first_row.keys())
    print("Detected schema:", schema)

    ds = load_dataset(
        args.dataset_name,
        args.config,
        split=args.split,
        cache_dir=args.cache_dir,
        streaming=True,
    )

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    save_metadata(out_dir, args.dataset_name, args.config, args.split)

    out_file = out_dir / "data.jsonl"
    count = 0

    with open(out_file, "w", encoding="utf-8") as f:
        for ex in ds:
            if args.max_rows is not None and count >= args.max_rows:
                break

            row = normalize_example(ex, schema)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    stats = {
        "language_code": LANG_CODE,
        "language_name": LANG_NAME,
        "num_rows": count,
    }
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Saved {count} rows -> {out_file}")


def process_non_streaming(args: argparse.Namespace) -> None:
    print(f"[NON-STREAMING] Loading {args.dataset_name} / {args.config} [{args.split}]")

    ds = load_dataset(
        args.dataset_name,
        args.config,
        split=args.split,
        cache_dir=args.cache_dir,
        streaming=False,
    )

    print("Columns:", ds.column_names)
    schema = detect_schema(ds.column_names)
    print("Detected schema:", schema)

    ds = ds.map(
        lambda ex: normalize_example(ex, schema),
        remove_columns=ds.column_names,
        num_proc=args.num_proc,
    )

    if args.max_rows is not None:
        n = min(len(ds), args.max_rows)
        ds = ds.select(range(n))

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    save_metadata(out_dir, args.dataset_name, args.config, args.split)

    out_file = out_dir / "data.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "language_code": LANG_CODE,
        "language_name": LANG_NAME,
        "num_rows": len(ds),
    }
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(ds)} rows -> {out_file}")


def main() -> None:
    args = parse_args()
    if args.streaming:
        process_streaming(args)
    else:
        process_non_streaming(args)


if __name__ == "__main__":
    main()