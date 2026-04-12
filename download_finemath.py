#!/usr/bin/env python3
"""
Download FineMath-4+ for the AfriqueLLM reproduction project.

Dataset:
    HuggingFaceTB/finemath
Config:
    finemath-4plus
Split:
    train

Normalized output schema:
{
    "text": "...",
    "language": "math",
    "source": "finemath",
    "doc_id": null,
    "url": null,
    "metadata": {...}
}

Examples:
    python download_finemath.py --out_dir data/raw/finemath --streaming --max_rows 1000
    python download_finemath.py --out_dir data/raw/finemath --max_rows 10000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from datasets import load_dataset


LANG_CODE = "math"
LANG_NAME = "Mathematics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="HuggingFaceTB/finemath",
        help="Hugging Face dataset name",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="finemath-4plus",
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
        default=1,
        help="Processes for non-streaming map",
    )
    return parser.parse_args()


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def detect_text_column(columns: list[str]) -> str:
    """
    FineMath may expose slightly different text-like column names depending on
    version/config. We prefer the most likely ones in order.
    """
    candidates = ["text", "document", "content", "markdown"]
    for col in candidates:
        if col in columns:
            return col
    raise ValueError(f"Could not find a text column. Available columns: {columns}")


def detect_optional_column(columns: list[str], candidates: list[str]) -> Optional[str]:
    for col in candidates:
        if col in columns:
            return col
    return None


def normalize_example(example: dict, text_col: str, id_col: Optional[str], metadata_col: Optional[str]) -> dict:
    return {
        "text": example.get(text_col),
        "language": LANG_CODE,
        "source": "finemath",
        "doc_id": example.get(id_col) if id_col else None,
        "url": None,
        "metadata": example.get(metadata_col) if metadata_col else None,
    }


def save_metadata(out_dir: Path, dataset_name: str, config: str, split: str) -> None:
    metadata = {
        "dataset": dataset_name,
        "config": config,
        "split": split,
        "language_code": LANG_CODE,
        "language_name": LANG_NAME,
        "description": "FineMath-4+ subset for AfriqueLLM reproduction",
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def dataset_loader(args: argparse.Namespace, streaming: bool):
    return load_dataset(
        args.dataset_name,
        args.config,
        split=args.split,
        cache_dir=args.cache_dir,
        streaming=streaming,
    )


def process_streaming(args: argparse.Namespace) -> None:
    print(f"[STREAMING] Loading {args.dataset_name} / {args.config} [{args.split}]")

    ds = dataset_loader(args, streaming=True)
    first_row = next(iter(ds))
    columns = list(first_row.keys())

    print("Columns:", columns)

    text_col = detect_text_column(columns)
    id_col = detect_optional_column(columns, ["id", "doc_id", "uuid"])
    metadata_col = detect_optional_column(columns, ["metadata", "meta"])

    print(
        "Detected schema:",
        {"text": text_col, "id": id_col, "metadata": metadata_col},
    )

    # Reload because we consumed one row
    ds = dataset_loader(args, streaming=True)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    save_metadata(out_dir, args.dataset_name, args.config, args.split)

    out_file = out_dir / "data.jsonl"
    count = 0

    with open(out_file, "w", encoding="utf-8") as f:
        for ex in ds:
            if args.max_rows is not None and count >= args.max_rows:
                break

            row = normalize_example(ex, text_col, id_col, metadata_col)
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

    ds = dataset_loader(args, streaming=False)
    columns = ds.column_names

    print("Columns:", columns)

    text_col = detect_text_column(columns)
    id_col = detect_optional_column(columns, ["id", "doc_id", "uuid"])
    metadata_col = detect_optional_column(columns, ["metadata", "meta"])

    print(
        "Detected schema:",
        {"text": text_col, "id": id_col, "metadata": metadata_col},
    )

    if args.max_rows is not None:
        n = min(len(ds), args.max_rows)
        ds = ds.select(range(n))

    ds = ds.map(
        lambda ex: normalize_example(ex, text_col, id_col, metadata_col),
        remove_columns=ds.column_names,
        num_proc=args.num_proc,
    )

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