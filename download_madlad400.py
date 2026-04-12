#!/usr/bin/env python3
"""
Download and filter MADLAD-400 for the AfriqueLLM reproduction project.

MADLAD-400 supports loading selected languages and separate clean/noisy splits.
This script uses the clean split by default.

Usage:
    python download_madlad400.py --out_dir data/raw/madlad400 --max_rows_per_lang 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

from datasets import load_dataset


TARGET_LANGUAGES = {
    # High-resource replay languages
    "eng": "English",
    "fra": "French",
    "por": "Portuguese",
    "arb": "Arabic",
    # African languages in AfriqueLLM
    "afr": "Afrikaans",
    "swh": "Swahili",
    "ary": "Moroccan Arabic",
    "som": "Somali",
    "amh": "Amharic",
    "arz": "Egyptian Arabic",
    "hau": "Hausa",
    "kin": "Kinyarwanda",
    "zul": "Zulu",
    "ibo": "Igbo",
    "plt": "Plateau Malagasy",
    "xho": "Xhosa",
    "sna": "Shona",
    "yor": "Yoruba",
    "nya": "Nyanja",
    "sot": "Southern Sotho",
    "tir": "Tigrinya",
    "aeb": "Tunisian Arabic",
    "gaz": "Oromo",
    "tsn": "Tswana",
}

TEXT_COLUMN_CANDIDATES = ["text", "content", "document"]
LANG_COLUMN_CANDIDATES = ["lang", "language", "lang_id"]
ID_COLUMN_CANDIDATES = ["id", "doc_id"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", default="allenai/MADLAD-400")
    p.add_argument("--split", default="clean")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--max_rows_per_lang", type=int, default=None)
    p.add_argument("--num_proc", type=int, default=4)
    return p.parse_args()


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def guess_column_name(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cols = set(columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def detect_schema(columns: Iterable[str]) -> dict:
    text_col = guess_column_name(columns, TEXT_COLUMN_CANDIDATES)
    lang_col = guess_column_name(columns, LANG_COLUMN_CANDIDATES)
    id_col = guess_column_name(columns, ID_COLUMN_CANDIDATES)

    if text_col is None:
        raise ValueError(f"Could not find text column. Columns: {list(columns)}")

    return {"text": text_col, "language": lang_col, "id": id_col}


def normalize_example(example: dict, schema: dict, fallback_lang: Optional[str] = None) -> dict:
    lang = example.get(schema["language"]) if schema["language"] else fallback_lang
    return {
        "text": example.get(schema["text"]),
        "language": lang,
        "doc_id": example.get(schema["id"]) if schema["id"] else None,
        "source": "madlad400",
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    target_codes = list(TARGET_LANGUAGES.keys())

    print(f"Loading {args.dataset_name} [{args.split}] for languages={target_codes}")
    ds = load_dataset(
        args.dataset_name,
        languages=target_codes,
        split=args.split,
        cache_dir=args.cache_dir,
    )

    print("Columns:", ds.column_names)
    schema = detect_schema(ds.column_names)
    print("Detected schema:", schema)

    ds = ds.map(
        lambda ex: normalize_example(ex, schema),
        remove_columns=ds.column_names,
        num_proc=args.num_proc,
    )

    if "language" in ds.column_names:
        ds = ds.filter(lambda ex: ex["language"] in TARGET_LANGUAGES, num_proc=args.num_proc)

    metadata = {
        "dataset": args.dataset_name,
        "split": args.split,
        "target_languages": TARGET_LANGUAGES,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    for lang_code, lang_name in TARGET_LANGUAGES.items():
        lang_ds = ds.filter(lambda ex, lc=lang_code: ex["language"] == lc, num_proc=args.num_proc)

        if args.max_rows_per_lang is not None:
            n = min(len(lang_ds), args.max_rows_per_lang)
            lang_ds = lang_ds.select(range(n))

        lang_dir = out_dir / lang_code
        ensure_dir(lang_dir)
        out_file = lang_dir / "data.jsonl"

        with open(out_file, "w", encoding="utf-8") as f:
            for row in lang_ds:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        stats = {
            "language_code": lang_code,
            "language_name": lang_name,
            "num_rows": len(lang_ds),
        }
        with open(lang_dir / "stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(lang_ds)} rows for {lang_code}")

    print("Done.")


if __name__ == "__main__":
    main()