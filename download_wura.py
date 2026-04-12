#!/usr/bin/env python3
"""
Download WURA language-by-language for the AfriqueLLM reproduction project.

WURA is organized with one config per language.
This script loads each language separately and normalizes rows to:

{
    "text": "...",
    "language": "...",
    "source": "wura",
    "doc_id": "...",
    "url": "...",
    "metadata": {...}
}

Usage:
    python download_wura.py --out_dir data/raw/wura --max_rows_per_lang 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

from datasets import load_dataset


# Current WURA loader configs observed from the dataset loader
TARGET_LANGUAGES = {
    # High-resource languages currently available
    "eng": "English",
    "fra": "French",
    "por": "Portuguese",

    # African languages available in WURA
    "afr": "Afrikaans",
    "amh": "Amharic",
    "arz": "Egyptian Arabic",
    "hau": "Hausa",
    "ibo": "Igbo",
    "kin": "Kinyarwanda",
    "mlg": "Malagasy",
    "nya": "Nyanja",
    "orm": "Oromo",
    "sna": "Shona",
    "som": "Somali",
    "sot": "Southern Sotho",
    "swa": "Swahili",
    "tir": "Tigrinya",
    "xho": "Xhosa",
    "yor": "Yoruba",
    "zul": "Zulu",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", default="castorini/wura")
    p.add_argument("--split", default="train")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--max_rows_per_lang", type=int, default=None)
    p.add_argument("--num_proc", type=int, default=1)
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
    text_col = guess_column_name(columns, ["content", "text", "document"])
    headline_col = guess_column_name(columns, ["headline", "title"])
    id_col = guess_column_name(columns, ["id", "doc_id"])
    url_col = guess_column_name(columns, ["url"])
    category_col = guess_column_name(columns, ["category"])

    if text_col is None:
        raise ValueError(f"Could not find text column. Columns: {list(columns)}")

    return {
        "text": text_col,
        "headline": headline_col,
        "id": id_col,
        "url": url_col,
        "category": category_col,
    }


def normalize_example(example: dict, schema: dict, lang_code: str) -> dict:
    metadata = {}
    if schema["headline"] and example.get(schema["headline"]) is not None:
        metadata["headline"] = example.get(schema["headline"])
    if schema["category"] and example.get(schema["category"]) is not None:
        metadata["category"] = example.get(schema["category"])

    return {
        "text": example.get(schema["text"]),
        "language": lang_code,
        "source": "wura",
        "doc_id": example.get(schema["id"]) if schema["id"] else None,
        "url": example.get(schema["url"]) if schema["url"] else None,
        "metadata": metadata if metadata else None,
    }


def save_metadata(out_dir: Path, dataset_name: str) -> None:
    metadata = {
        "dataset": dataset_name,
        "description": "WURA subset for AfriqueLLM reproduction",
        "languages": TARGET_LANGUAGES,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def process_language(
    dataset_name: str,
    lang_code: str,
    lang_name: str,
    split: str,
    out_dir: Path,
    cache_dir: Optional[str],
    max_rows_per_lang: Optional[int],
    num_proc: int,
) -> None:
    print(f"\nLoading {lang_code} ({lang_name})")

    ds = load_dataset(
        dataset_name,
        lang_code,
        split=split,
        cache_dir=cache_dir,
        verification_mode="no_checks",
    )

    print("Columns:", ds.column_names)
    schema = detect_schema(ds.column_names)
    print(f"Detected schema for {lang_code}: {schema}")

    # IMPORTANT: select first, then map
    if max_rows_per_lang is not None:
        n = min(len(ds), max_rows_per_lang)
        ds = ds.select(range(n))

    ds = ds.map(
        lambda ex: normalize_example(ex, schema, lang_code),
        remove_columns=ds.column_names,
        num_proc=num_proc,
    )

    lang_dir = out_dir / lang_code
    ensure_dir(lang_dir)
    out_file = lang_dir / "data.jsonl"

    with open(out_file, "w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "language_code": lang_code,
        "language_name": lang_name,
        "num_rows": len(ds),
    }
    with open(lang_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(ds)} rows for {lang_code}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    save_metadata(out_dir, args.dataset_name)

    for lang_code, lang_name in TARGET_LANGUAGES.items():
        try:
            process_language(
                dataset_name=args.dataset_name,
                lang_code=lang_code,
                lang_name=lang_name,
                split=args.split,
                out_dir=out_dir,
                cache_dir=args.cache_dir,
                max_rows_per_lang=args.max_rows_per_lang,
                num_proc=args.num_proc,
            )
        except Exception as e:
            print(f"ERROR for {lang_code} ({lang_name}): {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()