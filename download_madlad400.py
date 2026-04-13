#!/usr/bin/env python3
"""
Download MADLAD-400 language-by-language.

MADLAD-400 schema (two columns, consistent across all languages):
    text  : string  — the document text
    lang  : string  — the language code (e.g. "af", "sw")

Note: unlike FineWeb2 or WURA, MADLAD has no document IDs or URLs.
It is pure web crawl text, deduplicated at the document level.

The dataset is loaded one language at a time using its ISO 639-3 code
as the config name. The clean split is used by default (recommended).

Usage:
    python download_madlad400.py --out_dir data/raw/madlad400 --max_rows_per_lang 1000
    python download_madlad400.py --out_dir data/raw/madlad400
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from datasets import load_dataset


# MADLAD uses short ISO 639-3 codes as config names.
# Not all languages available in FineWeb2 are in MADLAD —
# Arabic dialects (ary, arz, aeb) are absent; we skip them.
TARGET_LANGUAGES = {
    # High-resource replay languages
    "eng": "English",
    "fra": "French",
    "por": "Portuguese",
    "arb": "Arabic",
    # African languages
    "afr": "Afrikaans",
    "swh": "Swahili",
    "som": "Somali",
    "amh": "Amharic",
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
    "gaz": "Oromo",
    "tsn": "Tswana",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", default="allenai/MADLAD-400")
    p.add_argument("--split", default="clean",
                   help="'clean' (filtered) or 'noisy' (raw LangID only)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--max_rows_per_lang", type=int, default=None)
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize(example: dict, lang_code: str) -> dict:
    # MADLAD schema: just "text" and "lang" — no IDs or URLs.
    return {
        "text": example["text"],
        "language": lang_code,
    }


def process_language(
    dataset_name: str,
    lang_code: str,
    lang_name: str,
    split: str,
    out_dir: Path,
    cache_dir: Optional[str],
    max_rows: Optional[int],
) -> None:
    print(f"[madlad] {lang_code} ({lang_name})")

    ds = load_dataset(dataset_name, lang_code, split=split, cache_dir=cache_dir)

    if max_rows is not None:
        ds = ds.select(range(min(len(ds), max_rows)))

    lang_dir = out_dir / lang_code
    ensure_dir(lang_dir)

    with open(lang_dir / "data.jsonl", "w", encoding="utf-8") as f:
        for example in ds:
            row = normalize(example, lang_code)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {"language_code": lang_code, "language_name": lang_name, "num_rows": len(ds)}
    (lang_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"  → {len(ds)} rows saved to {lang_dir / 'data.jsonl'}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    metadata = {"dataset": args.dataset_name, "split": args.split, "languages": TARGET_LANGUAGES}
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    for lang_code, lang_name in TARGET_LANGUAGES.items():
        try:
            process_language(
                dataset_name=args.dataset_name,
                lang_code=lang_code,
                lang_name=lang_name,
                split=args.split,
                out_dir=out_dir,
                cache_dir=args.cache_dir,
                max_rows=args.max_rows_per_lang,
            )
        except Exception as e:
            print(f"ERROR {lang_code} ({lang_name}): {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
