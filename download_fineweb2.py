#!/usr/bin/env python3
"""
Download FineWeb2 language-by-language.

Languages: 20 African languages + French, Portuguese, Arabic
(based on the AfriqueLLM paper)

Usage:
    # Streaming (memory-efficient, saves as .jsonl)
    python download_fineweb2.py --out_dir data/raw/fineweb2 --streaming --max_rows_per_lang 1000

    # Non-streaming (faster if you have RAM, saves as Arrow dataset)
    python download_fineweb2.py --out_dir data/raw/fineweb2 --max_rows_per_lang 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from datasets import Features, Value, load_dataset


FINEWEB2_FEATURES = Features({
    "text": Value("string"),
    "id": Value("string"),
    "dump": Value("string"),
    "url": Value("string"),
    "date": Value("string"),
    "file_path": Value("string"),
    "language": Value("string"),
    "language_score": Value("float64"),
    "language_script": Value("string"),
    "minhash_cluster_size": Value("int64"),
    "top_langs": Value("string"),
    "wordlist_ratio": Value("float64"),
})


# fmt: off
TARGET_LANGUAGES = {
    # High-resource replay languages
    "fra_Latn": "French",
    "por_Latn": "Portuguese",
    "arb_Arab": "Arabic",
    # African languages from AfriqueLLM Table 1
    "afr_Latn": "Afrikaans",
    "swh_Latn": "Swahili",
    "ary_Arab": "Moroccan Arabic",
    "som_Latn": "Somali",
    "amh_Ethi": "Amharic",
    "arz_Arab": "Egyptian Arabic",
    "hau_Latn": "Hausa",
    "kin_Latn": "Kinyarwanda",
    "zul_Latn": "Zulu",
    "ibo_Latn": "Igbo",
    "plt_Latn": "Plateau Malagasy",
    "xho_Latn": "Xhosa",
    "sna_Latn": "Shona",
    "yor_Latn": "Yoruba",
    "nya_Latn": "Nyanja",
    "sot_Latn": "Southern Sotho",
    "tir_Ethi": "Tigrinya",
    "aeb_Arab": "Tunisian Arabic",
    "gaz_Latn": "Oromo",
    "tsn_Latn": "Tswana",
}
# fmt: on

# Columns we actually want to keep in the output.
# Full FineWeb2 schema (from HuggingFace):
#   text, id, dump, url, date, file_path, language,
#   language_score, language_script, minhash_cluster_size, top_langs
# (some subsets also have wordlist_ratio, we just ignore it by selecting explicitly)
KEEP_COLUMNS = ["text", "id", "url", "language"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="HuggingFaceFW/fineweb-2")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--max_rows_per_lang", type=int, default=None)
    parser.add_argument("--num_proc", type=int, default=4,
                        help="Parallel workers for non-streaming map (ignored in streaming mode)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize(example: dict) -> dict:
    """Keep only the columns we care about, rename id -> doc_id."""
    return {
        "text": example["text"],
        "doc_id": example["id"],
        "url": example["url"],
        "language": example["language"],
    }


def write_stats(lang_dir: Path, lang_code: str, lang_name: str, count: int) -> None:
    stats = {"language_code": lang_code, "language_name": lang_name, "num_rows": count}
    (lang_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Streaming path  →  one .jsonl file per language
# ---------------------------------------------------------------------------

def process_streaming(
    dataset_name: str,
    lang_code: str,
    lang_name: str,
    split: str,
    out_dir: Path,
    cache_dir: Optional[str],
    max_rows: Optional[int],
) -> None:
    print(f"[stream] {lang_code} ({lang_name})")

    ds = load_dataset(dataset_name, lang_code, split=split,
                      cache_dir=cache_dir, streaming=True,
                      features=FINEWEB2_FEATURES)

    lang_dir = out_dir / lang_code
    ensure_dir(lang_dir)

    count = 0
    with open(lang_dir / "data.jsonl", "w", encoding="utf-8") as f:
        for example in ds:
            if max_rows is not None and count >= max_rows:
                break
            f.write(json.dumps(normalize(example), ensure_ascii=False) + "\n")
            count += 1

    write_stats(lang_dir, lang_code, lang_name, count)
    print(f"  → {count} rows saved to {lang_dir / 'data.jsonl'}")


# ---------------------------------------------------------------------------
# Non-streaming path  →  Arrow dataset on disk
# ---------------------------------------------------------------------------

def process_non_streaming(
    dataset_name: str,
    lang_code: str,
    lang_name: str,
    split: str,
    out_dir: Path,
    cache_dir: Optional[str],
    max_rows: Optional[int],
    num_proc: int,
) -> None:
    print(f"[full]   {lang_code} ({lang_name})")

    ds = load_dataset(dataset_name, lang_code, split=split,
                      cache_dir=cache_dir, streaming=False,
                      features=FINEWEB2_FEATURES)

    ds = ds.select_columns(KEEP_COLUMNS)  # drop columns we don't need first
    ds = ds.map(normalize, num_proc=num_proc, remove_columns=KEEP_COLUMNS)

    if max_rows is not None:
        ds = ds.select(range(min(len(ds), max_rows)))

    lang_dir = out_dir / lang_code
    ensure_dir(lang_dir)
    ds.save_to_disk(str(lang_dir))

    write_stats(lang_dir, lang_code, lang_name, len(ds))
    print(f"  → {len(ds)} rows saved to {lang_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # Save a human-readable record of what was downloaded
    metadata = {
        "dataset": args.dataset_name,
        "split": args.split,
        "languages": TARGET_LANGUAGES,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)
    )

    for lang_code, lang_name in TARGET_LANGUAGES.items():
        try:
            if args.streaming:
                process_streaming(
                    dataset_name=args.dataset_name,
                    lang_code=lang_code,
                    lang_name=lang_name,
                    split=args.split,
                    out_dir=out_dir,
                    cache_dir=args.cache_dir,
                    max_rows=args.max_rows_per_lang,
                )
            else:
                process_non_streaming(
                    dataset_name=args.dataset_name,
                    lang_code=lang_code,
                    lang_name=lang_name,
                    split=args.split,
                    out_dir=out_dir,
                    cache_dir=args.cache_dir,
                    max_rows=args.max_rows_per_lang,
                    num_proc=args.num_proc,
                )
        except Exception as e:
            print(f"ERROR {lang_code} ({lang_name}): {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
