#!/usr/bin/env python3
"""
Convert downloaded FineWeb2 JSONL files into a format LLaMA-Factory expects
for continued pre-training (stage: pt).

LLaMA-Factory CPT expects a JSONL where each row has a "text" field.
Our downloaded files already have this — this script just merges them
and registers them in data/dataset_info.json.

Usage:
    python prepare_data.py --data_dir data/raw/fineweb2 --out_dir data/lf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TARGET_LANGUAGES = {
    "fra_Latn": "French",
    "por_Latn": "Portuguese",
    "arb_Arab": "Arabic",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                        help="Root dir of downloaded FineWeb2 data (e.g. data/raw/fineweb2)")
    parser.add_argument("--out_dir", required=True,
                        help="Output dir for LLaMA-Factory data (e.g. data/lf)")
    parser.add_argument("--merge", action="store_true",
                        help="Merge all languages into a single file (simpler but loses per-lang control)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_info = {}
    total_rows = 0

    if args.merge:
        # Single merged file — simplest setup for a quick test
        merged_path = out_dir / "fineweb2_all.jsonl"
        count = 0
        with open(merged_path, "w", encoding="utf-8") as fout:
            for lang_code in TARGET_LANGUAGES:
                src = data_dir / lang_code / "data.jsonl"
                if not src.exists():
                    print(f"  WARNING: {src} not found, skipping")
                    continue
                with open(src, encoding="utf-8") as fin:
                    for line in fin:
                        row = json.loads(line)
                        # LLaMA-Factory CPT only needs "text"
                        fout.write(json.dumps({"text": row["text"]}, ensure_ascii=False) + "\n")
                        count += 1
        print(f"Merged {count} rows → {merged_path}")

        dataset_info["fineweb2"] = {
            "file_name": str(merged_path.resolve()),
            "columns": {"prompt": "text"},
        }
        total_rows = count

    else:
        # One registered dataset per language — matches what the paper does
        # (allows per-language upsampling via UniMax later)
        for lang_code, lang_name in TARGET_LANGUAGES.items():
            src = data_dir / lang_code / "data.jsonl"
            if not src.exists():
                print(f"  WARNING: {src} not found, skipping {lang_code}")
                continue

            # Read stats
            stats_path = data_dir / lang_code / "stats.json"
            num_rows = json.loads(stats_path.read_text())["num_rows"] if stats_path.exists() else "?"

            # LLaMA-Factory can read the file in-place — no need to copy
            dataset_info[f"fineweb2_{lang_code}"] = {
                "file_name": str(src.resolve()),
                "columns": {"prompt": "text"},
            }
            print(f"  Registered {lang_code} ({lang_name}): {num_rows} rows")
            total_rows += num_rows if isinstance(num_rows, int) else 0

    # Write dataset_info.json into the out_dir
    dataset_info_path = out_dir / "dataset_info.json"
    dataset_info_path.write_text(json.dumps(dataset_info, indent=2, ensure_ascii=False))
    print(f"\nWrote {dataset_info_path}")
    print(f"Total rows registered: {total_rows}")
    print(f"\nDataset names to use in train_cpt.yaml:")
    for name in dataset_info:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
