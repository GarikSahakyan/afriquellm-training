#!/usr/bin/env python3
"""
Translate WebOrganizer documents into African languages and normalize
the output to JSONL ready for CPT training.

Uses the web domain translation prompt from AfriqueLLM Appendix A.2,
which preserves formatting, inline markup, numerals, and named entities.

Usage:
    python translate_weborganizer.py \
        --csv_path data/raw/weborganizer/data_for_translation.csv \
        --out_dir data/raw/weborganizer/translated \
        --model translate_gemma_4b

    # Normalize only (if translation already done):
    python translate_weborganizer.py \
        --csv_path data/raw/weborganizer/data_for_translation.csv \
        --out_dir data/raw/weborganizer/translated \
        --model translate_gemma_4b \
        --normalize_only
"""

from __future__ import annotations

import argparse
import csv as csv_module
import json
import sys
from pathlib import Path

# Uses the web-specific function added to translate_data_samples.py,
# which applies the AfriqueLLM Appendix A.2 web domain prompt.
from translate_data_samples import (
    get_llm_model_and_tokenizer,
    translate_web_with_translate_gemma,
)


# 17 African languages — all except Arabic dialects (already well covered).
TARGET_LANGUAGES = {
    "afr_Latn": "af",   # Afrikaans
    "swh_Latn": "sw",   # Swahili
    "som_Latn": "so",   # Somali
    "amh_Ethi": "am",   # Amharic
    "hau_Latn": "ha",   # Hausa
    "kin_Latn": "rw",   # Kinyarwanda
    "zul_Latn": "zu",   # Zulu
    "ibo_Latn": "ig",   # Igbo
    "plt_Latn": "mg",   # Plateau Malagasy
    "xho_Latn": "xh",   # Xhosa
    "sna_Latn": "sn",   # Shona
    "yor_Latn": "yo",   # Yoruba
    "nya_Latn": "ny",   # Nyanja
    "sot_Latn": "st",   # Southern Sotho
    "tir_Ethi": "ti",   # Tigrinya
    "gaz_Latn": "om",   # Oromo
    "tsn_Latn": "tn",   # Tswana
}

MODEL_IDS = {
    "translate_gemma_4b":  "google/translategemma-4b-it",
    "translate_gemma_12b": "google/translategemma-12b-it",
    "translate_gemma_27b": "google/translategemma-27b-it",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", required=True,
                   help="Path to CSV prepared by download_weborganizer.py")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model", default="translate_gemma_4b",
                   choices=list(MODEL_IDS.keys()))
    p.add_argument("--normalize_only", action="store_true")
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1 — Translation
# ---------------------------------------------------------------------------

def run_translation(csv_path: Path, out_dir: Path, model_choice: str) -> None:
    model_id = MODEL_IDS[model_choice]
    translation_dir = model_id.split("/")[-1]

    print(f"Loading model: {model_id}")
    model, _ = get_llm_model_and_tokenizer(model_id)

    iso_codes = list(TARGET_LANGUAGES.values())
    print(f"Translating into {len(iso_codes)} languages")

    with open(csv_path, encoding="utf-8") as f:
        reader = csv_module.reader(f)
        next(reader)  # skip header: id, source_language, source_text, domain

        for i, row in enumerate(reader):
            src_lang = row[1]
            src_text = row[2]

            for iso_code in iso_codes:
                translate_web_with_translate_gemma(
                    gemma_model=model,
                    target_language=iso_code,
                    source_path=str(csv_path),
                    source_language=src_lang,
                    source_text=src_text,
                    output_path=str(out_dir),
                    translation_directory=translation_dir,
                )

            if (i + 1) % 100 == 0:
                print(f"  Translated {i + 1} documents...")

    print(f"Translation complete. Files in: {out_dir / translation_dir}/")


# ---------------------------------------------------------------------------
# Step 2 — Normalize .txt → JSONL
# ---------------------------------------------------------------------------

def normalize_to_jsonl(out_dir: Path, model_choice: str) -> None:
    model_id = MODEL_IDS[model_choice]
    translation_dir = out_dir / model_id.split("/")[-1]

    if not translation_dir.exists():
        print(f"ERROR: {translation_dir} not found. Run translation first.")
        sys.exit(1)

    iso_to_internal = {v: k for k, v in TARGET_LANGUAGES.items()}

    jsonl_dir = out_dir / "jsonl"
    ensure_dir(jsonl_dir)

    for txt_file in sorted(translation_dir.glob("*.txt")):
        iso_code = txt_file.stem
        lang_code = iso_to_internal.get(iso_code, iso_code)

        lines = txt_file.read_text(encoding="utf-8").strip().splitlines()
        out_file = jsonl_dir / f"{lang_code}.jsonl"

        with open(out_file, "w", encoding="utf-8") as f:
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                row = {
                    "text": line,
                    "language": lang_code,
                    "source": "weborganizer_translated",
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"  {lang_code}: {len(lines)} rows → {out_file}")

    print(f"Normalization complete. JSONL files in: {jsonl_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    if not args.normalize_only:
        run_translation(
            csv_path=Path(args.csv_path),
            out_dir=out_dir,
            model_choice=args.model,
        )

    normalize_to_jsonl(out_dir=out_dir, model_choice=args.model)


if __name__ == "__main__":
    main()
