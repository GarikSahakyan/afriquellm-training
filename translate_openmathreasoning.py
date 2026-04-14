#!/usr/bin/env python3
"""
Translate OpenMathReasoning samples into African languages and normalize
the output to JSONL ready for CPT training.

This script:
    1. Calls translate_data_samples.py functions directly (no subprocess)
    2. Translates the prepared CSV into each target language
    3. Normalizes the raw .txt output into JSONL with a "text" field,
       matching the format of all other training datasets

The paper (AfriqueLLM, Section 3.1) translates into 17 African languages —
all languages except Arabic dialects (ary, arz, aeb), which are already
well represented in the monolingual corpus.

Usage:
    python translate_openmathreasoning.py \
        --csv_path data/raw/openmathreasoning/data_for_translation.csv \
        --out_dir data/raw/openmathreasoning/translated \
        --model translate_gemma_4b

    # To normalize translated .txt files to JSONL after translation:
    python translate_openmathreasoning.py \
        --csv_path data/raw/openmathreasoning/data_for_translation.csv \
        --out_dir data/raw/openmathreasoning/translated \
        --model translate_gemma_4b \
        --normalize_only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Import from translate_data_samples.py (same directory).
# We use our added math-specific function which uses the prompt from
# AfriqueLLM Appendix A.2 and a sufficient max_new_tokens for full traces.
from translate_data_samples import (
    get_llm_model_and_tokenizer,
    translate_math_with_translate_gemma,
)


# 17 African languages from AfriqueLLM Table 1, excluding Arabic dialects.
# Keys are our internal codes (matching other download scripts),
# values are ISO 639-1 codes that translategemma understands.
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
                   help="Path to CSV prepared by download_openmathreasoning.py")
    p.add_argument("--out_dir", required=True,
                   help="Directory where translated .txt files and final JSONL are written")
    p.add_argument("--model", default="translate_gemma_4b",
                   choices=list(MODEL_IDS.keys()))
    p.add_argument("--normalize_only", action="store_true",
                   help="Skip translation, only normalize existing .txt files to JSONL")
    p.add_argument("--source_lang", default="en",
                   help="ISO 639-1 code of the source language (default: en)")
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1 — Translation
# ---------------------------------------------------------------------------

def run_translation(csv_path: Path, out_dir: Path, model_choice: str, source_lang: str) -> None:
    import csv as csv_module

    model_id = MODEL_IDS[model_choice]
    translation_dir = model_id.split("/")[-1]   # e.g. "translategemma-4b-it"

    print(f"Loading model: {model_id}")
    model, _ = get_llm_model_and_tokenizer(model_id)
    # translategemma uses a pipeline, tokenizer is None

    iso_codes = list(TARGET_LANGUAGES.values())
    print(f"Translating into {len(iso_codes)} languages: {iso_codes}")

    with open(csv_path, encoding="utf-8") as f:
        reader = csv_module.reader(f)
        next(reader)  # skip header

        for i, row in enumerate(reader):
            src_lang = row[1]   # "en"
            src_text = row[2]   # formatted <problem>...</problem><think>...</think>answer <eos>

            for iso_code in iso_codes:
                translate_math_with_translate_gemma(
                    gemma_model=model,
                    target_language=iso_code,
                    source_path=str(csv_path),
                    source_language=src_lang,
                    source_text=src_text,
                    output_path=str(out_dir),
                    translation_directory=translation_dir,
                )

            if (i + 1) % 100 == 0:
                print(f"  Translated {i + 1} samples...")

    print(f"Translation complete. Files in: {out_dir / translation_dir}/")


# ---------------------------------------------------------------------------
# Step 2 — Normalize .txt → JSONL
# ---------------------------------------------------------------------------

def normalize_to_jsonl(out_dir: Path, model_choice: str) -> None:
    """
    The translation script writes one .txt file per language, with one
    translated sample per line. We convert each to JSONL with a "text"
    field, matching the format used by all other datasets (FineWeb2, WURA, etc.)
    so everything can be merged cleanly for training.
    """
    model_id = MODEL_IDS[model_choice]
    translation_dir = out_dir / model_id.split("/")[-1]

    if not translation_dir.exists():
        print(f"ERROR: {translation_dir} not found. Run translation first.")
        sys.exit(1)

    # Map ISO code back to our internal lang_code for consistency
    iso_to_internal = {v: k for k, v in TARGET_LANGUAGES.items()}

    jsonl_dir = out_dir / "jsonl"
    ensure_dir(jsonl_dir)

    for txt_file in sorted(translation_dir.glob("*.txt")):
        iso_code = txt_file.stem          # filename is the ISO code, e.g. "sw"
        lang_code = iso_to_internal.get(iso_code, iso_code)

        lines = txt_file.read_text(encoding="utf-8").strip().splitlines()
        out_file = jsonl_dir / f"{lang_code}.jsonl"

        with open(out_file, "w", encoding="utf-8") as f:
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Each line is a fully translated <problem>...</problem><think>...</think>answer <eos>
                # This is the format the paper uses for math synthetic data in CPT.
                row = {"text": line, "language": lang_code, "source": "openmathreasoning_translated"}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"  {lang_code}: {len(lines)} rows → {out_file}")

    print(f"Normalization complete. JSONL files in: {jsonl_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    if not args.normalize_only:
        run_translation(
            csv_path=csv_path,
            out_dir=out_dir,
            model_choice=args.model,
            source_lang=args.source_lang,
        )

    normalize_to_jsonl(out_dir=out_dir, model_choice=args.model)


if __name__ == "__main__":
    main()
