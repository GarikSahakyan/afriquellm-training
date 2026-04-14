#!/usr/bin/env python3
"""
Download OpenMathReasoning (cot split) and prepare it for translation.

What the AfriqueLLM paper does (Section 3.1 + Appendix A.2):
    - Takes the cot split of OpenMathReasoning (Moshkov et al., 2025)
    - Formats each sample as:
        <problem>[problem]</problem>
        <think>[reasoning]</think>
        [answer] <eos>
    - Translates this into 17 African languages using GPT-4.1
      (we will use translate_gemma instead)

We save two outputs:
    1. Raw JSONL — the original fields kept for reference
    2. CSV — ready to feed into translate_data_samples.py
       Columns: id | source_language | source_text

The CSV source_text uses the exact format from Appendix A.2 of the paper,
which preserves the structure needed to reconstruct problem/think/answer
after translation.

Usage:
    python download_openmathreasoning.py --out_dir data/raw/openmathreasoning
    python download_openmathreasoning.py --out_dir data/raw/openmathreasoning --max_rows 1000
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

from datasets import load_dataset


# The paper uses the "cot" split — chain-of-thought reasoning traces
DATASET_NAME = "nvidia/OpenMathReasoning"
SPLIT = "cot"

# Source language for the translation script (ISO 639-1)
SOURCE_LANG = "en"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True,
                   help="Output directory for raw JSONL and translation CSV")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--max_rows", type=int, default=None,
                   help="Cap number of rows (for testing; remove for full dataset)")
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def format_for_translation(example: dict) -> str:
    """
    Format a sample using the math translation prompt structure from
    Appendix A.2 of the AfriqueLLM paper:

        <problem>[Original Problem]</problem>
        <think>[Original Reasoning]</think>
        [Final Answer] <eos>

    This structure is preserved through translation so we can parse
    problem / reasoning / answer back out of the translated text.
    """
    problem = (example.get("problem") or "").strip()
    reasoning = (example.get("generated_solution") or "").strip()
    answer = (example.get("expected_answer") or "").strip()

    return (
        f"<problem>{problem}</problem>\n"
        f"<think>{reasoning}</think>\n"
        f"{answer} <eos>"
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    print(f"[openmathreasoning] Loading {DATASET_NAME} / {SPLIT}")
    ds = load_dataset(DATASET_NAME, split=SPLIT, cache_dir=args.cache_dir)

    if args.max_rows is not None:
        ds = ds.select(range(min(len(ds), args.max_rows)))

    print(f"  {len(ds)} samples loaded")

    raw_path = out_dir / "data.jsonl"
    csv_path = out_dir / "data_for_translation.csv"

    # Save raw JSONL — keep only fields relevant to us
    with open(raw_path, "w", encoding="utf-8") as f:
        for example in ds:
            row = {
                "problem": example.get("problem"),
                "generated_solution": example.get("generated_solution"),
                "expected_answer": example.get("expected_answer"),
                "problem_type": example.get("problem_type"),
                "problem_source": example.get("problem_source"),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Save CSV for translate_data_samples.py
    # Columns: id | source_language | source_text
    # (the script reads line[1] as source_language, line[2] as source_text)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "source_language", "source_text"])
        for i, example in enumerate(ds):
            writer.writerow([i, SOURCE_LANG, format_for_translation(example)])

    stats = {
        "dataset": DATASET_NAME,
        "split": SPLIT,
        "num_rows": len(ds),
        "raw_jsonl": str(raw_path),
        "translation_csv": str(csv_path),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    print(f"  → Raw JSONL: {raw_path}")
    print(f"  → Translation CSV: {csv_path}")
    print(f"  → {len(ds)} rows ready for translation")


if __name__ == "__main__":
    main()
