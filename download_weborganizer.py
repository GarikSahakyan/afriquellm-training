#!/usr/bin/env python3
"""
Download WebOrganizer documents for the AfriqueLLM reproduction project.

What the AfriqueLLM paper does (Section 3.1 + Table 8):
    - Selects 10 domains from Web Organizer (Wettig et al., 2025),
      spanning 20 topics
    - Translates selected documents into 17 African languages using GPT-4.1
    - Domains: Food and Dining, Health, History, Industrial, Politics,
      Science and Technology, Software Development, Travel,
      Education and Jobs, Entertainment
    - Total synthetic data: ~324M tokens (Table 8)

Dataset: HuggingFaceFW/web-organizer (domain-classified CommonCrawl)

We save:
    1. Raw JSONL per domain
    2. A single CSV ready for translate_data_samples.py
       Columns: id | source_language | source_text

Usage:
    python download_weborganizer.py --out_dir data/raw/weborganizer --max_rows_per_domain 100
    python download_weborganizer.py --out_dir data/raw/weborganizer
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

from datasets import load_dataset


DATASET_NAME = "HuggingFaceFW/web-organizer"
SOURCE_LANG = "en"

# The 10 domains selected in the paper (Table 8).
# These are the config/subset names in the HuggingFace dataset.
TARGET_DOMAINS = [
    "food_and_dining",
    "health",
    "history",
    "industrial",
    "politics",
    "science_and_technology",
    "software_development",
    "travel",
    "education_and_jobs",
    "entertainment",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--max_rows_per_domain", type=int, default=None,
                   help="Cap per domain for testing")
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    csv_path = out_dir / "data_for_translation.csv"
    global_id = 0

    with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "source_language", "source_text", "domain"])

        for domain in TARGET_DOMAINS:
            print(f"[weborganizer] {domain}")

            try:
                ds = load_dataset(
                    DATASET_NAME,
                    domain,
                    split="train",
                    cache_dir=args.cache_dir,
                )
            except Exception as e:
                print(f"  ERROR loading {domain}: {e}")
                continue

            if args.max_rows_per_domain is not None:
                ds = ds.select(range(min(len(ds), args.max_rows_per_domain)))

            # Save raw JSONL per domain
            domain_dir = out_dir / "raw" / domain
            ensure_dir(domain_dir)

            with open(domain_dir / "data.jsonl", "w", encoding="utf-8") as f:
                for example in ds:
                    f.write(json.dumps({"text": example["text"], "domain": domain},
                                       ensure_ascii=False) + "\n")

            # Append to translation CSV
            for example in ds:
                writer.writerow([global_id, SOURCE_LANG, example["text"], domain])
                global_id += 1

            stats = {"domain": domain, "num_rows": len(ds)}
            (domain_dir / "stats.json").write_text(json.dumps(stats, indent=2))
            print(f"  → {len(ds)} documents")

    print(f"\nTotal: {global_id} documents across {len(TARGET_DOMAINS)} domains")
    print(f"Translation CSV: {csv_path}")


if __name__ == "__main__":
    main()
