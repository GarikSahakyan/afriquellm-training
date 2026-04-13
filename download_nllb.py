#!/usr/bin/env python3
"""
Download and filter NLLB parallel data for the AfriqueLLM reproduction project.

What the paper does (Section 3.1):
    1. Collect ~1B English-African bilingual pairs from NLLB (NLLB Team et al., 2022)
    2. Filter with SSA-COMET quality estimation at threshold 0.7
    3. Result: ~4M pairs / ~456M tokens

Dataset: allenai/nllb (HuggingFace)
Schema:
    translation          : dict  — {"eng_Latn": "...", "<lang>": "..."}
    laser_score          : float — LASER3 mining score (not the same as SSA-COMET)
    source_sentence_lid  : float — language ID confidence for source
    target_sentence_lid  : float — language ID confidence for target
    source_sentence_source, target_sentence_source, source_sentence_url,
    target_sentence_url  : str   — provenance metadata (ignored here)

We download English-centric pairs (eng_Latn ↔ African language) for each
language, save raw JSONL, then run SSA-COMET filtering in a second pass.

Usage:
    # Step 1 — download raw pairs
    python download_nllb.py --out_dir data/raw/nllb --max_rows_per_pair 10000

    # Step 2 — filter with SSA-COMET (requires ssa-comet installed)
    python download_nllb.py --out_dir data/raw/nllb --filter_only --threshold 0.7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from datasets import load_dataset


# African languages from AfriqueLLM Table 1 that have NLLB English-centric pairs.
# Arabic dialects (ary, arz, aeb) are excluded — they are not in NLLB English-centric data.
# The config name format is "<src>-<tgt>" using NLLB language codes.
# We always use eng_Latn as one side. HuggingFace picks the alphabetical order for the config name.
TARGET_LANGUAGES = {
    "afr_Latn": "Afrikaans",
    "swh_Latn": "Swahili",
    "som_Latn": "Somali",
    "amh_Ethi": "Amharic",
    "hau_Latn":  "Hausa",
    "kin_Latn":  "Kinyarwanda",
    "zul_Latn":  "Zulu",
    "ibo_Latn":  "Igbo",
    "plt_Latn":  "Plateau Malagasy",
    "xho_Latn":  "Xhosa",
    "sna_Latn":  "Shona",
    "yor_Latn":  "Yoruba",
    "nya_Latn":  "Nyanja",
    "sot_Latn":  "Southern Sotho",
    "tir_Ethi":  "Tigrinya",
    "gaz_Latn":  "Oromo",
    "tsn_Latn":  "Tswana",
}

ENGLISH = "eng_Latn"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", default="allenai/nllb")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--max_rows_per_pair", type=int, default=None,
                   help="Cap per language pair (for testing)")
    p.add_argument("--filter_only", action="store_true",
                   help="Skip download, only run SSA-COMET filtering on existing raw files")
    p.add_argument("--threshold", type=float, default=0.7,
                   help="SSA-COMET score threshold (paper uses 0.7)")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Batch size for SSA-COMET inference")
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def config_name(lang_code: str) -> str:
    # allenai/nllb config names are alphabetically ordered pairs, e.g. "afr_Latn-eng_Latn"
    pair = sorted([ENGLISH, lang_code])
    return f"{pair[0]}-{pair[1]}"


def english_key(lang_code: str) -> str:
    # In the translation dict, English is always keyed as "eng_Latn"
    return ENGLISH


def normalize(example: dict, lang_code: str) -> dict:
    translation = example["translation"]
    return {
        "src_lang": ENGLISH,
        "tgt_lang": lang_code,
        "src_text": translation[ENGLISH],
        "tgt_text": translation[lang_code],
        "laser_score": example.get("laser_score"),
    }


# ---------------------------------------------------------------------------
# Step 1 — Download
# ---------------------------------------------------------------------------

def download_language(
    dataset_name: str,
    lang_code: str,
    lang_name: str,
    out_dir: Path,
    cache_dir: Optional[str],
    max_rows: Optional[int],
) -> None:
    cfg = config_name(lang_code)
    print(f"[nllb] {cfg} ({lang_name})")

    try:
        ds = load_dataset(dataset_name, cfg, split="train", cache_dir=cache_dir)
    except Exception as e:
        print(f"  ERROR loading {cfg}: {e}")
        return

    if max_rows is not None:
        ds = ds.select(range(min(len(ds), max_rows)))

    lang_dir = out_dir / "raw" / lang_code
    ensure_dir(lang_dir)

    out_file = lang_dir / "data.jsonl"
    count = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for example in ds:
            row = normalize(example, lang_code)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    stats = {"lang_code": lang_code, "lang_name": lang_name,
             "config": cfg, "num_pairs": count}
    (lang_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"  → {count} pairs saved to {out_file}")


# ---------------------------------------------------------------------------
# Step 2 — SSA-COMET filtering
# ---------------------------------------------------------------------------

def filter_language(
    lang_code: str,
    lang_name: str,
    raw_dir: Path,
    filtered_dir: Path,
    threshold: float,
    batch_size: int,
    model,  # SSA-COMET model object
) -> None:
    raw_file = raw_dir / lang_code / "data.jsonl"
    if not raw_file.exists():
        print(f"  SKIP {lang_code}: raw file not found at {raw_file}")
        return

    print(f"[filter] {lang_code} ({lang_name})")

    # Load all pairs
    pairs = []
    with open(raw_file, encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))

    # Build SSA-COMET input format:
    # each sample needs {"src": English, "mt": African lang, "ref": None}
    # SSA-COMET is a reference-free QE model (no reference needed)
    samples = [{"src": p["src_text"], "mt": p["tgt_text"]} for p in pairs]

    # Run in batches
    scores = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        output = model.predict(batch, batch_size=batch_size, gpus=1)
        scores.extend(output.scores)

    # Filter
    kept = [p for p, s in zip(pairs, scores) if s >= threshold]

    out_dir = filtered_dir / lang_code
    ensure_dir(out_dir)
    out_file = out_dir / "data.jsonl"

    with open(out_file, "w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "lang_code": lang_code,
        "lang_name": lang_name,
        "num_pairs_before": len(pairs),
        "num_pairs_after": len(kept),
        "threshold": threshold,
        "retention_rate": round(len(kept) / len(pairs), 4) if pairs else 0,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"  → {len(kept)}/{len(pairs)} pairs kept ({stats['retention_rate']*100:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    raw_dir = out_dir / "raw"
    filtered_dir = out_dir / "filtered"

    if not args.filter_only:
        # Step 1: download raw pairs
        for lang_code, lang_name in TARGET_LANGUAGES.items():
            download_language(
                dataset_name=args.dataset_name,
                lang_code=lang_code,
                lang_name=lang_name,
                out_dir=out_dir,
                cache_dir=args.cache_dir,
                max_rows=args.max_rows_per_pair,
            )
        print("\nDownload complete. Run with --filter_only to apply SSA-COMET filtering.")
        return

    # Step 2: SSA-COMET filtering
    # Install SSA-COMET: pip install ssa-comet
    # Model: Unbabel/wmt22-comet-da is the base; SSA-COMET is a fine-tuned version
    # for African languages. See: https://huggingface.co/Unbabel/ssa-comet-qe
    try:
        from comet import load_from_checkpoint, download_model
    except ImportError:
        print("SSA-COMET not installed. Run: pip install unbabel-comet")
        print("Then download the model: python -c \"from comet import download_model; download_model('Unbabel/ssa-comet-qe')\"")
        return

    print("Loading SSA-COMET model...")
    model_path = download_model("Unbabel/ssa-comet-qe")
    model = load_from_checkpoint(model_path)

    ensure_dir(filtered_dir)
    for lang_code, lang_name in TARGET_LANGUAGES.items():
        filter_language(
            lang_code=lang_code,
            lang_name=lang_name,
            raw_dir=raw_dir,
            filtered_dir=filtered_dir,
            threshold=args.threshold,
            batch_size=args.batch_size,
            model=model,
        )

    # Write a global summary
    total_before = total_after = 0
    for lang_code in TARGET_LANGUAGES:
        stats_file = filtered_dir / lang_code / "stats.json"
        if stats_file.exists():
            s = json.loads(stats_file.read_text())
            total_before += s.get("num_pairs_before", 0)
            total_after += s.get("num_pairs_after", 0)

    summary = {
        "threshold": args.threshold,
        "total_pairs_before": total_before,
        "total_pairs_after": total_after,
        "overall_retention": round(total_after / total_before, 4) if total_before else 0,
    }
    (filtered_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nFiltering complete: {total_after}/{total_before} pairs kept overall")
    print(f"Summary written to {filtered_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
