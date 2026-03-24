#!/usr/bin/env python
"""Analyze token length of full dialogue JSON files.

Usage (from repo root):

  # Quick test: sample 3 files with local Qwen3-8B tokenizer
  python scripts/analyze_full_token_lengths.py --sample 3

  # Full scan
  python scripts/analyze_full_token_lengths.py

Model: Qwen3-8B (32k native context, 128k with YaRN)
Thresholds: 8k / 16k / 32k tokens

If --hf-model-name is provided and transformers is installed, the script will
use that HuggingFace tokenizer. Otherwise it falls back to a simple
character-based approximation (~4 characters per token).
"""

import argparse
import math
import random
from pathlib import Path
from typing import Callable, Optional, Tuple


def build_token_counter(hf_model_name: Optional[str]) -> Tuple[Callable[[str], int], str]:
    """Return a (counter_fn, description) pair.

    If hf_model_name is given and transformers is available, use that tokenizer.
    Otherwise, fall back to a simple char-length-based approximation.
    """

    if hf_model_name:
        try:
            from transformers import AutoTokenizer  # type: ignore

            tokenizer = AutoTokenizer.from_pretrained(hf_model_name, use_fast=True)

            def count_tokens(text: str) -> int:
                return len(tokenizer(text)["input_ids"])

            desc = f"HuggingFace tokenizer '{hf_model_name}'"
            return count_tokens, desc
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to load HuggingFace tokenizer '{hf_model_name}': {exc}")
            print("[WARN] Falling back to simple char-length-based approximation (len(text)/4).")

    def approx_count(text: str) -> int:
        # Very rough approximation used by many GPT-style models: ~4 chars per token
        return max(1, math.ceil(len(text) / 4)) if text else 0

    return approx_count, "approximate: len(text)/4 characters per token"


def analyze_files(data_dir: Path, pattern: str, hf_model_name: Optional[str], sample: Optional[int] = None) -> None:
    files = sorted(data_dir.rglob(pattern))
    # Filter out mapping / log files if matched accidentally
    files = [p for p in files if not p.name.endswith("session_id_mapping.json")]

    if not files:
        print(f"[INFO] No files found under {data_dir} matching pattern '{pattern}'.")
        return

    total_files = len(files)

    # Sample if requested
    if sample and sample < len(files):
        files = random.sample(files, sample)
        print(f"[INFO] Randomly sampled {sample} files from {total_files} total files.")
    else:
        print(f"[INFO] Analyzing all {total_files} files.")

    counter, counter_desc = build_token_counter(hf_model_name)
    print(f"[INFO] Using token counter: {counter_desc}")
    print(f"[INFO] Starting analysis...\n")

    lengths = []  # list of (path, char_len, token_len)

    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to read {path}: {exc}")
            continue

        char_len = len(text)
        token_len = counter(text)
        lengths.append((path, char_len, token_len))

    if not lengths:
        print("[INFO] No readable files.")
        return

    # Per-file report (sorted by token length descending)
    lengths.sort(key=lambda x: x[2], reverse=True)

    print("Top files by token length (descending):")
    for path, char_len, token_len in lengths[:20]:
        print(f"  - {path}: tokens={token_len}, chars={char_len}")

    # Aggregate stats
    token_values = [t for _, _, t in lengths]
    char_values = [c for _, c, _ in lengths]

    def pct(count: int, total: int) -> float:
        return 100.0 * count / total if total else 0.0

    n = len(token_values)
    max_tokens = max(token_values)
    min_tokens = min(token_values)
    avg_tokens = sum(token_values) / n
    total_tokens = sum(token_values)
    total_chars = sum(char_values)

    over_8k = sum(1 for t in token_values if t > 8192)
    over_16k = sum(1 for t in token_values if t > 16384)
    over_32k = sum(1 for t in token_values if t > 32768)

    print("\nSummary:")
    print(f"  Files analyzed: {n}")
    print(f"  Token length (min/avg/max): {min_tokens:.0f} / {avg_tokens:.1f} / {max_tokens:.0f}")
    print(f"  Char length  (min/avg/max): {min(char_values)} / {sum(char_values)/n:.1f} / {max(char_values)}")
    print(f"  Total tokens across all files: {total_tokens:.0f}")
    print(f"  Total chars  across all files: {total_chars}")
    print(f"  > 8,192 tokens:  {over_8k} files ({pct(over_8k, n):.1f}%)")
    print(f"  > 16,384 tokens: {over_16k} files ({pct(over_16k, n):.1f}%)")
    print(f"  > 32,768 tokens: {over_32k} files ({pct(over_32k, n):.1f}%) [Qwen3-8B native limit]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze token lengths of full dialogue JSON files.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="dataset/full",
        help="Root directory containing full dialogue JSON files (default: dataset/full)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="full_*_with_think.json",
        help=(
            "Glob pattern to match files under data-dir (default: 'full_*_with_think.json'). "
            "You can change this to e.g. 'full_*_improved.json'."
        ),
    )
    parser.add_argument(
        "--hf-model-name",
        type=str,
        default="models/Qwen3-8B",
        help=(
            "HuggingFace model name or local path for tokenization. "
            "Default: 'models/Qwen3-8B'. Set to empty string to use char-based approximation."
        ),
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Randomly sample N files instead of analyzing all. Useful for quick tests.",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"[ERROR] data-dir does not exist: {data_dir}")
        return

    hf_model = args.hf_model_name if args.hf_model_name else None
    analyze_files(data_dir=data_dir, pattern=args.pattern, hf_model_name=hf_model, sample=args.sample)


if __name__ == "__main__":  # pragma: no cover
    main()
