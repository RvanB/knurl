"""Prepare the same bounded Stack sample for several languages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_highlight.dataset.annotate import SUPPORTED_LANGUAGES
from neural_highlight.dataset.prepare import prepare_dataset
from neural_highlight.dataset.download import SMOL_XS_ONLY_LANGUAGES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", choices=SUPPORTED_LANGUAGES, default=list(SUPPORTED_LANGUAGES))
    parser.add_argument("--max-files", type=int, default=5000)
    parser.add_argument("--max-bytes", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source", choices=("smol", "smol-xs"), default="smol")
    parser.add_argument("--output-root", type=Path, default=Path("data/annotated"))
    args = parser.parse_args()
    manifests = {}
    for language in args.languages:
        source = "smol-xs" if args.source == "smol" and language in SMOL_XS_ONLY_LANGUAGES else args.source
        suffix = " (Smol-XS fallback)" if source != args.source else ""
        print(f"Preparing {language}{suffix}...", flush=True)
        manifests[language] = prepare_dataset(
            args.output_root / language,
            language,
            args.max_files,
            args.max_bytes,
            args.seed,
            source,
        )
    print(json.dumps(manifests, indent=2))


if __name__ == "__main__":
    main()
