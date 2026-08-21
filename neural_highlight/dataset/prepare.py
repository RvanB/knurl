"""Download, fully annotate, and repository-split a Stack Smol sample."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

from neural_highlight.dataset.annotate import SUPPORTED_LANGUAGES, annotate, replace_comment_bodies
from neural_highlight.dataset.download import stream_stack_smol, stream_stack_smol_xs, take_usable_files
from neural_highlight.dataset.split import repository_split
from neural_highlight.dataset.storage import StoredFile, write_record
from neural_highlight.languages import LANGUAGE_IDS


def prepare_dataset(
    output_dir: Path,
    language: str = "python",
    max_files: int = 1000,
    max_bytes: int = 1_000_000,
    seed: int = 0,
    source: str = "smol",
) -> dict[str, object]:
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"currently supported languages: {', '.join(SUPPORTED_LANGUAGES)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    repositories: dict[str, set[str]] = {name: set() for name in ("train", "validation", "test")}

    source_records = stream_stack_smol(language) if source == "smol" else stream_stack_smol_xs(language)
    records = take_usable_files(source_records, max_files, max_bytes)
    with ExitStack() as stack:
        streams = {
            name: stack.enter_context((output_dir / f"{name}.jsonl").open("w", encoding="utf-8"))
            for name in repositories
        }
        for raw in records:
            repository = str(raw["repository_name"])
            split = repository_split(repository, seed)
            generated_comments = replace_comment_bodies(str(raw["content"]), language)
            annotation = annotate(generated_comments, language, supervise_comment_bodies=True)
            write_record(
                streams[split],
                StoredFile(
                    repository=repository,
                    language=language,
                    path=str(raw["path"]),
                    source=annotation.source,
                    labels=annotation.labels,
                    region_labels=bytes([LANGUAGE_IDS[language]]) * len(annotation.source),
                    label_mask=annotation.label_mask,
                ),
            )
            counts[split] += 1
            repositories[split].add(repository)

    manifest: dict[str, object] = {
        "dataset": f"bigcode/the-stack-{source}",
        "split_unit": "repository" if source == "smol" else "content (XS has no repository metadata)",
        "language": language,
        "seed": seed,
        "max_bytes": max_bytes,
        "files": dict(counts),
        "repositories": {key: len(value) for key, value in repositories.items()},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=SUPPORTED_LANGUAGES, default="python")
    parser.add_argument("--max-files", type=int, default=1000)
    parser.add_argument("--max-bytes", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source", choices=("smol", "smol-xs"), default="smol")
    args = parser.parse_args(argv)
    output = args.output or Path("data/annotated") / args.language
    print(json.dumps(prepare_dataset(output, args.language, args.max_files, args.max_bytes, args.seed, args.source), indent=2))


if __name__ == "__main__":
    main()
