"""
Generate a shared clip-level train/val/test split manifest.

The manifest is intended to be reused by KD and ai8x QAT datasets so
overlapping frames from the same source recording never cross split boundaries 
and inflate val/test metrics.

Example:
    python data_splits.py --data_dir ./data --output ./data/splits.json
"""

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

AUDIO_EXTENSIONS = (".wav", ".flac")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a deterministic clip-level split manifest."
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("./data"),
        help="Dataset root containing guitar/ and piano/ subdirectories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <data_dir>/splits.json.",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Fraction of paired stems assigned to validation.",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.1,
        help="Fraction of paired stems assigned to final held-out test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=22,
        help="Random seed used to shuffle stems deterministically.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing split manifest.",
    )
    return parser.parse_args()

def collect_stems(directory: Path) -> set[str]:
    stems = set()
    for extension in AUDIO_EXTENSIONS:
        stems.update(path.stem for path in directory.glob(f"*{extension}"))
    return stems

def split_count(num_items: int, fraction: float) -> int:
    if fraction <= 0.0 or num_items == 0:
        return 0
    return max(1, int(round(num_items * fraction)))

def make_manifest(args: argparse.Namespace) -> dict:
    guitar_stems = collect_stems(args.data_dir / "guitar")
    piano_stems = collect_stems(args.data_dir / "piano")
    paired_stems = sorted(guitar_stems & piano_stems)

    if not paired_stems:
        raise ValueError("No matching guitar/piano stems found.")

    rng = random.Random(args.seed)
    shuffled = paired_stems[:]
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_test = split_count(n_total, args.test_split)
    n_val = split_count(n_total, args.val_split)

    if n_val + n_test >= n_total and n_total > 1:
        overflow = n_val + n_test - (n_total - 1)
        reduce_test = min(n_test, overflow)
        n_test -= reduce_test
        overflow -= reduce_test
        n_val = max(0, n_val - overflow)

    test_stems = sorted(shuffled[:n_test])
    val_stems = sorted(shuffled[n_test:n_test + n_val])
    train_stems = sorted(shuffled[n_test + n_val:])

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(args.data_dir),
        "seed": args.seed,
        "val_split": args.val_split,
        "test_split": args.test_split,
        "audio_extensions": list(AUDIO_EXTENSIONS),
        "splits": {
            "train": train_stems,
            "val": val_stems,
            "test": test_stems,
        },
        "counts": {
            "train": len(train_stems),
            "val": len(val_stems),
            "test": len(test_stems),
            "paired": len(paired_stems),
            "guitar_only": len(guitar_stems - piano_stems),
            "piano_only": len(piano_stems - guitar_stems),
        },
        "unmatched": {
            "guitar_only": sorted(guitar_stems - piano_stems),
            "piano_only": sorted(piano_stems - guitar_stems),
        },
    }

def find_split_manifest(data_dir: Path, split_manifest: str | Path | None) -> Path | None:
    if split_manifest is not None:
        return Path(split_manifest)

    default_manifest = data_dir / "splits.json"
    return default_manifest if default_manifest.exists() else None

def load_split_manifest(
    data_dir: str | Path,
    split_manifest: str | Path | None = None,
) -> dict[str, list[str]] | None:
    """
    Load and validate a clip-level split manifest.

    If split_manifest is None, <data_dir>/splits.json is used when present.
    Returns None when no manifest was requested or found.
    """
    data_dir = Path(data_dir)
    manifest_path = find_split_manifest(data_dir, split_manifest)
    if manifest_path is None:
        return None
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    raw_splits = payload.get("splits")
    if not isinstance(raw_splits, dict):
        raise ValueError(f"Split manifest {manifest_path} is missing a 'splits' object.")

    splits = {
        name: list(raw_splits.get(name, []))
        for name in ("train", "val", "test")
    }
    if not splits["train"]:
        raise ValueError(f"Split manifest {manifest_path} must contain a non-empty train split.")

    split_sets = {name: set(stems) for name, stems in splits.items()}
    overlap_messages = []
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(split_sets[left] & split_sets[right])
        if overlap:
            overlap_messages.append(f"{left}/{right}: {overlap}")
    if overlap_messages:
        raise ValueError(
            f"Split manifest {manifest_path} has overlapping stems: "
            + "; ".join(overlap_messages)
        )

    paired_stems = (
        collect_stems(data_dir / "guitar")
        & collect_stems(data_dir / "piano")
    )
    missing = sorted(set().union(*split_sets.values()) - paired_stems)
    if missing:
        raise ValueError(
            f"Split manifest {manifest_path} references stems without paired audio: {missing}"
        )

    print(f"Using split manifest: {manifest_path}")
    return splits

def main() -> None:
    args = parse_args()

    if args.val_split < 0.0 or args.test_split < 0.0:
        raise ValueError("--val_split and --test_split must be non-negative.")
    if args.val_split + args.test_split >= 1.0:
        raise ValueError("--val_split + --test_split must be less than 1.0.")

    guitar_dir = args.data_dir / "guitar"
    piano_dir = args.data_dir / "piano"
    if not guitar_dir.exists() or not piano_dir.exists():
        raise FileNotFoundError(
            f"Expected dataset directories {guitar_dir} and {piano_dir}."
        )

    output = args.output if args.output is not None else args.data_dir / "splits.json"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists. Pass --overwrite to replace it.")

    manifest = make_manifest(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    counts = manifest["counts"]
    print(f"Wrote split manifest: {output}")
    print(
        "Splits: "
        f"train={counts['train']} val={counts['val']} test={counts['test']} "
        f"paired={counts['paired']}"
    )
    if counts["guitar_only"] or counts["piano_only"]:
        print(
            "Unmatched stems: "
            f"guitar_only={counts['guitar_only']} piano_only={counts['piano_only']}"
        )

if __name__ == "__main__":
    main()
