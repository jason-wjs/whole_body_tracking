"""CLI for validating canonical G1-ready `.npz` motion clips."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

from whole_body_tracking.data_process.schema import validate_npz_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="whole_body_tracking.validate_dataset")
    parser.add_argument("paths", nargs="+", help="One or more .npz clips or directories to validate.")
    return parser


def _iter_npz_paths(paths: Iterable[str]) -> list[Path]:
    resolved: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file():
            resolved.append(path)
        elif path.is_dir():
            resolved.extend(sorted(path.rglob("*.npz")))
        else:
            raise FileNotFoundError(f"Path does not exist: {path}")
    if not resolved:
        raise RuntimeError("No .npz files found to validate")
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for path in _iter_npz_paths(args.paths):
        info = validate_npz_file(path)
        print(
            f"VALID {path} frames={info.num_frames} joints={info.num_joints} "
            f"bodies={info.num_bodies} fps={info.fps}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
