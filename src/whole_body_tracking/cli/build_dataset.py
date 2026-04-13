"""CLI for building compiled motion datasets."""

from __future__ import annotations

import argparse
from typing import Sequence

from whole_body_tracking.data_process.build_dataset import build_compiled_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="whole_body_tracking.build_dataset")
    parser.add_argument("--dataset-root", required=True, help="A .npz file or directory of .npz clips.")
    parser.add_argument("--output-dir", required=True, help="Output compiled dataset directory.")
    parser.add_argument("--target-fps", type=int, default=50, help="Expected clip fps.")
    parser.add_argument(
        "--storage-dtype",
        choices=("float16", "float32"),
        default="float32",
        help="Floating point dtype used for stored arrays.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = build_compiled_dataset(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        target_fps=args.target_fps,
        storage_dtype=args.storage_dtype,
    )
    print(
        "BUILD OK "
        f"output_dir={summary.output_dir} "
        f"clips={summary.num_clips} "
        f"total_frames={summary.total_frames} "
        f"fps={summary.fps} "
        f"dtype={summary.storage_dtype}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
