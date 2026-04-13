from __future__ import annotations

from pathlib import Path

from whole_body_tracking.cli.build_dataset import main as build_dataset_main
from whole_body_tracking.cli.evaluate import main as evaluate_main
from whole_body_tracking.cli.play import main as play_main
from whole_body_tracking.cli.train import main as train_main
from whole_body_tracking.runtime.mjlab_guard import is_disallowed_workspace_mjlab_path


LAFAN1_ROOT = Path("/home/humanoid/Downloads/Data/G1_retargeted/lafan1_npz")


def _build_dataset(tmp_path: Path, name: str) -> Path:
    output_dir = tmp_path / name
    rc = build_dataset_main(
        [
            "--dataset-root",
            str(LAFAN1_ROOT),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert rc == 0
    return output_dir


def test_is_disallowed_workspace_mjlab_path() -> None:
    assert is_disallowed_workspace_mjlab_path(
        "/home/humanoid/Projects/Junsong_WU/learning/locomotion/controller/mjlab/src/mjlab/__init__.py"
    )
    assert not is_disallowed_workspace_mjlab_path(
        "/tmp/fake-site-packages/mjlab/__init__.py"
    )


def test_shell_wrappers_do_not_reference_sparse_control() -> None:
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    for name in ("build_dataset.sh", "validate_dataset.sh", "train.sh", "play.sh", "evaluate.sh", "export.sh"):
        content = (scripts_dir / name).read_text(encoding="utf-8")
        assert "sparse_control" not in content


def test_train_cli_dry_run_smoke(tmp_path: Path) -> None:
    dataset_dir = _build_dataset(tmp_path, "train_compiled")
    rc = train_main(
        [
            "--dataset-path",
            str(dataset_dir),
            "--dataset-weight",
            "1.0",
            "--num-envs",
            "8",
            "--dry-run",
        ]
    )
    assert rc == 0


def test_play_cli_dry_run_smoke(tmp_path: Path) -> None:
    dataset_dir = _build_dataset(tmp_path, "play_compiled")
    rc = play_main(
        [
            "--dataset-path",
            str(dataset_dir),
            "--dry-run",
        ]
    )
    assert rc == 0


def test_evaluate_cli_dry_run_smoke(tmp_path: Path) -> None:
    dataset_dir = _build_dataset(tmp_path, "evaluate_compiled")
    output_file = tmp_path / "metrics.json"
    rc = evaluate_main(
        [
            "--dataset-path",
            str(dataset_dir),
            "--agent",
            "zero",
            "--device",
            "cpu",
            "--num-envs",
            "1",
            "--num-steps",
            "2",
            "--output-file",
            str(output_file),
        ]
    )
    assert rc == 0
    assert output_file.is_file()
