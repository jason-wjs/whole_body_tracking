from __future__ import annotations

import json
from pathlib import Path

from whole_body_tracking.tasks.general_tracking.scripts import play as play_script


def test_play_cli_extracts_dataset_args_and_delegates(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_delegate(argv: list[str]) -> int:
        captured["argv"] = argv
        captured["paths"] = json.loads(str(play_script.os.environ["WBT_COMPILED_DATASET_PATHS"]))
        captured["weights"] = json.loads(str(play_script.os.environ["WBT_COMPILED_DATASET_WEIGHTS"]))
        return 0

    monkeypatch.delenv("WBT_COMPILED_DATASET_PATHS", raising=False)
    monkeypatch.delenv("WBT_COMPILED_DATASET_WEIGHTS", raising=False)
    monkeypatch.setattr(play_script, "_delegate_to_mjlab_play", _fake_delegate)

    rc = play_script.main(
        [
            "Mjlab-GeneralTracking-Flat-Unitree-G1",
            "--dataset-path",
            "/tmp/ds_a",
            "--dataset-path=/tmp/ds_b",
            "--dataset-weight",
            "1.0",
            "--dataset-weight=3.0",
            "--viewer",
            "viser",
        ]
    )

    assert rc == 0
    assert captured["argv"] == [
        "Mjlab-GeneralTracking-Flat-Unitree-G1",
        "--viewer",
        "viser",
    ]
    assert captured["paths"] == ["/tmp/ds_a", "/tmp/ds_b"]
    assert captured["weights"] == [1.0, 3.0]


def test_shell_wrappers_use_new_entrypoints() -> None:
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    build_dataset_sh = (scripts_dir / "build_dataset.sh").read_text(encoding="utf-8")
    train_sh = (scripts_dir / "train.sh").read_text(encoding="utf-8")
    play_sh = (scripts_dir / "play.sh").read_text(encoding="utf-8")
    evaluate_sh = (scripts_dir / "evaluate.sh").read_text(encoding="utf-8")
    export_sh = (scripts_dir / "export.sh").read_text(encoding="utf-8")
    assert "WBT_RAW_DATASET_ROOT" in build_dataset_sh
    assert "WBT_COMPILED_DATASET_DIR" in build_dataset_sh
    assert "--dataset-root" in build_dataset_sh
    assert "--output-dir" in build_dataset_sh
    assert "uv run --project" in train_sh
    assert " train " in train_sh
    assert "Mjlab-GeneralTracking-Flat-Unitree-G1" in train_sh
    assert "--env.commands.motion.dataset-paths" in train_sh
    assert "--agent.experiment-name" in train_sh
    assert "--agent.run-name" in train_sh
    assert "WBT_MAX_ITERATIONS" in train_sh
    assert "uv run --project" in play_sh
    assert " wbt-play " in play_sh
    assert "WBT_COMPILED_DATASET_DIR" in play_sh
    assert "--experiment-name" in play_sh
    assert "--load-run" in play_sh
    assert "uv run --project" in evaluate_sh
    assert " wbt-evaluate " in evaluate_sh
    assert "WBT_EVAL_OUTPUT_FILE" in evaluate_sh
    assert "--output-file" in evaluate_sh
    assert "uv run --project" in export_sh
    assert " wbt-export " in export_sh
    assert "WBT_EXPORT_DIR" in export_sh
    assert "--output-dir" in export_sh


def test_play_cli_dispatches_headless_mode_locally(monkeypatch) -> None:
    called: dict[str, object] = {"mode": None}

    def _fake_delegate(_argv: list[str]) -> int:
      called["mode"] = "delegate"
      return 0

    def _fake_headless(_argv: list[str]) -> int:
      called["mode"] = "headless"
      return 0

    monkeypatch.setattr(play_script, "_delegate_to_mjlab_play", _fake_delegate)
    monkeypatch.setattr(play_script, "_run_headless_play", _fake_headless)

    rc = play_script.main(
        [
            "Mjlab-GeneralTracking-Flat-Unitree-G1",
            "--dataset-path",
            "/tmp/ds_a",
            "--viewer",
            "none",
            "--num-steps",
            "2",
        ]
    )

    assert rc == 0
    assert called["mode"] == "headless"
