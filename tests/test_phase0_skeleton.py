from __future__ import annotations

import importlib
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


EXPECTED_DIRS = [
    ROOT / "docs" / "plans",
    ROOT / "scripts",
    ROOT / "tests",
    SRC / "whole_body_tracking",
    SRC / "whole_body_tracking" / "cli",
    SRC / "whole_body_tracking" / "data_process",
    SRC / "whole_body_tracking" / "tracking",
    SRC / "whole_body_tracking" / "robot",
]


EXPECTED_FILES = [
    ROOT / "pyproject.toml",
    ROOT / "README.md",
    ROOT / "scripts" / "build_dataset.sh",
    ROOT / "scripts" / "evaluate.sh",
    ROOT / "scripts" / "validate_dataset.sh",
    ROOT / "scripts" / "train.sh",
    ROOT / "scripts" / "play.sh",
    ROOT / "scripts" / "export.sh",
    SRC / "whole_body_tracking" / "__init__.py",
    SRC / "whole_body_tracking" / "cli" / "__init__.py",
    SRC / "whole_body_tracking" / "cli" / "build_dataset.py",
    SRC / "whole_body_tracking" / "cli" / "evaluate.py",
    SRC / "whole_body_tracking" / "cli" / "export.py",
    SRC / "whole_body_tracking" / "cli" / "validate_dataset.py",
    SRC / "whole_body_tracking" / "cli" / "train.py",
    SRC / "whole_body_tracking" / "cli" / "play.py",
    SRC / "whole_body_tracking" / "data_process" / "__init__.py",
    SRC / "whole_body_tracking" / "data_process" / "schema.py",
    SRC / "whole_body_tracking" / "data_process" / "compiled_dataset.py",
    SRC / "whole_body_tracking" / "data_process" / "build_dataset.py",
    SRC / "whole_body_tracking" / "robot" / "__init__.py",
    SRC / "whole_body_tracking" / "robot" / "g1.py",
    SRC / "whole_body_tracking" / "runtime" / "__init__.py",
    SRC / "whole_body_tracking" / "runtime" / "mjlab_guard.py",
    SRC / "whole_body_tracking" / "tracking" / "__init__.py",
    SRC / "whole_body_tracking" / "tracking" / "checkpoints.py",
    SRC / "whole_body_tracking" / "tracking" / "command.py",
    SRC / "whole_body_tracking" / "tracking" / "env.py",
    SRC / "whole_body_tracking" / "tracking" / "env_cfg.py",
    SRC / "whole_body_tracking" / "tracking" / "math_utils.py",
    SRC / "whole_body_tracking" / "tracking" / "metrics.py",
    SRC / "whole_body_tracking" / "tracking" / "motion_source.py",
    SRC / "whole_body_tracking" / "tracking" / "observations.py",
    SRC / "whole_body_tracking" / "tracking" / "rl_cfg.py",
    SRC / "whole_body_tracking" / "tracking" / "runner.py",
    SRC / "whole_body_tracking" / "tracking" / "rewards.py",
    SRC / "whole_body_tracking" / "tracking" / "terminations.py",
]


IMPORT_MODULES = [
    "whole_body_tracking",
    "whole_body_tracking.cli.build_dataset",
    "whole_body_tracking.cli.evaluate",
    "whole_body_tracking.cli.export",
    "whole_body_tracking.cli.validate_dataset",
    "whole_body_tracking.data_process.schema",
    "whole_body_tracking.data_process.compiled_dataset",
    "whole_body_tracking.data_process.build_dataset",
    "whole_body_tracking.robot.g1",
    "whole_body_tracking.runtime.mjlab_guard",
    "whole_body_tracking.tracking.checkpoints",
    "whole_body_tracking.tracking.command",
    "whole_body_tracking.tracking.env",
    "whole_body_tracking.tracking.env_cfg",
    "whole_body_tracking.tracking.math_utils",
    "whole_body_tracking.tracking.metrics",
    "whole_body_tracking.tracking.motion_source",
    "whole_body_tracking.tracking.observations",
    "whole_body_tracking.tracking.rl_cfg",
    "whole_body_tracking.tracking.runner",
    "whole_body_tracking.tracking.rewards",
    "whole_body_tracking.tracking.terminations",
]


def test_expected_directories_exist() -> None:
    missing = [str(path) for path in EXPECTED_DIRS if not path.is_dir()]
    assert not missing, f"Missing directories: {missing}"


def test_expected_files_exist() -> None:
    missing = [str(path) for path in EXPECTED_FILES if not path.is_file()]
    assert not missing, f"Missing files: {missing}"


def test_shell_scripts_have_valid_bash_syntax() -> None:
    scripts = [
        ROOT / "scripts" / "build_dataset.sh",
        ROOT / "scripts" / "evaluate.sh",
        ROOT / "scripts" / "validate_dataset.sh",
        ROOT / "scripts" / "train.sh",
        ROOT / "scripts" / "play.sh",
        ROOT / "scripts" / "export.sh",
    ]
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_shell_scripts_are_self_contained_uv_wrappers() -> None:
    hardcoded_scripts = {
        ROOT / "scripts" / "validate_dataset.sh": [
            'python -m whole_body_tracking.cli.validate_dataset',
            "/home/humanoid/Downloads/Data/G1_retargeted/lafan1_npz",
        ],
        ROOT / "scripts" / "build_dataset.sh": [
            'python -m whole_body_tracking.cli.build_dataset',
            "--dataset-root /home/humanoid/Downloads/Data/G1_retargeted/lafan1_npz",
            "--output-dir /tmp/lafan1_compiled",
        ],
        ROOT / "scripts" / "train.sh": [
            'python -m whole_body_tracking.cli.train',
            "--logger wandb",
            "--wandb-project whole_body_tracking",
            "--run-name lafan1_g1_single_gpu",
            "--dataset-path /tmp/lafan1_compiled",
            "--device cuda:0",
            "--num-envs 4096",
            "--max-iterations 30000",
        ],
        ROOT / "scripts" / "play.sh": [
            'python -m whole_body_tracking.cli.play',
            "--dataset-path /tmp/lafan1_compiled",
            "--device cuda:0",
            "--num-envs 4",
            "--experiment-name g1_tracking",
            "--load-run '.*lafan1_g1_single_gpu'",
        ],
        ROOT / "scripts" / "evaluate.sh": [
            'python -m whole_body_tracking.cli.evaluate',
            "--dataset-path /tmp/lafan1_compiled",
            "--agent trained",
            "--device cuda:0",
            "--experiment-name g1_tracking",
            "--load-run '.*lafan1_g1_single_gpu'",
            "--output-file /tmp/wbt_eval/metrics.json",
        ],
        ROOT / "scripts" / "export.sh": [
            'python -m whole_body_tracking.cli.export',
            "--dataset-path /tmp/lafan1_compiled",
            "--experiment-name g1_tracking",
            "--load-run '.*lafan1_g1_single_gpu'",
            "--output-dir /tmp/wbt_export",
        ],
    }
    for script, expected_fragments in hardcoded_scripts.items():
        content = script.read_text(encoding="utf-8")
        assert "common.sh" not in content
        assert 'REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"' in content
        for fragment in expected_fragments:
            assert fragment in content


def test_pyproject_runtime_dependencies_match_official_mjlab() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]
    assert "mjlab==1.2.0" in dependencies
    assert "rsl-rl-lib==5.0.1" in dependencies


def test_core_modules_import() -> None:
    sys.path.insert(0, str(SRC))
    try:
        for module_name in IMPORT_MODULES:
            importlib.import_module(module_name)
    finally:
        sys.path.remove(str(SRC))
