from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import mjlab.tasks.registry as registry
import tomllib


def test_pyproject_declares_mjlab_task_entrypoint() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    value = pyproject["project"]["entry-points"]["mjlab.tasks"]["whole_body_tracking"]
    assert value == "whole_body_tracking._mjlab_tasks"


def test_root_package_import_has_no_registration_side_effect() -> None:
    code = (
        "import mjlab.tasks.registry as registry; "
        "registry._REGISTRY.clear(); "
        "import whole_body_tracking; "
        "print(registry.list_tasks())"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.stdout.strip() == "[]"


def test_general_tracking_tasks_register_with_mjlab_registry() -> None:
    registry._REGISTRY.clear()
    from whole_body_tracking.tasks.general_tracking.config import g1 as g1_config

    importlib.reload(g1_config)

    tasks = registry.list_tasks()
    assert "Mjlab-GeneralTracking-Flat-Unitree-G1" in tasks
    assert "Mjlab-GeneralTracking-Flat-Unitree-G1-No-State-Estimation" in tasks


def test_general_tracking_rl_cfg_uses_whole_body_tracking_wandb_project() -> None:
    registry._REGISTRY.clear()
    from whole_body_tracking.tasks.general_tracking.config import g1 as g1_config

    importlib.reload(g1_config)

    rl_cfg = registry.load_rl_cfg("Mjlab-GeneralTracking-Flat-Unitree-G1")
    assert rl_cfg.wandb_project == "whole_body_tracking"
