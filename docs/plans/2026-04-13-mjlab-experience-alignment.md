# MJLab Experience Alignment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align `Sparse/whole_body_tracking` user-facing train/play/evaluate/export workflows with the practical runtime experience of `mjlab` tracking while keeping this project independent.

**Architecture:** Reuse the simulator-backed env, local checkpoint resolution, and tracking-specific runner already present in this repo. Add missing evaluation and metrics entrypoints, extend shell wrappers, and update docs so the primary workflows mirror `mjlab`'s script-level ergonomics without reintroducing W&B motion artifact coupling.

**Tech Stack:** Python 3.13, official `mjlab==1.2.0`, `rsl-rl-lib==5.0.1`, PyTorch, shell wrappers, pytest.

---

### Task 1: Add evaluation entrypoints

**Files:**
- Create: `src/whole_body_tracking/tracking/metrics.py`
- Create: `src/whole_body_tracking/cli/evaluate.py`
- Create: `scripts/evaluate.sh`
- Modify: `pyproject.toml`
- Modify: `tests/test_phase0_skeleton.py`
- Modify: `tests/test_runtime_cli.py`
- Modify: `tests/test_tracking_runtime.py`

**Step 1: Write the failing tests**

- Add skeleton/import expectations for `metrics.py`, `evaluate.py`, and `evaluate.sh`.
- Add runtime smoke tests for `evaluate` with a zero policy and local checkpoint resolution.

**Step 2: Run tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --project . --with pytest pytest -q tests/test_tracking_runtime.py tests/test_phase0_skeleton.py tests/test_runtime_cli.py`

Expected: FAIL because the new files and CLI entrypoint do not exist yet.

**Step 3: Write minimal implementation**

- Port `mjlab` tracking metric formulas to `tracking/metrics.py`.
- Implement `cli/evaluate.py` using local checkpoint resolution and headless rollout.
- Add `evaluate.sh` and the corresponding `project.scripts` entry.

**Step 4: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --project . --with pytest pytest -q tests/test_tracking_runtime.py tests/test_phase0_skeleton.py tests/test_runtime_cli.py`

Expected: PASS

### Task 2: Verify end-to-end local workflow

**Files:**
- Modify: `README.md`

**Step 1: Write the failing test**

- Add or update smoke checks only if needed; otherwise use command verification for workflow.

**Step 2: Run verification commands**

Run:
- `./scripts/train.sh --dataset-path /tmp/wbt_script_smoke_compiled --device cpu --num-envs 1 --max-iterations 1 --logger tensorboard --run-name eval_smoke`
- `./scripts/play.sh --dataset-path /tmp/wbt_script_smoke_compiled --agent trained --device cpu --num-envs 1 --num-steps 2 --experiment-name g1_tracking --load-run '.*eval_smoke'`
- `./scripts/evaluate.sh --dataset-path /tmp/wbt_script_smoke_compiled --agent trained --device cpu --num-envs 1 --experiment-name g1_tracking --load-run '.*eval_smoke' --num-steps 2`
- `./scripts/export.sh --dataset-path /tmp/wbt_script_smoke_compiled --experiment-name g1_tracking --load-run '.*eval_smoke' --output-dir /tmp/wbt_export_eval_smoke`

Expected: all commands exit 0 and produce checkpoint / evaluation / export outputs.

**Step 3: Update docs**

- Document train/play/evaluate/export and local checkpoint lookup in `README.md`.

**Step 4: Run full test suite**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --project . --with pytest pytest -q tests`

Expected: PASS
