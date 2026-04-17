# General Tracking Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite `whole_body_tracking` into an external `mjlab` task package with official registry-driven training, local compiled-dataset support, and local play/evaluate/export helpers.

**Architecture:** Replace the legacy private runtime with a plugin-style package registered through `mjlab.tasks`. Keep a thin data subsystem and a custom motion command term for compiled local datasets. Use the official `train` entrypoint, plus task-specific `play`, `evaluate`, and `export` helpers.

**Tech Stack:** Python 3.10-3.13, `uv`, `mjlab>=1.3,<1.4`, PyTorch, `pytest`, `ruff`, JSON/NumPy compiled datasets.

---

### Task 1: Replace repository contract and packaging

**Files:**
- Modify: `pyproject.toml`
- Modify: `AGENTS.md`
- Modify: `README.md`
- Create: `src/whole_body_tracking/_mjlab_tasks.py`
- Modify: `src/whole_body_tracking/__init__.py`

**Steps:**
1. Write failing tests asserting the new task registry bootstrap path and package metadata assumptions.
2. Update `pyproject.toml` to:
   - use `uv_build`
   - depend on `mjlab[cu128]>=1.3,<1.4`
   - define `tool.uv.index` / `tool.uv.sources`
   - expose `wbt-build-dataset`, `wbt-play`, `wbt-evaluate`, `wbt-export`
   - register `whole_body_tracking._mjlab_tasks` as the `mjlab.tasks` entry point
3. Make `whole_body_tracking/__init__.py` side-effect free.
4. Add `_mjlab_tasks.py` that imports the G1 task registration module.
5. Update `AGENTS.md` and `README.md` to describe the new downstream-task-package role.
6. Run targeted tests for import/registry behavior.

### Task 2: Rebuild the data and robot schema layer

**Files:**
- Create: `src/whole_body_tracking/data/*`
- Create: `src/whole_body_tracking/robots/g1/*`
- Delete or replace: legacy `data_process/`, `robot/`
- Modify: `tests/test_data_pipeline.py`
- Create: `tests/test_schema.py`

**Steps:**
1. Write failing tests for:
   - raw clip validation
   - compiled dataset build/load
   - G1 schema constants
2. Move G1 naming constants into `robots/g1/schema.py`.
3. Recreate the data subsystem under `data/`:
   - `schema.py`
   - `compiled_dataset.py`
   - `build_dataset.py`
4. Remove the standalone validate CLI contract from tests and docs.
5. Run the data-focused test subset until green.

### Task 3: Build the `general_tracking` task family

**Files:**
- Create: `src/whole_body_tracking/tasks/general_tracking/**`
- Delete or replace: legacy `tracking/`
- Modify: `tests/test_motion_source.py`
- Modify: `tests/test_tracking_command.py`
- Modify: `tests/test_tracking_runtime.py`
- Create: `tests/test_registry.py`
- Create: `tests/test_env_smoke.py`

**Steps:**
1. Write failing tests for:
   - `MotionSource` start/uniform/adaptive behavior
   - task registration for both task IDs
   - env config creation for standard and no-state-estimation variants
   - CPU env reset/step smoke
2. Implement `mdp/motion_source.py` and `mdp/commands.py`.
3. Implement `general_tracking_env_cfg.py` and `config/g1/{env_cfgs.py,rl_cfg.py,__init__.py}`.
4. Re-export or wrap official tracking `observations`, `rewards`, `terminations`, and `metrics` as needed.
5. Register:
   - `Mjlab-GeneralTracking-Flat-Unitree-G1`
   - `Mjlab-GeneralTracking-Flat-Unitree-G1-No-State-Estimation`
6. Run the runtime and registry test subset until green.

### Task 4: Replace the CLI/runtime layer with task-specific scripts

**Files:**
- Create: `src/whole_body_tracking/tasks/general_tracking/scripts/{play.py,evaluate.py,export.py,__init__.py}`
- Modify: `scripts/{build_dataset.sh,train.sh,play.sh,evaluate.sh,export.sh}`
- Delete: legacy `cli/`, `runtime/`
- Modify: `tests/test_runtime_cli.py`

**Steps:**
1. Write failing tests for:
   - `wbt-play` dataset injection ordering
   - repeated `--dataset-path` / `--dataset-weight` parsing
   - local evaluate/export smoke helpers
2. Implement a shared dataset-env injection helper using:
   - `WBT_COMPILED_DATASET_PATHS`
   - `WBT_COMPILED_DATASET_WEIGHTS`
3. Implement `wbt-play` as a thin delegator to `mjlab.scripts.play`.
4. Implement `wbt-evaluate` and `wbt-export` using local checkpoints and loaded env/task configs.
5. Update shell wrappers to call the new entrypoints and the official `uv run train`.
6. Remove old CLI/runtime modules and update tests to the new surface.

### Task 5: Full verification and cleanup

**Files:**
- Modify: any remaining failing tests or docs identified by verification

**Steps:**
1. Run targeted TDD cycles until the new suite is green.
2. Run `uv sync --group dev` after the final dependency changes.
3. Run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q`.
4. Run at least one real smoke command for:
   - dataset build
   - `uv run train ... --agent.max-iterations 1`
   - `uv run wbt-play ...`
5. Record actual command outputs in the final summary.
