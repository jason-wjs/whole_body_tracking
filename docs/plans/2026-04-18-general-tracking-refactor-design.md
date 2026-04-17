# General Tracking Refactor Design

## Summary

`Sparse/whole_body_tracking` will be rewritten as a downstream `mjlab` task package.
It remains G1-only and whole-body-motion-tracking-only, but it stops owning a private
train/play runtime. Training will move to `mjlab`'s task registry flow, while local
dataset compilation, local dataset-backed play, evaluation, and export remain in this
repository.

Phase 1 ships two task IDs:

- `Mjlab-GeneralTracking-Flat-Unitree-G1`
- `Mjlab-GeneralTracking-Flat-Unitree-G1-No-State-Estimation`

## Repository Role

- `mjlab` is the framework dependency, installed through `uv`.
- `whole_body_tracking` is an external task package registered through
  `[project.entry-points."mjlab.tasks"]`.
- The repository owns only:
  - G1 tracking task registration and config
  - compiled dataset build/load logic
  - dataset-backed command sampling
  - local play/evaluate/export helpers

It does not own a separate RL framework, a simulator fork, or a custom long-term CLI
surface for training.

## Package Layout

The new package layout is:

```text
src/whole_body_tracking/
  __init__.py
  _mjlab_tasks.py
  data/
    __init__.py
    schema.py
    compiled_dataset.py
    build_dataset.py
  robots/
    __init__.py
    g1/
      __init__.py
      schema.py
  tasks/
    __init__.py
    general_tracking/
      __init__.py
      general_tracking_env_cfg.py
      mdp/
        __init__.py
        commands.py
        motion_source.py
        observations.py
        rewards.py
        terminations.py
        metrics.py
      config/
        __init__.py
        g1/
          __init__.py
          env_cfgs.py
          rl_cfg.py
      scripts/
        __init__.py
        play.py
        evaluate.py
        export.py
```

Deleted subsystems:

- `cli/`
- `runtime/`
- `robot/`
- legacy `tracking/`
- standalone dataset validate entrypoint

## Registration Model

`whole_body_tracking/__init__.py` must stay side-effect free.

Task registration moves to `whole_body_tracking/_mjlab_tasks.py`, and the package entry
point targets that module directly. This avoids the import-order bug where `wbt-play`
needs to set dataset environment variables before `mjlab` loads registered tasks.

`_mjlab_tasks.py` imports `whole_body_tracking.tasks.general_tracking.config.g1`, and
that module is the only place that calls `register_mjlab_task(...)`.

## Data Contract

Compiled datasets remain directory-based and clip-preserving.

Required structure:

- `meta_motion.json`
- `id_label.json`
- `arrays/joint_pos.npy`
- `arrays/joint_vel.npy`
- `arrays/body_pos_w.npy`
- `arrays/body_quat_w.npy`
- `arrays/body_lin_vel_w.npy`
- `arrays/body_ang_vel_w.npy`

`meta_motion.json` is the runtime contract and must include:

- `fps`
- `joint_names`
- `body_names`
- `clip_frame_starts`
- `clip_num_frames`
- `clip_weights`

`data/build_dataset.py` validates raw `.npz` clips during build. There is no standalone
validate CLI in the new layout.

`robots/g1/schema.py` is the single source of truth for G1 joint/body naming and anchor
semantics. The data layer and the task layer both consume it; it depends on neither.

## Runtime Flow

### Train

Training uses the official `mjlab` command:

```bash
uv run train Mjlab-GeneralTracking-Flat-Unitree-G1 \
  --env.commands.motion.dataset-paths "('/abs/dataset',)" \
  --env.commands.motion.dataset-weights "(1.0,)" \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 1
```

No `wbt-train` wrapper exists.

### Play

`mjlab`'s current `play` CLI cannot override nested `env.commands.motion.*` fields for a
custom command term. Therefore Phase 1 keeps a thin `wbt-play` wrapper that:

1. extracts repeated `--dataset-path` and optional repeated `--dataset-weight`
2. writes them into environment variables
3. delegates to `mjlab.scripts.play.main()`

The registered play env config reads:

- `WBT_COMPILED_DATASET_PATHS`
- `WBT_COMPILED_DATASET_WEIGHTS`

Both use JSON array strings.

### Evaluate / Export

Phase 1 keeps `wbt-evaluate` and `wbt-export` as task-specific scripts.

- `wbt-evaluate` uses local dataset paths and local checkpoints.
- `wbt-export` performs explicit export from a loaded checkpoint.
- Phase 1 does not restore training-time auto export or a custom runner.

## Task Composition

`GeneralTrackingCommandCfg` does not inherit `MotionCommandCfg`.

That is intentional. The task should not enter `mjlab`'s built-in tracking-specific
W&B motion artifact path. Instead it owns a local compiled-dataset command contract:

- `dataset_paths`
- `dataset_weights`
- `sampling_mode`
- adaptive sampling parameters
- `anchor_body_name`
- `body_names`

`MotionSource` remains a separate runtime helper in `mdp/motion_source.py`.
It handles:

- multi-dataset selection
- per-clip frame sampling
- adaptive failure-biased sampling
- reference frame retrieval

Observation, reward, termination, and metric logic stay aligned with official tracking
semantics. Phase 1 may re-export or thinly wrap official `mjlab.tasks.tracking.mdp`
helpers where interfaces already match the custom command term.

## Phase Boundaries

### Phase 1

- move to `mjlab` task registry architecture
- upgrade dependency model to `mjlab[cu128]>=1.3,<1.4`
- ship both G1 task IDs
- keep compiled dataset support
- keep `wbt-play`, `wbt-evaluate`, `wbt-export`
- remove legacy private runtime layout
- pass full local test suite

### Phase 2

Optional follow-up work:

- restore custom runner behavior
- auto-export ONNX/JIT during training
- attach task-specific ONNX metadata
- expand export metadata schema

