# History

This file consolidates the historical planning notes that used to live under
`docs/plans/`. It is a record of how the repository architecture moved toward
the current downstream `mjlab` task-package shape; it is not an active
implementation plan.

## 2026-04-13: MJLab Experience Alignment

The first planning pass focused on making the user-facing workflow feel closer
to `mjlab` tracking while keeping this repository independent.

The intended workflow surface was:

- train through the standard `mjlab` runtime conventions
- play with local checkpoint resolution
- evaluate with tracking metrics
- export trained policies locally

The plan also introduced the idea that this repository should avoid coupling to
`mjlab`'s W&B motion artifact path. Local motion data and local checkpoints were
treated as first-class runtime inputs.

Important outcomes from this planning phase:

- add `evaluate` and `export` entrypoints
- keep shell wrappers as thin conveniences
- use local checkpoint lookup for play/evaluate/export
- document train/play/evaluate/export as the core workflow

## 2026-04-18: General Tracking Refactor Design

The second planning pass reframed the repository as a downstream `mjlab` task
package under `Sparse/`.

The role boundary was:

- `mjlab` provides the framework, simulator-backed environment, registry, and
  core tracking semantics
- `whole_body_tracking` provides the G1 tracking task registration, compiled
  dataset handling, dataset-backed motion command, and local helper entrypoints

The repository was explicitly scoped to:

- Unitree G1 only
- whole-body motion tracking only
- local compiled datasets only
- no retargeting or motion reconstruction

The registration model was designed around a side-effect-free root package and a
single `mjlab.tasks` entry point:

```text
whole_body_tracking._mjlab_tasks
```

That module imports the G1 task registration package, which performs the actual
`register_mjlab_task(...)` calls.

The phase-one task IDs were:

- `Mjlab-GeneralTracking-Flat-Unitree-G1`
- `Mjlab-GeneralTracking-Flat-Unitree-G1-No-State-Estimation`

The compiled dataset contract was standardized as a directory containing:

- `meta_motion.json`
- `id_label.json`
- `arrays/joint_pos.npy`
- `arrays/joint_vel.npy`
- `arrays/body_pos_w.npy`
- `arrays/body_quat_w.npy`
- `arrays/body_lin_vel_w.npy`
- `arrays/body_ang_vel_w.npy`

The required metadata fields were:

- `fps`
- `joint_names`
- `body_names`
- `clip_frame_starts`
- `clip_num_frames`
- `clip_weights`

The design draft still mentioned a `robots/g1/schema.py` package and a separate
`MotionSource` helper. Both were later superseded by the final baseline
architecture: G1 schema constants live in `whole_body_tracking.data.g1_schema`,
`robots/` is removed, and dataset loading/sampling is owned by
`MultiMotionCommand`.

## 2026-04-18: General Tracking Refactor Implementation Plan

The implementation plan split the refactor into five work areas.

### Packaging And Registry

The package was to move to an official `uv`/`mjlab` dependency model, expose only
stable project-specific entrypoints, and register tasks through the
`mjlab.tasks` entry-point group.

Stable project entrypoints:

- `wbt-build-dataset`
- `wbt-play`
- `wbt-evaluate`
- `wbt-export`

Training intentionally uses the official `mjlab` flow:

```bash
uv run train <task-id>
```

### Data And G1 Schema

The data layer was rebuilt around raw G1-ready `.npz` clips and compiled
clip-preserving datasets. The repository validates and compiles local data; it
does not retarget, reconstruct, or reindex motions.

The final baseline keeps G1 naming and tracked-body constants in:

```text
src/whole_body_tracking/data/g1_schema.py
```

### General Tracking Task Family

The task family was rebuilt under:

```text
src/whole_body_tracking/tasks/general_tracking/
```

The task config reuses `mjlab` tracking environment composition where practical
and swaps the motion command term for the repository's local compiled-dataset
command.

The final command surface is:

- `MultiMotionCommandCfg`
- `MultiMotionCommand`

It supports:

- one or more compiled datasets
- dataset and clip weighting
- start, uniform, and adaptive sampling
- G1 anchor and tracked-body subsets
- local play/evaluate/export workflows

### Runtime Scripts

The old private CLI/runtime layer was replaced by task-specific scripts and thin
shell wrappers. These wrappers are conveniences around the stable entrypoints;
they are not a parallel training interface.

### Verification

The refactor plan required unit tests for registry behavior, data compilation,
schema constants, command sampling, runtime CLI wrappers, and CPU environment
smoke coverage. It also required at least one real local smoke run through
dataset build and minimal training.

## Current Baseline Outcome

The current baseline keeps the downstream package narrow:

- G1 only
- general whole-body tracking only
- local compiled datasets only
- no `robots/` package
- no local `controller/mjlab` path dependency
- no README dependency on machine-local dataset paths

The current registered repository task IDs are still:

- `Mjlab-GeneralTracking-Flat-Unitree-G1`
- `Mjlab-GeneralTracking-Flat-Unitree-G1-No-State-Estimation`

Those IDs now use `MultiMotionCommandCfg` internally for compiled multi-motion
dataset training. The upstream `mjlab` task IDs
`Mjlab-Tracking-Flat-Unitree-G1` and
`Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation` remain separate upstream
single-`motion_file` tracking tasks.

One naming decision remains worth revisiting: whether the repository should add
new `WBT-*` task IDs so dataset-backed downstream tasks are visually distinct
from upstream `mjlab` task names.
