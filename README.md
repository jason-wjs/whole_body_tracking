# whole_body_tracking

`whole_body_tracking` is a dataset-driven training project for the BeyondMimic whole-body tracking algorithm, adapted for local dataset workflows on top of `mjlab`. The original project is [HybridRobotics/whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking).

This repository focuses on:

- training on local motion datasets rather than a single W&B motion artifact
- compiling many G1-ready motion clips into reusable datasets
- running tracking training, playback, evaluation, and export.

## Input Data

This repository expects motion clips that are already retargeted and ready for G1 tracking. It does not perform retargeting or motion reconstruction.

Each input `.npz` clip is expected to contain the motion fields needed by the tracking task, including joint state, body pose, body orientation, and body velocity tensors.

## Main Interfaces

The stable user-facing entrypoints are the Python CLIs:

- `whole_body_tracking.cli.validate_dataset`
- `whole_body_tracking.cli.build_dataset`
- `whole_body_tracking.cli.train`
- `whole_body_tracking.cli.play`
- `whole_body_tracking.cli.evaluate`
- `whole_body_tracking.cli.export`

The shell scripts in `scripts/` are convenience launchers. For portable and reproducible usage, prefer the CLI forms below.

## Dependencies And Installation

This project is managed with `uv`.

Prerequisites:

- Python `>=3.10,<3.14`
- `uv`

Install runtime dependencies:

```bash
cd <repo_root>
uv sync
```

Install development dependencies as well:

```bash
cd <repo_root>
uv sync --extra dev
```

After installation, you can either:

- run commands through `uv run --project <repo_root> ...`
- or activate the project virtual environment and invoke the CLIs directly

## Common Workflows

### 1. Validate raw clips

Validate one clip:

```bash
uv run --project <repo_root> python -m whole_body_tracking.cli.validate_dataset \
  <path_to_motion_clip.npz>
```

Validate a directory of clips:

```bash
uv run --project <repo_root> python -m whole_body_tracking.cli.validate_dataset \
  <path_to_raw_dataset_dir>
```

### 2. Build a compiled dataset

```bash
uv run --project <repo_root> python -m whole_body_tracking.cli.build_dataset \
  --dataset-root <path_to_raw_dataset_dir_or_single_clip> \
  --output-dir <path_to_compiled_dataset_dir> \
  --target-fps <target_fps> \
  --storage-dtype <float16_or_float32>
```

On success, the builder prints a summary like:

```text
BUILD OK output_dir=<...> clips=<...> total_frames=<...> fps=<...> dtype=<...>
```

### 3. Train

```bash
uv run --project <repo_root> python -m whole_body_tracking.cli.train \
  --dataset-path <path_to_compiled_dataset_dir> \
  --device <device> \
  --num-envs <num_envs> \
  --max-iterations <max_iterations> \
  --logger wandb \
  --wandb-project <wandb_project> \
  --wandb-tag <wandb_tag> \
  --experiment-name <experiment_name> \
  --run-name <run_name>
```

### 4. Play

Play from an explicitly chosen checkpoint:

```bash
uv run --project <repo_root> python -m whole_body_tracking.cli.play \
  --dataset-path <path_to_compiled_dataset_dir> \
  --checkpoint-file <path_to_checkpoint.pt> \
  --device <device> \
  --num-envs <num_envs> \
  --viewer <auto_native_or_viser>
```


### 5. Evaluate

Evaluate runs the tracking task in finite-episode mode and reports task metrics. If `--num-steps` is omitted, evaluation runs until all parallel episodes finish, matching the original `mjlab` stopping rule.

```bash
uv run --project <repo_root> python -m whole_body_tracking.cli.evaluate \
  --dataset-path <path_to_compiled_dataset_dir> \
  --checkpoint-file <path_to_checkpoint.pt> \
  --device <device> \
  --num-envs <num_parallel_eval_episodes> \
  --output-file <path_to_metrics.json>
```

Reported metrics:

- `success_rate`
- `mpkpe`
- `r_mpkpe`
- `joint_vel_error`
- `ee_pos_error`
- `ee_ori_error`

### 6. Export

Export is optional because training already auto-exports artifacts at checkpoint save time. The explicit export CLI is mainly for re-exporting a selected checkpoint.

Export from an explicit checkpoint:

```bash
uv run --project <repo_root> python -m whole_body_tracking.cli.export \
  --dataset-path <path_to_compiled_dataset_dir> \
  --checkpoint-file <path_to_checkpoint.pt> \
  --output-dir <path_to_export_dir> \
  --device <device>
```

Export writes:

- `policy.onnx`
- `policy.pt`

## Checkpoint Resolution

`play`, `evaluate`, and `export` support two checkpoint selection modes:

- explicit: pass `--checkpoint-file <path>`
- local resolution: pass `--experiment-name`, `--load-run`, and optionally `--load-checkpoint`

The local resolution mode searches under:

```text
logs/rsl_rl/<experiment_name>/<run_dir>/model_*.pt
```

It selects the newest run directory matching `--load-run`, then the newest checkpoint file matching `--load-checkpoint`.

## Outputs

Typical training outputs are written under:

```text
logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/
```

These directories may contain:

- `model_*.pt` checkpoints
- TensorBoard or W&B logging artifacts
- auto-exported ONNX and TorchScript policy files
