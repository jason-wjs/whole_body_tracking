# whole_body_tracking

`whole_body_tracking` is a downstream `mjlab` task package for G1 whole-body motion
tracking with local compiled datasets.

This repository owns:

- compiled dataset build/load logic
- G1-specific task registration
- local dataset-backed play, evaluation, and export helpers

It does not own a separate simulator fork or a private training framework.

## Installation

```bash
cd <repo_root>
uv sync --dev
```

## Main Workflows

Project wrappers in `scripts/` carry the repository's default G1/LAFAN1 presets. Override them
with environment variables such as `WBT_RAW_DATASET_ROOT`, `WBT_COMPILED_DATASET_DIR`,
`WBT_EXPERIMENT_NAME`, `WBT_RUN_NAME`, `WBT_NUM_ENVS`, and `WBT_MAX_ITERATIONS`, or append
extra CLI flags after the script name.

Build a compiled dataset:

```bash
./scripts/build_dataset.sh
```

Train with the official `mjlab` registry flow:

```bash
./scripts/train.sh
```

Play from a local checkpoint:

```bash
./scripts/play.sh
```

Evaluate a checkpoint locally:

```bash
./scripts/evaluate.sh
```

Export ONNX and TorchScript artifacts:

```bash
./scripts/export.sh
```

For one-off overrides, either adjust the environment variables or call the lower-level commands
directly:

```bash
uv run train Mjlab-GeneralTracking-Flat-Unitree-G1 \
  --env.commands.motion.dataset-paths "('<compiled_dataset_dir>',)" \
  --env.commands.motion.dataset-weights "(1.0,)" \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 30000
```

## Registered Tasks

- `Mjlab-GeneralTracking-Flat-Unitree-G1`
- `Mjlab-GeneralTracking-Flat-Unitree-G1-No-State-Estimation`

## Dataset Contract

Raw input clips are already retargeted G1 `.npz` files. The builder validates and
compiles them into a clip-preserving directory containing:

- `meta_motion.json`
- `id_label.json`
- `arrays/*.npy`
