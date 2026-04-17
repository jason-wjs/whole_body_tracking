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

Build a compiled dataset:

```bash
uv run wbt-build-dataset \
  --dataset-root <raw_npz_dir_or_single_clip> \
  --output-dir <compiled_dataset_dir>
```

Train with the official `mjlab` registry flow:

```bash
uv run train Mjlab-GeneralTracking-Flat-Unitree-G1 \
  --env.commands.motion.dataset-paths "('<compiled_dataset_dir>',)" \
  --env.commands.motion.dataset-weights "(1.0,)" \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 30000
```

Play from a local checkpoint:

```bash
uv run wbt-play Mjlab-GeneralTracking-Flat-Unitree-G1 \
  --dataset-path <compiled_dataset_dir> \
  --checkpoint-file <checkpoint.pt> \
  --viewer viser
```

Evaluate a checkpoint locally:

```bash
uv run wbt-evaluate Mjlab-GeneralTracking-Flat-Unitree-G1 \
  --dataset-path <compiled_dataset_dir> \
  --checkpoint-file <checkpoint.pt> \
  --num-envs 1024 \
  --output-file <metrics.json>
```

Export ONNX and TorchScript artifacts:

```bash
uv run wbt-export Mjlab-GeneralTracking-Flat-Unitree-G1 \
  --dataset-path <compiled_dataset_dir> \
  --checkpoint-file <checkpoint.pt> \
  --output-dir <export_dir>
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
