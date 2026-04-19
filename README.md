# whole_body_tracking

`whole_body_tracking` is a downstream `mjlab` task package for G1 general motion
tracking with local compiled datasets.

## Baseline Branch Positioning

The `baseline` branch is positioned as the baseline for future general motion tracking
development. It keeps the original BeyondMimic algorithmic foundation from the official
[HybridRobotics/whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking)
repository, while adding support for training a general motion tracker on datasets.

This branch is intended to serve as the reference starting point for subsequent branches
that continue expanding general motion tracking capability.

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

For one-off overrides, either edit the hard-coded script arguments or call the lower-level commands
directly:

```bash
uv run train Mjlab-GeneralTracking-Flat-Unitree-G1 \
  --env.commands.motion.dataset-paths "('/tmp/lafan1_compiled',)" \
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
