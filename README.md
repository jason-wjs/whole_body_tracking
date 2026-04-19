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

Project wrappers in `scripts/` currently hard-code a local `lafan1` workflow:

- raw dataset root: `/home/humanoid/Downloads/Data/G1_retargeted/lafan1_npz`
- compiled dataset dir: `/tmp/lafan1_compiled`
- experiment: `g1_general_tracking`
- run name pattern: `lafan1_g1_single_gpu`

Adjust those command arguments directly if your local paths or run naming differ, then run the
script as-is or append extra CLI flags after the script name.

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
