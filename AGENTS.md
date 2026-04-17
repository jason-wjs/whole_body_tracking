# AGENTS.md

## Repository Role

- This repository is a downstream `mjlab` task package under `Sparse/`.
- It is not a fork of `mjlab` and should not depend on `controller/mjlab` by path.

## Dependency Rules

- Depend on official `mjlab` through `uv`.
- Prefer PyPI releases; only use pinned upstream git sources when PyPI is missing a
  required feature.
- Never modify the local `controller/mjlab` checkout to make this repository work.

## Scope

- G1 only
- whole-body motion tracking only
- local compiled datasets only

This repository does not perform retargeting or motion reconstruction.

## Interface Rules

- Training uses the official `uv run train <task-id>` flow.
- The stable project-specific interfaces are:
  - `wbt-build-dataset`
  - `wbt-play`
  - `wbt-evaluate`
  - `wbt-export`
- `scripts/` are thin wrappers around those entrypoints.
