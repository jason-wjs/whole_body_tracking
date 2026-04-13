# AGENTS.md

## Repository Role

- This repository is a standalone project located under `Sparse/`, but it is not part of `sparse_control`.
- README content is user-facing. Local workspace notes, implementation constraints, and author-only guidance belong here instead.

## Runtime Dependency Rules

- Runtime simulator dependency is official PyPI `mjlab==1.2.0` only.
- Local `controller/mjlab` is reference-only.
- Never add local `mjlab` as a path dependency.
- Never modify local `mjlab` to make this project work.
- Treat `mjlab` as a simulator/runtime backend, not as the task or training framework for this repository.

## Project Scope

- This repository targets dataset training for the BeyondMimic whole-body tracking algorithm.
- Input motions are already G1-ready `.npz` clips.
- The project does not perform retargeting, motion reconstruction, or schema-heavy conversion.

## Documentation Rules

- README should describe the repository for users, not for authors.
- Common workflow commands in README should use placeholders rather than hard-coded local paths or numeric presets.
- Author-facing local launch details can live in shell scripts and this file, not in README.

## Script Conventions

- `scripts/` may contain locally convenient hard-coded launchers.
- The stable user-facing interfaces are the Python CLIs under `src/whole_body_tracking/cli/`.
- When updating both CLI and scripts, keep CLI semantics authoritative and let scripts remain thin convenience wrappers.
