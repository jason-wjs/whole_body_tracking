"""Protect this project from accidentally importing the local workspace mjlab source tree."""

from __future__ import annotations

from pathlib import Path


def _workspace_controller_root() -> Path:
    return Path(__file__).resolve().parents[5]


def is_disallowed_workspace_mjlab_path(path: str | None) -> bool:
    if not path:
        return False
    raw = Path(path)
    try:
        resolved = raw.resolve()
    except FileNotFoundError:
        resolved = raw
    disallowed_root = (_workspace_controller_root() / "mjlab").resolve()
    try:
        resolved.relative_to(disallowed_root)
        return True
    except ValueError:
        return False


def assert_official_mjlab_import() -> object:
    import mjlab  # type: ignore

    module_path = getattr(mjlab, "__file__", None)
    if is_disallowed_workspace_mjlab_path(module_path):
        raise RuntimeError(
            "Refusing to use workspace-local mjlab source tree. "
            "Install and use official PyPI mjlab==1.2.0 instead."
        )
    return mjlab
