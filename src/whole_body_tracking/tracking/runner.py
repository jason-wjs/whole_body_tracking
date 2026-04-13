"""Tracking-specific runner with export metadata."""

from __future__ import annotations

import os

from mjlab.rl.exporter_utils import (
    attach_metadata_to_onnx,
    get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner

from whole_body_tracking.tracking.command import MotionSourceCommand


class WholeBodyTrackingOnPolicyRunner(MjlabOnPolicyRunner):
    def _build_export_metadata(self) -> dict[str, list | str | float]:
        motion_term = self.env.unwrapped.command_manager.get_term("motion")
        assert isinstance(motion_term, MotionSourceCommand)
        run_name = self.cfg.get("run_name") or "local"
        metadata = get_base_metadata(self.env.unwrapped, run_name)
        metadata.update(
            {
                "anchor_body_name": motion_term.cfg.anchor_body_name,
                "body_names": list(motion_term.cfg.body_names),
                "dataset_paths": list(motion_term.cfg.dataset_paths),
                "dataset_weights": list(motion_term.cfg.dataset_weights),
                "sampling_mode": motion_term.cfg.sampling_mode,
            }
        )
        return metadata

    def export_policy_artifacts(
        self,
        path: str,
        *,
        onnx_filename: str = "policy.onnx",
        jit_filename: str = "policy.pt",
        verbose: bool = False,
    ) -> tuple[str, str]:
        os.makedirs(path, exist_ok=True)
        self.export_policy_to_onnx(path, onnx_filename, verbose=verbose)
        self.export_policy_to_jit(path, jit_filename)
        attach_metadata_to_onnx(
            os.path.join(path, onnx_filename),
            self._build_export_metadata(),
        )
        return os.path.join(path, onnx_filename), os.path.join(path, jit_filename)

    def save(self, path: str, infos=None) -> None:
        super().save(path, infos)
        policy_path = path.split("model")[0]
        filename_stem = os.path.basename(os.path.dirname(policy_path)) or "policy"
        try:
            onnx_path, jit_path = self.export_policy_artifacts(
                policy_path,
                onnx_filename=f"{filename_stem}.onnx",
                jit_filename=f"{filename_stem}.pt",
            )
            if self.logger.logger_type in ["wandb"] and self.cfg["upload_model"]:
                import wandb

                wandb.save(onnx_path, base_path=os.path.dirname(policy_path))
                wandb.save(jit_path, base_path=os.path.dirname(policy_path))
        except Exception as exc:  # pragma: no cover - export failure should not break training
            print(f"[WARN] Policy export failed (training continues): {exc}")
