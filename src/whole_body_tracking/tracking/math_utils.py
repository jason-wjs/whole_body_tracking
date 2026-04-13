"""Minimal quaternion and frame-transform math for tracking terms."""

from __future__ import annotations

import torch


def normalize_quat(quat: torch.Tensor) -> torch.Tensor:
    return quat / torch.clamp(torch.linalg.norm(quat, dim=-1, keepdim=True), min=1e-8)


def quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    out = quat.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def quat_mul(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    w0, x0, y0, z0 = q0.unbind(dim=-1)
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    return torch.stack(
        (
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ),
        dim=-1,
    )


def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    qvec = quat[..., 1:]
    uv = torch.cross(qvec, vec, dim=-1)
    uuv = torch.cross(qvec, uv, dim=-1)
    return vec + 2.0 * (quat[..., :1] * uv + uuv)


def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    return quat_apply(quat_conjugate(quat), vec)


def quat_error_magnitude(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    rel = normalize_quat(quat_mul(q0, quat_conjugate(q1)))
    real = torch.clamp(rel[..., 0].abs(), 0.0, 1.0)
    return 2.0 * torch.arccos(real)


def matrix_from_quat(quat: torch.Tensor) -> torch.Tensor:
    quat = normalize_quat(quat)
    w, x, y, z = quat.unbind(dim=-1)
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return torch.stack(
        (
            torch.stack((1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)), dim=-1),
            torch.stack((2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)), dim=-1),
            torch.stack((2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)), dim=-1),
        ),
        dim=-2,
    )


def subtract_frame_transforms(
    frame_pos_w: torch.Tensor,
    frame_quat_w: torch.Tensor,
    target_pos_w: torch.Tensor,
    target_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pos_b = quat_apply_inverse(frame_quat_w, target_pos_w - frame_pos_w)
    quat_b = normalize_quat(quat_mul(quat_conjugate(frame_quat_w), target_quat_w))
    return pos_b, quat_b
