"""Runtime patches for DSGN++ inference on L40S (sm_89).

CUDA 11.0 + PT 1.7.1 routes torch.inverse() through cuSOLVER, which fails on Ada
(cusolver error 7). Inference itself stays on GPU; only the 4x4 calib inverse is
replaced with Gauss-Jordan using basic tensor ops (no cuSOLVER, no CPU fallback).
"""
import torch

import pcdet.utils.torch_utils as tu


def _inverse_4x4_gpu(P: torch.Tensor) -> torch.Tensor:
    """Invert a 4x4 matrix on P's device without cuSOLVER."""
    P4x4 = torch.eye(4, dtype=P.dtype, device=P.device)
    P4x4[:3, :] = P
    n = 4
    aug = torch.cat([P4x4, torch.eye(n, dtype=P.dtype, device=P.device)], dim=1)
    for col in range(n):
        pivot = aug[col, col]
        if pivot.abs() < 1e-12:
            swap_row = col + aug[col:, col].abs().argmax().item()
            aug[[col, swap_row]] = aug[[swap_row, col]]
            pivot = aug[col, col]
        aug[col] = aug[col] / pivot
        rows = torch.arange(n, device=P.device) != col
        aug[rows] = aug[rows] - aug[col] * aug[rows, col].unsqueeze(1)
    return aug[:, n:]


def unproject_image_to_rect(pts_image, P):
    pts_3d = torch.cat([pts_image[..., :2], torch.ones_like(pts_image[..., 2:3])], -1)
    pts_3d = pts_3d * pts_image[..., 2:3]
    pts_3d = torch.cat([pts_3d, torch.ones_like(pts_3d[..., 2:3])], -1)
    invP = _inverse_4x4_gpu(P)
    pts_3d = torch.matmul(pts_3d, torch.transpose(invP, 0, 1))
    return pts_3d[..., :3]


def unproject_image_to_pseudo_lidar(pts_image, P):
    pts_3d = torch.cat([pts_image[..., :2], torch.ones_like(pts_image[..., 2:3])], -1)
    pts_3d = pts_3d * pts_image[..., 2:3]
    pts_3d = torch.cat([pts_3d, torch.ones_like(pts_3d[..., 2:3])], -1)
    invP = _inverse_4x4_gpu(P)
    pts_3d = torch.matmul(pts_3d, torch.transpose(invP, 0, 1))
    pts_3d = pts_3d[..., [2, 0, 1]] * torch.as_tensor(
        [1.0, -1.0, -1.0], device=pts_3d.device, dtype=pts_3d.dtype
    )
    return pts_3d


tu.unproject_image_to_rect = unproject_image_to_rect
tu.unproject_image_to_pseudo_lidar = unproject_image_to_pseudo_lidar
