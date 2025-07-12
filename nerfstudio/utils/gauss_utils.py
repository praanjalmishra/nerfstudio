# Util functions for 3D Gaussians 

import numpy as np
import torch

def rot2quat(rotmat):
    """
    Rotation matrix to quaternion conversion

    Args:
        rotmat (B, 3, 3): rotation matrices

    Returns:
        quat (B, 4): quaternions in the order of wxyz
    """
    assert rotmat.shape[1:] == (3, 3), "Rotation mat must be (B, 3, 3)"
    # Extract the elements of the rotation matrix
    r11, r12, r13 = rotmat[:, 0, 0], rotmat[:, 0, 1], rotmat[:, 0, 2]
    r21, r22, r23 = rotmat[:, 1, 0], rotmat[:, 1, 1], rotmat[:, 1, 2]
    r31, r32, r33 = rotmat[:, 2, 0], rotmat[:, 2, 1], rotmat[:, 2, 2]
    # Compute the trace of the rotation matrix
    trace = r11 + r22 + r33
    # Using numerical stability trick to avoid division by zero
    def safe_sqrt(x):
        return torch.sqrt(torch.clamp(x, min=1e-10))
    # Quaternion calculation
    w = 0.5 * safe_sqrt(1 + trace)
    x = 0.5 * safe_sqrt(1 + r11 - r22 - r33) * torch.sign(r32 - r23)
    y = 0.5 * safe_sqrt(1 - r11 + r22 - r33) * torch.sign(r13 - r31)
    z = 0.5 * safe_sqrt(1 - r11 - r22 + r33) * torch.sign(r21 - r12)
    quat = torch.stack([w, x, y, z], dim=-1)
    return quat

def transform_gaussians(pose, means, quats):
    """
    Transform the means and quats of 3D Gaussians by a 6D pose

    Args:
        pose (4, 4): pose
        means (N, 3): Gaussian means
        quats (N, 4): Gaussian quaternions

    Returns:
        means_new (N, 3): transformed Gaussian means
        quats_new (N, 4): transformed Gaussian quaternions
    """
    assert pose.shape == (4, 4)
    assert means.shape[0] == quats.shape[0]
    if pose.device != means.device:
        pose = pose.to(means.device)
    means_new = (pose[:3, :3] @ means.T + pose[:3, 3:]).T
    # Rotate the Gaussians
    quats = quats / quats.norm(dim=-1, keepdim=True)
    rots = quat_to_rotmat(quats)
    obj_rots_new = pose[:3, :3] @ rots
    quats_new = rot2quat(obj_rots_new)
    return means_new, quats_new


def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert a quaternion to a rotation matrix.

    Args:
        quat (B, 4): Quaternions in (w, x, y, z) order

    Returns:
        rotmat (B, 3, 3): Corresponding rotation matrices
    """
    assert quat.shape[-1] == 4, "Quaternion must have shape (B, 4)"
    quat = quat / quat.norm(dim=-1, keepdim=True)

    w, x, y, z = quat.unbind(-1)

    B = quat.shape[0]
    rotmat = torch.empty((B, 3, 3), dtype=quat.dtype, device=quat.device)

    rotmat[:, 0, 0] = 1 - 2 * (y ** 2 + z ** 2)
    rotmat[:, 0, 1] = 2 * (x * y - z * w)
    rotmat[:, 0, 2] = 2 * (x * z + y * w)

    rotmat[:, 1, 0] = 2 * (x * y + z * w)
    rotmat[:, 1, 1] = 1 - 2 * (x ** 2 + z ** 2)
    rotmat[:, 1, 2] = 2 * (y * z - x * w)

    rotmat[:, 2, 0] = 2 * (x * z - y * w)
    rotmat[:, 2, 1] = 2 * (y * z + x * w)
    rotmat[:, 2, 2] = 1 - 2 * (x ** 2 + y ** 2)

    return rotmat
