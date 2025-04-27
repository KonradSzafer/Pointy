# Some of the augmentations inspired from:
# https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/provider.py

import torch


def random_rotate_pointcloud(
    pointcloud: torch.Tensor,
    axis: str = "z",
    angle_range: tuple = (-180, 180),
    has_colors: bool = False,
) -> torch.Tensor:
    """
    Randomly rotate a pointcloud around one of the axes (x, y, z)
    """
    xyz = pointcloud[:3, :]
    features = pointcloud[3:, :] if pointcloud.shape[0] > 3 else None

    # Generate a random rotation angle within the specified degree range
    min_angle, max_angle = angle_range
    angle_deg = torch.rand(1) * (max_angle - min_angle) + min_angle
    angle_rad = angle_deg * (torch.pi / 180.0)
    cos_angle = torch.cos(angle_rad)
    sin_angle = torch.sin(angle_rad)

    # Rotation matrix
    if axis == 'x':
        rotation_matrix = torch.tensor([
            [1, 0, 0],
            [0, cos_angle, -sin_angle],
            [0, sin_angle, cos_angle]
        ]).squeeze()
    elif axis == 'y':
        rotation_matrix = torch.tensor([
            [cos_angle, 0, sin_angle],
            [0, 1, 0],
            [-sin_angle, 0, cos_angle]
        ]).squeeze()
    elif axis == 'z':
        rotation_matrix = torch.tensor([
            [cos_angle, -sin_angle, 0],
            [sin_angle, cos_angle, 0],
            [0, 0, 1]
        ]).squeeze()
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")

    # Rotate the point cloud
    xyz = torch.matmul(rotation_matrix, xyz)
    
    if has_colors:
        return torch.cat((xyz, features), dim=0)
    
    if features is None:
        return xyz

    # Rotate the normals
    features = torch.matmul(rotation_matrix, features)
    return torch.cat((xyz, features), dim=0)


def jitter_pointcloud(
    pointcloud,
    sigma: float = 0.01,
    clip: float = 0.05,
    has_normals: bool = False
) -> torch.Tensor:
    """
    Randomly jitter points in a point cloud by adding Gaussian noise.
    """
    noise = sigma * torch.randn_like(pointcloud)
    jittered_noise = torch.clamp(noise, -clip, clip)

    if has_normals:
        # The same noise for xyz and normals to keep the normals orthogonal to the point cloud
        pointcloud = pointcloud[:3, :] + jittered_noise[:3, :]
        features = pointcloud[3:, :] + jittered_noise[:3, :]
        return torch.cat((pointcloud, features), dim=0)

    return pointcloud + jittered_noise


def random_scale_pointcloud(
    pointcloud: torch.Tensor,
    scale_range: tuple = (0.6, 1.0),
    has_colors: bool = False
) -> torch.Tensor:
    """
    Randomly scales a point cloud by a factor sampled uniformly from the specified range.
    """
    # Sample a random scale factor
    scale = torch.empty(1).uniform_(scale_range[0], scale_range[1]).item()
    
    # Scale only the xyz coordinates
    if has_colors:
        xyz = pointcloud[:3, :] * scale
        features = pointcloud[3:, :]
        return torch.cat((xyz, features), dim=0)

    # Scale the entire point cloud with normals
    return pointcloud * scale
