from itertools import product

import torch

from src.utils import setup_logger

logger = setup_logger(__name__)


def resample_pointcloud_random(
    pointcloud: torch.Tensor, num_points: int
) -> torch.Tensor:
    """ Resample a pointcloud randomly """
    N = pointcloud.size(1)
    indices = torch.randperm(N)[:num_points]
    return pointcloud[:, indices]


def resample_pointcloud_farthest(pointcloud: torch.Tensor, num_points: int) -> torch.Tensor:
    """
    Input:
        pointcloud (torch.Tensor): [6, N] - only the first 3 dimensions are used
        nponum_pointsint (int): number of samples
    Return:
        torch.Tensor: [6, num_points] - resampled pointcloud
    """
    _, N = pointcloud.shape
    
    if N < num_points:
        raise ValueError(
            f"Number of points in the pointcloud is less than the desired number of points: {N} < {num_points}"
        )
    elif N == num_points:
        return pointcloud
    
    centroids = torch.zeros(num_points, dtype=torch.long)
    distance = torch.ones(N) * 1e10
    farthest = torch.randint(0, N, (1,), dtype=torch.long)

    xyz = pointcloud[:3, :]  # Select only the first 3 dimensions [3, N]
    for i in range(num_points):
        centroids[i] = farthest  # Store the farthest point index
        centroid = xyz[:, farthest].view(
            3, 1
        )  # Get the coordinates of the farthest point
        dist = torch.sum(
            (xyz - centroid) ** 2, 0
        )  # Calculate squared Euclidean distance
        distance = torch.min(distance, dist)  # Update the minimum distance
        farthest = torch.max(distance, 0)[1]  # Find the next farthest point

    return pointcloud[:, centroids]


def normalize_pointcloud(pointcloud: torch.Tensor) -> torch.Tensor:
    """
    Normalize a pointcloud to fit within the range [-1, 1] along the largest dimension.
    """
    xyz = pointcloud[:3, :]
    if pointcloud.shape[0] == 6:
        colors = pointcloud[3:, :]
    else:
        colors = None

    # Compute min and max along each axis
    min_xyz = xyz.min(dim=1, keepdim=True).values
    max_xyz = xyz.max(dim=1, keepdim=True).values

    # Compute the center of the pointcloud
    center = (min_xyz + max_xyz) / 2.0
    xyz_centered = xyz - center

    # Compute the ranges along each axis
    ranges = (max_xyz - min_xyz).squeeze()

    # Find the maximum range among x, y, z
    max_range = ranges.max()

    # Avoid division by zero
    if max_range == 0:
        max_range = 1.0

    # Scale the xyz coordinates to fit within [-1, 1]
    xyz_normalized = xyz_centered / (max_range / 2.0)

    # If colors exist, normalize them to [0,1] if necessary
    if colors is not None:
        if colors.max() > 1:
            colors = colors / 255.0
        normalized_pointcloud = torch.cat([xyz_normalized, colors], dim=0)
    else:
        normalized_pointcloud = xyz_normalized

    return normalized_pointcloud


def sort_pointcloud_by_distance(pointcloud: torch.Tensor) -> torch.Tensor:
    """
    Sort points in a pointcloud based on their Euclidean distance from origin using PyTorch.
    """
    xyz = pointcloud[:3, :]
    distances = torch.sqrt(torch.sum(xyz * xyz, dim=0))
    sort_indices = torch.argsort(distances)
    sorted_pointcloud = pointcloud[:, sort_indices]
    return sorted_pointcloud


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """ Calculate squared Euclidean distance between each pair of points. """
    # src/dst shapes: [3, N] / [3, M]
    # Transposed to [N, 3], [M, 3] for easier broadcasting
    src_t = src.t()  # [N, 3]
    dst_t = dst.t()  # [M, 3]
    dist = (src_t.unsqueeze(1) - dst_t.unsqueeze(0)).pow(2).sum(-1)
    return dist


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Iterative Farthest Point Sampling.
    Returns the indices of the sampled points.
    """
    N = xyz.shape[1]
    
    distances = torch.ones(N, device=xyz.device) * 1e10
    idxs = torch.zeros(npoint, dtype=torch.long, device=xyz.device)

    # Initialize with a random points
    farthest = torch.randint(0, N, (1,), device=xyz.device).long()
    idxs[0] = farthest
    centroid = xyz[:, farthest].view(3)

    for i in range(1, npoint):
        # Compute distance
        dist = (xyz[0] - xyz[0, farthest]).pow(2) \
             + (xyz[1] - xyz[1, farthest]).pow(2) \
             + (xyz[2] - xyz[2, farthest]).pow(2)
        
        mask = dist < distances
        distances[mask] = dist[mask]

        farthest = torch.max(distances, dim=-1)[1]
        idxs[i] = farthest

    return idxs


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """ Equivalent to points[:, idx] but works with PyTorch JIT """
    return points.gather(1, idx.view(1, -1).expand(points.shape[0], -1))


def knn_query(x: torch.Tensor, anchor_idx: int, k: int) -> torch.Tensor:
    """
    Find k nearest neighbors of x[:, anchor_idx] in x (using the first xyz dims for distance).
    """
    xyz = x[:3, :]           # (3, N)
    anchor_xyz = xyz[:, anchor_idx].unsqueeze(1)  # (3, 1)
    dist = square_distance(anchor_xyz, xyz)        # (1, N)
    # topk() with k neighbors
    knn_idx = torch.topk(-dist[0], k, largest=True)[1]  # negative dist so largest => smallest real dist
    # Gather the actual points
    patch_points = index_points(x, knn_idx)        # (6, k)
    return patch_points


def create_patches_fps_knn(x: torch.Tensor, n_patches: int = 64, k: int = 32) -> torch.Tensor:
    """
    Create geometry-aware patches by:
      1) Farthest Point Sampling to pick `n_patches` anchors
      2) kNN around each anchor to form local patches

    Args:
        x: (6, N) input point cloud
        n_patches: number of patches/tokens (like 64)
        k: number of points in each patch

    Returns:
        patches: (n_patches, 6, k)  
    """
    assert x.dim() == 2 and x.shape[0] == 6, \
        "Input must be [6, N] shape, i.e. 6D for each of N points."

    xyz = x[:3, :]  # just the 3D coords for sampling
    
    # 1. Farthest-Point Sample
    anchor_indices = farthest_point_sample(xyz, n_patches)  # (n_patches,)

    # 2. For each anchor, get k nearest neighbors
    all_patches = []
    for i in range(n_patches):
        anchor_idx = anchor_indices[i].item()
        patch_points = knn_query(x, anchor_idx, k)
        all_patches.append(patch_points.unsqueeze(0))  # shape [1, 6, k]

    # Shape [n_patches, 6, k]
    patches = torch.cat(all_patches, dim=0)

    return patches
