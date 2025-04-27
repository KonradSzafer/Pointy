import torch
import torch.nn.functional as F


def furthest_point_sample(coords: torch.Tensor, s: int) -> torch.Tensor:
    """
    Furthest point sampling (FPS).

    Args:
        coords (torch.Tensor): Input point coordinates of shape (B, N, 3).
        s (int): Number of points to sample.

    Returns:
        torch.Tensor: Indices of the sampled points of shape (B, s), in long dtype.
    """
    B, N, _ = coords.shape
    # We will store the sampled point indices in the 'farthest_indices' tensor
    farthest_indices = torch.zeros((B, s), dtype=torch.long, device=coords.device)
    
    # distances[i, j] will hold the smallest distance of point j in batch i to any selected point
    distances = torch.ones((B, N), device=coords.device) * 1e10
    
    # We can start by picking a random initial point in each batch
    # You can also fix it to 0 or pick the point with the largest distance from the centroid, etc.
    farthest = torch.randint(low=0, high=N, size=(B,), device=coords.device)
    
    batch_range = torch.arange(B, dtype=torch.long, device=coords.device)
    
    for i in range(s):
        # Record the index of the farthest point chosen this round
        farthest_indices[:, i] = farthest
        
        # Gather the xyz coords for the newly selected farthest points
        centroid = coords[batch_range, farthest, :].unsqueeze(1)  # Shape: (B, 1, 3)
        
        # Compute squared distances from all points to this centroid
        # coords shape: (B, N, 3), centroid shape: (B, 1, 3)
        dist = torch.sum((coords - centroid) ** 2, dim=-1)  # (B, N)
        
        # Update the distances with the minimum distance to any of the sampled points
        mask = dist < distances
        distances[mask] = dist[mask]
        
        # Next farthest is the one which has the maximum distance to the selected set
        farthest = torch.argmax(distances, dim=1)
    
    return farthest_indices


def cal_loss(pred, ground_truth, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    ground_truth = ground_truth.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, ground_truth.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, ground_truth, reduction='mean')

    return loss


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]

    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Ball query.

    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    
    Output:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(k, xyz, new_xyz):
    """
    K nearest neighborhood.

    Input:
        k: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    
    Output:
        group_idx: grouped points index, [B, S, k]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, k, dim=-1, largest=False, sorted=False)
    return group_idx


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    
    Output:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def sample_and_ball_group(s, radius, n, coords, features):
    """
    Sampling by FPS and grouping by ball query.

    Input:
        s[int]: number of points to be sampled by FPS
        k[int]: number of points to be grouped into a neighbor by ball query
        n[int]: fix number of points in ball neighbor
        coords[tensor]: input points coordinates data with size of [B, N, 3]
        features[tensor]: input points features data with size of [B, N, D]
    
    Returns:
        new_coords[tensor]: sampled and grouped points coordinates by FPS with size of [B, s, k, 3]
        new_features[tensor]: sampled and grouped points features by FPS with size of [B, s, k, 2D]
    """
    batch_size = coords.shape[0]
    coords = coords.contiguous()

    # FPS sampling
    fps_idx = furthest_point_sample(coords, s).long()
    # fps_idx = pointnet2_utils.furthest_point_sample(coords, s).long()  # [B, s]
    new_coords = index_points(coords, fps_idx)                         # [B, s, 3]
    new_features = index_points(features, fps_idx)                     # [B, s, D]

    # ball_query grouping
    idx = query_ball_point(radius, n, coords, new_coords)              # [B, s, n]
    grouped_features = index_points(features, idx)                     # [B, s, n, D]
    
    # Matrix sub
    grouped_features_norm = grouped_features - new_features.view(batch_size, s, 1, -1)  # [B, s, n, D]

    # Concat, my be different in many networks
    aggregated_features = torch.cat([grouped_features_norm, new_features.view(batch_size, s, 1, -1).repeat(1, 1, n, 1)], dim=-1)  # [B, s, n, 2D]

    return new_coords, aggregated_features  # [B, s, 3], [B, s, n, 2D]


def sample_and_knn_group(s, k, coords, features):
    """
    Sampling by FPS and grouping by KNN.

    Input:
        s[int]: number of points to be sampled by FPS
        k[int]: number of points to be grouped into a neighbor by KNN
        coords[tensor]: input points coordinates data with size of [B, N, 3]
        features[tensor]: input points features data with size of [B, N, D]
    
    Returns:
        new_coords[tensor]: sampled and grouped points coordinates by FPS with size of [B, s, k, 3]
        new_features[tensor]: sampled and grouped points features by FPS with size of [B, s, k, 2D]
    """
    batch_size = coords.shape[0]
    coords = coords.contiguous()

    # FPS sampling
    fps_idx = furthest_point_sample(coords, s).long()
    # fps_idx = pointnet2_utils.furthest_point_sample(coords, s).long()  # [B, s]
    new_coords = index_points(coords, fps_idx)                         # [B, s, 3]
    new_features = index_points(features, fps_idx)                     # [B, s, D]

    # K-nn grouping
    idx = knn_point(k, coords, new_coords)                                              # [B, s, k]
    grouped_features = index_points(features, idx)                                      # [B, s, k, D]
    
    # Matrix sub
    grouped_features_norm = grouped_features - new_features.view(batch_size, s, 1, -1)  # [B, s, k, D]

    # Concat
    aggregated_features = torch.cat([grouped_features_norm, new_features.view(batch_size, s, 1, -1).repeat(1, 1, k, 1)], dim=-1)  # [B, s, k, 2D]

    return new_coords, aggregated_features  # [B, s, 3], [B, s, k, 2D]


class Logger():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


if __name__ == '__main__':
    points = torch.rand(32, 1024, 3).to('cuda')
    features = torch.rand(32, 1024, 128).to('cuda')
    new_points, new_features = sample_and_knn_group(512, 32, points, features)
    print(new_points.size())
    print(new_features.size())
