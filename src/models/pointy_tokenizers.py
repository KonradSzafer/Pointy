import math

import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import OmegaConf

from src.models.baselines.pointnet import (
    PointNetSetAbstraction, PointNetSetAbstractionMsg
)


class Linear(nn.Module):
    def __init__(self, config: OmegaConf, embedding_dim: int, seq_len: int):
        super(Linear, self).__init__()
        self.config = config
        self.input_size = int(6 * config.model.num_points / seq_len)
        self.linear = nn.Linear(self.input_size, embedding_dim)

    def forward(self, x):
        x = x[:, :, torch.randperm(self.config.model.num_points)]
        # [bs, 6, num_points] -> [bs, 6 * num_points]
        x = x.view(x.size(0), -1)
        x = x.view(x.size(0), -1, self.input_size)
        return self.linear(x)


class PointNetTinySkipConnection(nn.Module):
    def __init__(
        self,
        config: OmegaConf,
        embedding_dim: int
    ):
        super(PointNetTinySkipConnection, self).__init__()
        self.config = config
        self.num_points = self.config.model.num_points_per_patch
        self.normal_channel = self.config.model.normal_channel
        self.embedding_dim = embedding_dim

        self.in_channel = 3 if self.normal_channel else 0
        self.flattened_input_size = (6 if self.normal_channel else 3) * self.num_points
        
        # First set abstraction layer - reduced number of points and features
        self.sa1 = PointNetSetAbstractionMsg(
            256,
            [0.1, 0.2],          # Number of radius scales
            [16, 32],            # Number of samples
            self.in_channel,
            [[32, 32], [32, 64]] # MLPs
        )
        
        # Second set abstraction layer - reduced complexity
        self.sa2 = PointNetSetAbstractionMsg(
            64,
            [0.2, 0.4],
            [32, 64],
            96,                  # Input channels (32+64 from previous layer)
            [[64, 96], [64, 96]] # MLPs
        )
        
        # Final set abstraction layer
        self.sa3 = PointNetSetAbstraction(
            None, None, None, 
            192 + 3,                        # Input channels
            [256, 512, self.embedding_dim], # Final MLPs
            True
        )

        # Skip connection
        self.skip_connection = nn.Linear(
            self.flattened_input_size,
            self.embedding_dim,
            bias=False,
        )
        nn.init.kaiming_normal_(self.skip_connection.weight)
        # nn.init.xavier_uniform_(self.skip_connection.weight)

    def forward(self, pointcloud):
        # Batch size is: batch_size x seq_len, as we process each patch separately
        B, _, _ = pointcloud.shape
        if self.normal_channel:
            xyz = pointcloud[:, :3, :]
            norm = pointcloud[:, 3:, :]
        else:
            pointcloud = pointcloud[:, :3, :]
            xyz = pointcloud
            norm = None

        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        x = l3_points.view(B, self.embedding_dim) # Reshape to get final embedding

        # Skip connection
        x_skip = self.skip_connection(pointcloud.view(B, -1))
        x = x + x_skip

        return x


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device(x.get_device())

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


class PointNetEmbedding(nn.Module):
    def __init__(self, config: OmegaConf, embedding_dim: int):
        super(PointNetEmbedding, self).__init__()
        self.config = config
        self.seq_len = config.model.num_patches

        # EMBEDDING LAYER
        # self.embedding_layer = nn.Linear(6 * config.model.num_points_per_patch, embedding_dim)
        self.embedding_layer = PointNetTinySkipConnection(config, embedding_dim)

        # POSITIONAL EMBEDDING
        self.learnable_positional_embedding = nn.Parameter(torch.zeros(1, self.seq_len, embedding_dim))


    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor of shape [bs, seq_len, num_dims, num_points]
        """

        batch_size, seq_len, num_dims, num_points = x.shape

        # [bs, seq_len, num_dims, num_points] -> [bs, seq_len, num_dims * num_points]
        x_bsp = x.view(batch_size, seq_len, -1)

        # POINTNET EMBEDDING LAYER
        # Reshape to process each patch separately
        # [bs, seq_len, num_dims, num_points] -> [bs * seq_len, num_points, num_dims]
        x_input = x.view(-1, num_dims, num_points)
        x_patch_emb = self.embedding_layer(x_input)

        # Reshape back to original shape
        # [bs * seq_len, embedding_dim] -> [bs, seq_len, embedding_dim]
        x_patch_emb = x_patch_emb.view(batch_size, seq_len, -1)

        # POSITIONAL EMBEDDING
        x_patch_emb += self.learnable_positional_embedding 
        
        return x_patch_emb
