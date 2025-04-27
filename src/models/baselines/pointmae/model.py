# Source: https://github.com/Pang-Yatian/Point-MAE/blob/main/models/Point_MAE.py

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_

from src.models.baselines.pointmae import misc
from src.models.baselines.pointmae.logger import *


class KNNModule(torch.nn.Module):
    def __init__(self, group_size):
        super(KNNModule, self).__init__()
        self.group_size = group_size

    def knn(self, xyz, center, k):
        """
        Args:
            xyz:    Tensor of shape [B, N, 3] -- the entire point cloud.
            center: Tensor of shape [B, npoint, 3] -- query points.
            k:      Number of nearest neighbors to return.
            
        Returns:
            A tuple (dists, idx) where:
              - dists: Tensor of shape [B, npoint, k] containing the squared distances.
              - idx:   Tensor of shape [B, npoint, k] containing the indices of the k nearest points.
        """
        B, N, _ = xyz.shape
        _, npoint, _ = center.shape

        # Expand dimensions to enable broadcasting:
        #   center becomes [B, npoint, 1, 3]
        #   xyz becomes [B, 1, N, 3]
        center_expanded = center.unsqueeze(2)  # [B, npoint, 1, 3]
        xyz_expanded = xyz.unsqueeze(1)        # [B, 1, N, 3]

        # Compute squared Euclidean distance between each query point and all points:
        # Resulting shape: [B, npoint, N]
        dist_squared = torch.sum((center_expanded - xyz_expanded) ** 2, dim=-1)

        # For each query point, find the indices of the k smallest distances.
        # Note: largest=False gives us the smallest values.
        dists, idx = torch.topk(dist_squared, k=k, dim=-1, largest=False, sorted=False)

        return dists, idx

    def forward(self, xyz, center):
        # Call the knn function with k = self.group_size.
        _, idx = self.knn(xyz, center, k=self.group_size)
        return idx


# -------------------------------------------------------------------------
# Grouping utilities remain the same (FPS + KNN)
# -------------------------------------------------------------------------
class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        # self.knn = KNN(k=self.group_size, transpose_mode=True)
        self.knn = KNNModule(group_size=self.group_size)

    def forward(self, xyz):
        '''
            input xyz: B N 3
            output: neighborhood: B G M 3
                    center:       B G 3
        '''
        xyz = xyz.permute(0, 2, 1)[:, :, :3]  # B 3 N
        batch_size, num_points, _ = xyz.shape
        # fps to pick "center" points
        center = misc.fps(xyz, self.num_group)  # B G 3
        
        # KNN to find group neighborhoods
        idx = self.knn(xyz, center)  # B G M
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1)*num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.reshape(batch_size*num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, 
                                         self.group_size, 3).contiguous()
        # Normalize each group by subtracting its center
        neighborhood = neighborhood - center.unsqueeze(2)
        
        return neighborhood, center

# -------------------------------------------------------------------------
# The same original Embedding (Encoder) for tokens
# -------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            returns       : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs*g, n, 3)          # (B*G, N, 3)
        feature = self.first_conv(point_groups.transpose(2,1))   # (B*G, 256, N)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]   # (B*G, 256, 1)
        feature = torch.cat([feature_global.expand(-1, -1, n), 
                             feature], dim=1)  # (B*G, 512, N)
        feature = self.second_conv(feature)     # (B*G, encoder_channel, N)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # (B*G, encoder_channel)
        return feature_global.reshape(bs, g, self.encoder_channel)

# -------------------------------------------------------------------------
# Attention + Transformer Blocks
# -------------------------------------------------------------------------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, 
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C//self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, heads, N, c_per_head
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, 
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads,
                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., 
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., 
                 drop_path_rate=0.):
        super().__init__()
        dpr = ([x.item() for x in torch.linspace(0, drop_path_rate, depth)]
               if isinstance(drop_path_rate, float) else drop_path_rate)
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[i] if isinstance(dpr, list) else dpr)
            for i in range(depth)
        ])

    def forward(self, x, pos):
        """
        x:   [B, G, C]
        pos: [B, G, C] (positional embedding)
        """
        # simple addition of positional embedding, then block pass
        for block in self.blocks:
            x = block(x + pos)
        return x

# -------------------------------------------------------------------------
# A "Pure" Transformer-Backbone Model without Masking
# -------------------------------------------------------------------------
class PointMAE(nn.Module):
    """
    Uses the same grouping & transformer blocks as Point-MAE,
    but **no masking** and **no decoder**. 
    """
    def __init__(self, num_classes):
        super().__init__()
        
        trans_dim = 384
        depth = 12
        drop_path_rate = 0.1
        num_heads = 6
        group_size = 32
        num_group = 64
        encoder_dims = 384
        
        # 1) Grouping
        self.group_divider = Group(num_group=num_group, group_size=group_size)
        # 2) Local Encoder (point-wise MLP)
        self.encoder_dims = encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        # 3) Positional Embedding
        self.trans_dim = trans_dim
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        # 4) Transformer
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=depth,
            drop_path_rate=drop_path_rate,
            num_heads=num_heads
        )
        self.norm = nn.LayerNorm(self.trans_dim)
        
        self.cls_head = nn.Linear(self.trans_dim, num_classes)
        
    def encode(self, pts):
        """
        pts: [B, N, 3]  raw pointcloud
        Returns: global tokens of shape [B, G, C] 
                 which are the pure, final transformer features.
        """
        # (i) Group the input pointcloud
        neighborhood, center = self.group_divider(pts)  # [B, G, M, 3], [B, G, 3]

        # (ii) Encode each group to [B, G, encoder_dims]
        x = self.encoder(neighborhood)

        # (iii) Positional Embedding (based on group center)
        pos = self.pos_embed(center)  # [B, G, trans_dim]

        # (iv) Pass through transformer blocks
        x = self.blocks(x, pos)

        # (v) Final norm 
        x = self.norm(x)  # [B, G, trans_dim]

        # You could either return x for a downstream head 
        # (e.g. classification/segmentation) or do further MLP
        
        x = x.transpose(1,2)         # [B, trans_dim, G]
        x = x[..., -1]
        return x

    def forward(self, pts):
        x = self.encode(pts)
        x = self.cls_head(x)
        return x
