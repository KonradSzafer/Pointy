import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, config: OmegaConf, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.num_layers = config.model.num_layers
        self.embedding_dim = config.model.embed_dim
        self.num_heads = config.model.num_heads
        self.layers = nn.ModuleList([
            TransformerBlock(self.embedding_dim, self.num_heads, mlp_ratio, dropout) 
            for _ in range(self.num_layers)
        ])
        self.norm = nn.LayerNorm(self.embedding_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = x[:, -1]
        return x


if __name__ == "__main__":
    config = OmegaConf.create({
        "model": {
            "embed_dim": 768,
            "num_heads": 12,
            "num_layers": 2
        }
    })
    encoder = TransformerEncoder(config)
    x = torch.randn(64, 768)
    output = encoder(x)
    print(output.shape)
