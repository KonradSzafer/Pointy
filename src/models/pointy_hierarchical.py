import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.hidden_dim = int(dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, dim),
            nn.Dropout(dropout),
        )

        self._init_weights()

    def _init_weights(self):
        # bias is default to 0
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                # nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        # [batch_size, seq_len, dim]
        x_norm1 = self.norm1(x)
        attn_output, _ = self.attn(x_norm1, x_norm1, x_norm1)
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x


class HierarchicalTransformer(nn.Module):
    def __init__(
        self, 
        config,
        mlp_ratio: int = 4.0,
        dropout: int = 0.1,
    ):
        super(HierarchicalTransformer, self).__init__()
        self.embedding_dim = config.model.embed_dim
        self.num_heads = config.model.num_heads
        self.merge_ratios = config.model.merge_ratios
        self.depth_stages = config.model.depth_stages
        assert len(self.merge_ratios) == len(self.depth_stages), \
            "Number of merge ratios must match number of depth stages and vice versa."

        # Main Hierarchical Transformer
        self.stages = nn.ModuleList()
        for merge_ratio, depth_stage in zip(self.merge_ratios, self.depth_stages):
            self.stages.append(nn.ModuleList([
                TransformerBlock(self.embedding_dim, self.num_heads, mlp_ratio, dropout) 
                for _ in range(depth_stage)
            ]))

        self.norm = nn.LayerNorm(self.embedding_dim)

    @staticmethod
    def merge_tokens(x, merge_ratio):
        BS, n_tokens, dim = x.size()
        x = x.view(BS, n_tokens // merge_ratio, merge_ratio, dim).sum(dim=2)
        return x

    def forward(self, x):
        # Hierarchical Transformer
        for merge_ratio, stage in zip(self.merge_ratios, self.stages):  
            for block in stage:
                x = block(x)

            # Token merging
            x = self.merge_tokens(x, merge_ratio)

        # Final normalization
        x = self.norm(x)
        return x


def calculate_params(model: nn.Module) -> int:
    """ Calculates the number of trainable parameters in a model. """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    encoder = HierarchicalTransformer(
        None,
        embed_dim=768, 
        num_heads=12,
        mlp_ratio=4.0, 
        dropout=0.1, 
    )
    
    x = torch.randn(8, 64, 768)  # 64 patches
    encoded = encoder(x)
    print(f"Output shape: {encoded.shape}")

    print(f"Number of parameters: {calculate_params(encoder)/1e6:.2f}M")
