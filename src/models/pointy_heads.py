import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import OmegaConf


class Classification(nn.Module):
    def __init__(self, config: OmegaConf, embedding_dim: int):
        super(Classification, self).__init__()
        self.classifier = nn.Linear(embedding_dim, config.model.num_classes)
        
    def forward(self, x):
        x = x.squeeze(1)
        x = self.classifier(x)
        return x
