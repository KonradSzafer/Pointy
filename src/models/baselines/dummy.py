import torch
from torch import nn
from omegaconf import OmegaConf


class Dummy(nn.Module):
    def __init__(self, config: OmegaConf):
        super(Dummy, self).__init__()
        self.config = config
        self.vision_encoder = torch.nn.Sequential(
            nn.Linear(6 * config.model.num_points, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, config.model.num_classes),
        )

    def forward(self, x):
        x = x[:, :, torch.randperm(self.config.model.num_points)]
        x = x.view(x.size(0), -1)
        return self.vision_encoder(x)
