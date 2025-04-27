import torch
import torch.nn as nn
import torch.nn.functional as F


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    predicted = output.argmax(dim=1)
    correct = (predicted == target).sum()
    return correct / target.size(0)


def cosine_similarity(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cosine_similarity = F.cosine_similarity(pred, target, dim=-1)
    cosine_similarity = torch.sum(cosine_similarity) / pred.size(0)
    return cosine_similarity


def cosine_similarity_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cosine_similarity = F.cosine_similarity(pred, target, dim=-1)
    loss = 1 - cosine_similarity
    loss = torch.sum(loss) / pred.size(0) 
    return loss


class MSECosineLoss(nn.Module):
    def __init__(self, mse_weight: int = 1.0):
        super(MSECosineLoss, self).__init__()
        self.mse_weight = mse_weight
        self.mse = nn.MSELoss()
        self.cosine = nn.CosineSimilarity(dim=1)

    def forward(self, input, target):
        mse_loss = self.mse(input, target)

        cosine_similarity = self.cosine(input, target)
        cosine_loss = 1 - torch.mean(cosine_similarity)

        total_loss = self.mse_weight * mse_loss + cosine_loss
        return total_loss
