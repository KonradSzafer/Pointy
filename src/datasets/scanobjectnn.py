# source: https://github.com/ma-xu/pointMLP-pytorch/blob/main/classification_ScanObjectNN/ScanObjectNN.py

import os

import h5py
import torch
from torch.utils.data import Dataset
from omegaconf import OmegaConf

from src.data import TransformsCompose
from src.datasets.base import DatasetMetadata


CLASS_NAMES = None


def load_scanobjectnn_data(filename: str) -> tuple[torch.Tensor, torch.Tensor]:
    with h5py.File(filename, mode="r") as f:
        pointclouds = f["data"][:].astype("float32")
        labels = f["label"][:].astype("int64")
    pointclouds = torch.from_numpy(pointclouds)
    pointclouds = pointclouds.permute(0, 2, 1)
    labels = torch.from_numpy(labels)
    return pointclouds, labels


class ScanObjectNN(Dataset):
    def __init__(
        self,
        data_path: str,
        transforms: TransformsCompose = TransformsCompose(),
        index_range: tuple[int, int] = None,
        config: OmegaConf = None,
    ):
        self.data_path = data_path
        self.pointclouds, self.labels = load_scanobjectnn_data(self.data_path)
        self.transforms = transforms

        if index_range:
            self.pointclouds = self.pointclouds[index_range[0] : index_range[1]]
            self.labels = self.labels[index_range[0] : index_range[1]]

    def get_metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            filenames=None,
            num_samples=len(self.pointclouds),
            num_classes=15,
        )
        
    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx) -> tuple[torch.Tensor, str, str, int]:
        """
        Does not return object id, thus third values is empty string.
        """        
        xyz = self.pointclouds[idx]
        rgb = torch.zeros_like(xyz)  # TEMP - using black color for now
        pointcloud = torch.cat([xyz, rgb], dim=0)
        
        pointcloud = self.transforms(pointcloud)

        label = self.labels[idx]
        # class_name = CLASS_NAMES[label]

        return pointcloud, "class_name", "", label
