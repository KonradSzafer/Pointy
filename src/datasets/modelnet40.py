import os

import numpy as np
import torch
from torch.utils.data import Dataset
from omegaconf import OmegaConf

from src.data import TransformsCompose
from src.datasets.base import DatasetMetadata

CLASS_NAMES = [
    "airplane",    # 0
    "bathtub",     # 1 
    "bed",         # 2
    "bench",       # 3
    "bookshelf",   # 4
    "bottle",      # 5
    "bowl",        # 6
    "car",         # 7
    "chair",       # 8
    "cone",        # 9
    "cup",         # 10
    "curtain",     # 11
    "desk",        # 12
    "door",        # 13 
    "dresser",     # 14
    "flower_pot",  # 15
    "glass_box",   # 16
    "guitar",      # 17
    "keyboard",    # 18
    "lamp",        # 19
    "laptop",      # 20
    "mantel",      # 21
    "monitor",     # 22
    "night_stand", # 23
    "person",      # 24
    "piano",       # 25
    "plant",       # 26
    "radio",       # 27
    "range_hood",  # 28
    "sink",        # 29
    "sofa",        # 30
    "stairs",      # 31
    "stool",       # 32
    "table",       # 33
    "tent",        # 34
    "toilet",      # 35
    "tv_stand",    # 36
    "vase",        # 37
    "wardrobe",    # 38
    "xbox",        # 39
]


class ModelNet40(Dataset):
    def __init__(
        self,
        data_path: str,
        transforms: TransformsCompose = TransformsCompose(),
        index_range: tuple[int, int] = None,
        config: OmegaConf = None,
    ):
        self.data_path = data_path
        self.transforms = transforms
        self.files = os.listdir(self.data_path)

        if index_range:
            self.files = self.files[index_range[0] : index_range[1]]

    def get_metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            filenames=self.files,
            num_samples=len(self.files),
            num_classes=len(CLASS_NAMES),
        )
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> tuple[torch.Tensor, str, str, int]:
        """
        Does not return object id, thus third values is empty string.
        """
        filename = self.files[idx]
        filepath = self.data_path + filename
        sample_data = np.load(filepath)

        # transpose from (N, 6) to (6, N)
        pointcloud = np.concatenate(
            (sample_data["pointcloud"].T, sample_data["normal"].T), axis=0
        )
        pointcloud = torch.tensor(pointcloud, dtype=torch.float32)
        pointcloud = self.transforms(pointcloud)

        label = sample_data["label"][0]
        class_name = CLASS_NAMES[label]

        return pointcloud, class_name, "", label
