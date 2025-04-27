import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from omegaconf import OmegaConf

from src.data import TransformsCompose
from src.datasets.base import DatasetMetadata


def load_captions(filepath: str) -> pd.DataFrame:
    num_columns = len(pd.read_csv(filepath, nrows=1).columns)

    if num_columns == 2:
        df = pd.read_csv(filepath, names=["id", "caption"], header=None)
        df["labels"] = 0  # dummy labels column for compatibility
        return df

    if num_columns == 3:
        return pd.read_csv(filepath, names=["id", "caption", "labels"], header=None)
    
    raise ValueError(f"Unexpected number of columns: {num_columns}")


def filter_files_by_csv(files: list[str], captions_df: pd.DataFrame) -> list[str]:
    captions_ids_set = set(captions_df["id"].values)
    return [f for f in files if f.replace(".pt", "") in captions_ids_set]


class Cap3D(Dataset):
    def __init__(
        self,
        data_path: str,
        transforms: TransformsCompose = TransformsCompose(),
        index_range: tuple[int, int] = None,
        config: OmegaConf = None,
    ):
        self.data_path = data_path
        self.transforms = transforms
        self.config = config
        self.use_captions = True

        # Captions and labels (optional) - one dir lower
        self.captions_filepath = \
            Path(self.data_path).parent / self.config.dataset.captions_filename

        self.captions_filepath = str(self.captions_filepath)
        self.captions_df = load_captions(self.captions_filepath)

        # Pointcloud files
        self.files = os.listdir(self.data_path)
    
        # Filter files by available captions/labels from the csv
        self.files = filter_files_by_csv(self.files, self.captions_df)
        
        # Filter files by index range
        if index_range:
            self.files = self.files[index_range[0] : index_range[1]]

    def get_metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            filenames=self.files,
            num_samples=len(self.files),
            num_classes=len(self.captions_df["labels"].unique()),
        )

    def _get_row_value(self, id: str, col: str):
        return self.captions_df.loc[self.captions_df["id"] == id, col].iloc[0]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> tuple[torch.Tensor, str, str, int]:
        filename = self.files[idx]
        pointcloud_filepath = self.data_path + filename

        pointcloud = torch.load(pointcloud_filepath, weights_only=True)
        pointcloud = self.transforms(pointcloud)
        
        id = filename.replace(".pt", "")

        # captions and labels (optional)
        caption = self._get_row_value(id, "caption") if self.use_captions else ""
        label = self._get_row_value(id, "labels") if self.use_captions else 0
    
        return pointcloud, caption, id, label
