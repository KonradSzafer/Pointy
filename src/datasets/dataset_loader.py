import os
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset
from omegaconf import OmegaConf

from src.data import TransformsCompose
from src.datasets.cap3d import Cap3D
from src.datasets.modelnet40 import ModelNet40
from src.datasets.scanobjectnn import ScanObjectNN
from src.paths import PATHS


def calculate_splits(config: OmegaConf) -> dict[str, tuple[int,int]]:
    """
    Calculate the number of samples for each split of the dataset.
    
    Returns:
        dict[str, tuple[int,int]]: A dictionary with the number of samples for each split.
    """

    # Dataset is in the form of separate train and test directories
    if not config.dataset.train_path.endswith("/") and not config.dataset.test_path.endswith("/"):
        return {"train": None, "test": None}

    # Has a captions file to filter the samples
    if hasattr(config.dataset, "captions_filename"):
        df = pd.read_csv(Path(config.dataset.train_path).parent / config.dataset.captions_filename)
        num_samples = int(len(df) * config.dataset.dataset_pct)
        return {
            "train": [0, int(num_samples * config.dataset.train_pct)],
            "test": [int(num_samples * config.dataset.train_pct), num_samples]
        }

    # Single dataset directory
    if config.dataset.train_path == config.dataset.test_path:
        # Apply the percentage to the dataset
        num_samples = len(os.listdir(config.dataset.train_path))
        num_samples = int(num_samples * config.dataset.dataset_pct)
        return {
            "train": [0, int(num_samples * config.dataset.train_pct)],
            "test": [int(num_samples * config.dataset.train_pct), num_samples]
        }

    # Classic train - test split
    # Apply the percentage to the dataset
    num_samples_train = len(os.listdir(config.dataset.train_path))
    num_samples_train = int(num_samples_train * config.dataset.dataset_pct)
    num_samples_test = len(os.listdir(config.dataset.test_path))
    num_samples_test = int(num_samples_test * config.dataset.dataset_pct)
    return {
        "train": [0, num_samples_train],
        "test": [0, num_samples_test]
    }


def load_dataset(
    config: OmegaConf,
    train_transforms: TransformsCompose,
    test_transforms: TransformsCompose,
) -> tuple[Dataset, Dataset]:
    
    # Update the paths
    config.dataset.train_path = PATHS.data + config.dataset.train_path
    config.dataset.test_path = PATHS.data + config.dataset.test_path

    # Datasets
    datasets_factory = {
        "Cap3D": Cap3D,
        "ModelNet40": ModelNet40,
        "ScanObjectNN": ScanObjectNN,
    }

    dataset_class = datasets_factory.get(config.dataset.name, None)
    
    if dataset_class is None:
        raise ValueError(f"Dataset {config.dataset.name} not found.")

    # Splits
    splits_indexes = calculate_splits(config)
    
    return (
        dataset_class(
            data_path=config.dataset.train_path,
            transforms=train_transforms,
            index_range=splits_indexes["train"],
            config=config,
        ),
        dataset_class(
            data_path=config.dataset.test_path,
            transforms=test_transforms,
            index_range=splits_indexes["test"],
            config=config,
        )
    )
