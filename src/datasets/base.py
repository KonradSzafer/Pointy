from dataclasses import dataclass


@dataclass
class DatasetMetadata:
    filenames: list = None
    num_samples: int = None
    num_classes: int = None
