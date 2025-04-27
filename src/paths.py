import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class PATHS:
    data = os.environ.get("DATA_PATH", "data/")
    checkpoints = os.environ.get("CHECKPOINTS_PATH", "checkpoints/")
    results = os.environ.get("RESULTS_PATH", "results/")
