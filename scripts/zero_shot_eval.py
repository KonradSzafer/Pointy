import os
import argparse
from pathlib import Path

import wandb
import torch
from omegaconf import OmegaConf

from src.data.factory import get_transforms
from src.datasets import load_dataset
from src.models import load_model, load_pretrained
from src.trainers import ZeroShotEvaluation
from src.utils import setup_logger, log_config, set_seed
from src.paths import PATHS

logger = setup_logger(__name__)


def zero_shot_evaluation(
    config_eval_path: str = None,
    model: torch.nn.Module = None,
    config_model: OmegaConf = None,
    wandb_run: tuple[str, wandb.sdk.wandb_run.Run] = None,
) -> None:
    # Load eval config
    config_eval = OmegaConf.load(config_eval_path)
    config_eval.name = Path(config_eval_path).stem
    config_eval.pid = os.getpid()

    # Set the seed
    set_seed(config_eval.training.seed)
    
    # Logging the configuration
    log_config(config_eval, logger)
    
    if model is None or config_model is None:
        logger.warning("Loading model from checkpoint")

        # Load model config - for the transforms, and the model
        config_model_path = os.path.join(
            PATHS.checkpoints, config_eval.checkpoint_name, "config.json"
        )
        config_model = OmegaConf.load(config_model_path)
                
        # Load the model
        model = load_pretrained(config_eval.checkpoint_name)
    
    # Load the dataset
    train_dataset, test_dataset = load_dataset(
        config_eval,
        get_transforms(config_model),
        get_transforms(config_model),
    )

    # Update the eval config with the model config, for model name and other parameters
    config_eval.model = config_model.model
    trainer = ZeroShotEvaluation(
        model,
        train_dataset,
        test_dataset,
        config_eval,
        wandb_run,
    )
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/zero_shot_evaluation_modelnet40.yaml", type=str,
    )
    args = parser.parse_args()
    zero_shot_evaluation(config_eval_path=args.config,)
