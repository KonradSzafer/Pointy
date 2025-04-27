import os
import argparse
from pathlib import Path
from omegaconf import OmegaConf

from src.data.factory import get_transforms
from src.datasets import load_dataset
from src.models import load_model, load_pretrained
from src.models.pointy import get_component
from src.trainers import ClassifierTrainer, ZeroShotEvaluation
from src.utils import setup_logger, log_config, set_seed
from src.paths import PATHS
from scripts.zero_shot_eval import zero_shot_evaluation

logger = setup_logger(__name__)


def main(args: argparse.Namespace, additional_args: dict = None) -> ClassifierTrainer:
    
    # Load and update the configuration
    config = OmegaConf.load(args.config)
    config.name = Path(args.config).stem
    config.pid = os.getpid()
    if additional_args is not None:
        config.update(additional_args)

    # Set the seed
    set_seed(config.training.seed)

    # Logging the configuration
    log_config(config, logger)

    # Load the dataset
    train_dataset, test_dataset = load_dataset(
        config,
        get_transforms(config, train=True),
        get_transforms(config, train=False),
    )

    # Load the model
    if hasattr(config, "checkpoint_name"):
        logger.info(f"Loading checkpoint: {config.checkpoint_name}")
        model = load_pretrained(config.checkpoint_name)
        model.head = get_component(
            config, "head", embedding_dim=model.encoder.embedding_dim
        )
    else:
        model = load_model(config)

    # Train the model
    trainer = ClassifierTrainer(
        model,
        train_dataset,
        test_dataset,
        config,
    )

    trainer.run()
    
    if config.zero_shot_eval:
        zero_shot_evaluation(
            "configs/zero_shot_evaluation_modelnet40.yaml",
            model,
            config,
            (trainer.run_id, trainer.wandb)
        )
        zero_shot_evaluation(
            "configs/zero_shot_evaluation_scanobjectnn.yaml",
            model,
            config,
            (trainer.run_id, trainer.wandb)
        )

    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/classification.yaml", type=str)
    args = parser.parse_args()
    main(args)
