import time
import argparse

import torch
from thop import profile
from omegaconf import OmegaConf

from src.models import load_model
from src.models.utils import calculate_params
from src.utils import setup_logger

logger = setup_logger(__name__)


def main(args: argparse.Namespace) -> None:
    # Load model
    config = OmegaConf.load(args.config)
    model = load_model(config)
    model.eval()

    # Input
    if config.model.name == "Pointy":
        input_data = torch.randn(
            1, config.model.num_patches, 3, 2048 // config.model.num_patches
        )
    else:
        input_data = torch.randn(1, 3, 2048)

    # Inference time
    start = time.time()
    with torch.no_grad():
        model(input_data)
    end = time.time()
    logger.info(f"Inference time: {end - start:.3f} s")

    # thop.profile returns number of MACs (Multiply-Accumulate operations) and params
    macs, params = profile(model, inputs=(input_data, ), verbose=False)

    logger.info(f"Params (M) thop: {params / 1e6:.3f}")
    logger.info(f"Params (M) torch: {calculate_params(model) / 1e6:.3f}")

    # 1 MAC is usually counted as 2 FLOPs (1 multiply + 1 addition)
    flops = 2 * macs
    logger.info(f"MACs (G): {macs / 1e9:.3f}")
    logger.info(f"FLOPs (G): {flops / 1e9:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model inference benchmark")
    parser.add_argument("--config", type=str, default="configs/classification.yaml")
    args = parser.parse_args()
    main(args)
