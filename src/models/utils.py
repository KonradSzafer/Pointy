import os
import json
import logging

import torch
from torch import nn
from omegaconf import OmegaConf

from src.paths import PATHS

logger = logging.getLogger(__name__)


def reset_weights(model: nn.Module) -> None:
    """ Resets the weights of a model. """
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
        elif isinstance(layer, nn.Sequential) or isinstance(layer, nn.Module):
            reset_weights(layer)


def save_model(
    model: nn.Module,
    config: OmegaConf,
    model_name: str,
) -> None:
    """
    Saves the model and config to a checkpoint folder.
    Checkpoint name should follow the format: {model_name}_{num_steps}.
    """
    checkpoint_path = os.path.join(PATHS.checkpoints, model_name)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)

    logger.info(f"Saving model {model_name} to {checkpoint_path}")
    filepath_model = os.path.join(checkpoint_path, "model.pt")
    filepath_config = os.path.join(checkpoint_path, "config.json")

    # Save model
    torch.save(model.state_dict(), filepath_model)

    # Save config
    config_dict = OmegaConf.to_container(config, resolve=True)
    with open(filepath_config, 'w') as json_file:
        json.dump(config_dict, json_file, indent=2)


def calculate_params(model: nn.Module) -> int:
    """ Calculates the number of trainable parameters in a model. """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(device_id: int = 0, force_cpu: bool = False) -> torch.device:
    if force_cpu:
        logger.warning("Forcing CPU usage.")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_id}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compile_model(model: torch.nn.Module, config: OmegaConf) -> nn.Module:
    """
    Tries to compile the model using TorchScript if CUDA is available.
    """
    if config.debug:
        logger.info("Debug mode is enabled. Skipping model compilation.")
        return model
    
    if not torch.cuda.is_available():
        logger.info("CUDA is not used or available. Skipping model compilation.")
        return model

    try:
        logger.info("Compiling model.")
        model = torch.compile(model)
    except Exception as e:
        logger.warning(f"Error compiling model: {e}. Continuing without compilation.")
    return model


def data_parallel(
    model: torch.nn.Module,
    device: torch.device,
    config: OmegaConf
) -> nn.Module:
    """
    Parallelizes the data if multiple devices are available and specified in the config.
    """
    if config.force_cpu:
        logger.warning("Forcing CPU usage. Skipping data parallel.")
        return model

    if not torch.cuda.is_available():
        logger.warning(
            f"CUDA is not used or available. Using {device} without data parallel."
        )
        return model

    device_index = device.index
    device_indices = list(set([device_index] + config.model_devices))

    if torch.cuda.device_count() == 1:
        logger.warning(f"Single device detected. Using only device {device_index}.")
        return model
        
    if not len(device_indices) > 1:
        logger.warning(f"Single device specified. Using only device {device_index}.")
        return model

    logger.info(f"Data parallel devices: {device_indices}")
    model = nn.DataParallel(model, device_ids=device_indices)
    return model
