import os
import logging

import torch
from torch import nn
from omegaconf import OmegaConf

from src.paths import PATHS
from src.models.baselines.dummy import Dummy as DummyModel
from src.models.baselines.pointmlp import pointMLP as PointMLPModel
from src.models.baselines.pointnet import PointNet as PointNetModel
from src.models.baselines.pointnet2 import PointNet2 as PointNet2Model
from src.models.baselines.dgcnn import DGCNN as DGCNNModel
from src.models.baselines.pct import PCTCls as PCTModel
from src.models.baselines.pointcloud_transformer_hengshuang import \
    PointTransformerCls as PointTransformerHengshuangModel
from src.models.baselines.pointcloud_transformer_nico import \
    PointTransformerCls as PointTransformerNicoModel
from src.models.baselines.pointmae import PointMAE as PointMAEModel
from src.models.pointy import Pointy as PointyModel

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self):
        self.models: Dict[str, nn.Module] = {}
        
    def register(self, name: str, model: nn.Module) -> None:
        self.models[name] = model

    def load(self, name: str, config: OmegaConf) -> nn.Module:
        if name not in self.models:
            raise ValueError(f"Model {name} not found.")
        return self.models[name](config).model


MODEL_REGISTRY = ModelRegistry()


def register_model(name: str):
    def decorator(model_class):
        MODEL_REGISTRY.register(name, model_class)
        return model_class
    return decorator


@register_model("Dummy")
class Dummy(nn.Module):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.model = DummyModel(config)


@register_model("PointNet")
class PointNet(nn.Module):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.model = PointNetModel(config.model.num_classes)


@register_model("PointNet2")
class PointNet2(nn.Module):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.model = PointNet2Model(
            config.model.num_classes, normal_channel=config.model.normal_channel
        )


@register_model("PointMLP")
class PointMLP(nn.Module):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.model = PointMLPModel(config.model.num_classes, config.model.num_points)


@register_model("DGCNN")
class DGCNN(nn.Module):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.model = DGCNNModel(output_channels=config.model.num_classes)


@register_model("PCT")
class PCT(nn.Module):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.model = PCTModel(num_categories=config.model.num_classes)


@register_model("PointTransformerHengshuang")
class PointTransformerHengshuang(nn.Module):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.model = PointTransformerHengshuangModel(num_classes=config.model.num_classes)


@register_model("PointTransformerNico")
class PointTransformerNico(nn.Module):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.model = PointTransformerNicoModel(num_classes=config.model.num_classes)


@register_model("PointMAE")
class PointMAE(nn.Module):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.model = PointMAEModel(num_classes=config.model.num_classes)


@register_model("Pointy")
class Pointy(nn.Module):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.model = PointyModel(config)


def load_model(config: OmegaConf) -> nn.Module:
    """ Loads the model based on the config. """
    try:
        return MODEL_REGISTRY.load(config.model.name, config)
    except ValueError as e:
        raise ValueError(
            f"Failed to load model: {config.model.name}. Error: {e}"
        )


def clean_checkpoint_state_dict(model: nn.Module, checkpoint: dict) -> dict:
    """
    Cleans the state dict by removing keys that are not part of the model.
    Logs a warning for each key that is missing.
    """
    # Remove "module." if present in the keys - model was wrapped in DataParallel
    checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}

    # Warn about mismatching keys
    for key in model.state_dict().keys():
        if not any(key in checkpoint_key for checkpoint_key in checkpoint.keys()):
            logger.warning(f"Key not found in checkpoint: {key} - skipping")
            
    # Keep only the keys that are found in the model
    checkpoint_clean = {}
    for key, value in checkpoint.items():
        # Skip keys not found in the model
        if key not in model.state_dict().keys():
            logger.warning(f"Key not found in model: {key} - skipping")
            continue
        checkpoint_clean[key] = value

    return checkpoint_clean


def load_pretrained(checkpoint_name: str) -> nn.Module:
    """ Loads a model from a checkpoint. """
    # Get paths
    path_checkpoint = os.path.join(PATHS.checkpoints, checkpoint_name, "model.pt")
    path_config = os.path.join(PATHS.checkpoints, checkpoint_name, "config.json")
    
    # Load architecture
    config = OmegaConf.load(path_config)
    model = load_model(config)

    # Load weights
    checkpoint = torch.load(path_checkpoint, map_location=torch.device("cpu"))
    
    # Clean the checkpoint
    checkpoint = clean_checkpoint_state_dict(model, checkpoint)

    # Load model checkpoint
    model.load_state_dict(checkpoint, strict=False)
    return model
