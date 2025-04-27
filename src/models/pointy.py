import logging

from torch import nn
from omegaconf import OmegaConf

from src.models.pointy_tokenizers import Linear, PointNetEmbedding
from src.models.pointy_encoders import TransformerEncoder
from src.models.pointy_hierarchical import HierarchicalTransformer
from src.models.pointy_heads import Classification
from src.utils import log_table
from src.models.utils import calculate_params

logger = logging.getLogger(__name__)


def get_component(config: OmegaConf, component_type: str, **kwargs) -> nn.Module:
    """
    Factory function to get the component of the model based on the config.    
    """
    factories = {
        'tokenizer': {
            "Linear": Linear,
            "PointNetEmbedding": PointNetEmbedding,
        },
        'encoder': {
            "TransformerEncoder": TransformerEncoder,
            "HierarchicalTransformer": HierarchicalTransformer,
        },
        'head': {
            "Classification": Classification,
        }
    }

    component_factory = factories.get(component_type)
    if component_factory is None:
        raise ValueError(f"Unknown component type '{component_type}'.")

    component_name = getattr(config.model, component_type, None)
    if component_name is None:
        raise ValueError(f"No '{component_type}' found in config.model.")

    component_class = component_factory.get(component_name)
    if component_class is None:
        raise ValueError(f"Component '{component_name}' not found for type '{component_type}'.")

    return component_class(config, **kwargs)


class Pointy(nn.Module):
    def __init__(self, config: OmegaConf):
        super(Pointy, self).__init__()
        self.config = config
        self.encoder = get_component(config, 'encoder')
        self.tokenizer = get_component(
            config,
            'tokenizer',
            embedding_dim=self.encoder.embedding_dim,
        )
        self.head = get_component(
            config,
            'head',
            embedding_dim=self.encoder.embedding_dim
        )
        # Log the number of parameters
        params_dict = {
            "Tokenizer": f"{(calculate_params(self.tokenizer) / 1e6):.2f}m",
            "Encoder": f"{(calculate_params(self.encoder) / 1e6):.2f}m",
            "Head": f"{(calculate_params(self.head) / 1e6):.2f}m",
            "Total": f"{(calculate_params(self) / 1e6):.2f}m",
        }
        logger.info("Number of parameters:")
        log_table(params_dict, logger)

    def encode(self, x):
        x = self.tokenizer(x)
        x = self.encoder(x)
        return x

    def forward(self, x):
        x = self.tokenizer(x)
        x = self.encoder(x)
        x = self.head(x)
        return x
