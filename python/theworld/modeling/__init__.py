"""Modular components for TheWorld model."""

from .outputs import FusionOutput
from .cosmos_vae_encoder import CosmosVAEEncoder
from .world_projector import WorldProjector
from .spatial_reducer import SpatialReducer, WorldProjectionConfig
from .fusion import EmbeddingFusion
from .theworld import TheWorld
from transformers import AutoConfig, AutoModelForCausalLM
from .config import TheWorldConfig

__all__ = [
    "GemmaVisionOutput",
    "FusionOutput",
    "CosmosVAEEncoder",
    "WorldProjector",
    "SpatialReducer",
    "WorldProjectionConfig",
    "EmbeddingFusion",
    "TheWorld",
]

AutoConfig.register("the_world", TheWorldConfig)
AutoModelForCausalLM.register(TheWorldConfig, TheWorld)
