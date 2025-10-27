"""Modular components for TheWorld model."""

from .outputs import FusionOutput
from .cosmos_encoder import CosmosEncoder
from .fusion import EmbeddingFusion
from .theworld import TheWorld
from transformers import AutoConfig, AutoModelForCausalLM
from .config import TheWorldConfig

__all__ = [
    "GemmaVisionOutput",
    "FusionOutput",
    "CosmosEncoder",
    "EmbeddingFusion",
    "TheWorld",
]

AutoConfig.register("the_world", TheWorldConfig)
AutoModelForCausalLM.register(TheWorldConfig, TheWorld)
