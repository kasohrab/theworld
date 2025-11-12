"""
TheWorld: Fused Vision-Language-World Model

Combines Google's Gemma 3 vision-language model with NVIDIA's Cosmos world model
to enable reasoning about both static visual understanding and temporal dynamics.
"""

from .modeling import TheWorld, CosmosVAEEncoder, WorldProjector, SpatialReducer, WorldProjectionConfig
from .config import TrainingConfig
from .data import TheWorldDataset, HFDatasetWrapper, theworld_collate_fn, create_theworld_collator
from .datasets import DataCompDataset, load_datacomp, SpatialRGPTDataset
from . import constants

__version__ = "0.1.0"

__all__ = [
    "TheWorld",
    "CosmosVAEEncoder",
    "WorldProjector",
    "SpatialReducer",
    "WorldProjectionConfig",
    "TrainingConfig",
    "TheWorldDataset",
    "HFDatasetWrapper",
    "theworld_collate_fn",
    "create_theworld_collator",
    "DataCompDataset",
    "load_datacomp",
    "SpatialRGPTDataset",
    "constants",
]
