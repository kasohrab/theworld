"""
TheWorld: Fused Vision-Language-World Model

Combines Google's Gemma 3 vision-language model with NVIDIA's Cosmos world model
to enable reasoning about both static visual understanding and temporal dynamics.
"""

from .modeling import TheWorld
from .config import TrainingConfig
from .data import TheWorldDataset, HFDatasetWrapper, theworld_collate_fn, create_theworld_collator
from .datasets import DataCompDataset, load_datacomp
from .baselines import Gemma3Baseline

__version__ = "0.1.0"

__all__ = [
    "TheWorld",
    "TrainingConfig",
    "TheWorldDataset",
    "HFDatasetWrapper",
    "theworld_collate_fn",
    "create_theworld_collator",
    "DataCompDataset",
    "load_datacomp",
    "Gemma3Baseline",
]
