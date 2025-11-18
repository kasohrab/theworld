from theworld.constants import DEFAULT_COSMOS_MODEL, DEFAULT_GEMMA_MODEL
from transformers import Gemma3Config
from typing import Optional

class TheWorldConfig(Gemma3Config):
    model_type = "the_world"

    def __init__(
        self,
        gemma_model_name: Optional[str] = DEFAULT_GEMMA_MODEL,
        cosmos_model_name: str = DEFAULT_COSMOS_MODEL,
        enable_world: bool = True,
        world_projection_mode: str = "spatial",
        projection_architecture: str = "mlp",
        freeze_gemma_vision: bool = True,
        freeze_gemma_language: bool = True,
        freeze_cosmos_vae: bool = True,
        **kwargs,
    ):
        self.gemma_model_name = gemma_model_name
        self.cosmos_model_name = cosmos_model_name
        self.enable_world = enable_world
        self.world_projection_mode = world_projection_mode
        self.projection_architecture = projection_architecture
        self.freeze_gemma_vision = freeze_gemma_vision
        self.freeze_gemma_language = freeze_gemma_language
        self.freeze_cosmos_vae = freeze_cosmos_vae
        super().__init__(**kwargs)