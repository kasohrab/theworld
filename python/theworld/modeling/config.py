from theworld.constants import DEFAULT_COSMOS_MODEL
from transformers import Gemma3Config

class TheWorldConfig(Gemma3Config):
    model_type = "the_world"  # Register a new model type

    def __init__(
        self,
        cosmos_model_name: str = DEFAULT_COSMOS_MODEL,
        enable_world: bool = True,
        freeze_gemma_vision: bool = True,
        freeze_gemma_language: bool = True,
        freeze_cosmos_vae: bool = True,
        
        # Pass all original Gemma3Config args to the parent
        **kwargs,
    ):
        self.cosmos_model_name = cosmos_model_name
        self.enable_world = enable_world
        self.freeze_gemma_vision = freeze_gemma_vision
        self.freeze_gemma_language = freeze_gemma_language
        self.freeze_cosmos_vae = freeze_cosmos_vae
        super().__init__(**kwargs)