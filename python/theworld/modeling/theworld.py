"""Refactored TheWorld model - inherits from Gemma3ForConditionalGeneration."""

import torch
import torch.nn as nn
from PIL import Image
from typing import Any, Dict, List, Optional, Union
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
from transformers.cache_utils import Cache
from diffusers.pipelines.cosmos.pipeline_cosmos2_video2world import Cosmos2VideoToWorldPipeline
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from huggingface_hub import hf_hub_download

from .cosmos_encoder import CosmosEncoder
from .fusion import EmbeddingFusion
from ..constants import DEFAULT_COSMOS_MODEL, DEFAULT_GEMMA_MODEL
from .config import TheWorldConfig


class TheWorld(Gemma3ForConditionalGeneration):
    """
    TheWorld: Fused vision-language-world model combining Gemma 3 and Cosmos.

    Inherits from Gemma3ForConditionalGeneration to eliminate code duplication.
    When world tokens are not used, behavior is identical to pure Gemma3.
    """

    # Type annotations for instance variables
    gemma_model_name: Optional[str]
    cosmos_model_name: Optional[str]
    freeze_gemma_vision: bool
    freeze_gemma_language: bool
    freeze_cosmos_vae: bool
    random_projection_init: bool
    load_full_cosmos_pipeline: bool
    enable_world: bool

    processor: Optional[Any]  # AutoProcessor type not fully typed in transformers
    cosmos_pipe: Optional[Cosmos2VideoToWorldPipeline]
    cosmos_vae: Optional[AutoencoderKL]
    cosmos_vae_encoder: Optional[nn.Module]
    cosmos_encoder: Optional[CosmosEncoder]
    fusion: Optional[EmbeddingFusion]
    sow_token_id: Optional[int]
    eow_token_id: Optional[int]
    config_class = TheWorldConfig

    def __init__(self, config: TheWorldConfig):
        """
        Initialize TheWorld model structure.

        NOTE: Do not call this directly. Use `TheWorld.from_pretrained()` instead.
        This method only creates the model structure without loading weights.

        Args:
            config: Gemma3Config from pretrained model
        """
        super().__init__(config)

        # These will be set by from_pretrained()
        self.enable_world = False
        self.cosmos_pipe = None
        self.cosmos_vae = None
        self.cosmos_vae_encoder = None
        self.cosmos_encoder = None
        self.fusion = None
        self.sow_token_id = None
        self.eow_token_id = None
        self.processor = None

        # Configuration flags (set by from_pretrained)
        self.gemma_model_name = None
        self.cosmos_model_name = None
        self.freeze_gemma_vision = True
        self.freeze_gemma_language = True
        self.freeze_cosmos_vae = True
        self.random_projection_init = False
        self.load_full_cosmos_pipeline = True

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        enable_world: bool = True,
        cosmos_model_name: str = DEFAULT_COSMOS_MODEL,
        device: str = "cuda",
        freeze_gemma_vision: bool = True,
        freeze_gemma_language: bool = True,
        freeze_cosmos_vae: bool = True,
        random_projection_init: bool = False,
        load_full_cosmos_pipeline: bool = True,
        **kwargs: Any
    ) -> "TheWorld":
        """
        Loads TheWorld model.
        
        - Case A (New Model): Loads a base Gemma model and initializes
          new world model components.
        - Case B (Saved Checkpoint): Loads a saved TheWorld checkpoint.
        """
        # 1. Load config from path
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # 2. Check config type to see which case we are in
        if isinstance(config, TheWorldConfig):
            print(f"Loading saved TheWorld checkpoint from: {pretrained_model_name_or_path}")
            model = super(TheWorld, cls).from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **kwargs
            )
            
        else:
            print(f"Initializing new TheWorld model from base: {pretrained_model_name_or_path}")
            
            model = super(TheWorld, cls).from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )

            the_world_config = TheWorldConfig.from_dict(
                model.config.to_dict(),
                cosmos_model_name=cosmos_model_name,
                enable_world=enable_world,
                freeze_gemma_vision=freeze_gemma_vision,
                freeze_gemma_language=freeze_gemma_language,
                freeze_cosmos_vae=freeze_cosmos_vae,
                load_full_cosmos_pipeline=load_full_cosmos_pipeline,
                random_projection_init=random_projection_init,
            )
            the_world_config.gemma_model_name = pretrained_model_name_or_path
            
            model.config = the_world_config
            model.__init__(the_world_config) 

        return model

    def get_trainable_parameters(self):
        """Return count and percentage of trainable parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total, 100 * trainable / total

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Save only trainable parameters.

        Frozen pretrained models (Gemma, Cosmos) are reloaded from HuggingFace on load.
        This makes checkpoints much smaller (~300MB vs ~17GB) and avoids duplicate parameter issues.

        Returns:
            Dictionary with only trainable parameters (safe for safetensors)
        """
        state: Dict[str, Any] = {}

        # Save only trainable parameters
        for name, param in self.named_parameters():
            if param.requires_grad:
                state[name] = param

        return state

    def load_state_dict(
        self, state_dict: Dict[str, Any], strict: bool = False, assign: bool = False
    ) -> Any:
        """Load trainable parameters from checkpoint.

        Frozen parameters are already loaded from HuggingFace during __init__.
        Only trainable parameters need to be restored from checkpoint.

        Args:
            state_dict: State dictionary with trainable parameters
            strict: Whether to require exact key match (default: False for trainable-only checkpoints)
            assign: Whether to assign items in state_dict to corresponding keys in module

        Returns:
            NamedTuple with missing_keys and unexpected_keys
        """
        # Load only the trainable parameters (strict=False to ignore missing frozen params)
        # Suppress the "missing keys" warning - it's expected for trainable-only checkpoints
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*missing keys.*", category=UserWarning)
            warnings.filterwarnings(
                "ignore", message=".*were not found in the checkpoint.*", category=FutureWarning
            )
            return super().load_state_dict(state_dict, strict=False, assign=assign)

    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Save model to HuggingFace format.
        
        Saves the custom TheWorldConfig to config.json and
        only the trainable parameters to model.safetensors.
        """
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        self.config.save_pretrained(save_directory)
        
        state_dict = self.state_dict()
        
        if safe_serialization:
            from safetensors.torch import save_file
            save_path = os.path.join(save_directory, "model.safetensors")
            save_file(state_dict, save_path)
        else:
            save_path = os.path.join(save_directory, "pytorch_model.bin")
            torch.save(state_dict, save_path)
            
        print(f"✓ Model saved to {save_directory} (trainable parameters only)")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        images: Optional[List[Image.Image]] = None,
        **lm_kwargs: Any,
    ):
        """
        Forward pass for TheWorld model.

        Automatically handles world token injection and routing:
        - If enable_world=True and images provided: inject world tokens (if needed) + world-augmented forward
        - Otherwise: pure Gemma3 forward (identical to parent)

        Args:
            input_ids: Token IDs (B, seq_len)
            pixel_values: Preprocessed images for Gemma SigLIP (B, C, H, W)
            attention_mask: Attention mask (B, seq_len)
            position_ids: Position IDs for the input tokens
            past_key_values: Cached key/value states from previous forward passes
            token_type_ids: Token type IDs (for models that use them)
            cache_position: Position indices for cached generation
            inputs_embeds: Pre-computed input embeddings (alternative to input_ids)
            labels: Target labels for loss computation (B, label_len), optional
            use_cache: Whether to return past_key_values for generation
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return a dict or tuple
            logits_to_keep: Number of logits to keep (memory optimization)
            images: Raw PIL images for Cosmos (List of B images), TheWorld-specific parameter
            **lm_kwargs: Additional keyword arguments passed to the language model

        Returns:
            CausalLMOutputWithPast with loss, logits, and optionally past_key_values
        """
        # Route to world-augmented or pure Gemma path
        if self.enable_world and images is not None:
            # Inject world tokens if not present
            if (
                input_ids is not None
                and self.sow_token_id is not None
                and not (input_ids == self.sow_token_id).any()
            ):
                input_ids, attention_mask = self._inject_world_tokens(input_ids, attention_mask)

            # World-augmented path
            return self._forward_with_world(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                token_type_ids=token_type_ids,
                cache_position=cache_position,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                logits_to_keep=logits_to_keep,
                images=images,
                **lm_kwargs,
            )
        else:
            # Pure Gemma path - delegate to parent
            return super().forward(  # type: ignore
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,  # type: ignore
                token_type_ids=token_type_ids,
                cache_position=cache_position,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                logits_to_keep=logits_to_keep,
                **lm_kwargs,
            )

    def _forward_with_world(
        self,
        input_ids: Optional[torch.LongTensor],
        pixel_values: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]],
        token_type_ids: Optional[torch.LongTensor],
        cache_position: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.FloatTensor],
        labels: Optional[torch.LongTensor],
        use_cache: Optional[bool],
        output_attentions: Optional[bool],
        output_hidden_states: Optional[bool],
        return_dict: Optional[bool],
        logits_to_keep: Union[int, torch.Tensor],
        images: Optional[List[Image.Image]],
        **lm_kwargs: Any,
    ):
        """Forward pass with world model augmentation."""
        assert self.cosmos_encoder is not None, "Cosmos encoder must be loaded"
        assert self.fusion is not None, "Fusion module must be loaded"
        assert input_ids is not None, "input_ids is required for world-augmented forward"
        assert pixel_values is not None, "pixel_values is required for world-augmented forward"
        assert attention_mask is not None, "attention_mask is required for world-augmented forward"

        # 1. Get embeddings
        target_device = self.get_input_embeddings().weight.device
        input_ids = input_ids.to(target_device)  # type: ignore[assignment]
        pixel_values = pixel_values.to(target_device)  # type: ignore[assignment]
        attention_mask = attention_mask.to(target_device)  # type: ignore[assignment]

        inputs_embeds = self.model.language_model.embed_tokens(input_ids)
        assert inputs_embeds is not None, "inputs_embeds must not be None"

        # 2. Get vision features (reuse parent's method)
        image_features = self.model.get_image_features(pixel_values)
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        special_image_mask = self.model.get_placeholder_mask(input_ids, inputs_embeds, image_features)  # type: ignore
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)  # type: ignore[assignment]

        # 3. Get world embeddings
        world_embeds = self.cosmos_encoder(images=images)

        # 4. Fuse embeddings
        fusion_output = self.fusion(
            gemma_embeds=inputs_embeds,
            world_embeds=world_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # 5. Align labels with world tokens
        if labels is not None:
            start_positions = (input_ids == self.sow_token_id).nonzero(as_tuple=True)[1]
            end_positions = (input_ids == self.eow_token_id).nonzero(as_tuple=True)[1]

            if len(start_positions) > 0 and len(end_positions) > 0:
                start_pos = start_positions[0].item()
                end_pos = end_positions[0].item()
                batch_size = input_ids.size(0)
                num_world_tokens = world_embeds.size(1)

                # Build labels: [tokens_before | -100 for world | tokens_after]
                labels_before = input_ids[:, : start_pos + 1].to(target_device)
                labels_world = torch.full((batch_size, num_world_tokens), -100, dtype=torch.long, device=target_device)
                labels_after = input_ids[:, end_pos:].to(target_device)
                combined_labels = torch.cat([labels_before, labels_world, labels_after], dim=1)
            else:
                combined_labels = labels
        else:
            combined_labels = None

        # 6. Call parent forward with fused embeddings (passing inputs_embeds, NOT input_ids)
        return super().forward(  # type: ignore
            input_ids=None,  # We're using inputs_embeds instead
            pixel_values=None,  # Already processed into embeddings
            attention_mask=fusion_output.combined_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,  # type: ignore
            token_type_ids=token_type_ids,
            cache_position=cache_position,
            inputs_embeds=fusion_output.combined_embeds,
            labels=combined_labels,  # type: ignore
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            logits_to_keep=logits_to_keep,
            **lm_kwargs,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.LongTensor],
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = True,
        logits_to_keep: Optional[Union[int, torch.Tensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        images: Optional[List[Image.Image]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Prepare inputs for generation, injecting world tokens if enabled.

        This method is called by the parent's generate() to prepare inputs for each generation step.
        When enable_world=True, we automatically inject SOW/EOW tokens on the first step.

        Args:
            input_ids: Input token IDs
            past_key_values: Cached key/value states from previous generation steps
            inputs_embeds: Pre-computed input embeddings (alternative to input_ids)
            cache_position: Position indices for cached generation
            position_ids: Position IDs for the input tokens
            pixel_values: Preprocessed images for Gemma SigLIP encoder
            attention_mask: Attention mask for the input
            token_type_ids: Token type IDs (for models that use them)
            use_cache: Whether to use KV cache for generation
            logits_to_keep: Number of logits to keep (memory optimization)
            labels: Target labels (not used in generation)
            images: Raw PIL images for Cosmos world model (TheWorld-specific parameter)
            **kwargs: Additional keyword arguments
        """
        # If this is the first generation step and world is enabled, inject SOW/EOW
        if (
            self.enable_world
            and cache_position is not None
            and cache_position[0] == 0
            and self.sow_token_id is not None
            and input_ids is not None
        ):
            # Inject SOW, EOW after BOS token (position 1)
            batch_size = input_ids.shape[0]
            sow_eow = torch.tensor(
                [[self.sow_token_id, self.eow_token_id]] * batch_size,
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
            input_ids = torch.cat(
                [
                    input_ids[:, :1],  # BOS
                    sow_eow,  # SOW, EOW
                    input_ids[:, 1:],  # Rest
                ],
                dim=1,
            )

            # Update attention mask to match
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [
                        attention_mask[:, :1],
                        torch.ones(
                            (batch_size, 2), device=attention_mask.device, dtype=attention_mask.dtype
                        ),
                        attention_mask[:, 1:],
                    ],
                    dim=1,
                )

        # Call parent's prepare method
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            labels=labels,
            **kwargs,
        )

        # Pass images only on first step (not during cached decoding)
        if cache_position is not None and cache_position[0] == 0 and images is not None:
            model_inputs["images"] = images

        return model_inputs
