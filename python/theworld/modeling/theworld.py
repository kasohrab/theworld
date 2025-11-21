"""Refactored TheWorld model - inherits from Gemma3ForConditionalGeneration."""

import torch
import torch.nn as nn
from PIL import Image
from typing import Any, Dict, List, Optional, Union
from transformers import Gemma3ForConditionalGeneration, AutoConfig, AutoProcessor
from transformers.cache_utils import Cache
from diffusers.pipelines.cosmos.pipeline_cosmos2_video2world import Cosmos2VideoToWorldPipeline
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from .cosmos_vae_encoder import CosmosVAEEncoder
from .world_projector import WorldProjector
from .spatial_reducer import WorldProjectionConfig
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
    processor: Optional[Any]  # AutoProcessor type not fully typed in transformers
    cosmos_pipe: Optional[Cosmos2VideoToWorldPipeline]
    cosmos_vae: Optional[AutoencoderKL]
    cosmos_vae_encoder: Optional[CosmosVAEEncoder]
    world_projector: Optional[WorldProjector]
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
            config: TheWorldConfig with model hyperparameters
        """
        super().__init__(config)

        # Initialize placeholders for world model components
        # These will be populated by from_pretrained()
        self.processor = None
        self.cosmos_pipe = None
        self.cosmos_vae = None
        self.cosmos_vae_encoder = None
        self.world_projector = None
        self.fusion = None
        self.sow_token_id = None
        self.eow_token_id = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        enable_world: bool = True,
        cosmos_model_name: str = DEFAULT_COSMOS_MODEL,
        world_projection_mode: str = "spatial",
        projection_architecture: str = "mlp",
        device: Optional[str] = None,
        freeze_gemma_vision: bool = True,
        freeze_gemma_language: bool = True,
        freeze_cosmos_vae: bool = True,
        **kwargs: Any,
    ) -> "TheWorld":
        """
        Load TheWorld model from pretrained weights.

        - Case A (New Model): Loads a base Gemma model and initializes new world model components.
        - Case B (Saved Checkpoint): Loads a saved TheWorld checkpoint with world components.

        Args:
            pretrained_model_name_or_path: HuggingFace model ID or path
            enable_world: If True, add Cosmos world model. If False, Gemma-only baseline
            cosmos_model_name: HuggingFace model ID for Cosmos world model
            world_projection_mode: Projection mode for world tokens ("spatial" or "channel")
            projection_architecture: Projection architecture ("mlp", "mlp_no_final_gelu", "linear")
            device: Device to load Cosmos on ("cuda", "cpu", etc.)
            freeze_gemma_vision: If True, freeze Gemma's vision encoder (SigLIP)
            freeze_gemma_language: If True, freeze Gemma's language model
            freeze_cosmos_vae: If True, freeze Cosmos VAE encoder
            **kwargs: Passed to parent's from_pretrained (dtype, device_map, etc.)

        Returns:
            Initialized TheWorld model with loaded weights
        """
        import gc

        # Auto-detect device if not provided
        # Handle different distributed training scenarios
        if device is None:
            import torch.distributed as dist

            # Check if using FSDP (requires CPU init with cpu_ram_efficient_loading)
            using_fsdp = False
            try:
                from accelerate import PartialState

                state = PartialState()
                using_fsdp = hasattr(state, "distributed_type") and state.distributed_type.value == "FSDP"
            except Exception:
                # PartialState not available or not in distributed setup
                pass

            if using_fsdp:
                # FSDP with cpu_ram_efficient_loading needs CPU init
                # FSDP will handle device placement during wrapping
                device = "cpu"
            elif dist.is_initialized():
                # DDP - each rank uses its own GPU (rank 0 → cuda:0, rank 1 → cuda:1, etc.)
                local_rank = dist.get_rank() % torch.cuda.device_count()
                device = f"cuda:{local_rank}"
            elif torch.cuda.is_available():
                # Single GPU training
                device = "cuda"
            else:
                # CPU training
                device = "cpu"

        # Validate configuration
        if not enable_world and not freeze_cosmos_vae:
            raise ValueError(
                "Invalid configuration: enable_world=False but freeze_cosmos_vae=False. "
                "Cannot train Cosmos VAE when world model is not enabled."
            )

        # 1. Load config from path
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # 2. Check config type to determine loading path
        if isinstance(config, TheWorldConfig):
            # Case B: Loading saved TheWorld checkpoint
            # Use two-stage loading to avoid meta tensors
            import os
            from pathlib import Path

            # Check if local path or Hub repo ID
            if os.path.exists(pretrained_model_name_or_path):
                # Local checkpoint
                return cls.from_checkpoint(pretrained_model_name_or_path, device=device, **kwargs)
            else:
                # Hub checkpoint
                hf_token = kwargs.pop("token", None) or kwargs.pop("use_auth_token", None)
                return cls.from_checkpoint_hub(
                    repo_id=pretrained_model_name_or_path, device=device, hf_token=hf_token, **kwargs
                )

        else:
            # Case A: Initializing TheWorld from base Gemma (fresh or as part of checkpoint loading)
            print(f"Loading Gemma base: {pretrained_model_name_or_path}")

            # Load Gemma via parent's from_pretrained
            model = super(TheWorld, cls).from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
            print(f"✓ Gemma loaded (dtype: {next(model.parameters()).dtype})")

            # Create TheWorldConfig from Gemma config
            gemma_config_dict = model.config.to_dict()
            # Remove any conflicting keys that we're setting explicitly
            gemma_config_dict.pop("gemma_model_name", None)
            gemma_config_dict.pop("cosmos_model_name", None)
            gemma_config_dict.pop("enable_world", None)
            gemma_config_dict.pop("world_projection_mode", None)
            gemma_config_dict.pop("projection_architecture", None)
            gemma_config_dict.pop("freeze_gemma_vision", None)
            gemma_config_dict.pop("freeze_gemma_language", None)
            gemma_config_dict.pop("freeze_cosmos_vae", None)

            the_world_config = TheWorldConfig(
                gemma_model_name=pretrained_model_name_or_path,
                cosmos_model_name=cosmos_model_name,
                enable_world=enable_world,
                world_projection_mode=world_projection_mode,
                projection_architecture=projection_architecture,
                freeze_gemma_vision=freeze_gemma_vision,
                freeze_gemma_language=freeze_gemma_language,
                freeze_cosmos_vae=freeze_cosmos_vae,
                **gemma_config_dict,
            )

            model.config = the_world_config

            # Initialize placeholder attributes WITHOUT calling __init__
            # (calling __init__ would reinitialize the parent model and break device_map)
            model.processor = None
            model.cosmos_pipe = None
            model.cosmos_vae = None
            model.cosmos_vae_encoder = None
            model.cosmos_encoder = None
            model.fusion = None
            model.sow_token_id = None
            model.eow_token_id = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 3. Load processor (needed for tokenization)
        base_model = (
            model.config.gemma_model_name
            if hasattr(model.config, "gemma_model_name")
            else pretrained_model_name_or_path
        )
        model.processor = AutoProcessor.from_pretrained(
            base_model,
            local_files_only=False,
        )

        # 4. Load Cosmos (conditionally)
        if model.config.enable_world:
            print("Loading Cosmos pipeline...")
            from cosmos_guardrail import CosmosSafetyChecker

            safety_checker = CosmosSafetyChecker()
            model.cosmos_pipe = Cosmos2VideoToWorldPipeline.from_pretrained(
                model.config.cosmos_model_name,
                torch_dtype=torch.bfloat16,
                safety_checker=safety_checker,
                low_cpu_mem_usage=True,
                local_files_only=False,
            )
            # Move only VAE to GPU - keep safety checker in RAM to save memory
            model.cosmos_vae = model.cosmos_pipe.vae.to(device)
            model.cosmos_vae_encoder = model.cosmos_vae.encoder
            print(f"✓ Cosmos pipeline loaded (VAE on {device}, safety checker in RAM)")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 5. Create Cosmos components (encoder, fusion, tokens)
            assert model.cosmos_vae is not None, "Cosmos VAE must be loaded"
            cosmos_img_dim = getattr(model.cosmos_vae.config, "z_dim", 16)
            gemma_dim = model.config.text_config.hidden_size

            # Add world tokens
            custom_tokens = ["<start_of_world>", "<end_of_world>"]
            num_added = model.processor.tokenizer.add_special_tokens({"additional_special_tokens": custom_tokens})
            if num_added > 0:
                model.resize_token_embeddings(len(model.processor.tokenizer))
                print(f"✓ Added {num_added} custom tokens to vocabulary")

            model.sow_token_id = model.processor.tokenizer.convert_tokens_to_ids("<start_of_world>")
            model.eow_token_id = model.processor.tokenizer.convert_tokens_to_ids("<end_of_world>")
            print(f"✓ Custom token IDs: SOW={model.sow_token_id}, EOW={model.eow_token_id}")

            # Cosmos VAE Encoder (preprocessing + VAE encoding)
            model.cosmos_vae_encoder = CosmosVAEEncoder(
                cosmos_vae=model.cosmos_vae,
                device=device,
                freeze_vae=model.config.freeze_cosmos_vae,
            )

            # World Projector (reduction + projection)
            projection_config = WorldProjectionConfig(
                mode=model.config.world_projection_mode,
                architecture=model.config.projection_architecture,
            )
            model.world_projector = WorldProjector(
                config=projection_config,
                z_dim=cosmos_img_dim,
                gemma_dim=gemma_dim,
                device=device,
            )

            # EmbeddingFusion
            model.fusion = EmbeddingFusion(
                sow_token_id=model.sow_token_id,
                eow_token_id=model.eow_token_id,
            )
        else:
            print("⊘ Skipping Cosmos loading (Gemma-only mode)")

        # Workaround for RetinaFace bug (cosmos_guardrail dependency disables gradients globally)
        torch.set_grad_enabled(True)

        # 6. Apply freezing configuration
        model._apply_freezing()

        return model

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: Optional[str] = None, **kwargs: Any) -> "TheWorld":
        """
        Load TheWorld from a local checkpoint directory.

        This performs two-stage loading to avoid meta tensor issues:
        1. Initialize base models (Gemma + Cosmos) with full weights
        2. Apply checkpoint weights on top (only trainable parameters)

        Args:
            checkpoint_path: Path to checkpoint directory
            device: Device to load model on
            **kwargs: Additional arguments passed to from_pretrained

        Returns:
            Initialized TheWorld model with checkpoint weights loaded

        Example:
            >>> model = TheWorld.from_checkpoint("./checkpoints/checkpoint-1000")
        """
        from pathlib import Path
        from safetensors.torch import load_file

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")

        # Load config to get base model names and settings
        config = TheWorldConfig.from_pretrained(checkpoint_path)

        # Stage 1: Initialize from base models (gets all weights, no meta tensors)
        model = cls.from_pretrained(
            config.gemma_model_name,
            device=device,
            enable_world=config.enable_world,
            cosmos_model_name=config.cosmos_model_name,
            world_projection_mode=config.world_projection_mode,
            projection_architecture=getattr(config, "projection_architecture", "mlp"),
            freeze_gemma_vision=config.freeze_gemma_vision,
            freeze_gemma_language=config.freeze_gemma_language,
            freeze_cosmos_vae=config.freeze_cosmos_vae,
            **kwargs,
        )

        # Stage 2: Load checkpoint weights
        print(f"Loading checkpoint weights from: {checkpoint_path}")

        # Try safetensors first, then PyTorch
        if (checkpoint_path / "model.safetensors").exists():
            state_dict = load_file(checkpoint_path / "model.safetensors")
        elif (checkpoint_path / "pytorch_model.bin").exists():
            state_dict = torch.load(checkpoint_path / "pytorch_model.bin", map_location="cpu")
        else:
            raise ValueError(f"No checkpoint file found in {checkpoint_path}")

        # Load weights (strict=False allows missing keys for frozen components)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        print(f"✓ Loaded checkpoint weights ({len(state_dict)} tensors)")
        if missing_keys:
            print(f"  Note: {len(missing_keys)} weights loaded from base model (expected for frozen components)")

        return model

    @classmethod
    def from_checkpoint_hub(
        cls,
        repo_id: str,
        checkpoint_name: Optional[str] = None,
        device: Optional[str] = None,
        hf_token: Optional[str] = None,
        **kwargs: Any,
    ) -> "TheWorld":
        """
        Load TheWorld from a HuggingFace Hub checkpoint.

        This performs two-stage loading to avoid meta tensor issues:
        1. Initialize base models (Gemma + Cosmos) with full weights
        2. Apply checkpoint weights on top (only trainable parameters)

        Args:
            repo_id: HuggingFace Hub repository ID (e.g., "username/theworld-spatial")
            checkpoint_name: Specific checkpoint subdirectory (e.g., "checkpoint-1000").
                           If None, loads from root of repo.
            device: Device to load model on
            hf_token: HuggingFace API token for private repos
            **kwargs: Additional arguments passed to from_pretrained

        Returns:
            Initialized TheWorld model with checkpoint weights loaded

        Example:
            >>> # Load latest checkpoint from root
            >>> model = TheWorld.from_checkpoint_hub("kasohrab/theworld-spatial")

            >>> # Load specific checkpoint
            >>> model = TheWorld.from_checkpoint_hub(
            ...     "kasohrab/theworld-spatial",
            ...     checkpoint_name="checkpoint-1000"
            ... )
        """
        from huggingface_hub import snapshot_download
        import os

        # Download checkpoint from Hub
        print(f"Downloading checkpoint from Hub: {repo_id}")
        if checkpoint_name:
            print(f"  Checkpoint: {checkpoint_name}")

        local_dir = snapshot_download(
            repo_id=repo_id,
            allow_patterns=["*.json", "*.safetensors", "*.bin"] if not checkpoint_name else [f"{checkpoint_name}/*"],
            token=hf_token or os.environ.get("HF_TOKEN"),
        )

        # Determine checkpoint path
        if checkpoint_name:
            checkpoint_path = os.path.join(local_dir, checkpoint_name)
        else:
            checkpoint_path = local_dir

        # Use from_checkpoint for the actual loading
        return cls.from_checkpoint(checkpoint_path, device=device, **kwargs)

    def get_trainable_parameters(self):
        """Return count and percentage of trainable parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total, 100 * trainable / total

    def _apply_freezing(self):
        """Apply freezing configuration to model components."""
        # 1. Freeze/unfreeze Gemma vision encoder (SigLIP) + projector
        if self.config.freeze_gemma_vision:
            for param in self.model.vision_tower.parameters():
                param.requires_grad = False
            for param in self.model.multi_modal_projector.parameters():
                param.requires_grad = False
            print("✓ Gemma vision encoder (SigLIP) + projector frozen")
        else:
            for param in self.model.vision_tower.parameters():
                param.requires_grad = True
            for param in self.model.multi_modal_projector.parameters():
                param.requires_grad = True
            print("✓ Gemma vision encoder (SigLIP) + projector trainable")

        # 2. Freeze/unfreeze Gemma language model
        if self.config.freeze_gemma_language:
            for param in self.model.language_model.parameters():
                param.requires_grad = False
            for param in self.lm_head.parameters():
                param.requires_grad = False
            print("✓ Gemma language model frozen")
        else:
            for param in self.model.language_model.parameters():
                param.requires_grad = True
            for param in self.lm_head.parameters():
                param.requires_grad = True
            print("✓ Gemma language model trainable")

        # 3. Freeze/unfreeze Cosmos VAE (entire VAE, not just encoder)
        if self.config.enable_world and self.cosmos_vae is not None:
            if self.config.freeze_cosmos_vae:
                for param in self.cosmos_vae.parameters():
                    param.requires_grad = False
                print("✓ Cosmos VAE (encoder + decoder) frozen")
            else:
                for param in self.cosmos_vae.parameters():
                    param.requires_grad = True
                print("✓ Cosmos VAE (encoder + decoder) trainable")

            # 4. Always keep projection layer trainable
            if self.world_projector is not None:
                for param in self.world_projector.parameters():
                    param.requires_grad = True
                print("✓ World projection layer trainable")

        # Print final trainable parameter count
        trainable, total, pct = self.get_trainable_parameters()
        print(f"✓ Trainable parameters: {trainable:,} / {total:,} ({pct:.2f}%)")

    def _inject_world_tokens(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
    ) -> tuple[torch.LongTensor, Optional[torch.Tensor]]:
        """Inject SOW and EOW tokens at the beginning of the sequence.

        Inserts tokens as: [BOS, SOW, EOW, ...rest...]

        Args:
            input_ids: Token IDs (B, seq_len)
            attention_mask: Attention mask (B, seq_len)

        Returns:
            Modified input_ids and attention_mask with world tokens inserted
        """
        batch_size, seq_len = input_ids.shape

        # Handle empty batch (all samples filtered by collator)
        if batch_size == 0 or seq_len == 0:
            return input_ids, attention_mask

        device = input_ids.device

        # Create new input_ids with space for SOW and EOW tokens
        # Structure: [BOS, SOW, EOW, ...rest...]
        new_input_ids = torch.zeros((batch_size, seq_len + 2), dtype=input_ids.dtype, device=device)

        # Copy BOS token (position 0)
        new_input_ids[:, 0] = input_ids[:, 0]

        # Insert SOW and EOW tokens after BOS
        new_input_ids[:, 1] = self.sow_token_id  # type: ignore
        new_input_ids[:, 2] = self.eow_token_id  # type: ignore

        # Copy rest of sequence
        new_input_ids[:, 3:] = input_ids[:, 1:]

        # Update attention mask if provided
        if attention_mask is not None:
            new_attention_mask = torch.zeros((batch_size, seq_len + 2), dtype=attention_mask.dtype, device=device)
            # Copy attention for BOS
            new_attention_mask[:, 0] = attention_mask[:, 0]
            # Set attention for SOW and EOW
            new_attention_mask[:, 1] = 1
            new_attention_mask[:, 2] = 1
            # Copy rest of attention mask
            new_attention_mask[:, 3:] = attention_mask[:, 1:]
            attention_mask = new_attention_mask

        return new_input_ids, attention_mask

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

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = False, assign: bool = False) -> Any:
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
            warnings.filterwarnings("ignore", message=".*were not found in the checkpoint.*", category=FutureWarning)
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

        # Only save trainable parameters (those with requires_grad=True)
        trainable_keys = {name for name, param in self.named_parameters() if param.requires_grad}
        trainable_state_dict = {k: v for k, v in self.state_dict().items() if k in trainable_keys}

        if safe_serialization:
            from safetensors.torch import save_file

            save_path = os.path.join(save_directory, "model.safetensors")
            save_file(trainable_state_dict, save_path)
        else:
            save_path = os.path.join(save_directory, "pytorch_model.bin")
            torch.save(trainable_state_dict, save_path)

        print(f"✓ Model saved to {save_directory}")
        print(f"  Trainable parameters: {len(trainable_state_dict)}")
        print(f"  Total parameters in model: {len(self.state_dict())}")

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
        if self.config.enable_world and images is not None:
            # Inject world tokens if not present
            if input_ids is not None and self.sow_token_id is not None and not (input_ids == self.sow_token_id).any():
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
        assert self.cosmos_vae_encoder is not None, "Cosmos VAE encoder must be loaded"
        assert self.world_projector is not None, "World projector must be loaded"
        assert self.fusion is not None, "Fusion module must be loaded"
        assert input_ids is not None, "input_ids is required for world-augmented forward"
        assert pixel_values is not None, "pixel_values is required for world-augmented forward"
        assert attention_mask is not None, "attention_mask is required for world-augmented forward"

        # 1. Get embeddings
        target_device = self.get_input_embeddings().weight.device
        input_ids = input_ids.to(target_device, non_blocking=True)  # type: ignore[assignment]
        pixel_values = pixel_values.to(target_device, non_blocking=True)  # type: ignore[assignment]
        attention_mask = attention_mask.to(target_device, non_blocking=True)  # type: ignore[assignment]

        inputs_embeds = self.model.language_model.embed_tokens(input_ids)
        assert inputs_embeds is not None, "inputs_embeds must not be None"

        # 2. Get vision features (reuse parent's method)
        image_features = self.model.get_image_features(pixel_values)
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype, non_blocking=True)
        special_image_mask = self.model.get_placeholder_mask(input_ids, inputs_embeds, image_features)  # type: ignore
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)  # type: ignore[assignment]

        # 3. Get world embeddings (two-step pipeline)
        latents = self.cosmos_vae_encoder(images)  # (B, z_dim, H, W)
        world_embeds = self.world_projector(latents)  # (B, num_tokens, gemma_dim)

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
                # Use tensor-based indexing to avoid .item() CPU-GPU sync
                batch_size = input_ids.size(0)
                num_world_tokens = world_embeds.size(1)

                # Get first positions as tensors (no .item() call)
                start_pos = start_positions[0]
                end_pos = end_positions[0]

                # Build labels: [tokens_before | -100 for world | tokens_after]
                # Use index_select to avoid integer indexing which would require .item()
                labels_before = input_ids[:, : start_pos + 1].to(target_device, non_blocking=True)
                labels_world = torch.full((batch_size, num_world_tokens), -100, dtype=torch.long, device=target_device)
                labels_after = input_ids[:, end_pos:].to(target_device, non_blocking=True)
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
            self.config.enable_world
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
                        torch.ones((batch_size, 2), device=attention_mask.device, dtype=attention_mask.dtype),
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
