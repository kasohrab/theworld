"""Refactored TheWorld model - inherits from Gemma3ForConditionalGeneration."""

import torch
import torch.nn as nn
from PIL import Image
from typing import Any, Dict, List, Optional, Union
from transformers import Gemma3ForConditionalGeneration, Gemma3Config, AutoProcessor
from transformers.cache_utils import Cache
from diffusers.pipelines.cosmos.pipeline_cosmos2_video2world import Cosmos2VideoToWorldPipeline
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from huggingface_hub import hf_hub_download

from .cosmos_encoder import CosmosEncoder
from .fusion import EmbeddingFusion
from ..constants import DEFAULT_COSMOS_MODEL, DEFAULT_GEMMA_MODEL


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

    def __init__(self, config: Gemma3Config):
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
    def from_pretrained(  # type: ignore[override]
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
        **kwargs: Any  # dtype, device_map, low_cpu_mem_usage, etc.
    ) -> "TheWorld":
        """
        Load TheWorld model from pretrained Gemma3 weights, then add Cosmos components.

        This is the recommended way to create a TheWorld model. It properly handles:
        - Loading pretrained Gemma3 weights with correct dtype
        - Adding Cosmos world model components
        - Setting up special tokens
        - Applying freezing configuration

        Args:
            pretrained_model_name_or_path: HuggingFace model ID for Gemma (e.g., "google/gemma-3-4b-it")
            enable_world: If True, add Cosmos world model. If False, Gemma-only baseline
            cosmos_model_name: HuggingFace model ID for Cosmos world model
            device: Device to load Cosmos on ("cuda", "cpu", etc.)
            freeze_gemma_vision: If True, freeze Gemma's vision encoder (SigLIP)
            freeze_gemma_language: If True, freeze Gemma's language model
            freeze_cosmos_vae: If True, freeze Cosmos VAE encoder
            random_projection_init: If True, randomly initialize projection (ablation)
            load_full_cosmos_pipeline: If True, load full pipeline; else VAE only
            **kwargs: Passed to parent's from_pretrained (dtype, device_map, etc.)

        Returns:
            Initialized TheWorld model with loaded weights

        Example:
            >>> model = TheWorld.from_pretrained(
            ...     "google/gemma-3-4b-it",
            ...     enable_world=True,
            ...     dtype=torch.bfloat16,
            ...     device_map="auto"
            ... )
        """
        import gc

        # Validate configuration
        if not enable_world and not freeze_cosmos_vae:
            raise ValueError(
                "Invalid configuration: enable_world=False but freeze_cosmos_vae=False. "
                "Cannot train Cosmos VAE when world model is not enabled."
            )

        # 1. Load Gemma fully via parent's from_pretrained
        # This handles dtype, device_map, weight loading, and buffer dtypes correctly
        # NOTE: We pass *model_args and **kwargs to parent, which handles HF parameters
        print("Loading Gemma 3 vision-language model...")
        model = super(TheWorld, cls).from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        print(f"✓ Gemma loaded (dtype: {next(model.parameters()).dtype})")

        # 2. Store configuration
        model.gemma_model_name = pretrained_model_name_or_path
        model.cosmos_model_name = cosmos_model_name
        model.enable_world = enable_world
        model.freeze_gemma_vision = freeze_gemma_vision
        model.freeze_gemma_language = freeze_gemma_language
        model.freeze_cosmos_vae = freeze_cosmos_vae
        model.random_projection_init = random_projection_init
        model.load_full_cosmos_pipeline = load_full_cosmos_pipeline

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 3. Load processor (needed for tokenization)
        model.processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=False,
        )

        # 4. Load Cosmos (conditionally)
        if enable_world:
            if load_full_cosmos_pipeline:
                print("Loading full Cosmos pipeline...")
                from cosmos_guardrail import CosmosSafetyChecker

                safety_checker = CosmosSafetyChecker()
                model.cosmos_pipe = Cosmos2VideoToWorldPipeline.from_pretrained(
                    cosmos_model_name,
                    torch_dtype=torch.bfloat16,
                    safety_checker=safety_checker,
                    low_cpu_mem_usage=True,
                    local_files_only=False,
                )
                # Move only VAE to GPU - keep safety checker in RAM to save memory
                model.cosmos_vae = model.cosmos_pipe.vae.to(device)
                print(f"✓ Full Cosmos pipeline loaded (VAE on {device}, safety checker in RAM)")
            else:
                print("Loading Cosmos VAE only...")
                cosmos_vae = AutoencoderKL.from_pretrained(
                    cosmos_model_name,
                    subfolder="vae",
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    local_files_only=False,
                )
                model.cosmos_vae = cosmos_vae.to(device)  # type: ignore
                model.cosmos_pipe = None
                print(f"✓ Cosmos VAE loaded to {device}")

            assert model.cosmos_vae is not None, "Cosmos VAE must be loaded"
            model.cosmos_vae_encoder = model.cosmos_vae.encoder
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            print("⊘ Skipping Cosmos loading (Gemma-only mode)")
            model.cosmos_pipe = None
            model.cosmos_vae = None
            model.cosmos_vae_encoder = None

        # 5. Create Cosmos components (encoder, fusion, tokens)
        if enable_world:
            assert model.cosmos_vae is not None, "Cosmos VAE must be loaded"
            cosmos_img_dim = getattr(model.cosmos_vae.config, "z_dim", 16)
            gemma_dim = model.config.text_config.hidden_size

            # Add world tokens
            custom_tokens = ["<start_of_world>", "<end_of_world>"]
            num_added = model.processor.tokenizer.add_special_tokens(
                {"additional_special_tokens": custom_tokens}
            )
            if num_added > 0:
                model.resize_token_embeddings(len(model.processor.tokenizer))
                print(f"✓ Added {num_added} custom tokens to vocabulary")

            model.sow_token_id = model.processor.tokenizer.convert_tokens_to_ids("<start_of_world>")
            model.eow_token_id = model.processor.tokenizer.convert_tokens_to_ids("<end_of_world>")
            print(f"✓ Custom token IDs: SOW={model.sow_token_id}, EOW={model.eow_token_id}")

            # CosmosEncoder
            model.cosmos_encoder = CosmosEncoder(
                cosmos_vae=model.cosmos_vae,
                cosmos_dim=cosmos_img_dim,
                gemma_dim=gemma_dim,
                device=device,
                freeze_vae=freeze_cosmos_vae,
            )

            if random_projection_init:
                nn.init.xavier_uniform_(model.cosmos_encoder.world_projection.weight)
                if model.cosmos_encoder.world_projection.bias is not None:
                    nn.init.zeros_(model.cosmos_encoder.world_projection.bias)
                print("⚠ Projection layer randomly initialized (ablation mode)")

            # EmbeddingFusion
            model.fusion = EmbeddingFusion(
                sow_token_id=model.sow_token_id,
                eow_token_id=model.eow_token_id,
            )
        else:
            model.cosmos_encoder = None
            model.fusion = None
            model.sow_token_id = None
            model.eow_token_id = None

        # CRITICAL FIX: Re-enable gradients after Cosmos import
        torch.set_grad_enabled(True)

        # 6. Apply freezing configuration
        model._apply_freezing()

        return model

    def _apply_freezing(self):
        """Apply freezing configuration to model components."""
        # 1. Freeze/unfreeze Gemma vision encoder (SigLIP)
        if self.freeze_gemma_vision:
            for param in self.model.vision_tower.parameters():
                param.requires_grad = False
            print("✓ Gemma vision encoder (SigLIP) frozen")
        else:
            for param in self.model.vision_tower.parameters():
                param.requires_grad = True
            print("✓ Gemma vision encoder (SigLIP) trainable")

        # 2. Freeze/unfreeze Gemma language model
        if self.freeze_gemma_language:
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

        # 3. Freeze/unfreeze Cosmos VAE encoder (only if Cosmos loaded)
        if self.enable_world and self.cosmos_vae_encoder is not None:
            if self.freeze_cosmos_vae:
                for param in self.cosmos_vae_encoder.parameters():
                    param.requires_grad = False
                print("✓ Cosmos VAE encoder frozen")
            else:
                for param in self.cosmos_vae_encoder.parameters():
                    param.requires_grad = True
                print("✓ Cosmos VAE encoder trainable")

            # 4. Always keep projection layer trainable
            if self.cosmos_encoder is not None:
                for param in self.cosmos_encoder.world_projection.parameters():
                    param.requires_grad = True
                print("✓ World projection layer trainable")

        # Print final trainable parameter count
        trainable, total, pct = self.get_trainable_parameters()
        print(f"✓ Trainable parameters: {trainable:,} / {total:,} ({pct:.2f}%)")

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
        is_main_process: bool = True,
        state_dict: Optional[Dict[str, Any]] = None,
        save_function: Any = None,
        push_to_hub: bool = False,
        max_shard_size: str = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[str] = None,
        save_peft_format: bool = True,
        **kwargs: Any,
    ) -> None:
        """Save model for HuggingFace Trainer compatibility.

        Saves only trainable parameters + metadata to the directory.
        Frozen models (Gemma, Cosmos) are not saved - they will be reloaded from HuggingFace.

        Args:
            save_directory: Directory to save the model
            is_main_process: Whether this is the main process (for distributed training)
            state_dict: Optional pre-computed state dict (uses self.state_dict() if None)
            safe_serialization: If True, save as safetensors format (faster, safer)
            **kwargs: Additional arguments passed to parent (ignored for trainable-only saves)

        Example:
            >>> model.save_pretrained("./checkpoints/checkpoint-1000")
            >>> # Save as pickle format for backward compatibility
            >>> model.save_pretrained("./checkpoints/checkpoint-1000", safe_serialization=False)
        """
        import os

        if not is_main_process:
            return

        os.makedirs(save_directory, exist_ok=True)

        # Use provided state_dict or get trainable params only
        if state_dict is None:
            state = self.state_dict()
        else:
            state = state_dict

        # Save in requested format
        if safe_serialization:
            from safetensors.torch import save_file
            import json

            save_path = os.path.join(save_directory, "model.safetensors")
            save_file(state, save_path)

            # Save config separately for safetensors (needed for loading)
            config_data = {
                "model_config": {
                    "gemma_model_name": self.gemma_model_name,
                    "cosmos_model_name": self.cosmos_model_name,
                    "enable_world": self.enable_world,
                    "load_full_cosmos_pipeline": self.load_full_cosmos_pipeline,
                },
                "freeze_config": {
                    "freeze_gemma_vision": self.freeze_gemma_vision,
                    "freeze_gemma_language": self.freeze_gemma_language,
                    "freeze_cosmos_vae": self.freeze_cosmos_vae,
                },
            }
            config_path = os.path.join(save_directory, "checkpoint_config.json")
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)

            print(f"✓ Model saved to {save_directory} (safetensors format, trainable parameters only)")
        else:
            save_path = os.path.join(save_directory, "pytorch_model.bin")
            torch.save(state, save_path)
            print(f"✓ Model saved to {save_directory} (pickle format, trainable parameters only)")

    @classmethod
    def from_checkpoint_hub(
        cls,
        model_id: str,
        checkpoint_name: Optional[str] = None,
        device: str = "cuda",
        hf_token: Optional[str] = None,
    ) -> "TheWorld":
        """Load a trained TheWorld checkpoint from HuggingFace Hub.

        NOTE: This loads a trained checkpoint of TheWorld, not a base model.
        To create a new TheWorld model from Gemma3, use `from_pretrained()`.

        Args:
            model_id: HuggingFace Hub model ID (e.g., "username/theworld-datacomp")
            checkpoint_name: Name of checkpoint file to load. If None, tries safetensors first, then .bin
                            Can also use "checkpoint-1000/model.safetensors" for specific checkpoint
            device: Device to load model on (default: "cuda")
            hf_token: HuggingFace API token for private models

        Returns:
            TheWorld model instance with loaded weights

        Example:
            >>> model = TheWorld.from_checkpoint_hub("username/theworld-datacomp")
            >>> # Use parent's generate() method
            >>> outputs = model.generate(...)

            # Load specific checkpoint
            >>> model = TheWorld.from_checkpoint_hub(
            ...     "username/theworld-datacomp",
            ...     checkpoint_name="checkpoint-1000/model.safetensors"
            ... )
        """
        from safetensors.torch import load_file as load_safetensors

        print(f"Downloading checkpoint from HuggingFace Hub: {model_id}")

        # Auto-detect format if not specified
        if checkpoint_name is None:
            # Try safetensors first (faster), fallback to pickle
            try:
                checkpoint_path = hf_hub_download(
                    repo_id=model_id,
                    filename="model.safetensors",
                    token=hf_token,
                )
                checkpoint_name = "model.safetensors"
                print("✓ Using safetensors format (faster loading)")
            except Exception:
                try:
                    checkpoint_path = hf_hub_download(
                        repo_id=model_id,
                        filename="pytorch_model.bin",
                        token=hf_token,
                    )
                    checkpoint_name = "pytorch_model.bin"
                    print("⚠ Using pickle format (consider re-saving with safe_serialization=True)")
                except Exception as e:
                    raise ValueError(
                        f"Could not find checkpoint in {model_id}. "
                        f"Tried: model.safetensors, pytorch_model.bin. Error: {e}"
                    )
        else:
            # Download specified checkpoint
            checkpoint_path = hf_hub_download(
                repo_id=model_id,
                filename=checkpoint_name,
                token=hf_token,
            )

        # Load checkpoint based on file extension
        if checkpoint_name.endswith(".safetensors"):
            # Load safetensors format
            state_dict = load_safetensors(checkpoint_path)
            # For safetensors, we need to extract config from a separate file or use defaults
            # Try to download config if it exists
            try:
                config_path = hf_hub_download(
                    repo_id=model_id,
                    filename="checkpoint_config.json",
                    token=hf_token,
                )
                import json

                with open(config_path, "r") as f:
                    checkpoint_config = json.load(f)
                model_config = checkpoint_config.get("model_config", {})
                freeze_config = checkpoint_config.get("freeze_config", {})
            except Exception:
                # Use defaults if config not found
                model_config = {}
                freeze_config = {}
                print("⚠ Config file not found, using defaults")

            checkpoint = {"model_state_dict": state_dict}
        else:
            # Load pickle format
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model_config = checkpoint.get("model_config", {})
            freeze_config = checkpoint.get("freeze_config", {})

        # Get model names from checkpoint
        gemma_model_name = model_config.get("gemma_model_name", DEFAULT_GEMMA_MODEL)
        cosmos_model_name = model_config.get("cosmos_model_name", DEFAULT_COSMOS_MODEL)
        load_full_cosmos_pipeline = model_config.get("load_full_cosmos_pipeline", True)
        enable_world = model_config.get("enable_world", True)

        # Get freeze configuration
        freeze_gemma_vision = freeze_config.get("freeze_gemma_vision", True)
        freeze_gemma_language = freeze_config.get("freeze_gemma_language", True)
        freeze_cosmos_vae = freeze_config.get("freeze_cosmos_vae", True)

        print(f"Initializing TheWorld model with:")
        print(f"  Gemma: {gemma_model_name}")
        print(f"  Cosmos: {cosmos_model_name}")
        print(f"  Load full Cosmos pipeline: {load_full_cosmos_pipeline}")
        print(f"  Enable world: {enable_world}")

        # Initialize model with configuration from checkpoint
        model = cls.from_pretrained(
            gemma_model_name,
            enable_world=enable_world,
            cosmos_model_name=cosmos_model_name,
            device=device,
            freeze_gemma_vision=freeze_gemma_vision,
            freeze_gemma_language=freeze_gemma_language,
            freeze_cosmos_vae=freeze_cosmos_vae,
            load_full_cosmos_pipeline=load_full_cosmos_pipeline,
            dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        # Load trainable parameters
        print("Loading checkpoint weights...")
        missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        if missing:
            print(f"⚠ Missing keys (expected for frozen components): {len(missing)} keys")
        if unexpected:
            print(f"⚠ Unexpected keys: {unexpected}")

        print(f"✓ Model loaded from {model_id}")
        return model

    def save_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        **kwargs: Union[str, int, float, bool],
    ) -> None:
        """Save training checkpoint with trainable parameters and optimizer state.

        Includes model state, optimizer state, training metadata (epoch/step), and config.
        Use this for resuming training. For inference-only deployment, use save_pretrained().

        Args:
            path: Path to save checkpoint (.pt file)
            optimizer: Optional optimizer to save state
            epoch: Current epoch number
            step: Current training step
            **kwargs: Additional metadata to save

        Example:
            >>> model.save_checkpoint("checkpoint.pt", optimizer=optimizer, epoch=5)
        """
        checkpoint = {
            "model_state_dict": {},
            "freeze_config": {
                "freeze_gemma_vision": self.freeze_gemma_vision,
                "freeze_gemma_language": self.freeze_gemma_language,
                "freeze_cosmos_vae": self.freeze_cosmos_vae,
            },
            "model_config": {
                "gemma_model_name": self.gemma_model_name,
                "cosmos_model_name": self.cosmos_model_name,
                "load_full_cosmos_pipeline": self.load_full_cosmos_pipeline,
                "enable_world": self.enable_world,
            },
            "epoch": epoch,
            "step": step,
            **kwargs,
        }

        # Save only trainable parameters (much smaller checkpoint)
        for name, param in self.named_parameters():
            if param.requires_grad:
                checkpoint["model_state_dict"][name] = param.data.cpu()

        # Save optimizer state
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint, path)
        print(f"✓ Checkpoint saved to {path}")

    def load_checkpoint(
        self, path: str, optimizer: Optional[torch.optim.Optimizer] = None, strict: bool = False
    ) -> Dict[str, int]:
        """Load training checkpoint and resume training.

        Args:
            path: Path to checkpoint file
            optimizer: Optional optimizer to load state into
            strict: Whether to strictly enforce all keys match

        Returns:
            Dictionary with epoch and step if available

        Example:
            >>> info = model.load_checkpoint("checkpoint.pt", optimizer=optimizer)
            >>> print(f"Resuming from epoch {info['epoch']}")
        """
        checkpoint = torch.load(path, map_location=self.device)

        # Load trainable parameters
        missing, unexpected = self.load_state_dict(checkpoint["model_state_dict"], strict=strict)

        if missing and strict:
            print(f"⚠ Missing keys: {missing}")
        if unexpected and strict:
            print(f"⚠ Unexpected keys: {unexpected}")

        # Load optimizer state
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("✓ Optimizer state loaded")

        print(f"✓ Checkpoint loaded from {path}")

        return {"epoch": checkpoint.get("epoch", 0), "step": checkpoint.get("step", 0)}

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for memory-efficient training.

        This reduces activation memory by 4-8x at the cost of 30-40% slower training.
        Only beneficial when training large portions of the model (e.g., unfrozen language model).

        Automatically enables for:
        - Gemma language model (via parent's gradient_checkpointing_enable())
        - Cosmos VAE encoder (if unfrozen)

        Example:
            >>> model = TheWorld("google/gemma-3-4b-it", freeze_gemma_language=False)
            >>> model.enable_gradient_checkpointing()  # Reduces memory usage
        """
        # Enable for Gemma (language model + vision tower) via parent class method
        if not self.freeze_gemma_language:
            super().gradient_checkpointing_enable()
            print("✓ Gradient checkpointing enabled for Gemma language model")

        # Enable for Cosmos VAE encoder (if unfrozen and supports it)
        if not self.freeze_cosmos_vae and self.cosmos_vae_encoder is not None:
            if hasattr(self.cosmos_vae_encoder, "gradient_checkpointing_enable"):
                self.cosmos_vae_encoder.gradient_checkpointing_enable()
                print("✓ Gradient checkpointing enabled for Cosmos VAE encoder")
            elif hasattr(self.cosmos_vae_encoder, "enable_gradient_checkpointing"):
                self.cosmos_vae_encoder.enable_gradient_checkpointing()
                print("✓ Gradient checkpointing enabled for Cosmos VAE encoder")

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
        device = input_ids.device

        # print(f"[TheWorld] Injecting world tokens (SOW={self.sow_token_id}, EOW={self.eow_token_id}) into batch of {batch_size} sequences")

        # Create new input_ids with space for SOW and EOW tokens
        # Structure: [BOS, SOW, EOW, ...rest...]
        new_input_ids = torch.zeros(
            (batch_size, seq_len + 2), dtype=input_ids.dtype, device=device
        )

        # Copy BOS token (position 0)
        new_input_ids[:, 0] = input_ids[:, 0]

        # Insert SOW and EOW tokens after BOS
        new_input_ids[:, 1] = self.sow_token_id
        new_input_ids[:, 2] = self.eow_token_id

        # Copy rest of sequence
        new_input_ids[:, 3:] = input_ids[:, 1:]

        # Update attention mask if provided
        if attention_mask is not None:
            new_attention_mask = torch.zeros(
                (batch_size, seq_len + 2), dtype=attention_mask.dtype, device=device
            )
            # Copy attention for BOS
            new_attention_mask[:, 0] = attention_mask[:, 0]
            # Set attention for SOW and EOW
            new_attention_mask[:, 1] = 1
            new_attention_mask[:, 2] = 1
            # Copy rest of attention mask
            new_attention_mask[:, 3:] = attention_mask[:, 1:]
            attention_mask = new_attention_mask

        return new_input_ids, attention_mask

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
