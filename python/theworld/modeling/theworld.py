"""Main TheWorld model class."""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import List, Optional, Union, Dict
from torch import Tensor
from transformers import Gemma3ForConditionalGeneration, AutoProcessor, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from diffusers.pipelines.cosmos.pipeline_cosmos2_video2world import Cosmos2VideoToWorldPipeline
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from huggingface_hub import hf_hub_download

from .cosmos_encoder import CosmosEncoder
from .gemma_vision import GemmaVisionEncoder
from .fusion import EmbeddingFusion


class TheWorld(nn.Module):
    def __init__(
        self,
        gemma_model_name,
        cosmos_model_name="nvidia/Cosmos-Predict2-2B-Video2World",
        device="cuda",
        freeze_gemma_vision=True,
        freeze_gemma_language=True,
        freeze_cosmos_vae=True,
        random_projection_init=False,
        load_full_cosmos_pipeline=True,
    ):
        """
        TheWorld: Fused vision-language-world model combining Gemma 3 and Cosmos.

        Args:
            gemma_model_name: HuggingFace model ID for Gemma/PaliGemma vision-language model
            cosmos_model_name: HuggingFace model ID for Cosmos world model
                               (2B: nvidia/Cosmos-Predict2-2B-Video2World,
                                7B: nvidia/Cosmos-Predict2-7B-Video2World,
                                14B: nvidia/Cosmos-Predict2-14B-Video2World)
            device: Device to load models on
            freeze_gemma_vision: If True, freeze Gemma's vision encoder (SigLIP) [default: True]
            freeze_gemma_language: If True, freeze Gemma's language model [default: True]
            freeze_cosmos_vae: If True, freeze Cosmos VAE encoder [default: True]
            random_projection_init: If True, randomly initialize projection layer (for ablation) [default: False]
            load_full_cosmos_pipeline: If True (default), load full Cosmos pipeline (transformer+VAE+scheduler).
                                      If False, attempt to load only VAE (currently not supported due to custom architecture) [default: True]

        Trainable components (by default):
            - world_projection: Projects Cosmos latents (16-dim) to Gemma dimension (2304-dim)

        Note: Cosmos encoder now uses single-frame encoding only (no temporal prediction).
              Cosmos VAE uses custom architecture, so full pipeline is required (load_full_cosmos_pipeline must be True).
        """
        super(TheWorld, self).__init__()

        self.device = device
        self.random_projection_init = random_projection_init  # For ablation studies
        self.load_full_cosmos_pipeline = load_full_cosmos_pipeline  # Feature flag for optimization

        # Store model names for checkpointing
        self.gemma_model_name = gemma_model_name
        self.cosmos_model_name = cosmos_model_name

        # Store freeze configuration
        self.freeze_gemma_vision = freeze_gemma_vision
        self.freeze_gemma_language = freeze_gemma_language
        self.freeze_cosmos_vae = freeze_cosmos_vae

        # --- 1. Load Gemma first (uses device_map="auto" for efficient GPU loading) ---
        print("Loading Gemma 3 vision-language model...")
        import gc

        self.gemma = Gemma3ForConditionalGeneration.from_pretrained(
            gemma_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=False,
        )
        print(f"✓ Gemma loaded to {self.gemma.device if hasattr(self.gemma, 'device') else 'auto'}")

        # Aggressive memory cleanup before loading Cosmos
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- 2. Load Cosmos (conditionally: full pipeline or VAE only) ---
        if load_full_cosmos_pipeline:
            # Load full pipeline (includes transformer, VAE, scheduler, safety checker)
            # Useful for future features like autoregressive rollout
            print("Loading full Cosmos pipeline (transformer + VAE + scheduler)...")
            from cosmos_guardrail import CosmosSafetyChecker

            safety_checker = CosmosSafetyChecker()
            self.cosmos_pipe: Optional[Cosmos2VideoToWorldPipeline] = Cosmos2VideoToWorldPipeline.from_pretrained(
                cosmos_model_name,
                torch_dtype=torch.bfloat16,
                safety_checker=safety_checker,
                low_cpu_mem_usage=True,
                local_files_only=False,
            )
            self.cosmos_pipe = self.cosmos_pipe.to(device)
            self.cosmos_vae = self.cosmos_pipe.vae
            print(f"✓ Full Cosmos pipeline loaded to {device}")
        else:
            # Optimized: Load only VAE (saves 2B-14B params from transformer/decoder)
            print("Loading Cosmos VAE only (optimized, skipping transformer)...")

            self.cosmos_vae = AutoencoderKL.from_pretrained(
                cosmos_model_name,
                subfolder="vae",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                local_files_only=False,
            )
            target_device: torch.device = torch.device(device)
            _ = self.cosmos_vae.to(target_device)
            self.cosmos_pipe = None  # Not loaded in optimized mode
            print(f"✓ Cosmos VAE loaded to {device} (transformer not loaded)")

        # Store reference to VAE encoder for freezing logic
        self.cosmos_vae_encoder = self.cosmos_vae.encoder

        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.processor = AutoProcessor.from_pretrained(
            gemma_model_name,
            local_files_only=False,
        )

        # Get the output dimensions from the Cosmos encoder and Gemma
        # We use the true latent space dimension (z_dim=16), not the encoder output (32)
        # See docs/world_model_latent_space.md for explanation
        cosmos_img_dim = getattr(self.cosmos_vae.config, "z_dim", 16)  # 16-dim latent space
        # Gemma3 uses 'hidden_size' in text_config
        gemma_dim = self.gemma.config.text_config.hidden_size

        # Add bracket special tokens for world model embeddings (before creating modules)
        special_tokens = {"additional_special_tokens": ["<the_world_start>", "<the_world_end>"]}
        num_added = self.processor.tokenizer.add_special_tokens(special_tokens)
        if num_added > 0:
            self.gemma.resize_token_embeddings(len(self.processor.tokenizer))

        # Store token IDs for use in modules
        self.world_start_id = self.processor.tokenizer.convert_tokens_to_ids("<the_world_start>")
        self.world_end_id = self.processor.tokenizer.convert_tokens_to_ids("<the_world_end>")

        # --- Create modular components ---

        # CosmosEncoder: Handles VAE encoding and projection (single-frame only)
        self.cosmos_encoder = CosmosEncoder(
            cosmos_vae=self.cosmos_vae,
            cosmos_dim=cosmos_img_dim,
            gemma_dim=gemma_dim,
            device=device,
            freeze_vae=freeze_cosmos_vae,
        )

        # Optionally reinitialize projection with random weights (for ablation studies)
        if random_projection_init:
            nn.init.xavier_uniform_(self.cosmos_encoder.world_projection.weight)
            if self.cosmos_encoder.world_projection.bias is not None:
                nn.init.zeros_(self.cosmos_encoder.world_projection.bias)
            print("⚠ Projection layer randomly initialized (ablation mode)")

        # GemmaVisionEncoder: Handles SigLIP encoding + embedding fusion
        self.gemma_vision = GemmaVisionEncoder(gemma_model=self.gemma)

        # EmbeddingFusion: Handles inserting world tokens between brackets
        self.fusion = EmbeddingFusion(
            world_start_id=self.world_start_id,
            world_end_id=self.world_end_id,
        )

        # Expose device map from Gemma to TheWorld (for HuggingFace Trainer compatibility)
        # This prevents Trainer from trying to move the model when device_map="auto" is used
        if hasattr(self.gemma, "hf_device_map"):
            self.hf_device_map = self.gemma.hf_device_map

        # Add minimal config for HuggingFace Hub compatibility
        # This allows Trainer to create model cards and upload to Hub
        class TheWorldConfig:
            def __init__(self, gemma_model_name, cosmos_model_name):
                # Use base Gemma model as _name_or_path for HuggingFace validation
                self._name_or_path = gemma_model_name
                self.model_type = "theworld"
                self.gemma_model_name = gemma_model_name
                self.cosmos_model_name = cosmos_model_name

        self.config = TheWorldConfig(gemma_model_name, cosmos_model_name)

        # CRITICAL FIX: Cosmos2VideoToWorldPipeline import globally disables gradients
        # Re-enable them for training (this is a bug in Cosmos diffusers implementation)
        torch.set_grad_enabled(True)

        # Apply freezing configuration (after all components are created)
        self._apply_freezing()

    def _apply_freezing(self):
        """Apply freezing configuration to model components."""

        # 1. Freeze/unfreeze Gemma vision encoder (SigLIP)
        if self.freeze_gemma_vision:
            for param in self.gemma.model.vision_tower.parameters():
                param.requires_grad = False
            print("✓ Gemma vision encoder (SigLIP) frozen")
        else:
            for param in self.gemma.model.vision_tower.parameters():
                param.requires_grad = True
            print("✓ Gemma vision encoder (SigLIP) trainable")

        # 2. Freeze/unfreeze Gemma language model
        if self.freeze_gemma_language:
            for param in self.gemma.model.language_model.parameters():
                param.requires_grad = False
            for param in self.gemma.lm_head.parameters():
                param.requires_grad = False
            print("✓ Gemma language model frozen")
        else:
            for param in self.gemma.model.language_model.parameters():
                param.requires_grad = True
            for param in self.gemma.lm_head.parameters():
                param.requires_grad = True
            print("✓ Gemma language model trainable")

        # 3. Freeze/unfreeze Cosmos VAE encoder
        if self.freeze_cosmos_vae:
            for param in self.cosmos_vae_encoder.parameters():
                param.requires_grad = False
            print("✓ Cosmos VAE encoder frozen")
        else:
            for param in self.cosmos_vae_encoder.parameters():
                param.requires_grad = True
            print("✓ Cosmos VAE encoder trainable")

        # 4. Always keep projection layer trainable (core trainable component)
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

    def state_dict(self, *args, **kwargs):
        """Save only trainable parameters.

        Frozen pretrained models (Gemma, Cosmos) are reloaded from HuggingFace on load.
        This makes checkpoints much smaller (~300MB vs ~17GB) and avoids duplicate parameter issues.

        Returns:
            Dictionary with only trainable parameters (safe for safetensors)
        """
        state = {}

        # Save only trainable parameters
        for name, param in self.named_parameters():
            if param.requires_grad:
                state[name] = param

        return state

    def load_state_dict(self, state_dict, strict=False):
        """Load trainable parameters from checkpoint.

        Frozen parameters are already loaded from HuggingFace during __init__.
        Only trainable parameters need to be restored from checkpoint.

        Args:
            state_dict: State dictionary with trainable parameters
            strict: Whether to require exact key match (default: False for trainable-only checkpoints)

        Returns:
            NamedTuple with missing_keys and unexpected_keys
        """
        # Load only the trainable parameters (strict=False to ignore missing frozen params)
        return super().load_state_dict(state_dict, strict=False)

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for memory-efficient training.

        This reduces activation memory by 4-8x at the cost of 30-40% slower training.
        Only beneficial when training large portions of the model.
        """
        # Enable for Gemma language model (if unfrozen)
        if not self.freeze_gemma_language:
            if hasattr(self.gemma, "gradient_checkpointing_enable") and callable(
                getattr(self.gemma, "gradient_checkpointing_enable")
            ):
                getattr(self.gemma, "gradient_checkpointing_enable")()
                print("✓ Gradient checkpointing enabled for Gemma language model")
            elif hasattr(self.gemma, "enable_gradient_checkpointing") and callable(
                getattr(self.gemma, "enable_gradient_checkpointing")
            ):
                getattr(self.gemma, "enable_gradient_checkpointing")()
                print("✓ Gradient checkpointing enabled for Gemma language model")

        # Enable for Cosmos VAE encoder (if unfrozen)
        if not self.freeze_cosmos_vae:
            if hasattr(self.cosmos_vae_encoder, "gradient_checkpointing_enable") and callable(
                getattr(self.cosmos_vae_encoder, "gradient_checkpointing_enable")
            ):
                getattr(self.cosmos_vae_encoder, "gradient_checkpointing_enable")()
                print("✓ Gradient checkpointing enabled for Cosmos VAE encoder")
            elif hasattr(self.cosmos_vae_encoder, "enable_gradient_checkpointing") and callable(
                getattr(self.cosmos_vae_encoder, "enable_gradient_checkpointing")
            ):
                getattr(self.cosmos_vae_encoder, "enable_gradient_checkpointing")()
                print("✓ Gradient checkpointing enabled for Cosmos VAE encoder")

    def save_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        **kwargs: Union[str, int, float, bool],
    ) -> None:
        """Save training checkpoint with trainable parameters and optimizer state.

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

    def save_pretrained(self, save_directory: str) -> None:
        """Save model for HuggingFace Trainer compatibility.

        Saves only trainable parameters + metadata to the directory.
        Frozen models (Gemma, Cosmos) are not saved - they will be reloaded from HuggingFace.

        Args:
            save_directory: Directory to save the model

        Example:
            >>> model.save_pretrained("./checkpoints/checkpoint-1000")
        """
        import os

        os.makedirs(save_directory, exist_ok=True)

        # Save state dict (trainable params only)
        state = self.state_dict()
        torch.save(state, os.path.join(save_directory, "pytorch_model.bin"))

        print(f"✓ Model saved to {save_directory} (trainable parameters only)")

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        checkpoint_name: str = "pytorch_model.bin",
        device: str = "cuda",
        hf_token: Optional[str] = None,
    ) -> "TheWorld":
        """Load a trained TheWorld model from HuggingFace Hub.

        Args:
            model_id: HuggingFace Hub model ID (e.g., "username/theworld-datacomp")
            checkpoint_name: Name of checkpoint file to load (default: "pytorch_model.bin")
                            Can also use "checkpoint-1000/pytorch_model.bin" for specific checkpoint
            device: Device to load model on (default: "cuda")
            hf_token: HuggingFace API token for private models

        Returns:
            TheWorld model instance with loaded weights

        Example:
            >>> model = TheWorld.from_pretrained("username/theworld-datacomp")
            >>> outputs = model.generate(image, "What is in this image?")

            # Load specific checkpoint
            >>> model = TheWorld.from_pretrained(
            ...     "username/theworld-datacomp",
            ...     checkpoint_name="checkpoint-1000/pytorch_model.bin"
            ... )
        """
        print(f"Downloading checkpoint from HuggingFace Hub: {model_id}")

        # Download checkpoint file from Hub
        checkpoint_path = hf_hub_download(
            repo_id=model_id,
            filename=checkpoint_name,
            token=hf_token,
        )

        # Load checkpoint to extract configuration
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Extract model configuration
        model_config = checkpoint.get("model_config", {})
        freeze_config = checkpoint.get("freeze_config", {})

        # Get model names from checkpoint
        gemma_model_name = model_config.get("gemma_model_name", "google/gemma-3-4b-it")
        cosmos_model_name = model_config.get("cosmos_model_name", "nvidia/Cosmos-Predict2-2B-Video2World")
        load_full_cosmos_pipeline = model_config.get("load_full_cosmos_pipeline", True)

        # Get freeze configuration
        freeze_gemma_vision = freeze_config.get("freeze_gemma_vision", True)
        freeze_gemma_language = freeze_config.get("freeze_gemma_language", True)
        freeze_cosmos_vae = freeze_config.get("freeze_cosmos_vae", True)

        print(f"Initializing TheWorld model with:")
        print(f"  Gemma: {gemma_model_name}")
        print(f"  Cosmos: {cosmos_model_name}")
        print(f"  Load full Cosmos pipeline: {load_full_cosmos_pipeline}")

        # Initialize model with configuration from checkpoint
        model = cls(
            gemma_model_name=gemma_model_name,
            cosmos_model_name=cosmos_model_name,
            device=device,
            freeze_gemma_vision=freeze_gemma_vision,
            freeze_gemma_language=freeze_gemma_language,
            freeze_cosmos_vae=freeze_cosmos_vae,
            load_full_cosmos_pipeline=load_full_cosmos_pipeline,
        )

        # Load trainable parameters
        print("Loading checkpoint weights...")
        missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        if missing:
            print(f"⚠ Missing keys (expected for frozen components): {len(missing)} keys")
        if unexpected:
            print(f"⚠ Unexpected keys: {unexpected}")

        print(f"✓ Model loaded from {model_id}")

        # Get training info if available
        epoch = checkpoint.get("epoch")
        step = checkpoint.get("step")
        if epoch is not None or step is not None:
            print(f"  Checkpoint info: epoch={epoch}, step={step}")

        return model

    def forward(
        self,
        input_ids: Tensor,
        pixel_values: Tensor,
        attention_mask: Tensor,
        images: List[Image.Image],
        labels: Optional[Tensor] = None,
    ):
        """Forward pass for TheWorld model using modular components.

        Args:
            input_ids: Token IDs from collator (B, seq_len)
            pixel_values: Preprocessed images for Gemma SigLIP (B, C, H, W)
            attention_mask: Attention mask (B, seq_len)
            images: Raw PIL images for Cosmos (List of B images)
            labels: Target labels for loss computation (B, label_len), optional

        Returns:
            CausalLMOutputWithPast with loss, logits, and other outputs
        """
        # ========================================
        # STEP 1: Encode world via Cosmos (single-frame only)
        # ========================================
        world_embeds = self.cosmos_encoder(images=images)
        # world_embeds: (B, num_world_tokens, 2304) where num_world_tokens = H × W (spatial tokens)

        # ========================================
        # STEP 2: Encode vision+text via Gemma
        # ========================================
        # Move inputs to device where Gemma is (may be distributed across GPUs)
        target_device = self.gemma.get_input_embeddings().weight.device
        input_ids = input_ids.to(target_device)
        pixel_values = pixel_values.to(target_device)
        attention_mask = attention_mask.to(target_device)

        gemma_output = self.gemma_vision(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
        )
        # gemma_output.embeddings: (B, seq_len, 2304) - combined vision+text embeddings

        # ========================================
        # STEP 3: Fuse embeddings (insert world tokens)
        # ========================================
        fusion_output = self.fusion(
            gemma_embeds=gemma_output.embeddings,
            world_embeds=world_embeds,
            input_ids=gemma_output.input_ids,
            attention_mask=gemma_output.attention_mask,
        )
        # fusion_output.combined_embeds: (B, combined_len, 2304)
        # fusion_output.combined_attention_mask: (B, combined_len)

        # ========================================
        # STEP 4: Forward through language model
        # ========================================
        lm_outputs = self.gemma.language_model(
            inputs_embeds=fusion_output.combined_embeds,
            attention_mask=fusion_output.combined_attention_mask,
            return_dict=True,
        )

        # ========================================
        # STEP 5: Apply LM head
        # ========================================
        logits = self.gemma.lm_head(lm_outputs.last_hidden_state)

        # ========================================
        # STEP 6: Compute loss
        # ========================================
        loss = None
        if labels is not None:
            # Align labels with combined sequence
            # Find bracket positions to create label mask
            start_positions = (input_ids == self.world_start_id).nonzero(as_tuple=True)[1]
            end_positions = (input_ids == self.world_end_id).nonzero(as_tuple=True)[1]

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

                # Shift for causal LM: predict token n from tokens < n
                shift_logits = logits[..., :-1, :].contiguous().float()
                shift_labels = combined_labels[..., 1:].contiguous()

                # Compute cross-entropy loss
                loss_fct = nn.CrossEntropyLoss()
                vocab_size = self.gemma.config.text_config.vocab_size
                loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1).to(shift_logits.device))

        # Return in HuggingFace format
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=lm_outputs.past_key_values,
            hidden_states=lm_outputs.hidden_states if hasattr(lm_outputs, "hidden_states") else None,
            attentions=lm_outputs.attentions if hasattr(lm_outputs, "attentions") else None,
        )

    def _prepare_image(self, image):
        """Convert image to appropriate format for model."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        elif isinstance(image, torch.Tensor):
            if image.ndim == 4:
                image = image[0]  # Take first if batched
            img_np = image.permute(1, 2, 0).cpu().numpy()
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            return Image.fromarray(img_np).convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def generate(
        self,
        image: Union[Image.Image, np.ndarray, Tensor],
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        skip_world_tokens: bool = False,
        **kwargs: Union[str, int, float, bool],
    ) -> str:
        """
        Generate text response for image and prompt.

        Args:
            image: Input image (PIL, numpy, or tensor)
            prompt: Text prompt/question
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            skip_world_tokens: If True, completely skip world model (ablation mode)
            **kwargs: Additional generation parameters

        Returns:
            Generated text response

        Note: World model now uses single-frame encoding only (no temporal prediction).
        """

        # Prepare image
        pil_image = self._prepare_image(image)

        # Format with chat template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Process inputs with Gemma processor
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        # Move to device
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.gemma.device)
        elif isinstance(inputs, dict):
            inputs = {k: v.to(self.gemma.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        # For ablation: skip world tokens entirely
        if skip_world_tokens:
            # Use Gemma directly without world model
            with torch.no_grad():
                if temperature == 0.0:
                    outputs = self.gemma.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                    )
                else:
                    outputs = self.gemma.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                    )
        else:
            # Use TheWorld forward pass (includes world tokens)
            # Note: This is a simplified implementation
            # For full generation, we'd need to implement autoregressive decoding
            # For now, delegate to Gemma's generate with world embeddings
            # TODO: Implement proper autoregressive generation with world tokens
            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else 1.0,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

            # For now, use Gemma's generate (this is a limitation - doesn't fully use world tokens)
            # A proper implementation would require custom generation loop
            with torch.no_grad():
                outputs = self.gemma.generate(
                    **inputs,
                    generation_config=generation_config,
                )

        # Decode output (skip input tokens)
        if isinstance(inputs, dict) and "input_ids" in inputs:
            input_ids_tensor = inputs["input_ids"]
            assert isinstance(input_ids_tensor, Tensor), "Expected input_ids to be a Tensor"
            input_len = input_ids_tensor.shape[1]
        elif isinstance(inputs, Tensor):
            input_len = inputs.shape[1]
        else:
            raise ValueError(f"Unsupported inputs type: {type(inputs)}")

        generated_ids = outputs[:, input_len:]
        response = self.processor.decode(generated_ids[0], skip_special_tokens=True)

        return response.strip()
