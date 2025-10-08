import torch
import torch.nn as nn
from diffusers import Cosmos2VideoToWorldPipeline
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download
import os


class TheWorld(nn.Module):
    def __init__(
        self,
        gemma_model_name,
        cosmos_model_name="nvidia/Cosmos-Predict2-2B-Video2World",
        device="cuda",
        num_world_steps=0,
        max_world_steps=16,
        freeze_gemma_vision=True,
        freeze_gemma_language=True,
        freeze_cosmos_vae=True,
        random_projection_init=False,
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
            num_world_steps: Default number of future frames to predict (0 = current only)
            max_world_steps: Maximum frames for temporal embeddings
            freeze_gemma_vision: If True, freeze Gemma's vision encoder (SigLIP) [default: True]
            freeze_gemma_language: If True, freeze Gemma's language model [default: True]
            freeze_cosmos_vae: If True, freeze Cosmos VAE encoder [default: True]
            random_projection_init: If True, randomly initialize projection layer (for ablation) [default: False]

        Trainable components (by default):
            - temporal_embedding: Distinguishes between timesteps
            - world_projection: Projects Cosmos latents to Gemma dimension
        """
        super(TheWorld, self).__init__()

        self.device = device
        self.num_world_steps = num_world_steps  # How many future frames to predict (0 = current frame only)
        self.max_world_steps = max_world_steps  # Maximum for temporal embeddings
        self.random_projection_init = random_projection_init  # For ablation studies

        # Store model names for checkpointing
        self.gemma_model_name = gemma_model_name
        self.cosmos_model_name = cosmos_model_name

        # Store freeze configuration
        self.freeze_gemma_vision = freeze_gemma_vision
        self.freeze_gemma_language = freeze_gemma_language
        self.freeze_cosmos_vae = freeze_cosmos_vae

        # --- 1. Load the full Cosmos Pipeline to access all components ---
        # https://arxiv.org/pdf/2503.15558 Their own reasoning model doesn't even use world model!
        self.cosmos_pipe: Cosmos2VideoToWorldPipeline = Cosmos2VideoToWorldPipeline.from_pretrained(
            cosmos_model_name,
            torch_dtype=torch.bfloat16,
            safety_checker=None,
            requires_safety_checker=False,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        self.cosmos_pipe = self.cosmos_pipe.to(device)

        self.cosmos_vae_encoder = self.cosmos_pipe.components["vae"].encoder

        # Load Gemma 3 vision-language model and processor
        # Use device_map="auto" for automatic tensor parallelism across GPUs
        self.gemma = Gemma3ForConditionalGeneration.from_pretrained(
            gemma_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            gemma_model_name,
            local_files_only=True,
        )

        # Get the output dimensions from the Cosmos encoder and Gemma
        # We use the true latent space dimension (z_dim=16), not the encoder output (32)
        # See docs/world_model_latent_space.md for explanation
        cosmos_img_dim = self.cosmos_pipe.vae.config.z_dim  # 16-dim latent space
        # Gemma3 uses 'hidden_size' in text_config
        gemma_dim = self.gemma.config.text_config.hidden_size

        # Temporal embeddings for multi-step world model rollout
        # Helps distinguish between current frame (t=0) and future predictions (t=1,2,...)
        self.temporal_embedding = nn.Embedding(max_world_steps + 1, cosmos_img_dim, dtype=torch.bfloat16).to(device)

        # Create projection layer for world model embeddings -> Gemma
        self.world_projection = nn.Linear(cosmos_img_dim, gemma_dim, dtype=torch.bfloat16).to(device)

        # Optionally reinitialize with random weights (for ablation studies)
        if random_projection_init:
            nn.init.xavier_uniform_(self.world_projection.weight)
            if self.world_projection.bias is not None:
                nn.init.zeros_(self.world_projection.bias)
            print("⚠ Projection layer randomly initialized (ablation mode)")

        # Add bracket special tokens for world model embeddings
        special_tokens = {"additional_special_tokens": ["<the_world_start>", "<the_world_end>"]}
        num_added = self.processor.tokenizer.add_special_tokens(special_tokens)
        if num_added > 0:
            self.gemma.resize_token_embeddings(len(self.processor.tokenizer))

        # Store token IDs for use in forward pass
        self.world_start_id = self.processor.tokenizer.convert_tokens_to_ids("<the_world_start>")
        self.world_end_id = self.processor.tokenizer.convert_tokens_to_ids("<the_world_end>")

        # Expose device map from Gemma to TheWorld (for HuggingFace Trainer compatibility)
        # This prevents Trainer from trying to move the model when device_map="auto" is used
        if hasattr(self.gemma, "hf_device_map"):
            self.hf_device_map = self.gemma.hf_device_map

        # Apply freezing configuration (after all components are created)
        self._apply_freezing()

    def _apply_freezing(self):
        """Apply freezing configuration to model components."""
        # Always freeze all Cosmos components first
        for component_name in self.cosmos_pipe.components:
            component = self.cosmos_pipe.components[component_name]
            if component is not None and hasattr(component, "parameters"):
                for param in component.parameters():
                    param.requires_grad = False

        # Then selectively unfreeze VAE encoder if requested
        if not self.freeze_cosmos_vae:
            # Unfreeze only VAE encoder (for world model latent training)
            # Keep transformer/scheduler/decoder frozen
            if hasattr(self.cosmos_pipe, "vae"):
                for param in self.cosmos_pipe.vae.encoder.parameters():
                    param.requires_grad = True

        # Freeze Gemma components
        if self.freeze_gemma_vision:
            # Freeze vision tower (SigLIP encoder)
            if hasattr(self.gemma, "model") and hasattr(self.gemma.model, "vision_tower"):
                for param in self.gemma.model.vision_tower.parameters():
                    param.requires_grad = False

        if self.freeze_gemma_language:
            # Freeze language model layers
            if hasattr(self.gemma, "model") and hasattr(self.gemma.model, "language_model"):
                for param in self.gemma.model.language_model.parameters():
                    param.requires_grad = False
            # Also freeze LM head
            if hasattr(self.gemma, "lm_head"):
                for param in self.gemma.lm_head.parameters():
                    param.requires_grad = False

        # Projection layers are always trainable (not frozen)
        for param in self.temporal_embedding.parameters():
            param.requires_grad = True
        for param in self.world_projection.parameters():
            param.requires_grad = True

    def get_trainable_parameters(self):
        """Return count and percentage of trainable parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total, 100 * trainable / total

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory-efficient training.

        This reduces activation memory by 4-8x at the cost of 30-40% slower training.
        Only beneficial when training large portions of the model.
        """
        # Enable for Gemma language model (if unfrozen)
        if not self.freeze_gemma_language:
            if hasattr(self.gemma, "gradient_checkpointing_enable"):
                self.gemma.gradient_checkpointing_enable()
                print("✓ Gradient checkpointing enabled for Gemma language model")
            elif hasattr(self.gemma, "enable_gradient_checkpointing"):
                self.gemma.enable_gradient_checkpointing()
                print("✓ Gradient checkpointing enabled for Gemma language model")

        # Enable for Cosmos VAE encoder (if unfrozen)
        if not self.freeze_cosmos_vae:
            if hasattr(self.cosmos_vae_encoder, "gradient_checkpointing_enable"):
                self.cosmos_vae_encoder.gradient_checkpointing_enable()
                print("✓ Gradient checkpointing enabled for Cosmos VAE encoder")
            elif hasattr(self.cosmos_vae_encoder, "enable_gradient_checkpointing"):
                self.cosmos_vae_encoder.enable_gradient_checkpointing()
                print("✓ Gradient checkpointing enabled for Cosmos VAE encoder")

    def save_checkpoint(self, path: str, optimizer=None, epoch: int = None, step: int = None, **kwargs):
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
                "num_world_steps": self.num_world_steps,
                "max_world_steps": self.max_world_steps,
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

    def load_checkpoint(self, path: str, optimizer=None, strict: bool = False):
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

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        checkpoint_name: str = "pytorch_model.bin",
        device: str = "cuda",
        hf_token: str = None,
    ):
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
        num_world_steps = model_config.get("num_world_steps", 0)
        max_world_steps = model_config.get("max_world_steps", 16)

        # Get freeze configuration
        freeze_gemma_vision = freeze_config.get("freeze_gemma_vision", True)
        freeze_gemma_language = freeze_config.get("freeze_gemma_language", True)
        freeze_cosmos_vae = freeze_config.get("freeze_cosmos_vae", True)

        print(f"Initializing TheWorld model with:")
        print(f"  Gemma: {gemma_model_name}")
        print(f"  Cosmos: {cosmos_model_name}")
        print(f"  num_world_steps: {num_world_steps}")
        print(f"  max_world_steps: {max_world_steps}")

        # Initialize model with configuration from checkpoint
        model = cls(
            gemma_model_name=gemma_model_name,
            cosmos_model_name=cosmos_model_name,
            device=device,
            num_world_steps=num_world_steps,
            max_world_steps=max_world_steps,
            freeze_gemma_vision=freeze_gemma_vision,
            freeze_gemma_language=freeze_gemma_language,
            freeze_cosmos_vae=freeze_cosmos_vae,
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

    def forward(self, input_ids=None, pixel_values=None, attention_mask=None,
                images=None, texts=None, labels=None, num_world_steps=None,
                input_pixels=None, text=None, **kwargs):
        """Forward pass for TheWorld model.

        Accepts two input formats:
        1. Preprocessed (from collator): input_ids, pixel_values, attention_mask, images, texts, labels
        2. Raw (for direct inference): input_pixels, text, labels

        Args:
            input_ids: Preprocessed token IDs from collator
            pixel_values: Preprocessed image tensors from collator (for Gemma)
            attention_mask: Attention mask from collator
            images: Raw PIL images (for Cosmos)
            texts: Raw text prompts (for Cosmos)
            labels: Target labels
            num_world_steps: Override for number of future steps
            input_pixels: Raw image input (backward compatibility)
            text: Raw text input (backward compatibility)
        """
        # Use instance default if not provided
        if num_world_steps is None:
            num_world_steps = self.num_world_steps

        # Handle backward compatibility: direct inference calls
        if input_pixels is not None and text is not None:
            # Direct call with raw inputs - need to preprocess
            images = [input_pixels] if not isinstance(input_pixels, list) else input_pixels
            texts = [text] if not isinstance(text, list) else text
            # Will do full preprocessing below
            preprocessed = False
        else:
            # Called from Trainer with preprocessed inputs
            preprocessed = True

        # Extract single items from batch (we only support batch_size=1 currently)
        pil_image = images[0] if isinstance(images, list) else images
        text_prompt = texts[0] if isinstance(texts, list) else texts

        # Convert PIL image to tensor for Cosmos
        if isinstance(pil_image, Image.Image):
            img_np = np.array(pil_image.convert("RGB"))
            tensor_image = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        elif isinstance(pil_image, np.ndarray):
            tensor_image = torch.from_numpy(pil_image).permute(2, 0, 1).unsqueeze(0)
            pil_image = Image.fromarray(pil_image.astype(np.uint8))
        else:
            # Already a tensor
            tensor_image = pil_image

        # Ensure tensor is on correct device and dtype
        tensor_image = tensor_image.to(self.device, dtype=torch.bfloat16)

        # ========================================
        # STEP 1: Process Cosmos world model FIRST
        # ========================================
        # Get Cosmos world model embeddings with autoregressive rollout
        # This must happen BEFORE chat template so we know how many world tokens we have
        cosmos_input_5d = tensor_image.unsqueeze(2) if tensor_image.ndim == 4 else tensor_image

        if num_world_steps == 0:
            # Single-step: just encode current frame
            with torch.no_grad():
                latent_dist = self.cosmos_pipe.vae.encode(cosmos_input_5d).latent_dist
                latent_img_embeds = latent_dist.mean  # (B, 16, 1, H, W)
        else:
            # Multi-step: use Cosmos to predict future frames with text conditioning
            with torch.no_grad():
                # Ensure pipeline is on correct device (workaround for device migration issues)
                self.cosmos_pipe = self.cosmos_pipe.to(self.device)

                output = self.cosmos_pipe(
                    prompt=text_prompt,  # Use actual text prompt
                    image=pil_image,  # PIL Image for pipeline
                    num_frames=1 + num_world_steps,
                    num_inference_steps=10,
                    output_type="latent",
                    return_dict=True,
                )
                latent_img_embeds = output.frames

        # Process Cosmos latents
        b, c, t, h, w = latent_img_embeds.shape
        latent_img_embeds = latent_img_embeds.permute(0, 2, 3, 4, 1)  # (B, T, H, W, 16)

        # Add temporal embeddings
        temporal_ids = torch.arange(t, device=self.device)
        temporal_embeds = self.temporal_embedding(temporal_ids)
        latent_img_embeds = latent_img_embeds + temporal_embeds.view(1, t, 1, 1, c)

        # Reshape and project Cosmos world embeddings
        reshaped_world_embeds = latent_img_embeds.reshape(b, t * h * w, c)
        # Ensure correct dtype for projection layer
        reshaped_world_embeds = reshaped_world_embeds.to(dtype=torch.bfloat16)
        projected_world_embeds = self.world_projection(reshaped_world_embeds)
        # Now we have: projected_world_embeds shape: [B, num_world_tokens, 2560]

        # ========================================
        # STEP 2: Get preprocessed Gemma inputs
        # ========================================
        if not preprocessed:
            # Direct inference call - need to preprocess
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "<the_world_start> <the_world_end>"},  # Bracket markers
                        {"type": "image", "image": pil_image},  # Image will be processed by SigLIP
                        {"type": "text", "text": text_prompt},  # User prompt
                    ],
                }
            ]

            # Apply chat template - this handles image token insertion automatically
            gemma_inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=False,  # No generation prompt for embeddings
                tokenize=True,
                return_tensors="pt",
                return_dict=True,  # Returns input_ids, attention_mask, pixel_values, etc.
            )

            # Move inputs to the device where Gemma is
            target_device = self.gemma.get_input_embeddings().weight.device
            gemma_inputs = {k: v.to(target_device) for k, v in gemma_inputs.items()}

            input_ids = gemma_inputs["input_ids"]
            pixel_values = gemma_inputs["pixel_values"]
        else:
            # Training call - use preprocessed inputs from collator
            target_device = self.gemma.get_input_embeddings().weight.device
            input_ids = input_ids.to(target_device)
            pixel_values = pixel_values.to(target_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(target_device)

        # ========================================
        # STEP 3: Manually construct input embeddings with vision features
        # ========================================
        # This replicates what Gemma3Model.forward() does internally (modeling_gemma3.py:826-956),
        # but we do it manually so we can inject world embeddings before the language model processes them.
        # By doing this, world embeddings flow through ALL transformer layers alongside vision and text.

        # 3a. Get text token embeddings (image tokens are placeholders at this point)
        # Reference: Gemma3Model.forward() line 887-888
        inputs_embeds = self.gemma.model.language_model.embed_tokens(input_ids)

        # 3b. Process vision through SigLIP + multi-modal projector
        # Reference: Gemma3Model.forward() line 897-903
        with torch.no_grad():
            # Line 898: image_features = self.get_image_features(pixel_values)
            image_features = self.gemma.model.get_image_features(pixel_values)
            # Line 899: image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)

            # 3c. Replace image token placeholders with real SigLIP vision features
            # Reference: Line 900-903 in Gemma3Model.forward()
            # Line 900-902: special_image_mask = self.get_placeholder_mask(...)
            special_image_mask = self.gemma.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            # Line 903: inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # Now inputs_embeds contains: [text tokens + real vision features from SigLIP + text tokens]
        # This is equivalent to what Gemma3Model produces before calling language_model (line 937)
        embeddings = inputs_embeds  # [B, seq_len, 2560]

        # ========================================
        # STEP 4: Insert world embeddings between bracket tokens
        # ========================================
        # Find bracket positions (input_ids already extracted above)
        start_positions = (input_ids == self.world_start_id).nonzero(as_tuple=True)[1]
        end_positions = (input_ids == self.world_end_id).nonzero(as_tuple=True)[1]

        if len(start_positions) > 0 and len(end_positions) > 0:
            start_pos = start_positions[0].item()
            end_pos = end_positions[0].item()

            # Slice embeddings: [before_start] + [<start>] + [WORLD] + [<end>] + [after_end]
            embeddings_before = embeddings[:, : start_pos + 1, :]  # Up to and including <start>
            embeddings_after = embeddings[:, end_pos:, :]  # From <end> onwards

            # Move world embeddings to target device
            projected_world_embeds = projected_world_embeds.to(target_device)

            # Concatenate to insert world tokens between brackets
            combined_embeds = torch.cat([embeddings_before, projected_world_embeds, embeddings_after], dim=1)

            # ========================================
            # STEP 5: Update attention mask
            # ========================================
            # Get attention mask from preprocessing
            if not preprocessed:
                attn_mask = gemma_inputs["attention_mask"]
            else:
                attn_mask = attention_mask

            attention_mask_before = attn_mask[:, : start_pos + 1]
            attention_mask_after = attn_mask[:, end_pos:]
            world_attention_mask = torch.ones(
                (b, projected_world_embeds.size(1)), dtype=torch.long, device=target_device
            )
            combined_attention_mask = torch.cat(
                [attention_mask_before, world_attention_mask, attention_mask_after], dim=1
            )
        else:
            assert False, "No world brackets found (shouldn't happen, but safety)"

        # ========================================
        # STEP 6: Prepare labels for training (if provided)
        # ========================================
        if labels is not None and len(start_positions) > 0:
            # For training: Use standard causal LM approach - predict next token
            # Labels are just the shifted input_ids with world tokens masked out

            num_world = projected_world_embeds.size(1)

            # Build labels matching combined_embeds structure:
            # [tokens_before | world_tokens | tokens_after]
            # Mask out world tokens with -100

            labels_before = input_ids[:, :start_pos + 1].to(target_device)  # Up to <start>
            labels_world = torch.full((b, num_world), -100, dtype=torch.long, device=target_device)  # Mask world
            labels_after = input_ids[:, end_pos:].to(target_device)  # After <end>

            combined_labels = torch.cat([labels_before, labels_world, labels_after], dim=1)
        else:
            combined_labels = None

        # ========================================
        # STEP 7: Forward through language model (single pass with world embeddings)
        # ========================================
        # Now we pass combined_embeds (vision + world + text) through the language model.
        # This is a SINGLE pass where world embeddings flow through ALL transformer layers.
        # Reference: Gemma3Model.forward() line 937-948 calls language_model with inputs_embeds
        lm_outputs = self.gemma.language_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            return_dict=True,
        )

        # ========================================
        # STEP 8: Apply LM head and compute loss
        # ========================================
        # Reference: Gemma3ForConditionalGeneration.forward() line 1094-1119
        hidden_states = lm_outputs.last_hidden_state

        # Apply LM head to get logits
        logits = self.gemma.lm_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        if combined_labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits_float = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits_float[..., :-1, :].contiguous()
            shift_labels = combined_labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.gemma.config.text_config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        # Return in compatible format
        from transformers.modeling_outputs import CausalLMOutputWithPast

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
        image,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        num_world_steps: int = None,
        skip_world_tokens: bool = False,
        **kwargs,
    ) -> str:
        """
        Generate text response for image and prompt.

        Args:
            image: Input image (PIL, numpy, or tensor)
            prompt: Text prompt/question
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            num_world_steps: Override number of world steps (None = use default)
            skip_world_tokens: If True, completely skip world model (ablation mode)
            **kwargs: Additional generation parameters

        Returns:
            Generated text response
        """
        # Use default num_world_steps if not specified
        if num_world_steps is None:
            num_world_steps = self.num_world_steps

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
            from transformers import GenerationConfig

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
            input_len = inputs["input_ids"].shape[1]
        else:
            input_len = inputs.shape[1]

        generated_ids = outputs[:, input_len:]
        response = self.processor.decode(generated_ids[0], skip_special_tokens=True)

        return response.strip()
