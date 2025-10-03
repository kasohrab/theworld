import torch
import torch.nn as nn
from diffusers import Cosmos2VideoToWorldPipeline
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
from PIL import Image
import numpy as np


class TheWorld(nn.Module):
    def __init__(
        self,
        gemma_model_name,
        device="cuda",
        num_world_steps=0,
        max_world_steps=16,
        freeze_gemma_vision=True,
        freeze_gemma_language=True,
        freeze_cosmos_vae=True,
    ):
        """
        TheWorld: Fused vision-language-world model combining Gemma 3 and Cosmos.

        Args:
            gemma_model_name: HuggingFace model ID for Gemma 3
            device: Device to load models on
            num_world_steps: Default number of future frames to predict (0 = current only)
            max_world_steps: Maximum frames for temporal embeddings
            freeze_gemma_vision: If True, freeze Gemma's vision encoder (SigLIP) [default: True]
            freeze_gemma_language: If True, freeze Gemma's language model [default: True]
            freeze_cosmos_vae: If True, freeze Cosmos VAE encoder [default: True]

        Trainable components (by default):
            - temporal_embedding: Distinguishes between timesteps
            - world_projection: Projects Cosmos latents to Gemma dimension
        """
        super(TheWorld, self).__init__()

        self.device = device
        self.num_world_steps = num_world_steps  # How many future frames to predict (0 = current frame only)
        self.max_world_steps = max_world_steps  # Maximum for temporal embeddings

        # Store freeze configuration
        self.freeze_gemma_vision = freeze_gemma_vision
        self.freeze_gemma_language = freeze_gemma_language
        self.freeze_cosmos_vae = freeze_cosmos_vae

        # --- 1. Load the full Cosmos Pipeline to access all components ---
        # https://arxiv.org/pdf/2503.15558 Their own reasoning model doesn't even use world model!
        self.cosmos_pipe: Cosmos2VideoToWorldPipeline = Cosmos2VideoToWorldPipeline.from_pretrained(
            "nvidia/Cosmos-Predict2-2B-Video2World",
            torch_dtype=torch.bfloat16,
            safety_checker=None,
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

        # Add bracket special tokens for world model embeddings
        special_tokens = {"additional_special_tokens": ["<the_world_start>", "<the_world_end>"]}
        num_added = self.processor.tokenizer.add_special_tokens(special_tokens)
        if num_added > 0:
            self.gemma.resize_token_embeddings(len(self.processor.tokenizer))

        # Store token IDs for use in forward pass
        self.world_start_id = self.processor.tokenizer.convert_tokens_to_ids("<the_world_start>")
        self.world_end_id = self.processor.tokenizer.convert_tokens_to_ids("<the_world_end>")

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

    def forward(self, input_pixels, text, labels=None, num_world_steps=None):
        # input_pixels: PIL Image, numpy array (H,W,C), or torch.Tensor (B,C,H,W)
        # text: The text prompt/question (string or list of strings)
        # num_world_steps: Override for number of future steps to predict (default: use self.num_world_steps)

        # Use instance default if not provided
        if num_world_steps is None:
            num_world_steps = self.num_world_steps

        text_prompt = text if isinstance(text, str) else text[0]

        # Handle different input types - convert to both PIL and tensor
        if isinstance(input_pixels, Image.Image):
            # PIL Image input (from HuggingFace datasets)
            pil_image = input_pixels
            # Convert to tensor for Cosmos VAE
            img_np = np.array(pil_image.convert("RGB"))
            tensor_image = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        elif isinstance(input_pixels, np.ndarray):
            # NumPy array (H, W, C)
            pil_image = Image.fromarray(input_pixels.astype(np.uint8))
            tensor_image = torch.from_numpy(input_pixels).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        else:
            # Torch tensor (B, C, H, W) - current format
            tensor_image = input_pixels
            # Convert to PIL for Gemma processor
            img_tensor = input_pixels[0].float()
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np)

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
        # STEP 2: Create chat template with bracket tokens
        # ========================================
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

        # ========================================
        # STEP 3: Manually construct input embeddings with vision features
        # ========================================
        # This replicates what Gemma3Model.forward() does internally (modeling_gemma3.py:826-956),
        # but we do it manually so we can inject world embeddings before the language model processes them.
        # By doing this, world embeddings flow through ALL transformer layers alongside vision and text.
        input_ids = gemma_inputs["input_ids"]
        pixel_values = gemma_inputs["pixel_values"]

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
            attention_mask = gemma_inputs["attention_mask"]
            attention_mask_before = attention_mask[:, : start_pos + 1]
            attention_mask_after = attention_mask[:, end_pos:]
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
            # Structure: [tokens_before | <start> | world | <end> | tokens_after]
            # Only compute loss on text tokens, not special tokens or world/image
            num_before_start = start_pos + 1
            num_world = projected_world_embeds.size(1)

            # Create labels: -100 for all non-text tokens
            labels_before = input_ids[:, :num_before_start].to(target_device)
            labels_world = torch.full((b, num_world), -100, dtype=torch.long, device=target_device)
            labels_after = input_ids[:, end_pos:].to(target_device)

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
