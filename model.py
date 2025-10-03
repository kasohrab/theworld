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

        # Apply freezing configuration (after all components are created)
        self._apply_freezing()

    def _apply_freezing(self):
        """Apply freezing configuration to model components."""
        # Freeze Cosmos components
        if self.freeze_cosmos_vae:
            # Freeze all Cosmos components
            for component_name in self.cosmos_pipe.components:
                component = self.cosmos_pipe.components[component_name]
                if component is not None and hasattr(component, "parameters"):
                    for param in component.parameters():
                        param.requires_grad = False
        else:
            # Unfreeze only VAE encoder (for world model latent training)
            # Keep transformer/scheduler/etc frozen
            if hasattr(self.cosmos_pipe, "vae"):
                for param in self.cosmos_pipe.vae.encoder.parameters():
                    param.requires_grad = True
                # Keep decoder frozen even when training encoder
                for param in self.cosmos_pipe.vae.decoder.parameters():
                    param.requires_grad = False

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

        # 1. Process image through Gemma's vision encoder (training-compatible format)
        # This gives us static visual understanding (objects, scenes)
        with torch.no_grad():
            # Use proper Gemma 3 multimodal training format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},  # Pass PIL image directly
                        {"type": "text", "text": text_prompt},
                    ],
                }
            ]

            # Apply chat template - this handles image token insertion automatically
            gemma_inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=False,  # No generation prompt for embeddings
                tokenize=True,
                return_tensors="pt",
                return_dict=True,  # Return dict with input_ids, attention_mask, etc.
            )

            # Move inputs to the device where Gemma's embedding layer is
            target_device = self.gemma.get_input_embeddings().weight.device
            gemma_inputs = {k: v.to(target_device) for k, v in gemma_inputs.items()}

            # Get vision + text embeddings from Gemma
            gemma_vision_embeds = self.gemma.get_input_embeddings()(gemma_inputs["input_ids"])

        # 2. Get Cosmos world model embeddings with autoregressive rollout
        # This gives us physics-aware temporal dynamics
        # Cosmos VAE expects 5D input (B, C, T, H, W) for video
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

        # 3. Combine all embeddings: [Gemma vision, Cosmos world, rest of text]
        # When using device_map="auto", ensure tensors are on same device
        target_device = gemma_vision_embeds.device
        projected_world_embeds = projected_world_embeds.to(target_device)

        combined_embeds = torch.cat(
            [
                gemma_vision_embeds,  # Gemma's vision understanding
                projected_world_embeds,  # Cosmos world dynamics (already includes text from processor)
            ],
            dim=1,
        )

        # 4. Create attention mask for all tokens
        gemma_attention_mask = gemma_inputs["attention_mask"].to(target_device)
        world_attention_mask = torch.ones(projected_world_embeds.size()[:2], dtype=torch.long, device=target_device)
        combined_attention_mask = torch.cat([gemma_attention_mask, world_attention_mask], dim=1)

        # 5. Prepare labels for training (if provided)
        if labels is not None:
            # Labels should align with combined_embeds sequence
            # Structure: [Gemma vision + text tokens | Cosmos world tokens]
            # Only compute loss on Gemma text tokens, not vision or world tokens
            num_gemma_tokens = gemma_vision_embeds.size(1)
            num_world_tokens = projected_world_embeds.size(1)

            # Pad labels to match combined sequence length
            # -100 = ignore index (no loss computed for these tokens)
            labels = torch.cat(
                [
                    gemma_inputs["input_ids"].to(target_device),  # Gemma vision + text
                    torch.full(
                        (labels.size(0), num_world_tokens), -100, dtype=torch.long, device=target_device
                    ),  # World tokens (ignored)
                ],
                dim=1,
            )

        # 6. Forward through Gemma with combined vision understanding
        outputs = self.gemma(inputs_embeds=combined_embeds, attention_mask=combined_attention_mask, labels=labels)
        return outputs
