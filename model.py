import torch
import torch.nn as nn
from diffusers import Cosmos2VideoToWorldPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

class TheWorld(nn.Module):
    def __init__(self, gemma_model_name, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(TheWorld, self).__init__()

        self.device = device

        # --- 1. Load the full Cosmos Pipeline to access all components ---
        # https://arxiv.org/pdf/2503.15558 Their own reasoning model doesn't even use world model!
        self.cosmos_pipe: Cosmos2VideoToWorldPipeline = Cosmos2VideoToWorldPipeline.from_pretrained(
            "nvidia/Cosmos-Predict2-2B-Video2World",
            torch_dtype=torch.bfloat16,
            safety_checker=None,
        )
        self.cosmos_pipe = self.cosmos_pipe.to(device)

        self.cosmos_vae_encoder = self.cosmos_pipe.components["vae"].encoder

        # Load Gemma model and tokenizer
        self.gemma = AutoModelForCausalLM.from_pretrained(
            gemma_model_name,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(gemma_model_name)

        # Freeze Cosmos VAE encoder
        for param in self.cosmos_vae_encoder.parameters():
            param.requires_grad = False
        # Freeze Gemma
        for param in self.gemma.parameters():
            param.requires_grad = False

        # Get the output dimensions from the Cosmos encoder and Gemma
        # We use the true latent space dimension (z_dim=16), not the encoder output (32)
        # See docs/world_model_latent_space.md for explanation
        cosmos_img_dim = self.cosmos_pipe.vae.config.z_dim  # 16-dim latent space
        gemma_dim = self.gemma.config.hidden_size

        # Create projection layer for world model embeddings -> Gemma
        self.world_projection = nn.Linear(cosmos_img_dim, gemma_dim, dtype=torch.bfloat16).to(device)


    def forward(self, input_pixels, input_ids, text_attention_mask, labels=None):
        # input_pixels: The image for the world model
        # input_ids: The tokenized text (question/prompt)
        # We embed image via Cosmos world model, text via Gemma, then concat

        # Ensure inputs are on correct device and dtype
        input_pixels = input_pixels.to(self.device, dtype=torch.bfloat16)
        input_ids = input_ids.to(self.device)
        text_attention_mask = text_attention_mask.to(self.device)

        # Cosmos VAE expects 5D input (B, C, T, H, W) for video
        # Add temporal dimension for single frame
        if input_pixels.ndim == 4:
            input_pixels = input_pixels.unsqueeze(2)  # (B, C, H, W) -> (B, C, 1, H, W)

        # 1. Get Cosmos world model embeddings (Option 3: latent mean)
        # Use proper VAE encode to get 16-dim normalized latent space
        # See docs/world_model_latent_space.md for why we use mean instead of sample
        with torch.no_grad():
            latent_dist = self.cosmos_pipe.vae.encode(input_pixels).latent_dist
            latent_img_embeds = latent_dist.mean  # Deterministic, normalized 16-dim latents

        # latent_img_embeds shape: (B, 16, T, H, W) - squeeze temporal dimension for single frame
        b, c, t, h, w = latent_img_embeds.shape
        latent_img_embeds = latent_img_embeds.squeeze(2)  # (B, 16, H, W)

        # Reshape to sequence: (B, H×W, 16)
        reshaped_world_embeds = latent_img_embeds.permute(0, 2, 3, 1).reshape(b, h * w, c)

        # Project to Gemma dimension: (B, H×W, gemma_dim)
        projected_world_embeds = self.world_projection(reshaped_world_embeds)

        # 2. Get Gemma's text embeddings
        text_embeds = self.gemma.get_input_embeddings()(input_ids)

        # 3. Concatenate world model embeddings + text embeddings
        combined_embeds = torch.cat([projected_world_embeds, text_embeds], dim=1)

        # 4. Create attention mask for combined sequence
        world_attention_mask = torch.ones(projected_world_embeds.size()[:2], dtype=torch.long, device=self.device)
        combined_attention_mask = torch.cat([world_attention_mask, text_attention_mask], dim=1)

        # 5. Pass through Gemma
        outputs = self.gemma(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            labels=labels
        )
        return outputs