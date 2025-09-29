import torch
import torch.nn as nn
from diffusers import Cosmos2VideoToWorldPipeline
from transformers import AutoModelForCausalLM

class TheWorld(nn.Module):
    def __init__(self, gemma_model_name):
        super(TheWorld, self).__init__()

        # --- 1. Load the full Cosmos Pipeline to access all components ---
        # https://arxiv.org/pdf/2503.15558 Their own reasoning model doesn't even use world model!
        self.cosmos_pipe: Cosmos2VideoToWorldPipeline = Cosmos2VideoToWorldPipeline.from_pretrained(
            "nvidia/Cosmos-Predict2-2B-Video2World",
            torch_dtype=torch.bfloat16,
            safety_checker=None,
        )
        
        print(self.cosmos_pipe.components)
        self.cosmos_vae_encoder = self.cosmos_pipe.components["vae"].encoder
        self.cosmos_text_encoder = self.cosmos_pipe.text_encoder
        self.gemma = AutoModelForCausalLM.from_pretrained(gemma_model_name)

        for param in self.cosmos_pipe.parameters():
            param.requires_grad = False
        for param in self.gemma.parameters():
            param.requires_grad = False

        # Get the output dimensions from the Cosmos encoders
        cosmos_img_dim = self.cosmos_pipe.vae.config.latent_channels
        cosmos_text_dim = self.cosmos_text_encoder.config.hidden_size
        gemma_dim = self.gemma.config.hidden_size
        
        # Create a separate adapter for each modality to map 
        self.image_projection = nn.Linear(cosmos_img_dim, gemma_dim)
        self.text_projection = nn.Linear(cosmos_text_dim, gemma_dim)


    def forward(self, input_pixels, input_ids, text_attention_mask, labels=None):
        # input_pixels: The image for the VQA task
        # input_ids: The tokenized question about the image
        
        with torch.no_grad():
            latent_img_embeds = self.cosmos_vae_encoder(input_pixels).latent_dist.sample()
        b, c, h, w = latent_img_embeds.shape
        reshaped_img_embeds = latent_img_embeds.permute(0, 2, 3, 1).reshape(b, h * w, c)
        projected_img_embeds = self.image_projection(reshaped_img_embeds)
        
        with torch.no_grad():
            text_encoder_outputs = self.cosmos_text_encoder(input_ids=input_ids, attention_mask=text_attention_mask)
        txt_embeds = text_encoder_outputs.last_hidden_state
        projected_txt_embeds = self.text_projection(txt_embeds)

        combined_embeds = torch.cat([projected_img_embeds, projected_txt_embeds], dim=1)
        
        img_attention_mask = torch.ones(projected_img_embeds.size()[:2], dtype=torch.long, device=input_pixels.device)
        combined_attention_mask = torch.cat([img_attention_mask, text_attention_mask], dim=1)

        outputs = self.gemma(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            labels=labels
        )
        return outputs