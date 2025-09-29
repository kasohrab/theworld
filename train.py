import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# 1. Define the Composite Model
class CompositeModel(nn.Module):
    def __init__(self, cosmos_model_name, gemma_model_name):
        super(CompositeModel, self).__init__()
        self.cosmos = AutoModel.from_pretrained(cosmos_model_name)
        self.gemma = AutoModel.from_pretrained(gemma_model_name)

        # Freeze the base models
        for param in self.cosmos.parameters():
            param.requires_grad = False
        for param in self.gemma.parameters():
            param.requires_grad = False

        self.projection = nn.Linear(self.cosmos.config.hidden_size, self.gemma.config.hidden_size)

    def forward(self, input_ids, attention_mask=None, labels=None):
        cosmos_outputs = self.cosmos(
            input_ids, 
            attention_mask=attention_mask, 
            output_hidden_states=True
        )
        last_hidden_state = cosmos_outputs.hidden_states[-1]
        projected_hidden_state = self.projection(last_hidden_state)
        
        # When training, you'll need to pass labels to Gemma
        gemma_outputs = self.gemma(
            inputs_embeds=projected_hidden_state,
            labels=labels
        )
        return gemma_outputs

# 2. Set up your models and tokenizer
cosmos_model_name = "bert-base-uncased"  # Replace with your actual Cosmos model
gemma_model_name = "google/gemma-2b"  # Replace with your actual Gemma model
tokenizer = AutoTokenizer.from_pretrained(cosmos_model_name)

model = CompositeModel(cosmos_model_name, gemma_model_name)
optimizer = torch.optim.Adam(model.projection.parameters(), lr=1e-4)

# 3. Prepare some dummy data for demonstration
dummy_text = ["Here is an example sentence.", "Here is another one."]
inputs = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# For a language modeling task, the labels are typically the input_ids shifted
labels = input_ids.clone()

# 4. Training loop
model.train()
for epoch in range(3):  # A few epochs for demonstration
    optimizer.zero_grad()
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    
    # The loss is calculated by the Gemma model itself
    loss = outputs.loss
    
    loss.backward()
    optimizer.step()
    
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")