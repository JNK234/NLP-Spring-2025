import torch
import torch.nn as nn
from transformers import BertModel

class PrefixEncoder(nn.Module):
    def __init__(self, prefix_length, hidden_size):
        super().__init__()
        self.prefix_embeddings = nn.Parameter(torch.randn(prefix_length, hidden_size))

    def forward(self, batch_size):
        # Repeat prefix for every item in batch
        return self.prefix_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # [B, prefix_len, H]

class BertPrefix(nn.Module):
    def __init__(self, prefix_length=10):
        super().__init__()
        self.prefix_length = prefix_length
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.hidden_size = self.bert.config.hidden_size

        # Freeze all BERT weights
        for param in self.bert.parameters():
            param.requires_grad = False

        # One prefix encoder per layer
        self.prefix_encoders = nn.ModuleList([
            PrefixEncoder(prefix_length, self.hidden_size)
            for _ in range(self.bert.config.num_hidden_layers)
        ])

        self.classifier = nn.Linear(self.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        B, C, L = input_ids.shape
        input_ids = input_ids.view(B * C, L)
        attention_mask = attention_mask.view(B * C, L)

        # Get embeddings from input
        inputs_embeds = self.bert.embeddings(input_ids)

        # Add prefix tokens to input
        prefix_embed = self.prefix_encoders[0](inputs_embeds.size(0))  # [B*C, prefix_len, H]
        inputs_with_prefix = torch.cat([prefix_embed, inputs_embeds], dim=1)  # [B*C, prefix + L, H]

        # Extend attention mask to match prefix
        prefix_mask = torch.ones((attention_mask.size(0), self.prefix_length), device=attention_mask.device)
        extended_mask = torch.cat([prefix_mask, attention_mask], dim=1)  # [B*C, prefix + L]

        # Forward through BERT using hidden_states injection
        output = self.bert(inputs_embeds=inputs_with_prefix, attention_mask=extended_mask)
        cls_embedding = output.last_hidden_state[:, self.prefix_length, :]  # CLS is now shifted
        logits = self.classifier(cls_embedding)
        return logits.view(B, C)
