import torch
import torch.nn as nn
from transformers import BertModel

class Adapter(nn.Module):
    def __init__(self, hidden_size, adapter_dim=128):
        super().__init__()
        self.down = nn.Linear(hidden_size, adapter_dim)
        self.activation = nn.ReLU()
        self.up = nn.Linear(adapter_dim, hidden_size)

    def forward(self, x):
        return x + self.up(self.activation(self.down(x)))  # Residual connection

class BertAdapter(nn.Module):
    def __init__(self, adapter_dim=64):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.adapters = nn.ModuleList([
            Adapter(self.bert.config.hidden_size, adapter_dim)
            for _ in range(self.bert.config.num_hidden_layers)
        ])
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

        # Freeze all BERT weights
        for param in self.bert.parameters():
            param.requires_grad = False

        # Only adapters and classifier are trainable
        for param in self.adapters.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        B, C, L = input_ids.shape
        input_ids = input_ids.view(B * C, L)
        attention_mask = attention_mask.view(B * C, L)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        x = outputs.last_hidden_state  # shape: (B*C, L, H)

        # Apply adapters to each hidden layer output
        for i, adapter in enumerate(self.adapters):
            x = adapter(x)

        cls_embeddings = x[:, 0, :]  # CLS token
        logits = self.classifier(cls_embeddings)  # (B*C, 1)
        return logits.view(B, C)
