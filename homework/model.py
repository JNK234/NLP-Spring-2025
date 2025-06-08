import torch
import torch.nn as nn
from transformers import BertModel

class BertInitial(nn.Module):
    def __init__(self, dropout=0.1):  # Accept dropout as argument
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(dropout)  # Apply dropout to CLS token
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        """
        input_ids: Tensor of shape [batch_size, 4, seq_len]
        attention_mask: Tensor of shape [batch_size, 4, seq_len]
        """
        B, C, L = input_ids.shape  # B=batch_size, C=4 choices, L=seq_len

        # Flatten input for BERT
        input_ids = input_ids.view(B * C, L)
        attention_mask = attention_mask.view(B * C, L)

        # Pass through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [B*C, hidden]

        # Apply dropout before classification
        cls_embeddings = self.dropout(cls_embeddings)

        # Get logits
        logits = self.classifier(cls_embeddings)  # [B*C, 1]
        logits = logits.view(B, C)  # Reshape to [B, 4]

        return logits
