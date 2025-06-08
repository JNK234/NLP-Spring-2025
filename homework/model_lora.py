import torch
import torch.nn as nn
from transformers import BertModel

class LoRALayer(nn.Module):
    def __init__(self, original_linear, r=8, alpha=16):
        super().__init__()
        self.original = original_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)

        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=0.01)
        nn.init.zeros_(self.lora_B.weight)

        # Freeze original linear weights
        for param in self.original.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.original(x) + self.scaling * self.lora_B(self.lora_A(x))


class BertLoRA(nn.Module):
    def __init__(self, r=8, alpha=16):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # Replace attention Q and V projections with LoRA-wrapped ones
        for layer in self.bert.encoder.layer:
            attn = layer.attention.self
            attn.query = LoRALayer(attn.query, r=r, alpha=alpha)
            attn.value = LoRALayer(attn.value, r=r, alpha=alpha)

        # Freeze all other BERT weights
        for name, param in self.bert.named_parameters():
            if "lora_" not in name and "classifier" not in name:
                param.requires_grad = False

        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        B, C, L = input_ids.shape
        input_ids = input_ids.view(B * C, L)
        attention_mask = attention_mask.view(B * C, L)

        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = output.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embeddings)
        return logits.view(B, C)
