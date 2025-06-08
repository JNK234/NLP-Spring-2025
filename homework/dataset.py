from torch.utils.data import Dataset
import torch

class OpenBookQADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.samples = []

        for entry in data:
            if "answerKey" not in entry:
                continue  # Skip examples without labels

            stem = entry["question"]["stem"]
            fact = entry.get("fact1", "")
            choices = entry["question"]["choices"]
            
            try:
                label = "ABCD".index(entry["answerKey"])
            except ValueError:
                continue  # Skip invalid labels

            encoded_choices = []
            for choice in choices:
                text = f"[CLS] {fact} {stem} {choice['text']} [SEP]"
                enc = tokenizer.encode_plus(
                    text,
                    add_special_tokens=False,
                    padding='max_length',
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                encoded_choices.append(enc)

            self.samples.append((encoded_choices, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        encoded_choices, label = self.samples[idx]
        input_ids = torch.stack([ec["input_ids"].squeeze(0) for ec in encoded_choices])         # (4, L)
        attention_mask = torch.stack([ec["attention_mask"].squeeze(0) for ec in encoded_choices])  # (4, L)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label),
        }
