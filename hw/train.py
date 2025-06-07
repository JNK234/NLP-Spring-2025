import argparse
import logging
import os

# === Parse arguments first! ===
parser = argparse.ArgumentParser()

parser.add_argument("--mode", type=str, default="full", choices=["full", "adapter", "lora", "prefix"])
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--log_dir", type=str, default="logs/")
parser.add_argument("--model_dir", type=str, default="models/")

args = parser.parse_args()

args = parser.parse_args()

# === Logging setup ===
os.makedirs("logs", exist_ok=True)
log_filename = f"logs/{args.mode}_run.log"

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

logging.info(f"Started training with mode: {args.mode}")

# === Imports that depend on args.mode ===
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from dataset import OpenBookQADataset
import json
import time
from tqdm import tqdm

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === Load Tokenizer and Dataset ===
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def load_jsonl(path):
    return [json.loads(line) for line in open(path)]

train_data = load_jsonl("/home/kqm0007/northwestern/hw/data/train_complete.jsonl")
val_data = load_jsonl("/home/kqm0007/northwestern/hw/data/dev_complete.jsonl")

train_dataset = OpenBookQADataset(train_data, tokenizer)
val_dataset = OpenBookQADataset(val_data, tokenizer)

print("Train samples:", len(train_dataset))
print("Val samples:", len(val_dataset))

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# === Model and Optimizer ===
if args.mode == "adapter":
    from model_adapter import BertWithAdapter
    model = BertWithAdapter().to(device)
elif args.mode == "lora":
    from model_lora import BertWithLoRA
    model = BertWithLoRA().to(device)
elif args.mode == "prefix":
    from model_prefix import BertWithPrefix
    model = BertWithPrefix().to(device)
else:
    from model import BertMC
    model = BertMC(dropout=args.dropout).to(device)



optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

print("Example label from dataset:", train_dataset[0]["label"])
print("Example input IDs shape:", train_dataset[0]["input_ids"].shape)


# === Training ===
for epoch in range(num_epochs = args.epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    print(f"\nEpoch {epoch+1}/{num_epochs}")
    start_time = time.time()
    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)            # [B, 4, L]
        attention_mask = batch["attention_mask"].to(device)  # [B, 4, L]
        labels = batch["label"].to(device)                   # [B]

        logits = model(input_ids, attention_mask)            # [B, 4]
        loss = F.cross_entropy(logits, labels)
        if step % 50 == 0:
            logging.info(f"[Epoch {epoch+1} | Step {step}/{len(train_loader)}] Loss: {loss.item():.4f}")


        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    end_time = time.time()
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total * 100
    logging.info(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_loss:.4f} | Train Accuracy: {accuracy:.2f}% | Time: {end_time - start_time:.2f}s")

# === Save Model ===
import os
os.makedirs("models", exist_ok=True)

save_path = f"models/{args.mode}_model.pt"
torch.save(model.state_dict(), save_path)
logging.info(f"Model saved to {save_path}")



def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total * 100

val_acc = evaluate(model, val_loader, device)
logging.info(f"Validation Accuracy: {val_acc:.2f}%")

start = time.time()
test_data = load_jsonl("/home/kqm0007/northwestern/hw/data/test_complete.jsonl")
test_dataset = OpenBookQADataset(test_data, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=8)
test_acc = evaluate(model, test_loader, device)
end = time.time()
logging.info(f"Inference time on test set: {end - start:.2f}s")
logging.info(f"Test Accuracy: {test_acc:.2f}%")

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logging.info(f"Trainable Parameters: {num_params}")


