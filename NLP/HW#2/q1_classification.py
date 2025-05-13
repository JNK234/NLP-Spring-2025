"""
MSAI-337, Spring 2025
Homework #2: Question 1 - Classification Approach for OpenBookQA
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
import torch.optim as optim
import json
import sys
import os
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)

set_seed(42)

# Define device - automatically use CUDA if available, fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Label mapping
answers = ['A', 'B', 'C', 'D']


class MCQADataset(Dataset):
    """Dataset for Multiple Choice Question Answering"""
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        obs = self.data[idx]
        texts = [item[0] for item in obs]
        labels = [item[1] for item in obs]

        encoding = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        label_idx = labels.index(1)  # find correct answer index

        return {
            'input_ids': encoding['input_ids'],          # shape: (4, seq_len)
            'attention_mask': encoding['attention_mask'],
            'label': torch.tensor(label_idx, dtype=torch.long)
        }


class BERTClassifier(nn.Module):
    """BERT-based classifier for multiple choice questions"""
    def __init__(self):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(768, 1)  # Score per choice
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model
        
        Args:
            input_ids: Tensor of shape [batch_size, 4, seq_len]
            attention_mask: Tensor of shape [batch_size, 4, seq_len]
            
        Returns:
            scores: Tensor of shape [batch_size, 4] containing logits for each choice
        """
        bsz, num_choices, seq_len = input_ids.shape

        # Reshape to process all choices at once
        input_ids = input_ids.view(bsz * num_choices, seq_len)
        attention_mask = attention_mask.view(bsz * num_choices, seq_len)

        # Get BERT representations
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding

        # Get score for each choice
        scores = self.classifier(cls_output)  # [bsz*4, 1]
        scores = scores.view(bsz, num_choices)  # [bsz, 4]

        return scores


def load_data(file_path):
    """
    Load OpenBookQA data from JSONL file
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        data: List of examples formatted for the dataset
    """
    data = []
    try:
        with open(file_path, 'r') as f:
            json_list = list(f)
        
        for line in json_list:
            result = json.loads(line.strip())
            base = result['fact1'] + ' [SEP] ' + result['question']['stem']
            ans = answers.index(result['answerKey'])

            obs = []
            for j in range(4):
                # Format: fact [SEP] question [SEP] choice [END]
                text = base + ' [SEP] ' + result['question']['choices'][j]['text'] + ' [END]'
                label = 1 if j == ans else 0
                obs.append([text, label])
            data.append(obs)
            
        print(f"Loaded {len(data)} examples from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        sys.exit(1)


def evaluate(model, dataloader):
    """
    Evaluate the model on a dataset
    
    Args:
        model: The BERTClassifier model
        dataloader: DataLoader for the dataset
        
    Returns:
        accuracy: Accuracy of the model on the dataset
        predictions: List of model predictions
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, all_preds


def train_model(model, train_loader, valid_loader, epochs=3, lr=2e-5, model_save_path='best_bert_classifier.pth'):
    """
    Train the model
    
    Args:
        model: The BERTClassifier model
        train_loader: DataLoader for training data
        valid_loader: DataLoader for validation data
        epochs: Number of epochs to train
        lr: Learning rate
        model_save_path: Path to save the best model
        
    Returns:
        model: The trained model
        history: Dictionary containing training history
    """
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    # Training history
    history = {
        'train_loss': [],
        'valid_acc': []
    }
    
    best_valid_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        # Training loop
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update progress
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate average loss for the epoch
        avg_epoch_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        history['train_loss'].append(avg_epoch_loss)
        
        # Evaluate on validation set
        valid_acc, _ = evaluate(model, valid_loader)
        history['valid_acc'].append(valid_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_epoch_loss:.4f} - Valid Acc: {valid_acc:.4f}")
        
        # Save best model
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved to {model_save_path} with validation accuracy: {valid_acc:.4f}")
    
    return model, history

def plot_training_history(history, save_path):
    """Plot training history and save to specified path"""
    plt.figure(figsize=(12, 4))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['valid_acc'])
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    """Main function"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='BERT Classification for OpenBookQA')
    
    # File paths
    parser.add_argument('--train_file', type=str, default='train_complete.jsonl', 
                        help='Path to training data')
    parser.add_argument('--valid_file', type=str, default='dev_complete.jsonl', 
                        help='Path to validation data')
    parser.add_argument('--test_file', type=str, default='test_complete.jsonl', 
                        help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='/home/wnn7240/JNK/NLP-Spring-2025/NLP/HW#2/Logs/output_q1', 
                        help='Directory to save models and results')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=3, 
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='Batch size for training and evaluation')
    parser.add_argument('--lr', type=float, default=2e-5, 
                        help='Learning rate')
    parser.add_argument('--max_seq_len', type=int, default=512, 
                        help='Maximum sequence length')
    parser.add_argument('--no_cuda', action='store_true', 
                        help='Disable CUDA even if available')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Model paths
    model_save_path = os.path.join(args.output_dir, 'best_bert_classifier.pth')
    history_plot_path = os.path.join(args.output_dir, 'training_history.png')
    
    # Override device if no_cuda is set
    global device
    if args.no_cuda:
        device = torch.device("cpu")
        print(f"Manually set device to: {device}")
    
    print("Loading datasets...")
    train_data = load_data(args.train_file)
    valid_data = load_data(args.valid_file)
    test_data = load_data(args.test_file)

    print("Initializing tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BERTClassifier().to(device)

    print("Creating datasets and loaders...")
    train_dataset = MCQADataset(train_data, tokenizer, max_length=args.max_seq_len)
    valid_dataset = MCQADataset(valid_data, tokenizer, max_length=args.max_seq_len)
    test_dataset = MCQADataset(test_data, tokenizer, max_length=args.max_seq_len)

    # Use batch size from arguments
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    print(f"Starting training for {args.epochs} epochs...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=args.epochs,
        lr=args.lr,
        model_save_path=model_save_path # Pass model save path
    )
    
    # Plot training history
    plot_training_history(history, history_plot_path) # Pass history plot path
    print(f"Training history plot saved to {history_plot_path}")

    # Evaluate zero-shot performance (before fine-tuning)
    print("\nEvaluating zero-shot performance (using a fresh model)...")
    zero_shot_model = BERTClassifier().to(device)
    zero_shot_valid_acc, _ = evaluate(zero_shot_model, valid_loader)
    zero_shot_test_acc, _ = evaluate(zero_shot_model, test_loader)
    print(f"Zero-shot Validation Accuracy: {zero_shot_valid_acc:.4f}")
    print(f"Zero-shot Test Accuracy: {zero_shot_test_acc:.4f}")
    
    # Load best model and evaluate on test set
    print("\nEvaluating fine-tuned model on test set...")
    try:
        model.load_state_dict(torch.load(model_save_path))
        print(f"Loaded best model from {model_save_path}")
    except Exception as e:
        print(f"Could not load best model from {model_save_path}: {e}. Using current model state.")
    
    test_acc, test_preds = evaluate(model, test_loader)
    print(f"Fine-tuned Test Accuracy: {test_acc:.4f}")
    
    # Save results to file
    results_file = os.path.join(args.output_dir, 'results.txt')
    with open(results_file, 'w') as f:
        f.write("--- BERT Classification Approach Results ---\n")
        f.write(f"Zero-shot Validation Accuracy: {zero_shot_valid_acc:.4f}\n")
        f.write(f"Zero-shot Test Accuracy: {zero_shot_test_acc:.4f}\n")
        if history['valid_acc']: # Check if history is not empty
             f.write(f"Fine-tuned Validation Accuracy: {history['valid_acc'][-1]:.4f}\n")
        else:
             f.write("Fine-tuned Validation Accuracy: N/A (No validation results)\n")
        f.write(f"Fine-tuned Test Accuracy: {test_acc:.4f}\n\n")
        f.write("Limitations and possible improvements:\n")
        f.write("1. Limited by BERT's maximum sequence length (512 tokens)\n")
        f.write("2. Could improve with better prompt engineering\n")
        f.write("3. Potential for improvement with larger batch sizes or longer training\n")
        f.write("4. Could experiment with other BERT variants (larger models)\n")
    
    # Report results to console
    print(f"\nResults saved to {results_file}")
    print("\n--- BERT Classification Approach Results ---")
    print(f"Zero-shot Validation Accuracy: {zero_shot_valid_acc:.4f}")
    print(f"Zero-shot Test Accuracy: {zero_shot_test_acc:.4f}")
    if history['valid_acc']:
        print(f"Fine-tuned Validation Accuracy: {history['valid_acc'][-1]:.4f}")
    else:
        print("Fine-tuned Validation Accuracy: N/A (No validation results)")
    print(f"Fine-tuned Test Accuracy: {test_acc:.4f}")
    print("\nLimitations and possible improvements:")
    print("1. Limited by BERT's maximum sequence length (512 tokens)")
    print("2. Could improve with better prompt engineering")
    print("3. Potential for improvement with larger batch sizes or longer training")
    print("4. Could experiment with other BERT variants (larger models)")

if __name__ == "__main__":
    main()
