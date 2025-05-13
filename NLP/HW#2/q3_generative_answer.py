"""
MSAI-337, Spring 2025
Homework #2: Question 3 - Generative Approach for OpenBookQA (Generated Answer)
Fine-tuning script for a pre-trained transformer model that generates full answers.
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
import random
import math
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import BERTScorer
import nltk
import warnings

# Import the transformer model components from starter_q1.py
sys.path.append('NLP/HW#2')
from q2_pretraining import (
    Embedder, PositionalEncoder, Norm, MultiHeadAttention, FeedForward,
    DecoderLayer, Decoder, Transformer, get_clones, create_masks,
    CosineWithRestarts, get_model
)

# Set seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# Suppress warnings
warnings.filterwarnings("ignore")

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Make sure nltk packages are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Special tokens for the prompt
START_TOKEN = "[START]"
ANSWER_TOKEN = "[ANSWER]"
CHOICE_TOKENS = ["[A]", "[B]", "[C]", "[D]"]

class OpenBookQAGenerativeTextDataset(Dataset):
    """Dataset for generative approach to OpenBookQA with full text generation"""
    def __init__(self, data, tokenizer, max_length=512):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.answers = ['A', 'B', 'C', 'D']

        # Process all examples
        for example in data:
            processed = self.process_example(example)
            if processed:
                self.examples.append(processed)

    def process_example(self, example):
        """Process a single OpenBookQA example"""
        # Extract answer index and text
        answer_idx = self.answers.index(example['answerKey'])
        answer_text = example['question']['choices'][answer_idx]['text']

        # Format the prompt:
        # [START] <fact> <stem> [A] <choice1> [B] <choice2> [C] <choice3> [D] <choice4> [ANSWER] <answer text>
        prompt = f"{START_TOKEN} {example['fact1']} {example['question']['stem']}"

        # Add each choice with its label
        for i, choice in enumerate(example['question']['choices']):
            prompt += f" {CHOICE_TOKENS[i]} {choice['text']}"

        prompt += f" {ANSWER_TOKEN} {answer_text}"

        # Tokenize and ensure it fits within max length
        tokens = self.tokenizer(prompt, return_tensors='pt', truncation=True,
                                max_length=self.max_length)

        # If the sequence is too long even after truncation, skip it
        if tokens['input_ids'].size(1) > self.max_length:
            return None

        # Find the position of the [ANSWER] token
        input_ids = tokens['input_ids'][0].tolist()
        try:
            answer_pos = input_ids.index(self.tokenizer.convert_tokens_to_ids("[ANSWER]"))
        except ValueError:
            # If [ANSWER] token is truncated, skip this example
            return None

        # Check if there's at least one token after [ANSWER]
        if answer_pos >= len(input_ids) - 1:
            return None

        return {
            'input_ids': tokens['input_ids'][0],  # Remove batch dimension
            'attention_mask': tokens['attention_mask'][0],
            'answer_position': answer_pos,
            'answer_text': answer_text,
            'full_prompt': prompt # Keep original prompt for GPT-2 comparison
        }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def load_json_data(file_path):
    """Load data from OpenBookQA jsonl files"""
    data = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                example = json.loads(line.strip())
                data.append(example)
        print(f"Loaded {len(data)} examples from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        sys.exit(1)

def prepare_batch(batch, device):
    """
    Prepare a batch (already collated and padded) for training/inference.
    Moves tensors to the specified device.
    """
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    answer_positions = batch['answer_positions'].to(device) # Already a tensor
    answer_texts = batch['answer_texts'] # Keep as a list of strings

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'answer_positions': answer_positions,
        'answer_texts': answer_texts
    }

def pad_collate_fn(batch, pad_token_id):
    """
    Custom collate function to pad sequences within a batch for Q3.
    """
    input_ids_list = [item['input_ids'] for item in batch]
    attention_mask_list = [item['attention_mask'] for item in batch]
    answer_positions = torch.tensor([item['answer_position'] for item in batch])
    answer_texts = [item['answer_text'] for item in batch] # Keep as list of strings

    input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id)
    attention_mask_padded = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded,
        'answer_positions': answer_positions,
        'answer_texts': answer_texts
    }

def train_model(model, train_loader, valid_loader, tokenizer, args):
    """
    Fine-tune the pre-trained transformer model on OpenBookQA for text generation
    """
    print("Starting fine-tuning...")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineWithRestarts(optimizer, T_max=len(train_loader) * args.epochs) if args.use_scheduler else None

    history = {'train_loss': [], 'valid_metrics': []}
    best_valid_rouge = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in progress_bar:
            batch_data = prepare_batch(batch, args.device)
            input_ids = batch_data['input_ids']
            answer_positions = batch_data['answer_positions']
            src_mask = create_masks(input_ids, args.device)

            optimizer.zero_grad()
            output = model(input_ids, src_mask)

            # Calculate loss only for tokens AFTER the [ANSWER] token
            batch_size = input_ids.size(0)
            seq_length = input_ids.size(1)
            loss_mask = torch.zeros_like(input_ids, dtype=torch.float)
            for i in range(batch_size):
                start_pos = answer_positions[i] + 1
                if start_pos < seq_length:
                    loss_mask[i, start_pos:] = 1.0

            logits = output.view(-1, output.size(-1))
            targets = input_ids.view(-1)
            mask = loss_mask.view(-1).bool()
            logits = logits[mask]
            targets = targets[mask]

            # Avoid calculating loss if there are no target tokens (e.g., if all sequences end exactly at [ANSWER])
            if logits.size(0) == 0:
                continue

            loss = F.cross_entropy(logits, targets)
            loss.backward()

            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()
            if scheduler:
                scheduler.step()

            total_loss += loss.item()
            avg_loss = total_loss / (progress_bar.n + 1)
            progress_bar.set_postfix({'loss': avg_loss})

        epoch_avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        history['train_loss'].append(epoch_avg_loss)

        # Evaluate on validation set
        valid_metrics = evaluate_model(
            model, valid_loader, tokenizer, args.device,
            log_examples=(epoch % args.log_every == 0),
            max_gen_length=args.max_gen_length
        )
        history['valid_metrics'].append(valid_metrics)

        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {epoch_avg_loss:.4f}")
        print(f"Validation Metrics: BLEU: {valid_metrics['bleu']:.4f}, ROUGE-L: {valid_metrics['rouge_l']:.4f}, BERTScore: {valid_metrics['bertscore']:.4f}")

        if valid_metrics['rouge_l'] > best_valid_rouge:
            best_valid_rouge = valid_metrics['rouge_l']
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_qa_text_model.pth'))
            print(f"New best model saved with validation ROUGE-L: {valid_metrics['rouge_l']:.4f}")

        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f'qa_text_model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")

    plot_history(history, args.output_dir)
    return model, history

def evaluate_model(model, dataloader, tokenizer, device, log_examples=False, max_gen_length=50):
    """
    Evaluate the model on the given dataloader
    """
    model.eval()
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True, device=device) # Specify device for BERTScorer
    smooth = SmoothingFunction().method1

    bleu_scores, rouge1_scores, rouge2_scores, rougeL_scores, bertscore_scores = [], [], [], [], []
    all_predictions, all_references = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            batch_data = prepare_batch(batch, device)
            input_ids = batch_data['input_ids']
            answer_positions = batch_data['answer_positions']
            reference_texts = batch_data['answer_texts']
            batch_size = input_ids.size(0)

            batch_preds = []
            batch_refs = []

            for i in range(batch_size):
                seq = input_ids[i, :answer_positions[i]+1].unsqueeze(0) # Include [ANSWER]
                # Ensure max_length for generation doesn't exceed model capacity
                # Use the actual embedding layer's max position embeddings
                model_max_pos = model.decoder.pe.pe.size(1) # Corrected path to positional encoding tensor
                gen_max_len = min(max_gen_length, model_max_pos - seq.size(1))


                if gen_max_len <= 0:
                    generated_text = "" # Cannot generate if prompt is already too long
                else:
                    generated_ids = generate_text(
                        model,
                        seq,
                        tokenizer, # Pass tokenizer for EOS id
                        device,
                        max_length=gen_max_len
                    )
                    # Decode only the generated part (after [ANSWER])
                    generated_text = tokenizer.decode(generated_ids[0][seq.size(1):], skip_special_tokens=True).strip()

                reference_text = reference_texts[i].strip()
                batch_preds.append(generated_text)
                batch_refs.append(reference_text)

                # Calculate BLEU per example
                try:
                    if generated_text and reference_text:
                        ref_tok = nltk.word_tokenize(reference_text.lower())
                        pred_tok = nltk.word_tokenize(generated_text.lower())
                        if pred_tok:
                            bleu = sentence_bleu([ref_tok], pred_tok, smoothing_function=smooth, weights=(0.25, 0.25, 0.25, 0.25))
                            bleu_scores.append(bleu)
                except Exception as e:
                    # print(f"Warning: BLEU calculation failed for example: {e}")
                    pass

                # Calculate ROUGE per example
                try:
                    if generated_text and reference_text:
                        scores = rouge_scorer_obj.score(reference_text, generated_text)
                        rouge1_scores.append(scores['rouge1'].fmeasure)
                        rouge2_scores.append(scores['rouge2'].fmeasure)
                        rougeL_scores.append(scores['rougeL'].fmeasure)
                except Exception as e:
                    # print(f"Warning: ROUGE calculation failed for example: {e}")
                    pass

            # Calculate BERTScore for the batch
            if batch_preds and batch_refs:
                try:
                    # Filter out empty strings which cause issues with BERTScore
                    filtered_preds = [p for p, r in zip(batch_preds, batch_refs) if p and r]
                    filtered_refs = [r for p, r in zip(batch_preds, batch_refs) if p and r]
                    if filtered_preds: # Only score if there are non-empty pairs
                        P, R, F1 = bert_scorer.score(filtered_preds, filtered_refs)
                        bertscore_scores.extend(F1.tolist())
                    # Add 0.0 for pairs that were filtered out
                    bertscore_scores.extend([0.0] * (len(batch_preds) - len(filtered_preds)))
                except Exception as e:
                    print(f"Warning: BERTScore calculation failed for batch: {e}")
                    bertscore_scores.extend([0.0] * len(batch_preds)) # Append zeros if fails

            all_predictions.extend(batch_preds)
            all_references.extend(batch_refs)

            # Log examples from the first batch
            if log_examples and batch_idx == 0:
                print("\n--- Example Predictions (First Batch) ---")
                for i in range(min(3, len(batch_preds))):
                    print(f"Reference: {batch_refs[i]}")
                    print(f"Prediction: {batch_preds[i]}")
                    print("-" * 50)

    # Calculate average metrics
    metrics = {
        'bleu': np.mean(bleu_scores) if bleu_scores else 0.0,
        'rouge1': np.mean(rouge1_scores) if rouge1_scores else 0.0,
        'rouge2': np.mean(rouge2_scores) if rouge2_scores else 0.0,
        'rouge_l': np.mean(rougeL_scores) if rougeL_scores else 0.0,
        'bertscore': np.mean(bertscore_scores) if bertscore_scores else 0.0,
        'predictions': all_predictions,
        'references': all_references
    }
    return metrics


def generate_text(model, input_ids, tokenizer, device, max_length=50, temperature=1.0, top_k=50, top_p=0.9):
    """
    Generate text using the model with sampling options.
    """
    model.eval()
    curr_ids = input_ids.clone()
    eos_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        for _ in range(max_length):
            mask = create_masks(curr_ids, device)
            outputs = model(curr_ids, mask)
            next_token_logits = outputs[:, -1, :]

            # Apply temperature
            if temperature != 1.0:
                 next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                # Get top k logits and indices
                top_k_vals, top_k_indices = torch.topk(next_token_logits, top_k)
                # Create a mask filled with -inf
                filter_mask = torch.full_like(next_token_logits, -float('Inf'))
                # Scatter the top k logits back into the mask
                filter_mask.scatter_(1, top_k_indices, top_k_vals)
                next_token_logits = filter_mask


            # Apply top-p (nucleus) filtering
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift to keep the first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits = torch.where(indices_to_remove, torch.ones_like(next_token_logits) * -float('Inf'), next_token_logits)

            # Sample from the filtered distribution
            probs = F.softmax(next_token_logits, dim=-1)
            # Handle potential NaN/Inf in probs after filtering
            probs = torch.nan_to_num(probs)
            if torch.sum(probs) <= 0: # If all probabilities are zero or negative
                 # Fallback: Use argmax of the original logits before filtering
                 next_token = torch.argmax(outputs[:, -1, :], dim=-1).unsqueeze(1)
                 # print("Warning: Probabilities sum to zero, falling back to argmax.")
            else:
                 next_token = torch.multinomial(probs, num_samples=1)


            curr_ids = torch.cat([curr_ids, next_token], dim=1)

            if next_token.item() == eos_token_id:
                break

    return curr_ids


def evaluate_gpt2(dataloader, tokenizer, device, model_name="gpt2", max_gen_length=50, log_examples=False):
    """
    Evaluate a standard GPT-2 model on the dataset for comparison.
    Uses the 'full_prompt' from the dataset item.
    """
    print(f"Loading {model_name} for comparison...")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # Use the standard GPT-2 tokenizer for this evaluation
    gpt2_tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    # Ensure pad token is set for generation
    if gpt2_tokenizer.pad_token_id is None:
        gpt2_tokenizer.pad_token_id = gpt2_tokenizer.eos_token_id

    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True, device=device)
    smooth = SmoothingFunction().method1

    bleu_scores, rouge1_scores, rouge2_scores, rougeL_scores, bertscore_scores = [], [], [], [], []
    all_predictions, all_references = [], []

    # Need to iterate through the dataset directly to access 'full_prompt'
    dataset = dataloader.dataset

    with torch.no_grad():
        for idx, item in enumerate(tqdm(dataset, desc=f"Evaluating {model_name}")):
            # Prepare prompt for standard GPT-2 (remove custom tokens)
            prompt_text = item['full_prompt']
            prompt_text = prompt_text.split(ANSWER_TOKEN)[0] # Take part before [ANSWER]
            for token in [START_TOKEN] + CHOICE_TOKENS:
                prompt_text = prompt_text.replace(token, " ") # Replace with space
            prompt_text = ' '.join(prompt_text.split()) # Clean up whitespace
            # Decide whether to include ANSWER_TOKEN as part of the prompt for GPT-2
            # prompt_text += " " + ANSWER_TOKEN # Option 1: Include trigger
            prompt_text += " The answer is:" # Option 2: More natural prompt


            reference_text = item['answer_text'].strip()

            # Tokenize with standard GPT-2 tokenizer
            input_ids = gpt2_tokenizer.encode(prompt_text, return_tensors="pt").to(device)

            # Ensure generation length is valid
            gen_max_len = min(max_gen_length, gpt2_tokenizer.model_max_length - input_ids.size(1))

            if gen_max_len <= 0:
                 generated_text = ""
            else:
                # Generate text using standard GPT-2 generate method
                output_sequences = model.generate(
                    input_ids=input_ids,
                    max_length=input_ids.size(1) + gen_max_len,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=gpt2_tokenizer.eos_token_id,
                )

                # Decode generated sequence, removing the prompt part
                generated_sequence = output_sequences[0].tolist()
                prompt_len = len(input_ids[0])
                generated_text = gpt2_tokenizer.decode(generated_sequence[prompt_len:], skip_special_tokens=True).strip()


            batch_preds = [generated_text]
            batch_refs = [reference_text]
            all_predictions.extend(batch_preds)
            all_references.extend(batch_refs)

            # Calculate metrics per example
            try:
                if generated_text and reference_text:
                    ref_tok = nltk.word_tokenize(reference_text.lower())
                    pred_tok = nltk.word_tokenize(generated_text.lower())
                    if pred_tok:
                        bleu = sentence_bleu([ref_tok], pred_tok, smoothing_function=smooth, weights=(0.25, 0.25, 0.25, 0.25))
                        bleu_scores.append(bleu)
            except Exception: pass
            try:
                if generated_text and reference_text:
                    scores = rouge_scorer_obj.score(reference_text, generated_text)
                    rouge1_scores.append(scores['rouge1'].fmeasure)
                    rouge2_scores.append(scores['rouge2'].fmeasure)
                    rougeL_scores.append(scores['rougeL'].fmeasure)
            except Exception: pass

            # Calculate BERTScore (can do individually or batch later)
            if generated_text and reference_text:
                 try:
                     # Ensure inputs are lists
                     P, R, F1 = bert_scorer.score([generated_text], [reference_text])
                     bertscore_scores.append(F1.item())
                 except Exception as e:
                     # print(f"Warning: BERTScore failed for GPT-2 example: {e}")
                     bertscore_scores.append(0.0)

            # Log first few examples
            if log_examples and idx < 3:
                print(f"\n--- Example {model_name} Prediction {idx+1} ---")
                print(f"Prompt used: {prompt_text}")
                print(f"Reference: {reference_text}")
                print(f"Prediction: {generated_text}")
                print("-" * 50)

    # Calculate average metrics
    metrics = {
        'bleu': np.mean(bleu_scores) if bleu_scores else 0.0,
        'rouge1': np.mean(rouge1_scores) if rouge1_scores else 0.0,
        'rouge2': np.mean(rouge2_scores) if rouge2_scores else 0.0,
        'rouge_l': np.mean(rougeL_scores) if rougeL_scores else 0.0,
        'bertscore': np.mean(bertscore_scores) if bertscore_scores else 0.0,
        'predictions': all_predictions,
        'references': all_references
    }
    return metrics


def plot_history(history, output_dir):
    """Plot training history"""
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], 'b-')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.subplot(1, 3, 2)
    valid_metrics = history.get('valid_metrics', [])
    if valid_metrics:
        plt.plot([m.get('bleu', 0) for m in valid_metrics], 'g-', label='BLEU')
        plt.plot([m.get('rouge_l', 0) for m in valid_metrics], 'r-', label='ROUGE-L')
    plt.title('Validation Scores')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 3, 3)
    if valid_metrics:
        plt.plot([m.get('bertscore', 0) for m in valid_metrics], 'c-')
    plt.title('BERTScore')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'qa_text_training_history.png'))
    print(f"Training history plot saved to {os.path.join(output_dir, 'qa_text_training_history.png')}")

def save_examples(metrics, output_dir, filename):
    """Save examples of predictions and references"""
    path = os.path.join(output_dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write("Reference,Prediction\n")
        for ref, pred in zip(metrics.get('references', []), metrics.get('predictions', [])):
            ref_clean = ref.replace('"', '""').replace('\n', ' ') # Escape quotes for CSV
            pred_clean = pred.replace('"', '""').replace('\n', ' ')
            f.write(f"\"{ref_clean}\",\"{pred_clean}\"\n")
    print(f"Examples saved to {path}")

def main():
    parser = argparse.ArgumentParser(description='Fine-tune transformer model for OpenBookQA text generation')
    # Model parameters
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of model embeddings')
    parser.add_argument('--n_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    # Training parameters
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-6, help='Learning rate')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--use_scheduler', action='store_true', help='Use learning rate scheduler')
    parser.add_argument('--save_every', type=int, default=1, help='Save model every N epochs')
    parser.add_argument('--log_every', type=int, default=1, help='Log examples every N epochs')
    # Generation parameters
    parser.add_argument('--max_gen_length', type=int, default=50, help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p (nucleus) sampling')
    # File paths - Updated defaults to remove NLP/HW#2 prefix assuming script runs from NLP/HW#2
    parser.add_argument('--train_file', type=str, default='data/openbookqa/train_complete.jsonl', help='Path to training data')
    parser.add_argument('--valid_file', type=str, default='data/openbookqa/dev_complete.jsonl', help='Path to validation data')
    parser.add_argument('--test_file', type=str, default='data/openbookqa/test_complete.jsonl', help='Path to test data')
    parser.add_argument('--model_path', type=str, default='Logs/NLP_H2_Q2_pretraining/pretrained_wiki103_ep20.pth',
                        help='Path to pre-trained model')
    parser.add_argument('--output_dir', type=str, default='Logs/Q3_text_generation', help='Output directory')
    # Comparison options
    parser.add_argument('--compare_gpt2', action='store_true', help='Compare with GPT-2')
    parser.add_argument('--gpt2_model', type=str, default='gpt2', help='GPT-2 model to compare with')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Set seed and device
    set_seed(args.seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {args.device}")

    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # --- Cleaned Model Loading and Resizing ---

    # 1. Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    # 2. Get original vocab size
    original_vocab_size = len(tokenizer)
    print(f"Original tokenizer vocab size: {original_vocab_size}") # Should be 50257

    # 3. Add Special Tokens
    special_tokens_dict = {
        'additional_special_tokens': [START_TOKEN, ANSWER_TOKEN] + CHOICE_TOKENS
    }
    # Ensure EOS token is treated as pad token if none exists
    if tokenizer.pad_token_id is None:
        special_tokens_dict['pad_token'] = tokenizer.eos_token
        print(f"Set pad_token to EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")

    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added_toks} special tokens.")

    # 4. Get new vocab size
    new_vocab_size = len(tokenizer)
    print(f"New tokenizer vocab size: {new_vocab_size}") # Should be 50257 + num_added_toks

    # 5. Define Model Hyperparameters
    model_opt = argparse.Namespace(
        d_model=args.d_model,
        n_layers=args.n_layers,
        heads=args.heads,
        dropout=args.dropout,
        tied=1,  # Use tied weights (important for resizing logic)
        device=args.device,
        loadname=args.model_path # Keep for reference
    )

    # 6. Create Model Structure with ORIGINAL vocab size
    print(f"Creating model structure with original vocab size ({original_vocab_size})...")
    # Pass initialize_weights=False to prevent random initialization before loading
    model = get_model(model_opt, original_vocab_size, initialize_weights=False)
    # Note: Model is NOT on device yet

    # 7. Load Pre-trained Weights (if path provided)
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading pre-trained weights from {args.model_path} with strict=False...")
        try:
            # Load state dict to CPU first
            pretrained_state_dict = torch.load(args.model_path, map_location='cpu')
            load_result = model.load_state_dict(pretrained_state_dict, strict=False)
            print(f"Weight loading result:")
            print(f"  Missing keys: {load_result.missing_keys}")
            print(f"  Unexpected keys: {load_result.unexpected_keys}")
            if not any('decoder.layers' in key for key in model.state_dict() if key not in load_result.missing_keys):
                 print("Warning: Core decoder layers might not have been loaded correctly.")
            else:
                 print("Successfully loaded compatible pre-trained weights.")
        except FileNotFoundError:
            print(f"Error: Pretrained weights file not found at {args.model_path}. Initializing randomly.")
            for p in model.parameters():
                if p.dim() > 1: nn.init.xavier_uniform_(p)
        except Exception as e:
             print(f"Error loading weights: {e}. Initializing randomly.")
             for p in model.parameters():
                 if p.dim() > 1: nn.init.xavier_uniform_(p)
    else:
        print("No model_path provided or file doesn't exist. Initializing randomly.")
        for p in model.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    # 8. Manually Resize Embeddings and Output Layer
    if original_vocab_size != new_vocab_size:
        print(f"Resizing model embeddings and output layer from {original_vocab_size} to {new_vocab_size}...")
        old_embeddings = model.decoder.embed.embed
        embedding_dim = old_embeddings.embedding_dim
        new_embeddings = nn.Embedding(new_vocab_size, embedding_dim)
        new_embeddings.weight.data[:original_vocab_size, :] = old_embeddings.weight.data[:original_vocab_size, :]
        new_embeddings.weight.data[original_vocab_size:].normal_(mean=0.0, std=0.02)
        model.decoder.embed.embed = new_embeddings
        print("  - Token embeddings resized.")

        if model_opt.tied == 1:
            model.out.weight = model.decoder.embed.embed.weight
            print("  - Output layer weights tied and updated.")
            if hasattr(model.out, 'bias') and model.out.bias is not None:
                old_bias = model.out.bias
                if old_bias.size(0) == original_vocab_size:
                    new_bias = nn.Parameter(torch.Tensor(new_vocab_size))
                    new_bias.data[:original_vocab_size] = old_bias.data[:original_vocab_size]
                    new_bias.data[original_vocab_size:].zero_()
                    model.out.bias = new_bias
                    print(f"  - Output layer bias resized from {original_vocab_size} to {new_vocab_size}.")
                else:
                     print(f"  - Warning: Output bias size ({old_bias.size(0)}) != original vocab size ({original_vocab_size}). Bias not resized.")
            else:
                print("  - Warning: model.out.bias not found or is None. Bias not resized.")
        else:
            print("  - Warning: Weights are not tied. Resizing logic for untied output weights needs implementation.")
        print("Model resizing complete.")
    else:
        print("Vocab size unchanged. Skipping model resizing.")

    # 9. Move Model to Device
    model.to(args.device)
    print(f"Model moved to device: {args.device}")

    # --- End Cleaned Model Loading and Resizing ---

    # --- Verification Step ---
    print(f"--- Verification AFTER resize & move ---")
    final_embedding_size = model.decoder.embed.embed.num_embeddings
    print(f"    model.decoder.embed.embed.num_embeddings: {final_embedding_size}")
    print(f"    Expected new_vocab_size: {new_vocab_size}")
    if hasattr(model, 'out') and hasattr(model.out, 'bias') and model.out.bias is not None:
        print(f"    model.out.bias size: {model.out.bias.size(0)}")
    if final_embedding_size != new_vocab_size:
        print(f"    !!! VERIFICATION FAILED: Embedding size {final_embedding_size} != Expected size {new_vocab_size}")
    else:
        print(f"    Verification PASSED.")
    print(f"--- End Verification ---")

    # Load data
    print("Loading data...")
    try:
        train_data = load_json_data(args.train_file)
        valid_data = load_json_data(args.valid_file)
        test_data = load_json_data(args.test_file)
    except FileNotFoundError as e:
        print(f"Error loading data file: {e}. Please check file paths.")
        sys.exit(1)

    # Create datasets
    train_dataset = OpenBookQAGenerativeTextDataset(train_data, tokenizer)
    valid_dataset = OpenBookQAGenerativeTextDataset(valid_data, tokenizer)
    test_dataset = OpenBookQAGenerativeTextDataset(test_data, tokenizer)

    print(f"Train examples: {len(train_dataset)}")
    print(f"Valid examples: {len(valid_dataset)}")
    print(f"Test examples: {len(test_dataset)}")

    # Get pad token ID (should be set now)
    pad_token_id = tokenizer.pad_token_id
    print(f"Using pad token ID: {pad_token_id}")

    # Create data loaders
    collate_fn_with_padding = lambda batch: pad_collate_fn(batch, pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_with_padding)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_with_padding)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_with_padding)

    # Evaluate zero-shot performance
    print("\nEvaluating zero-shot performance...")
    zero_shot_valid_metrics = evaluate_model(
        model, valid_loader, tokenizer, args.device,
        log_examples=True, max_gen_length=args.max_gen_length
    )
    print("Zero-shot Validation Metrics:")
    print(f"  BLEU: {zero_shot_valid_metrics['bleu']:.4f}")
    print(f"  ROUGE-L: {zero_shot_valid_metrics['rouge_l']:.4f}")
    print(f"  BERTScore: {zero_shot_valid_metrics['bertscore']:.4f}")

    # Fine-tune model
    model, history = train_model(model, train_loader, valid_loader, tokenizer, args)

    # Evaluate on test set
    print("\nEvaluating fine-tuned model on test set...")
    try:
        best_model_path = os.path.join(args.output_dir, 'best_qa_text_model.pth')
        if os.path.exists(best_model_path):
             model.load_state_dict(torch.load(best_model_path, map_location=args.device))
             print("Loaded best model for final evaluation.")
        else:
             print("Best model checkpoint not found. Evaluating with the model from the last epoch.")
    except Exception as e:
        print(f"Could not load best model. Evaluating with the model from the last epoch. Error: {e}")

    test_metrics = evaluate_model(
        model, test_loader, tokenizer, args.device,
        log_examples=True, max_gen_length=args.max_gen_length
    )

    # Compare with GPT-2 if requested
    gpt2_metrics = None
    if args.compare_gpt2:
        print(f"\nComparing with {args.gpt2_model} on test set...")
        # Re-use the test_dataset which contains 'full_prompt'
        gpt2_test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False) # Simple loader ok

        gpt2_metrics = evaluate_gpt2(
            gpt2_test_loader, # Pass loader based on dataset containing 'full_prompt'
            tokenizer, # Pass original tokenizer for item access
            args.device,
            model_name=args.gpt2_model,
            max_gen_length=args.max_gen_length,
            log_examples=True
        )
        print(f"\n{args.gpt2_model} Test Metrics:")
        print(f"  BLEU: {gpt2_metrics['bleu']:.4f}")
        print(f"  ROUGE-L: {gpt2_metrics['rouge_l']:.4f}")
        print(f"  BERTScore: {gpt2_metrics['bertscore']:.4f}")
        save_examples(gpt2_metrics, args.output_dir, f"{args.gpt2_model}_test_predictions.csv")


    # Print summary results
    print("\n--- Final Results Summary ---")
    print("Zero-shot Validation Metrics:")
    print(f"  BLEU: {zero_shot_valid_metrics['bleu']:.4f}")
    print(f"  ROUGE-L: {zero_shot_valid_metrics['rouge_l']:.4f}")
    print(f"  BERTScore: {zero_shot_valid_metrics['bertscore']:.4f}")

    print("\nFine-tuned Test Metrics:")
    print(f"  BLEU: {test_metrics['bleu']:.4f}")
    print(f"  ROUGE-L: {test_metrics['rouge_l']:.4f}")
    print(f"  BERTScore: {test_metrics['bertscore']:.4f}")

    if gpt2_metrics:
        print(f"\n{args.gpt2_model} Test Metrics (Comparison):")
        print(f"  BLEU: {gpt2_metrics['bleu']:.4f}")
        print(f"  ROUGE-L: {gpt2_metrics['rouge_l']:.4f}")
        print(f"  BERTScore: {gpt2_metrics['bertscore']:.4f}")

    # Save examples from our fine-tuned model
    save_examples(test_metrics, args.output_dir, "finetuned_test_predictions.csv")

    # Save results summary
    results_summary_path = os.path.join(args.output_dir, 'results_summary.txt')
    with open(results_summary_path, 'w') as f:
        f.write("Zero-shot Validation Metrics:\n")
        f.write(f"  BLEU: {zero_shot_valid_metrics['bleu']:.4f}\n")
        f.write(f"  ROUGE-L: {zero_shot_valid_metrics['rouge_l']:.4f}\n")
        f.write(f"  BERTScore: {zero_shot_valid_metrics['bertscore']:.4f}\n\n")

        f.write("Fine-tuned Test Metrics:\n")
        f.write(f"  BLEU: {test_metrics['bleu']:.4f}\n")
        f.write(f"  ROUGE-L: {test_metrics['rouge_l']:.4f}\n")
        f.write(f"  BERTScore: {test_metrics['bertscore']:.4f}\n")

        if gpt2_metrics:
            f.write(f"\n{args.gpt2_model} Test Metrics (Comparison):\n")
            f.write(f"  BLEU: {gpt2_metrics['bleu']:.4f}\n")
            f.write(f"  ROUGE-L: {gpt2_metrics['rouge_l']:.4f}\n")
            f.write(f"  BERTScore: {gpt2_metrics['bertscore']:.4f}\n")

    print(f"\nResults summary saved to {results_summary_path}")

if __name__ == "__main__":
    main()
