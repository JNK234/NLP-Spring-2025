"""
MSAI-337, Spring 2025
Homework #2: Question 2 - Generative Approach for OpenBookQA (Multiple Choice)
Fine-tuning script for a pre-trained transformer model.
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
from transformers import GPT2TokenizerFast

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

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Special tokens for the prompt
START_TOKEN = "[START]"
ANSWER_TOKEN = "[ANSWER]"
CHOICE_TOKENS = ["[A]", "[B]", "[C]", "[D]"]
LABELS = ["A", "B", "C", "D"]

class OpenBookQADataset(Dataset):
    """Dataset for generative approach to OpenBookQA"""
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
        # Extract answer index
        answer_idx = self.answers.index(example['answerKey'])

        # Format the prompt as specified in the homework
        # [START] <fact> <stem> [A] <choice1> [B] <choice2> [C] <choice3> [D] <choice4> [ANSWER] <correct label>
        prompt = f"{START_TOKEN} {example['fact1']} {example['question']['stem']}"

        # Add each choice with its label
        for i, choice in enumerate(example['question']['choices']):
            prompt += f" {CHOICE_TOKENS[i]} {choice['text']}"

        prompt += f" {ANSWER_TOKEN} {LABELS[answer_idx]}"

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
            'correct_label': LABELS[answer_idx]
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
    correct_labels = batch['correct_labels'] # Keep as a list of strings

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'answer_positions': answer_positions,
        'correct_labels': correct_labels
    }

def pad_collate_fn(batch, pad_token_id):
    """
    Custom collate function to pad sequences within a batch.
    """
    input_ids_list = [item['input_ids'] for item in batch]
    attention_mask_list = [item['attention_mask'] for item in batch]
    answer_positions = torch.tensor([item['answer_position'] for item in batch])
    correct_labels = [item['correct_label'] for item in batch]

    input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id)
    attention_mask_padded = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded,
        'answer_positions': answer_positions,
        'correct_labels': correct_labels
    }

def train_model(model, train_loader, valid_loader, tokenizer, args):
    """
    Fine-tune the pre-trained transformer model on OpenBookQA
    """
    print("Starting fine-tuning...")
    # Add weight_decay to the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineWithRestarts(optimizer, T_max=len(train_loader) * args.epochs) if args.use_scheduler else None

    history = {'train_loss': [], 'valid_accuracy': []}
    best_valid_acc = 0.0

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
            output = model(input_ids, src_mask) # Shape: (batch_size, seq_len, vocab_size)

            batch_size = input_ids.size(0)
            # Logits at position `t` predict token at position `t+1`.
            # We want to predict the token at `answer_positions + 1` (the label 'A'/'B'/'C'/'D').
            # So, we need the logits from the preceding position: `answer_positions`.
            logits_for_answer = output[torch.arange(batch_size), answer_positions] # Shape: (batch_size, vocab_size)

            # The target token ID is the actual label token at `answer_positions + 1`.
            target_token_ids = input_ids[torch.arange(batch_size), answer_positions + 1] # Shape: (batch_size)

            # Calculate loss using logits for the answer position and the target token ID.
            loss = F.cross_entropy(logits_for_answer, target_token_ids)
            loss.backward()

            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()
            if scheduler:
                scheduler.step()

            total_loss += loss.item()
            avg_loss = total_loss / (progress_bar.n + 1)
            progress_bar.set_postfix({'loss': avg_loss})

        epoch_avg_loss = total_loss / len(train_loader)
        history['train_loss'].append(epoch_avg_loss)

        valid_acc, _ = evaluate_model(model, valid_loader, tokenizer, args.device, log_examples=(epoch % 5 == 0))
        history['valid_accuracy'].append(valid_acc)

        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {epoch_avg_loss:.4f} - Valid Acc: {valid_acc:.4f}")

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_qa_model.pth'))
            print(f"New best model saved with validation accuracy: {valid_acc:.4f}")

        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f'qa_model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")

    plot_history(history, args.output_dir)
    return model, history

def evaluate_model(model, dataloader, tokenizer, device, log_examples=False):
    """
    Evaluate the model on the given dataloader
    """
    model.eval()
    model.eval()
    model.eval()
    correct = 0
    total = 0
    predictions = []

    # No explicit mapping needed here anymore, we will decode and compare strings.

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            batch_data = prepare_batch(batch, device)
            input_ids = batch_data['input_ids']
            answer_positions = batch_data['answer_positions']
            correct_labels = batch_data['correct_labels']

            # --- Debugging Check ---
            if batch_idx == 0:
                print(f"\n--- Debugging evaluate_model (batch {batch_idx}) ---")
                max_val = torch.max(input_ids).item()
                num_embeddings = model.decoder.embed.embed.num_embeddings
                print(f"    Max ID in batch: {max_val}")
                print(f"    Model embedding size: {num_embeddings}")
                if max_val >= num_embeddings:
                    print(f"    !!! ALARM: Max token ID ({max_val}) >= num_embeddings ({num_embeddings})")
                print(f"--- End Debugging ---")
            # --- End Debugging ---

            src_mask = create_masks(input_ids, device)
            output = model(input_ids, src_mask) # Shape: (batch_size, seq_len, vocab_size)

            batch_size = input_ids.size(0)
            # Select logits from the position *before* the target label token (i.e., at answer_positions)
            logits_for_answer = output[torch.arange(batch_size), answer_positions] # Shape: (batch_size, vocab_size)

            # Get the token ID with the highest probability at that position
            predicted_token_ids = torch.argmax(logits_for_answer, dim=1) # Shape: (batch_size)

            batch_predictions = []
            for i, pred_id_tensor in enumerate(predicted_token_ids):
                pred_id = pred_id_tensor.item() # Get the integer token ID

                # Decode the single predicted token ID into text and strip whitespace
                decoded_token_text = tokenizer.decode(pred_id).strip()

                extracted_answer = ""
                # Check if the decoded token *is* one of the special choice tokens
                # CHOICE_TOKENS = ["[A]", "[B]", "[C]", "[D]"]
                # LABELS = ["A", "B", "C", "D"]
                if decoded_token_text == CHOICE_TOKENS[0]: # Decoded to "[A]"
                    extracted_answer = LABELS[0]          # Interpret as "A"
                elif decoded_token_text == CHOICE_TOKENS[1]: # Decoded to "[B]"
                    extracted_answer = LABELS[1]          # Interpret as "B"
                elif decoded_token_text == CHOICE_TOKENS[2]: # Decoded to "[C]"
                    extracted_answer = LABELS[2]          # Interpret as "C"
                elif decoded_token_text == CHOICE_TOKENS[3]: # Decoded to "[D]"
                    extracted_answer = LABELS[3]          # Interpret as "D"
                elif decoded_token_text: # Fallback: If not a special token, take its first character
                    extracted_answer = decoded_token_text[0]
                # If decoded_token_text was empty, extracted_answer remains ""

                batch_predictions.append(extracted_answer)

                # Compare the *interpreted* answer with the correct label string
                if extracted_answer == correct_labels[i]:
                    correct += 1
                total += 1 # Always increment total for every example evaluated

            predictions.extend(batch_predictions)

            if log_examples and batch_idx == 0 and len(batch_predictions) > 0: # Log first batch only
                print("\n--- Example Predictions (First Batch) ---")
                for i in range(min(3, len(batch_predictions))): # Log up to 3 examples
                    # Decode the full input for context
                    input_text = tokenizer.decode(input_ids[i].cpu())
                    # Get the prediction stored (which is now the first char)
                    pred = batch_predictions[i]
                    # Get the correct label
                    corr = correct_labels[i]
                    # Split for cleaner logging
                    input_parts = input_text.split(ANSWER_TOKEN)
                    question_part = input_parts[0] + ANSWER_TOKEN # Keep token for clarity
                    print(f"Q: {question_part}")
                    print(f"Prediction: {pred} (Correct: {corr})")
                    print("-" * 50)

    accuracy = correct / total if total > 0 else 0
    return accuracy, predictions

def generate_answers(model, dataloader, tokenizer, device, max_new_tokens=5):
    """
    Generates answers for OpenBookQA examples using the fine-tuned model.

    This function performs iterative decoding (greedy search) starting from the
    prompt which includes the fact, question, choices, and the '[ANSWER]' token.
    It generates tokens one by one until a stopping condition is met or
    `max_new_tokens` are generated. The final prediction is extracted as the
    first non-space character from the generated sequence.

    Args:
        model (nn.Module): The fine-tuned transformer model.
        dataloader (DataLoader): DataLoader providing batches of OpenBookQA examples.
        tokenizer (GPT2TokenizerFast): The tokenizer used for encoding/decoding.
        device (torch.device): The device (CPU/GPU) to perform computations on.
        max_new_tokens (int): The maximum number of tokens to generate after the [ANSWER] token.
                              Default is 5, which is usually enough for "A", "B", "C", "D" + potential space/EOS.

    Returns:
        list: A list of dictionaries, where each dictionary contains:
            - 'input' (str): The original input prompt (up to [ANSWER]).
            - 'prediction' (str): The model's predicted answer (single character A, B, C, or D, or empty if none).
            - 'gold' (str): The ground truth answer.

    Architectural Choices & Reasoning:
    - Greedy Decoding: For simplicity and speed, greedy decoding (torch.argmax) is used.
      This selects the most probable next token at each step. While effective, it might
      not always find the globally optimal sequence.
      Alternative: Beam search could be implemented for potentially better quality by
      keeping track of multiple hypotheses, but at a higher computational cost.
    - Stopping Conditions:
        1. `max_new_tokens`: A hard limit to prevent excessively long or runaway generation.
        2. EOS token: The standard way to stop generation when the model signals the end
           of its intended sequence. The EOS token itself is not part of the answer.
        3. Space after first generated token: A heuristic tailored for this specific task.
           Since the expected answer is a single character (A, B, C, D), if the model
           generates "A" then " " (space), it's a strong signal the answer "A" is complete.
           The space token itself is not part of the answer. This helps prevent generating
           extraneous text like "A and B are correct." when only "A" is desired.
    - Token Cleaning: Before final decoding of the generated part, `tokenizer.pad_token_id`
      and `tokenizer.eos_token_id` are explicitly removed. This is a safeguard, as EOS
      should ideally be handled by the stopping condition.
    - Answer Extraction: `generated_text.strip()[0] if generated_text.strip() else ""`
      is a robust way to get the first non-whitespace character from the decoded generated
      string. It handles cases like " A", "B ", or just "C", converting them to "A", "B", "C".
      If the stripped string is empty (e.g., only EOS or padding was generated), it defaults
      to an empty string for the prediction.
    """
    model.eval()  # Set the model to evaluation mode (disables dropout, etc.)
    results = []

    with torch.no_grad():  # Disable gradient calculations during inference
        for batch in tqdm(dataloader, desc="Generating Answers"):
            # Prepare batch data and move to the specified device
            batch_data = prepare_batch(batch, device)
            input_ids = batch_data['input_ids']             # Shape: (batch_size, seq_len)
            answer_positions = batch_data['answer_positions'] # Shape: (batch_size,)
            correct_labels = batch_data['correct_labels']   # List of strings (ground truth labels)
            batch_size = input_ids.size(0)

            for i in range(batch_size):
                # --- 1. Initialize Generation ---
                # Get the prompt for the current example. This includes the input sequence
                # up to and including the [ANSWER] token.
                # `answer_positions[i]` is the index of the [ANSWER] token.
                # `+1` makes the slice inclusive of the [ANSWER] token.
                current_prompt_ids_tensor = input_ids[i, :answer_positions[i] + 1] # Shape: (prompt_len,)
                # `seq` will be the input to the model, starting with the prompt and then appended with generated tokens.
                # Add a batch dimension for model input: (1, prompt_len)
                seq = current_prompt_ids_tensor.unsqueeze(0)

                # Store only the newly generated token IDs for this example
                generated_token_ids_list = []

                # --- 2. Iterative Token Generation ---
                for _ in range(max_new_tokens):
                    # Create attention mask for the current sequence `seq`.
                    # The mask ensures the model attends only to valid tokens.
                    seq_mask = create_masks(seq, device)

                    # Forward pass: Get model logits for the current sequence `seq`.
                    # `output` shape: (1, current_seq_len, vocab_size)
                    output = model(seq, seq_mask)

                    # Get logits for the *next* token prediction. These are the logits
                    # from the last token position in the current sequence `seq`.
                    # `output[:, -1, :]` selects logits for all vocab tokens at the last time step.
                    # Shape: (1, vocab_size)
                    next_token_logits = output[:, -1, :]

                    # Greedy decoding: Select the token ID with the highest logit.
                    # `next_token_id_tensor` shape: (1, 1)
                    next_token_id_tensor = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
                    next_token_id = next_token_id_tensor.item() # Convert to Python integer

                    # --- Stopping Conditions ---
                    # Check stopping conditions *before* adding the current `next_token_id`
                    # to `generated_token_ids_list` if it's a special stopping token (EOS/space).

                    # Condition 2: Stop if the End-of-Sequence (EOS) token is generated.
                    # The EOS token itself is not part of the answer.
                    if next_token_id == tokenizer.eos_token_id:
                        break # Exit generation loop

                    # Condition 3: Stop if a space token is generated *after* the first actual answer token.
                    # This heuristic helps capture single-character answers like "A", "B", "C", "D".
                    # `len(generated_token_ids_list) >= 1` ensures at least one non-special token has been generated.
                    # (Using GPT-2's space token 'Ġ' which is ID 220, or a regular space if tokenizer handles it differently)
                    # It's important to check the actual ID for space, as `tokenizer.convert_tokens_to_ids(" ")` might vary.
                    # A common ID for 'Ġ' (space prefixed) in GPT2 is 220.
                    # We assume the first generated token is the answer character.
                    is_space_token = (next_token_id == tokenizer.convert_tokens_to_ids(" ") or
                                      next_token_id == tokenizer.convert_tokens_to_ids("Ġ") or # Common GPT2 space
                                      tokenizer.decode(next_token_id).strip() == "") # General check for space-like tokens

                    if len(generated_token_ids_list) >= 1 and is_space_token:
                        break # Exit generation loop, space is not part of the answer

                    # If not a stopping token, add to list and append to sequence for next iteration
                    generated_token_ids_list.append(next_token_id)
                    seq = torch.cat([seq, next_token_id_tensor], dim=1) # Append for next model input

                # End of generation loop (max_new_tokens reached or a stopping condition met)

                # --- 3. Decode and Process Generated Output ---
                # Decode the original input prompt part (up to [ANSWER]) for context in results
                input_text_prompt = tokenizer.decode(current_prompt_ids_tensor.cpu())

                # Convert the list of *actually generated* token IDs (excluding stopping tokens like EOS/trailing space) to a tensor
                if not generated_token_ids_list: # Handle case where no tokens were generated (e.g., EOS right away)
                    generated_text = ""
                else:
                    generated_ids_tensor = torch.tensor(generated_token_ids_list, dtype=torch.long, device='cpu')

                    # Safeguard: Remove any padding tokens if they somehow appeared, though unlikely with this logic.
                    generated_ids_tensor = generated_ids_tensor[generated_ids_tensor != tokenizer.pad_token_id]
                    # EOS should have been handled by stopping condition, but as a safeguard:
                    generated_ids_tensor = generated_ids_tensor[generated_ids_tensor != tokenizer.eos_token_id]

                    # Decode the cleaned generated token IDs into text
                    generated_text = tokenizer.decode(generated_ids_tensor)

                # --- 3. Decode and Process Generated Output ---
                # Decode the original input prompt part (up to [ANSWER]) for context in results
                input_text_prompt = tokenizer.decode(current_prompt_ids_tensor.cpu())

                predicted_answer_char = ""
                # Check the *first* generated token ID if any were generated
                if generated_token_ids_list:
                    first_generated_id = generated_token_ids_list[0]
                    decoded_first_token = tokenizer.decode(first_generated_id).strip()

                    # Apply the same interpretation logic as in evaluate_model
                    if decoded_first_token == CHOICE_TOKENS[0]: # "[A]"
                        predicted_answer_char = LABELS[0]      # "A"
                    elif decoded_first_token == CHOICE_TOKENS[1]: # "[B]"
                        predicted_answer_char = LABELS[1]      # "B"
                    elif decoded_first_token == CHOICE_TOKENS[2]: # "[C]"
                        predicted_answer_char = LABELS[2]      # "C"
                    elif decoded_first_token == CHOICE_TOKENS[3]: # "[D]"
                        predicted_answer_char = LABELS[3]      # "D"
                    elif decoded_first_token: # Fallback: Take first char of the first decoded token
                         predicted_answer_char = decoded_first_token[0]
                # If no tokens were generated, or the first token didn't map, predicted_answer_char remains ""

                results.append({
                    'input': input_text_prompt,         # The prompt part (up to [ANSWER])
                    'prediction': predicted_answer_char, # The interpreted single character prediction
                    'gold': correct_labels[i]           # The ground truth label
                })
    return results


def plot_history(history, output_dir):
    """Plot training history"""
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], 'b-')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history['valid_accuracy'], 'g-')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'qa_training_history.png'))
    print(f"Training history plot saved to {os.path.join(output_dir, 'qa_training_history.png')}")

def main():
    parser = argparse.ArgumentParser(description='Fine-tune transformer model for OpenBookQA')
    # Model parameters
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of model embeddings')
    parser.add_argument('--n_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    # Training parameters
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    # Add weight decay argument
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for AdamW optimizer')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--use_scheduler', action='store_true', help='Use learning rate scheduler')
    parser.add_argument('--save_every', type=int, default=1, help='Save model every N epochs')
    # File paths
    parser.add_argument('--train_file', type=str, default='train_complete.jsonl', help='Path to training data')
    parser.add_argument('--valid_file', type=str, default='dev_complete.jsonl', help='Path to validation data')
    parser.add_argument('--test_file', type=str, default='test_complete.jsonl', help='Path to test data')
    parser.add_argument('--model_path', type=str, default='Logs/NLP_H2_Q2_pretraining/pretrained_wiki103_ep20.pth',
                        help='Path to pre-trained model')
    parser.add_argument('--output_dir', type=str, default='Logs/Q2_finetuning', help='Output directory')
    # Other options
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
            # Load state dict to CPU first to avoid GPU memory issues if model is large
            pretrained_state_dict = torch.load(args.model_path, map_location='cpu')
            # Load weights, ignoring layers with size mismatches (embedding, output bias)
            load_result = model.load_state_dict(pretrained_state_dict, strict=False)
            print(f"Weight loading result:")
            print(f"  Missing keys: {load_result.missing_keys}") # Should include embedding/output layers
            print(f"  Unexpected keys: {load_result.unexpected_keys}")
            if not any('decoder.layers' in key for key in model.state_dict() if key not in load_result.missing_keys):
                 print("Warning: Core decoder layers might not have been loaded correctly.")
            else:
                 print("Successfully loaded compatible pre-trained weights.")
        except FileNotFoundError:
            print(f"Error: Pretrained weights file not found at {args.model_path}. Model weights remain uninitialized.")
            # Initialize randomly if loading fails
            print("Initializing model weights randomly using Xavier uniform.")
            for p in model.parameters():
                if p.dim() > 1: nn.init.xavier_uniform_(p)
        except Exception as e:
             print(f"Error loading weights: {e}. Model may be partially initialized or uninitialized.")
             # Initialize randomly on error
             print("Initializing model weights randomly using Xavier uniform due to loading error.")
             for p in model.parameters():
                 if p.dim() > 1: nn.init.xavier_uniform_(p)
    else:
        print("No model_path provided or file doesn't exist.")
        # Initialize randomly if no pre-trained weights are loaded
        print("Initializing model weights randomly using Xavier uniform.")
        for p in model.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    # 8. Manually Resize Embeddings and Output Layer
    if original_vocab_size != new_vocab_size:
        print(f"Resizing model embeddings and output layer from {original_vocab_size} to {new_vocab_size}...")

        # Resize Token Embeddings
        old_embeddings = model.decoder.embed.embed
        embedding_dim = old_embeddings.embedding_dim
        new_embeddings = nn.Embedding(new_vocab_size, embedding_dim)
        # Copy old weights
        new_embeddings.weight.data[:original_vocab_size, :] = old_embeddings.weight.data[:original_vocab_size, :]
        # Initialize new token weights
        new_embeddings.weight.data[original_vocab_size:].normal_(mean=0.0, std=0.02)
        model.decoder.embed.embed = new_embeddings
        print("  - Token embeddings resized.")

        # Handle Output Layer (Tied Weights assumed based on model_opt.tied=1)
        if model_opt.tied == 1:
            # Weights are tied, just update the reference
            model.out.weight = model.decoder.embed.embed.weight
            print("  - Output layer weights tied and updated.")

            # Resize Output Bias (if it exists)
            if hasattr(model.out, 'bias') and model.out.bias is not None:
                old_bias = model.out.bias
                if old_bias.size(0) == original_vocab_size:
                    new_bias = nn.Parameter(torch.Tensor(new_vocab_size))
                    # Copy old bias values
                    new_bias.data[:original_vocab_size] = old_bias.data[:original_vocab_size]
                    # Initialize new bias values (e.g., to zero)
                    new_bias.data[original_vocab_size:].zero_()
                    model.out.bias = new_bias
                    print(f"  - Output layer bias resized from {original_vocab_size} to {new_vocab_size}.")
                else:
                     print(f"  - Warning: Output bias size ({old_bias.size(0)}) != original vocab size ({original_vocab_size}). Bias not resized.")
            else:
                print("  - Warning: model.out.bias not found or is None. Bias not resized.")
        else:
            # Handle untied weights case (similar logic but resize model.out.weight too)
            print("  - Warning: Weights are not tied (model_opt.tied != 1). Resizing logic for untied output weights needs implementation if required.")
            # Add logic here if untied weights are ever used for this model structure

        print("Model resizing complete.")
    else:
        print("Vocab size unchanged. Skipping model resizing.")

    # 9. Move Model to Device (AFTER potential resizing)
    model.to(args.device)
    print(f"Model moved to device: {args.device}")

    # --- End Cleaned Model Loading and Resizing ---

    # --- Verification Step (Optional but recommended) ---
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
    train_dataset = OpenBookQADataset(train_data, tokenizer)
    valid_dataset = OpenBookQADataset(valid_data, tokenizer)
    test_dataset = OpenBookQADataset(test_data, tokenizer)

    print(f"Train examples: {len(train_dataset)}")
    print(f"Valid examples: {len(valid_dataset)}")
    print(f"Test examples: {len(test_dataset)}")

    # Get pad token ID
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        print(f"Warning: Using EOS token (ID: {pad_token_id}) as padding token.")

    # Create data loaders
    collate_fn_with_padding = lambda batch: pad_collate_fn(batch, pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_with_padding)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_with_padding)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_with_padding)

    # --- Inspect First Training Batch ---
    print("\n--- Inspecting First Training Batch ---")
    try:
        first_batch = next(iter(train_loader))
        print(f"Batch size: {len(first_batch['correct_labels'])}")
        num_to_print = min(3, len(first_batch['correct_labels'])) # Print up to 3 examples
        print(f"Printing details for the first {num_to_print} examples:")
        for i in range(num_to_print):
            input_ids_example = first_batch['input_ids'][i]
            # Decode, skipping special tokens used by the tokenizer itself (like padding/eos if they are special)
            # but keeping our added special tokens ([START], [ANSWER], [A] etc.)
            decoded_text = tokenizer.decode(input_ids_example, skip_special_tokens=False)
            # Remove padding tokens manually if they weren't skipped
            decoded_text = decoded_text.replace(tokenizer.pad_token, "").strip()

            correct_label = first_batch['correct_labels'][i]
            answer_pos = first_batch['answer_positions'][i].item() # Get scalar value

            # Find the token ID at answer_pos + 1 (the target token)
            target_token_id = input_ids_example[answer_pos + 1].item()
            target_token_decoded = tokenizer.decode([target_token_id]).strip()

            print(f"\nExample {i+1}:")
            print(f"  Input Text (Decoded): {decoded_text}")
            print(f"  Correct Label: '{correct_label}'")
            print(f"  [ANSWER] Token Position: {answer_pos}")
            print(f"  Target Token ID (at pos {answer_pos + 1}): {target_token_id}")
            print(f"  Target Token Decoded: '{target_token_decoded}'") # Should match Correct Label
            print("-" * 20)
    except StopIteration:
        print("Warning: Training loader is empty, cannot inspect batch.")
    except Exception as e:
        print(f"Error inspecting training batch: {e}")
    print("--- End Inspection ---")
    # --- End Inspection ---


    # Evaluate zero-shot performance
    print("\nEvaluating zero-shot performance...")
    zero_shot_valid_acc, _ = evaluate_model(model, valid_loader, tokenizer, args.device, log_examples=True)
    print(f"Zero-shot validation accuracy: {zero_shot_valid_acc:.4f}")

    # Fine-tune model
    model, history = train_model(model, train_loader, valid_loader, tokenizer, args)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    try:
        # Load best model saved during training
        best_model_path = os.path.join(args.output_dir, 'best_qa_model.pth')
        if os.path.exists(best_model_path):
             model.load_state_dict(torch.load(best_model_path, map_location=args.device))
             print("Loaded best model for final evaluation.")
        else:
             print("Best model checkpoint not found. Evaluating with the model from the last epoch.")
    except Exception as e:
        print(f"Could not load best model. Evaluating with the model from the last epoch. Error: {e}")

    test_acc, test_preds = evaluate_model(model, test_loader, tokenizer, args.device, log_examples=True)
    print(f"Final Test accuracy: {test_acc:.4f}")

    # Generate some example predictions
    print("\nGenerating example predictions from test set...")
    results = generate_answers(model, test_loader, tokenizer, args.device)

    # Print summary results
    print("\n--- Final Results Summary ---")
    print(f"Zero-shot validation accuracy: {zero_shot_valid_acc:.4f}")
    # Ensure history is not empty before accessing
    final_valid_acc = history['valid_accuracy'][-1] if history['valid_accuracy'] else "N/A"
    print(f"Fine-tuned validation accuracy (last epoch): {final_valid_acc}")
    print(f"Fine-tuned test accuracy: {test_acc:.4f}")

    # Print 5 correct and 5 incorrect examples
    print("\n--- Example Test Set Predictions ---")
    correct_examples = [r for r in results if r['prediction'] == r['gold']]
    incorrect_examples = [r for r in results if r['prediction'] != r['gold']]

    print("\nCorrect Predictions:")
    for i, example in enumerate(correct_examples[:5]):
        print(f"{i+1}. Input: {example['input']}")
        print(f"   Prediction: {example['prediction']} (Gold: {example['gold']})")

    print("\nIncorrect Predictions:")
    for i, example in enumerate(incorrect_examples[:5]):
        print(f"{i+1}. Input: {example['input']}")
        print(f"   Prediction: {example['prediction']} (Gold: {example['gold']})")

    # Save results summary
    results_summary_path = os.path.join(args.output_dir, 'results_summary.txt')
    with open(results_summary_path, 'w') as f:
        f.write(f"Zero-shot validation accuracy: {zero_shot_valid_acc:.4f}\n")
        f.write(f"Fine-tuned validation accuracy (last epoch): {final_valid_acc}\n")
        f.write(f"Fine-tuned test accuracy: {test_acc:.4f}\n")
    print(f"\nResults summary saved to {results_summary_path}")

    # Save all predictions
    predictions_path = os.path.join(args.output_dir, 'test_predictions.jsonl')
    with open(predictions_path, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')
    print(f"All test predictions saved to {predictions_path}")


if __name__ == "__main__":
    main()
