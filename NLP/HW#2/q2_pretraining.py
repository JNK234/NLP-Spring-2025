import argparse
import os
import sys
import shutil
import random
import numpy as np
import time
import copy
import math
import pickle
from tqdm import tqdm 

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from transformers import GPT2TokenizerFast
import matplotlib.pyplot as plt # Import for plotting

def read_corpus(filename,tokenizer):
    """Reads a corpus file, tokenizes it, and returns a flat list of token IDs."""
    seq = []
    # Ensure file exists before opening
    if not os.path.exists(filename):
        print(f"Error: File not found at {filename}")
        sys.exit(1)
    try:
        with open(filename,'rt', encoding='utf-8') as f: # Specify encoding
            for line in f:
                line = line.strip() # Use strip() to remove leading/trailing whitespace including newline
                if line: # Process non-empty lines
                    tokens = tokenizer(line)
                    for t in tokens['input_ids']:
                        seq.append(t)
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        sys.exit(1)
    return(seq)

class Embedder(nn.Module):
    """Input Embedding layer"""
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        # nn.Embedding expects LongTensor input. Input 'x' should already be LongTensor.
        return self.embed(x)

class PositionalEncoder(nn.Module):
    """Adds positional encoding to the input embeddings."""
    def __init__(self, d_model, max_seq_len = 4096, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        # Ensure positional encoding is on the same device as input x
        pe_slice = self.pe[:, :seq_len].to(x.device)  # Directly move to x's device
        x = x + pe_slice
        return self.dropout(x)

class Norm(nn.Module):
    """Layer Normalization module."""
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

def attention(q, k, v, d_k, mask=None, dropout=None):
    """Core Scaled Dot-Product Attention calculation."""
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    # Using Euclidean Distance instead of dot product
    # q_norm = torch.sum(q**2, dim=-1, keepdim=True)
    # k_norm = torch.sum(k**2, dim=-1, keepdim=True)
    
    # qk_dot = torch.matmul(q, k.transpose(-2, -1))
    
    # distances = q_norm + k_norm.transpose(-2, -1) - 2 * qk_dot
    # scores = -distances / math.sqrt(d_k) # Negative distances for similarity
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module."""
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        assert d_model % heads == 0
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output

class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network."""
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
    
def get_clones(module, N):
    """Produces N identical copies of the module."""
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class CosineWithRestarts(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing with restarts scheduler."""
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 T_max: int,
                 eta_min: float = 0.,
                 last_epoch: int = -1,
                 factor: float = 1.) -> None:
        self.T_max = T_max
        self.eta_min = eta_min
        self.factor = factor
        self._last_restart: int = 0
        self._cycle_counter: int = 0
        self._cycle_factor: float = 1.
        self._updated_cycle_len: int = T_max
        self._initialized: bool = False
        super(CosineWithRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._initialized:
            self._initialized = True
            return self.base_lrs
        step = self.last_epoch + 1
        self._cycle_counter = step - self._last_restart
        lrs = [
            (
                self.eta_min + ((lr - self.eta_min) / 2) *
                (np.cos(np.pi * ((self._cycle_counter) % self._updated_cycle_len) / self._updated_cycle_len) + 1)
            ) for lr in self.base_lrs
        ]
        if self._cycle_counter % self._updated_cycle_len == 0:
            self._cycle_factor *= self.factor
            self._cycle_counter = 0
            self._updated_cycle_len = int(self._cycle_factor * self.T_max)
            self._last_restart = step
        return lrs

class DecoderLayer(nn.Module):
    """A single layer of the Decoder stack."""
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_3 = Norm(d_model) 
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout) 
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout) 
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x    
    
class Decoder(nn.Module):
    """The Decoder stack."""
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model) 
        
    def forward(self, trg, mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)

class Transformer(nn.Module):
    """ GPT2-style Decoder-Only Transformer model. """
    def __init__(self, vocab_size, d_model, N, heads, dropout, tied_weights=True):
        super().__init__()
        self.decoder = Decoder(vocab_size, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, vocab_size)
        if tied_weights:
            print("Tying input embedding and output linear layer weights.")
            self.out.weight = self.decoder.embed.embed.weight

    def forward(self, x, mask):
        d_output = self.decoder(x, mask)
        output = self.out(d_output)
        return output

def get_model(opt, vocab_size, initialize_weights=True):
    """
    Builds the Transformer model based on options.
    Weight loading is now handled separately after model creation.
    """
    assert opt.d_model % opt.heads == 0, "d_model must be divisible by heads"
    assert opt.dropout < 1, "dropout probability must be less than 1"
    
    print(f"Creating Transformer model structure with vocab_size={vocab_size}, d_model={opt.d_model}, n_layers={opt.n_layers}, heads={opt.heads}, tied={opt.tied==1}")
    model = Transformer(vocab_size, opt.d_model, opt.n_layers, opt.heads, opt.dropout, tied_weights=(opt.tied == 1))
    
    if initialize_weights:
        print("Initializing model weights randomly using Xavier uniform.")
        for p in model.parameters():
            if p.dim() > 1: 
                nn.init.xavier_uniform_(p)
    else:
        # Weights will be loaded externally or initialized if loading fails
        print("Skipping automatic weight initialization in get_model.")
        
    model.to(opt.device)
    return model

# --- Utility functions for masking ---
def nopeak_mask(size, device):
    """ Creates an upper triangular mask to prevent attention to future positions. """
    mask = torch.triu(torch.ones((1, size, size), device=device), diagonal=1).bool()
    return (~mask)  # Invert mask to allow attention to valid positions

def create_masks(src, device, pad_token=0):
    """ Creates the self-attention mask for the decoder (lookahead mask). """
    size = src.size(1) 
    np_mask = nopeak_mask(size, device) 
    return np_mask.unsqueeze(1) 

# --- Training and Testing Functions ---
    
def train_model(model, opt):
    """ 
    Trains the Transformer model. 
    Returns lists of training and validation perplexities per epoch.
    """
    print("--- Starting Training ---")
    model.train() 
    
    train_data = opt.train
    num_sequences = (len(train_data) - 1) // opt.seqlen
    num_batches = num_sequences // opt.batchsize
    
    if num_batches == 0:
        print(f"Warning: Not enough data ({len(train_data)} tokens) for a single batch. Exiting training.")
        return [], [] # Return empty lists

    print(f"Training data: {len(train_data)} tokens | Batches per epoch: {num_batches}")
    opt.train_len = num_batches 

    train_perplexities = []
    val_perplexities = []

    for epoch in range(opt.epochs):
        epoch_total_loss = 0.0 # Accumulate loss over the entire epoch
        epoch_total_tokens = 0 # Accumulate tokens for accurate epoch perplexity
        epoch_interval_loss = 0.0 # Accumulate loss for interval logging
        epoch_start_time = time.time()
        batch_indices = list(range(0, num_sequences * opt.seqlen, opt.seqlen * opt.batchsize))
        random.shuffle(batch_indices) 

        pbar = tqdm(enumerate(batch_indices), total=num_batches, desc=f"Epoch {epoch+1}/{opt.epochs}", leave=False)
        for i, batch_start_token_idx in pbar:
            
            src_list, trg_list = [], []
            current_batch_tokens = 0
            for b in range(opt.batchsize):
                start_idx = batch_start_token_idx + b * opt.seqlen
                if start_idx + opt.seqlen + 1 > len(train_data): continue 
                src_seq = train_data[start_idx : start_idx + opt.seqlen]
                trg_seq = train_data[start_idx + 1 : start_idx + opt.seqlen + 1]
                src_list.append(src_seq)
                trg_list.append(trg_seq)
                current_batch_tokens += len(trg_seq)

            if not src_list: continue 

            src = torch.LongTensor(src_list).to(opt.device)
            trg = torch.LongTensor(trg_list).to(opt.device)

            trg_mask = create_masks(src, opt.device)
            preds = model(src, trg_mask)
            # Use reduction='sum' to sum loss over batch, then average later for epoch ppl
            loss = F.cross_entropy(preds.view(-1, opt.vocab_size), 
                                   trg.contiguous().view(-1), 
                                   ignore_index=opt.trg_pad,
                                   reduction='sum') 
            
            opt.optimizer.zero_grad()
            # Scale loss by number of sequences in batch for gradient calculation if desired, or use mean loss
            # Using sum loss directly might lead to large gradients with large batches
            # Let's use mean loss for backprop:
            mean_loss = loss / current_batch_tokens 
            mean_loss.backward()
            
            if opt.norm > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), opt.norm)
            opt.optimizer.step()
            
            if opt.SGDR: opt.sched.step()

            epoch_total_loss += loss.item() # Accumulate sum loss for epoch ppl
            epoch_total_tokens += current_batch_tokens
            epoch_interval_loss += loss.item() # Accumulate sum loss for interval logging
            
            if (i + 1) % opt.printevery == 0: 
                 # Calculate interval PPL based on average loss over interval tokens
                 interval_tokens = epoch_total_tokens - (epoch_total_tokens - current_batch_tokens * opt.printevery) # Estimate tokens in interval
                 avg_loss_interval = epoch_interval_loss / (interval_tokens + 1e-9) # Avoid division by zero
                 perplexity_interval = math.exp(avg_loss_interval) if avg_loss_interval < 70 else float('inf')
                 current_lr = opt.optimizer.param_groups[0]['lr']
                 pbar.set_postfix(Loss=f"{avg_loss_interval:.4f}", PPL=f"{perplexity_interval:.2f}", LR=f"{current_lr:.6f}")
                 epoch_interval_loss = 0.0 # Reset interval loss accumulator

        pbar.close() 
        # --- End of Epoch ---
        epoch_duration = time.time() - epoch_start_time
        # Calculate average training loss and perplexity for the whole epoch
        avg_epoch_train_loss = epoch_total_loss / epoch_total_tokens
        epoch_train_perplexity = math.exp(avg_epoch_train_loss) if avg_epoch_train_loss < 70 else float('inf')
        train_perplexities.append(epoch_train_perplexity)
        
        print(f"--- Epoch {epoch+1} Finished | Time: {epoch_duration/60:.2f} min | Train PPL: {epoch_train_perplexity:.2f} ---")

        # --- Validation Step ---
        val_perplexity = test_model(model, opt.valid, opt, epoch) 
        val_perplexities.append(val_perplexity)
        print(f"Epoch {epoch+1} Validation Perplexity: {val_perplexity:.2f}")
        print("-" * 60)
        
        # --- Save Model ---
        if opt.savename:
            save_filename = f"{opt.savename}_epoch_{epoch+1}.pth"
            save_path = os.path.join(opt.dir_name, save_filename)
            print(f"Saving model checkpoint to {save_path}")
            torch.save(model.state_dict(), save_path)
        else:
             print("Warning: opt.savename not specified. Model weights will not be saved.")
        
        model.train() 

    print("--- Training Finished ---")
    return train_perplexities, val_perplexities # Return lists

    
def test_model(model, data, opt, epoch=-1): 
    """ Evaluates the model on a given dataset (validation or test). """
    mode = "Validation" if epoch != -1 else "Final Test"
    print(f"\n--- Running {mode} ---")
    model.eval() 
    total_loss = 0
    total_tokens = 0 

    num_sequences = (len(data) - 1) // opt.seqlen
    num_batches = num_sequences // opt.batchsize

    if num_batches == 0:
        print(f"Warning: Not enough data ({len(data)} tokens) for a single {mode} batch. Returning inf perplexity.")
        model.train() 
        return float('inf')

    print(f"{mode} data: {len(data)} tokens, Batches: {num_batches}")
    
    with torch.no_grad(): 
        batch_indices = list(range(0, num_sequences * opt.seqlen, opt.seqlen * opt.batchsize))
        # Add tqdm for evaluation as well
        pbar_test = tqdm(batch_indices, desc=f"{mode} Evaluation", leave=False) 

        for batch_start_token_idx in pbar_test:
            src_list, trg_list = [], []
            current_batch_tokens = 0
            for b in range(opt.batchsize):
                start_idx = batch_start_token_idx + b * opt.seqlen
                if start_idx + opt.seqlen + 1 > len(data): continue 
                src_seq = data[start_idx : start_idx + opt.seqlen]
                trg_seq = data[start_idx + 1 : start_idx + opt.seqlen + 1]
                src_list.append(src_seq)
                trg_list.append(trg_seq)
                current_batch_tokens += len(trg_seq) 

            if not src_list: continue

            src = torch.LongTensor(src_list).to(opt.device)
            trg = torch.LongTensor(trg_list).to(opt.device)

            trg_mask = create_masks(src, opt.device)
            preds = model(src, trg_mask)
            
            loss = F.cross_entropy(preds.view(-1, opt.vocab_size), 
                                   trg.contiguous().view(-1), 
                                   ignore_index=opt.trg_pad,
                                   reduction='sum') 
            
            total_loss += loss.item()
            total_tokens += current_batch_tokens 
        
        pbar_test.close()

    perplexity = float('inf')
    if total_tokens > 0:
        avg_loss = total_loss / total_tokens 
        perplexity = math.exp(avg_loss) if avg_loss < 70 else float('inf')
        print(f"{mode} Results: Average Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
    else:
        print(f"Warning: No tokens processed in {mode} set.")

    model.train() 
    print("---------------------------\n")
    return perplexity

def plot_perplexities(train_ppl, val_ppl, test_ppl, save_path):
    """ Plots training, validation, and test perplexities. """
    epochs = range(1, len(train_ppl) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_ppl, 'bo-', label='Training Perplexity')
    plt.plot(epochs, val_ppl, 'go-', label='Validation Perplexity')
    # Plot test perplexity as a horizontal line or a single point
    plt.axhline(y=test_ppl, color='r', linestyle='--', label=f'Final Test Perplexity ({test_ppl:.2f})')
    # Or plot as a point: plt.plot(len(epochs), test_ppl, 'r*', markersize=10, label=f'Final Test PPL ({test_ppl:.2f})')
    
    plt.title('Training, Validation, and Test Perplexity')
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0) # Ensure y-axis starts at 0 or slightly above min perplexity
    
    try:
        plt.savefig(save_path)
        print(f"Perplexity plot saved to {save_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    # plt.show() # Optionally display the plot interactively

def main():
    
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(10)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(10)

    parser = argparse.ArgumentParser(description="GPT2-Style Autoregressive Language Model Training")
    # Model Hyperparameters
    parser.add_argument('-d_model', type=int, default=512, help='Embedding dimension size')
    parser.add_argument('-n_layers', type=int, default=6, help='Number of decoder layers')
    parser.add_argument('-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('-dropout', type=float, default=0.1, help='Dropout rate') 
    parser.add_argument('-tied', type=int, default=1, choices=[0, 1], help='Tie input/output embeddings (1=True, 0=False)')

    # Training Hyperparameters
    parser.add_argument('-epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('-batchsize', type=int, default=16, help='Batch size for training') 
    parser.add_argument('-lr', type=float, default=0.0001, help='Learning rate') 
    parser.add_argument('-seqlen', type=int, default=512, help='Sequence length for training')
    parser.add_argument('-norm', type=float, default=1.0, help='Gradient clipping norm (0 to disable)') 
    parser.add_argument('-SGDR', action='store_true', help='Use Cosine Annealing with Restarts scheduler')
    
    # Environment and I/O
    parser.add_argument('-no_cuda', action='store_true', help='Disable CUDA even if available')
    parser.add_argument('-printevery', type=int, default=100, help='Log training status every N batches')
    parser.add_argument('-savename', type=str, default='gpt2_style_model', help='Base name for saving model checkpoints')    
    parser.add_argument('-loadname', type=str, default=None, help='Path to load pretrained model weights')    
    parser.add_argument('-dir_name', type=str, default='model_checkpoints', help='Directory name prefix to save checkpoints and logs')
    parser.add_argument('-train_file', type=str, default='wiki2.train.txt', help='Path to training data file') # Relative path
    parser.add_argument('-valid_file', type=str, default='wiki2.valid.txt', help='Path to validation data file') # Relative path
    parser.add_argument('-test_file', type=str, default='wiki2.test.txt', help='Path to test data file') # Relative path
                
    opt = parser.parse_args()
    
    # Setup Device
    use_cuda = not opt.no_cuda and torch.cuda.is_available()
    opt.device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {opt.device}")
    if use_cuda: print(f"CUDA Device Name: {torch.cuda.get_device_name(opt.device)}")

    # Setup Logging and Saving Directory
    # Use the directory name provided directly, without adding timestamp
    dir_name = opt.dir_name 
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"Created directory for logs and checkpoints: {dir_name}")
    else:
         print(f"Using existing directory: {dir_name}")
         
    source_name = sys.argv[0]
    try:
        shutil.copy(source_name, os.path.join(dir_name, os.path.basename(source_name)))
    except Exception as e:
        print(f"Warning: Could not copy script file: {e}")
        
    opt.dir_name = dir_name 
    opt.log_file = os.path.join(dir_name, "training_log.txt") 
    
    print("--- Options ---")
    print(vars(opt))
    print("-" * 30)
    
    # Load Tokenizer and Data
    print("Loading tokenizer and data...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    opt.trg_pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0 
    
    try:
        opt.train = read_corpus(opt.train_file, tokenizer)
        opt.valid = read_corpus(opt.valid_file, tokenizer)
        opt.test = read_corpus(opt.test_file, tokenizer)
    except SystemExit: # Catch exit from read_corpus on file not found
        sys.exit(1)
    except Exception as e: # Catch other potential errors
        print(f"An unexpected error occurred during data loading: {e}")
        sys.exit(1)
        
    print(f"Training tokens: {len(opt.train):,}")
    print(f"Validation tokens: {len(opt.valid):,}")
    print(f"Test tokens: {len(opt.test):,}")
    
    opt.vocab_size = tokenizer.vocab_size 
    print(f"Vocabulary Size: {opt.vocab_size}")
    
    # Build Model
    print("Building model...")
    model = get_model(opt, opt.vocab_size) 
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])        
    print(f'Total Trainable Parameters: {params:,}') 

    # Setup Optimizer and Scheduler
    print("Setting up optimizer...")
    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    
    opt.sched = None 
    if opt.SGDR:
        print("Using Cosine Annealing with Restarts scheduler.")
        num_sequences_train = (len(opt.train) - 1) // opt.seqlen
        num_batches_train = num_sequences_train // opt.batchsize
        if num_batches_train > 0:
             opt.sched = CosineWithRestarts(opt.optimizer, T_max=num_batches_train) 
        else:
             print("Warning: Not enough batches for SGDR scheduler.")

    # Start Training and collect perplexities
    train_perplexities, val_perplexities = train_model(model, opt)
    
    # Final Evaluation on Test Set
    print("\n--- Running Final Evaluation on Test Set ---")
    final_test_perplexity = test_model(model, opt.test, opt, epoch=-1) 
    
    # Plotting
    if train_perplexities and val_perplexities: # Check if lists are not empty
        plot_save_path = os.path.join(opt.dir_name, "perplexity_plot.png")
        plot_perplexities(train_perplexities, val_perplexities, final_test_perplexity, plot_save_path)
    else:
        print("Skipping plotting as no training/validation results were generated.")

    print("--- Script Finished ---")
        
if __name__ == "__main__":
    main()
