import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import tiktoken
from torch.utils.data import DataLoader, random_split
import math

# Model configuration parameters
CONFIG = {
    # Model architecture
    "vocab_size": 100277,  # Size of vocabulary (GPT-2 tokenizer)
    "d_model": 384,        # Embedding dimension
    "num_heads": 6,        # Number of attention heads
    "num_layers": 6,       # Number of transformer layers
    "dim_feedforward": 1536,  # Hidden layer size in feedforward network
    "dropout": 0.1,        # Dropout rate for regularization
    "max_seq_length": 128, # Maximum sequence length
    
    # Training parameters
    "batch_size": 16,
    "gradient_accumulation_steps": 4,
    "max_iters": 60000,
    "eval_interval": 500,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "checkpoint_dir": "checkpoints",
    "use_amp": True,       # Automatic Mixed Precision
    "warmup_iters": 2000,
    
    # Generation parameters
    "temperature": 0.7,    # Controls randomness in generation
    "top_k": 40,          # Number of highest probability tokens to keep
    "top_p": 0.9,         # Nucleus sampling threshold
    "max_new_tokens": 100
}

# Use GPU if available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_positional_encoding(max_seq_length, d_model):
    # Create sinusoidal positional encodings for transformer
    pos = torch.arange(max_seq_length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    pe = torch.zeros(max_seq_length, d_model)
    pe[:, 0::2] = torch.sin(pos * div_term)  # Even indices: sine
    pe[:, 1::2] = torch.cos(pos * div_term)  # Odd indices: cosine
    return pe.unsqueeze(0)

class SimpleTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embedding layer
        self.embedding = nn.Embedding(config["vocab_size"], config["d_model"])
        
        # Positional encoding (fixed, not learned)
        self.register_buffer('pos_encoding', get_positional_encoding(config["max_seq_length"], config["d_model"]))
        
        # Normalization and dropout
        self.embed_norm = nn.LayerNorm(config["d_model"])
        self.dropout = nn.Dropout(config["dropout"])
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["d_model"],
            nhead=config["num_heads"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"],
            batch_first=True,
            norm_first=True
        )
        
        # Stack multiple transformer layers
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config["num_layers"])
        
        # Output projection to vocabulary size
        self.fc_out = nn.Linear(config["d_model"], config["vocab_size"])
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Initialize weights with small random values
        with torch.no_grad():
            self.embedding.weight.data.uniform_(-0.1, 0.1)
            self.fc_out.weight.data.uniform_(-0.1, 0.1)
            self.fc_out.bias.data.zero_()
    
    def forward(self, x, mask=None):
        seq_len = x.size(1)
        x = x.to(DEVICE)
        
        # Handle sequences longer than positional encoding
        if seq_len > self.pos_encoding.size(1):
            pos_encoding = self.pos_encoding.repeat(1, (seq_len // self.pos_encoding.size(1)) + 1, 1)
            pos_encoding = pos_encoding[:, :seq_len, :]
        else:
            pos_encoding = self.pos_encoding[:, :seq_len, :]
        
        pos_encoding = pos_encoding.to(DEVICE)
        
        # Combine token embeddings with positional encoding
        x = self.embedding(x) + pos_encoding
        x = self.embed_norm(x)
        x = self.dropout(x)
        
        # Create causal mask if not provided (prevents attending to future tokens)
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(DEVICE)
        
        # Process through transformer and project to vocabulary
        x = self.transformer(x, mask=mask)
        return self.fc_out(x)
    
    def generate(self, prompt_ids, max_new_tokens, temperature=0.7, top_k=40, top_p=0.9):
        self.eval()
        with torch.no_grad():
            # Handle long prompts
            if prompt_ids.size(1) > self.config["max_seq_length"]:
                print(f"Warning: Prompt exceeds maximum sequence length ({self.config['max_seq_length']}). Truncating.")
                prompt_ids = prompt_ids[:, -self.config["max_seq_length"]:]
            
            prompt_ids = prompt_ids.to(DEVICE)
            generated = prompt_ids.clone()
            
            # Generate tokens one at a time
            for _ in range(max_new_tokens):
                # Get last max_seq_length tokens as input
                input_ids = generated[:, -self.config["max_seq_length"]:]
                output = self(input_ids)
                
                # Get logits for next token and apply temperature
                next_token_logits = output[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_values, _ = torch.topk(next_token_logits, top_k)
                    indices_to_remove = next_token_logits < top_k_values[..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply nucleus sampling (top-p)
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                next_token = next_token.to(generated.device)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if EOS token is generated
                if next_token.item() == 0:
                    break
            
            return generated

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, text, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        # Handle different input types
        if isinstance(text, str):
            self.text = text
        elif isinstance(text, list):
            self.text = " ".join(text)
        else:
            self.text = str(text)
        
        # Tokenize text
        tokens = tokenizer.encode(self.text)
        self.sequences = []
        
        # Create overlapping sequences for training
        for i in range(0, len(tokens) - max_seq_length, max_seq_length // 2):
            sequence = tokens[i:i + max_seq_length]
            if len(sequence) == max_seq_length:
                self.sequences.append(torch.tensor(sequence, dtype=torch.long))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # For next token prediction: input is all tokens except last, target is all tokens except first
        sequence = self.sequences[idx]
        input_ids = sequence[:-1]
        targets = sequence[1:]
        return input_ids, targets

def get_lr(step, warmup_steps, total_steps):
    # Learning rate schedule with warmup and cosine decay
    if step < warmup_steps:
        return CONFIG["learning_rate"] * (step + 1) / warmup_steps
    else:
        decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
        return CONFIG["learning_rate"] * 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

def save_checkpoint(model, optimizer, epoch, loss, config):
    # Save model checkpoint
    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_model.pt"
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }, checkpoint_path)
    print(f"Best model saved with loss: {loss:.4f}")

def train_model():
    # Setup tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")
    CONFIG["vocab_size"] = tokenizer.n_vocab
    print(f"Using vocabulary size: {CONFIG['vocab_size']}")
    
    # Load Wikipedia dataset
    print("Loading Wikipedia dataset...")
    dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True, trust_remote_code=True)
    print("Wikipedia dataset loaded successfully")
    
    # Process dataset in chunks
    print("Processing Wikipedia text...")
    chunk_size = 2000000
    all_text = ""
    for i, example in enumerate(dataset.take(2000)):
        all_text += example["text"] + " "
        if len(all_text) >= chunk_size:
            break
    
    print(f"Processed {len(all_text)} characters from Wikipedia")
    
    # Create dataset and dataloader
    train_dataset = TextDataset(all_text, tokenizer, CONFIG["max_seq_length"])
    print(f"Created dataset with {len(train_dataset)} training examples")
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    
    # Initialize model and optimizer
    model = SimpleTransformer(CONFIG).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=CONFIG["use_amp"])
    
    # Training loop
    print("Starting training...")
    model.train()
    best_loss = float('inf')
    iter_num = 0
    train_iter = iter(train_loader)
    
    while iter_num < CONFIG["max_iters"]:
        # Update learning rate
        lr = get_lr(iter_num, CONFIG["warmup_iters"], CONFIG["max_iters"])
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        optimizer.zero_grad()
        
        # Gradient accumulation
        for _ in range(CONFIG["gradient_accumulation_steps"]):
            try:
                input_ids, targets = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                input_ids, targets = next(train_iter)
            
            input_ids = input_ids.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=CONFIG["use_amp"]):
                logits = model(input_ids)
                loss = nn.functional.cross_entropy(logits.view(-1, CONFIG["vocab_size"]), targets.view(-1))
                loss = loss / CONFIG["gradient_accumulation_steps"]
            
            scaler.scale(loss).backward()
        
        # Gradient clipping
        if CONFIG["gradient_clip"] > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip"])
        
        # Update weights
        scaler.step(optimizer)
        scaler.update()
        
        # Log progress
        if iter_num % 100 == 0:
            print(f"Iteration {iter_num}/{CONFIG['max_iters']}, Loss: {loss.item() * CONFIG['gradient_accumulation_steps']:.4f}, LR: {lr:.6f}")
        
        # Save checkpoint if loss improved
        if iter_num > 0 and iter_num % CONFIG["eval_interval"] == 0:
            avg_loss = loss.item() * CONFIG["gradient_accumulation_steps"]
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(model, optimizer, iter_num, avg_loss, CONFIG)
                print(f"Checkpoint saved at iteration {iter_num} with loss: {avg_loss:.4f}")
        
        iter_num += 1
    
    print("Training complete!")
    return model

def generate_text(prompt, max_length=100):
    print(f"Generating text from prompt: '{prompt}'")
    
    # Setup model and tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")
    model = SimpleTransformer(CONFIG).to(DEVICE)
    
    # Load best checkpoint if available
    checkpoint_dir = Path(CONFIG["checkpoint_dir"])
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        latest_checkpoint = max(checkpoint_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
        print(f"Loading checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=DEVICE)
        model.load_state_dict(checkpoint['model'])
        model = model.to(DEVICE)
    else:
        print("No checkpoint found. Using untrained model.")
    
    model.eval()
    
    # Tokenize prompt
    prompt_tokens = tokenizer.encode(prompt)
    if len(prompt_tokens) > CONFIG["max_seq_length"]:
        print(f"Warning: Prompt exceeds maximum sequence length ({CONFIG['max_seq_length']}). Truncating.")
        prompt_tokens = prompt_tokens[:CONFIG["max_seq_length"]]
    
    prompt_ids = torch.tensor(prompt_tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)
    
    # Generate text
    with torch.no_grad():
        generated_ids = model.generate(
            prompt_ids, 
            max_new_tokens=max_length,
            temperature=CONFIG["temperature"],
            top_k=CONFIG["top_k"],
            top_p=CONFIG["top_p"]
        )
    
    # Decode and return generated text
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    print("\nGenerated text:")
    print(generated_text)
    return generated_text

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Simple Transformer Example")
    parser.add_argument("--mode", choices=["train", "generate"], default="generate", help="Mode to run the model in")
    parser.add_argument("--prompt", type=str, default="Hello", help="Prompt for text generation")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of generated text")
    args = parser.parse_args()
    
    # Run in specified mode
    if args.mode == "train":
        train_model()
    else:
        generate_text(args.prompt, args.max_length) 