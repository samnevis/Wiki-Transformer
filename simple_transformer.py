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
from torch.utils.data import DataLoader

# Configuration
CONFIG = {
    # Model parameters
    "vocab_size": 50257,
    "d_model": 128,
    "num_heads": 2,
    "num_layers": 2,
    "dim_feedforward": 256,
    "dropout": 0.1,
    "max_seq_length": 64,
    
    # Training parameters
    "batch_size": 32,
    "num_epochs": 1,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "checkpoint_dir": "checkpoints",
    "use_amp": True,
    
    # Generation parameters
    "temperature": 0.6,
    "top_k": 20,
    "top_p": 0.8,
    "max_new_tokens": 100
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_positional_encoding(max_seq_length, d_model):
    """Create positional encoding for transformer."""
    pos = torch.arange(max_seq_length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    pe = torch.zeros(max_seq_length, d_model)
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    return pe.unsqueeze(0)

class SimpleTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding and positional encoding
        self.embedding = nn.Embedding(config["vocab_size"], config["d_model"])
        self.register_buffer('pos_encoding', get_positional_encoding(config["max_seq_length"], config["d_model"]))
        self.embed_norm = nn.LayerNorm(config["d_model"])
        self.dropout = nn.Dropout(config["dropout"])
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["d_model"],
            nhead=config["num_heads"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"],
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config["num_layers"])
        
        # Output layer
        self.fc_out = nn.Linear(config["d_model"], config["vocab_size"])
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        with torch.no_grad():
            self.embedding.weight.data.uniform_(-0.1, 0.1)
            self.fc_out.weight.data.uniform_(-0.1, 0.1)
            self.fc_out.bias.data.zero_()
    
    def resize_embedding(self, new_vocab_size):
        """Resize embedding and output layers for different vocabulary size."""
        old_embedding = self.embedding
        old_fc_out = self.fc_out
        
        self.embedding = nn.Embedding(new_vocab_size, old_embedding.embedding_dim)
        self.fc_out = nn.Linear(old_fc_out.in_features, new_vocab_size)
        
        with torch.no_grad():
            if new_vocab_size > old_embedding.num_embeddings:
                # Expand vocabulary
                self.embedding.weight.data[:old_embedding.num_embeddings] = old_embedding.weight.data
                self.embedding.weight.data[old_embedding.num_embeddings:].uniform_(-0.1, 0.1)
                self.fc_out.weight.data[:old_fc_out.out_features] = old_fc_out.weight.data
                self.fc_out.weight.data[old_fc_out.out_features:].uniform_(-0.1, 0.1)
                self.fc_out.bias.data[:old_fc_out.out_features] = old_fc_out.bias.data
                self.fc_out.bias.data[old_fc_out.out_features:].zero_()
            else:
                # Shrink vocabulary
                self.embedding.weight.data = old_embedding.weight.data[:new_vocab_size]
                self.fc_out.weight.data = old_fc_out.weight.data[:new_vocab_size]
                self.fc_out.bias.data = old_fc_out.bias.data[:new_vocab_size]
    
    def forward(self, x, mask=None):
        """Forward pass through the model."""
        seq_len = x.size(1)
        x = x.to(self.embedding.weight.device)
        
        # Handle positional encoding
        if seq_len > self.pos_encoding.size(1):
            pos_encoding = self.pos_encoding.repeat(1, (seq_len // self.pos_encoding.size(1)) + 1, 1)
            pos_encoding = pos_encoding[:, :seq_len, :]
        else:
            pos_encoding = self.pos_encoding[:, :seq_len, :]
        
        # Ensure positional encoding is on the same device as the input
        pos_encoding = pos_encoding.to(x.device)
        
        # Embedding and positional encoding
        x = self.embedding(x) + pos_encoding
        x = self.embed_norm(x)
        x = self.dropout(x)
        
        # Create causal mask if not provided
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        
        # Transformer and output
        x = self.transformer(x, mask=mask)
        return self.fc_out(x)
    
    def generate(self, prompt_ids, max_new_tokens, temperature=0.7, top_k=40, top_p=0.9):
        """Generate text from a prompt."""
        self.eval()
        with torch.no_grad():
            # Truncate prompt if needed
            if prompt_ids.size(1) > self.config["max_seq_length"]:
                print(f"Warning: Prompt exceeds maximum sequence length ({self.config['max_seq_length']}). Truncating.")
                prompt_ids = prompt_ids[:, -self.config["max_seq_length"]:]
            
            prompt_ids = prompt_ids.to(self.embedding.weight.device)
            generated = prompt_ids.clone()
            
            for _ in range(max_new_tokens):
                # Get the last max_seq_length tokens
                input_ids = generated[:, -self.config["max_seq_length"]:]
                
                # Get model output
                output = self(input_ids)
                next_token_logits = output[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from the filtered distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                next_token = next_token.to(generated.device)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if we generate an EOS token (0)
                if next_token.item() == 0:
                    break
            
            return generated

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, text, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        # Handle different types of text input
        if isinstance(text, str):
            self.text = text
        elif isinstance(text, list):
            self.text = " ".join(text)
        else:
            self.text = str(text)
        
        # Tokenize the text
        tokens = tokenizer.encode(self.text)
        self.sequences = []
        
        # Create sequences of max_seq_length
        for i in range(0, len(tokens) - max_seq_length, max_seq_length // 2):
            sequence = tokens[i:i + max_seq_length]
            if len(sequence) == max_seq_length:
                self.sequences.append(torch.tensor(sequence, dtype=torch.long))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]

def train_model():
    """Train the transformer model."""
    print("Starting training...")
    try:
        # Setup tokenizer and dataset
        tokenizer = tiktoken.get_encoding("cl100k_base")
        CONFIG["vocab_size"] = tokenizer.n_vocab
        print(f"Using vocabulary size: {CONFIG['vocab_size']}")
        
        # Load dataset
        dataset = load_dataset("tiny_shakespeare", trust_remote_code=True)
        print("Dataset structure:")
        for split in dataset.keys():
            print(f"  {split}: {len(dataset[split])} examples")
        
        # Prepare text data
        train_text = dataset["train"]["text"]
        if isinstance(train_text, list):
            train_text = " ".join(train_text)
        
        # Create datasets
        text_dataset = TextDataset(train_text, tokenizer, CONFIG["max_seq_length"])
        train_size = int(0.9 * len(text_dataset))
        train_dataset, val_dataset = torch.utils.data.random_split(
            text_dataset, [train_size, len(text_dataset) - train_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"])
        
        # Initialize model and optimizer
        model = SimpleTransformer(CONFIG).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
        criterion = nn.CrossEntropyLoss()
        os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
        scaler = torch.cuda.amp.GradScaler() if CONFIG["use_amp"] else None
        
        # Training loop
        best_val_loss = float('inf')
        for epoch in range(CONFIG["num_epochs"]):
            # Training phase
            model.train()
            total_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
                batch = batch.to(DEVICE)
                input_ids = batch[:, :-1]
                target_ids = batch[:, 1:]
                
                # Forward pass with mixed precision if enabled
                if CONFIG["use_amp"]:
                    with torch.cuda.amp.autocast():
                        outputs = model(input_ids)
                        loss = criterion(outputs.reshape(-1, outputs.size(-1)), target_ids.reshape(-1))
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(input_ids)
                    loss = criterion(outputs.reshape(-1, outputs.size(-1)), target_ids.reshape(-1))
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip"])
                    optimizer.step()
                
                total_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Evaluating"):
                    batch = batch.to(DEVICE)
                    input_ids = batch[:, :-1]
                    target_ids = batch[:, 1:]
                    outputs = model(input_ids)
                    val_loss += criterion(outputs.reshape(-1, outputs.size(-1)), target_ids.reshape(-1)).item()
            
            val_loss /= len(val_loader)
            print(f"Validation loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss
                }, os.path.join(CONFIG["checkpoint_dir"], "best_model.pt"))
        
        print("Training completed!")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

def generate_text(prompt, max_length=100):
    """Generate text from a prompt."""
    print(f"Generating text from prompt: '{prompt}'")
    
    # Setup tokenizer and model
    tokenizer = tiktoken.get_encoding("cl100k_base")
    model = SimpleTransformer(CONFIG).to(DEVICE)
    
    # Load checkpoint if available
    checkpoint_dir = Path(CONFIG["checkpoint_dir"])
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        latest_checkpoint = max(checkpoint_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
        print(f"Loading checkpoint: {latest_checkpoint}")
        
        checkpoint = torch.load(latest_checkpoint, map_location=DEVICE)
        if checkpoint['model']['embedding.weight'].shape[0] != CONFIG["vocab_size"]:
            model.resize_embedding(checkpoint['model']['embedding.weight'].shape[0])
        model.load_state_dict(checkpoint['model'])
        # Ensure model is on the correct device
        model = model.to(DEVICE)
    else:
        print("No checkpoint found. Using untrained model.")
    
    model.eval()
    
    # Tokenize and prepare prompt
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
    
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    print("\nGenerated text:")
    print(generated_text)
    return generated_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Transformer Example")
    parser.add_argument("--mode", choices=["train", "generate"], default="generate", help="Mode to run the model in")
    parser.add_argument("--prompt", type=str, default="Hello", help="Prompt for text generation")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of generated text")
    
    args = parser.parse_args()
    if args.mode == "train":
        train_model()
    else:
        generate_text(args.prompt, args.max_length) 