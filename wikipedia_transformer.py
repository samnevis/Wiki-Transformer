# Import necessary libraries
import torch  # PyTorch for deep learning operations - provides tensors, autograd, and neural network modules
import torch.nn as nn  # Neural network modules - contains layers, loss functions, and other building blocks
import torch.optim as optim  # Optimization algorithms - provides optimizers like Adam, SGD, etc.
import argparse  # Command line argument parsing - allows users to specify parameters when running the script
import os  # Operating system operations - for file and directory handling
from pathlib import Path  # Path manipulation - provides an object-oriented interface to filesystem paths
import numpy as np  # Numerical operations - provides mathematical functions and array operations
from tqdm import tqdm  # Progress bar - displays a progress bar for loops
from datasets import load_dataset  # Hugging Face datasets - loads pre-defined datasets
import tiktoken  # OpenAI's tokenizer - converts text to token IDs using GPT-2's tokenization
from torch.utils.data import DataLoader, random_split  # Data loading utilities - creates batches and handles data iteration
import math  # Mathematical operations - provides mathematical functions

# Global configuration dictionary containing all model and training parameters
# This centralizes all hyperparameters in one place for easy modification
CONFIG = {
    # Model architecture parameters - define the structure and capacity of the model
    "vocab_size": 100277,  # Size of the vocabulary (GPT-2 tokenizer size) - number of unique tokens the model can process
    "d_model": 384,  # Dimension of the model (embedding size) - size of vectors used throughout the network
    "num_heads": 6,  # Number of attention heads - allows the model to focus on different parts of the input
    "num_layers": 6,  # Number of transformer layers - depth of the model, more layers = more capacity
    "dim_feedforward": 1536,  # Dimension of feedforward network - size of the hidden layer in the feedforward network
    "dropout": 0.1,  # Dropout rate for regularization - randomly drops neurons during training to prevent overfitting
    "max_seq_length": 128,  # Maximum sequence length for input/output - maximum number of tokens the model can process at once
    
    # Training hyperparameters - control how the model is trained
    "batch_size": 16,  # Number of samples per batch - smaller batch size for better generalization
    "gradient_accumulation_steps": 4,  # Number of steps to accumulate gradients - effective batch size = batch_size * gradient_accumulation_steps
    "max_iters": 60000,  # Maximum number of iterations - total number of training steps
    "eval_interval": 500,  # Evaluate and save model every N iterations
    "learning_rate": 1e-4,  # Learning rate for optimizer - reduced for more stable training
    "weight_decay": 0.01,  # L2 regularization factor - penalizes large weights to prevent overfitting
    "gradient_clip": 1.0,  # Maximum gradient norm - prevents exploding gradients by scaling them down if too large
    "checkpoint_dir": "checkpoints",  # Directory to save model checkpoints - where trained models are stored
    "use_amp": True,  # Whether to use automatic mixed precision - uses FP16 for faster training with less memory
    "warmup_iters": 2000,  # Number of warmup iterations for learning rate schedule
    
    # Text generation parameters - control how text is generated
    "temperature": 0.7,  # Sampling temperature (higher = more random) - controls randomness in generation
    "top_k": 40,  # Number of highest probability tokens to keep - limits token selection to top K most likely
    "top_p": 0.9,  # Cumulative probability threshold for nucleus sampling - limits token selection to tokens with cumulative probability < p
    "max_new_tokens": 100  # Maximum number of tokens to generate - limits the length of generated text
}

# Set device to GPU if available, otherwise CPU
# This allows the model to run on a GPU if one is available, which is much faster than CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_positional_encoding(max_seq_length, d_model):
    """
    Create positional encoding matrix for transformer.
    This helps the model understand token positions in the sequence.
    
    Positional encoding is crucial because transformers process all tokens in parallel
    and need a way to know the order of tokens. This function creates a unique encoding
    for each position using sine and cosine functions of different frequencies.
    
    Args:
        max_seq_length: Maximum sequence length - the longest sequence the model can handle
        d_model: Dimension of the model - must match the embedding dimension
        
    Returns:
        Positional encoding matrix of shape (1, max_seq_length, d_model)
        This matrix is added to the token embeddings to provide position information
    """
    # Create position indices - a tensor of integers from 0 to max_seq_length-1
    # These represent the position of each token in the sequence
    pos = torch.arange(max_seq_length).unsqueeze(1)  # Shape: (max_seq_length, 1)
    
    # Create division term for sinusoidal encoding
    # This creates different frequencies for the sine and cosine functions
    # The formula is: div_term = exp(i * -log(10000) / d_model)
    # where i ranges from 0 to d_model/2
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    
    # Initialize positional encoding matrix with zeros
    # This will be filled with sine and cosine values
    pe = torch.zeros(max_seq_length, d_model)
    
    # Apply sine to even indices and cosine to odd indices
    # This creates a unique pattern for each position
    # Even indices (0, 2, 4, ...) use sine
    pe[:, 0::2] = torch.sin(pos * div_term)
    # Odd indices (1, 3, 5, ...) use cosine
    pe[:, 1::2] = torch.cos(pos * div_term)
    
    # Add batch dimension and return
    # The final shape is (1, max_seq_length, d_model)
    return pe.unsqueeze(0)  # Add batch dimension

class SimpleTransformer(nn.Module):
    """
    A simplified transformer model for text generation.
    Implements the core transformer architecture with encoder-only design.
    
    This model is based on the transformer architecture from "Attention is All You Need"
    but simplified for text generation. It uses only the encoder part of the transformer
    and generates text autoregressively (one token at a time).
    
    The model consists of:
    1. Token embeddings
    2. Positional encoding
    3. Multiple transformer layers
    4. Output projection to vocabulary size
    """
    def __init__(self, config):
        """
        Initialize the transformer model.
        
        This method sets up all the components of the model:
        - Embedding layer for token representations
        - Positional encoding for sequence order
        - Transformer encoder layers for processing
        - Output layer for predicting the next token
        
        Args:
            config: Configuration dictionary containing model parameters
                  This includes vocab_size, d_model, num_heads, etc.
        """
        super().__init__()  # Initialize the parent nn.Module class
        self.config = config  # Store the configuration for later use
        
        # Token embedding layer
        # This converts token IDs (integers) to dense vectors (embeddings)
        # Shape: (vocab_size, d_model)
        self.embedding = nn.Embedding(config["vocab_size"], config["d_model"])
        
        # Register positional encoding as a buffer (not a parameter)
        # Buffers are tensors that are part of the model but not updated during training
        # This is more efficient than computing positional encoding on every forward pass
        self.register_buffer('pos_encoding', get_positional_encoding(config["max_seq_length"], config["d_model"]))
        
        # Layer normalization for embeddings
        # This normalizes the embeddings to have mean 0 and variance 1
        # This helps with training stability
        self.embed_norm = nn.LayerNorm(config["d_model"])
        
        # Dropout layer for regularization
        # This randomly drops neurons during training to prevent overfitting
        self.dropout = nn.Dropout(config["dropout"])
        
        # Create transformer encoder layer
        # This is the core of the transformer architecture
        # It processes the embeddings through self-attention and feedforward networks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["d_model"],  # Dimension of the model
            nhead=config["num_heads"],  # Number of attention heads
            dim_feedforward=config["dim_feedforward"],  # Dimension of feedforward network
            dropout=config["dropout"],  # Dropout rate
            batch_first=True,  # Input shape: (batch, seq, feature) instead of (seq, batch, feature)
            norm_first=True  # Apply normalization before attention (more stable training)
        )
        
        # Stack multiple transformer layers
        # This creates a deeper model with more capacity
        # Each layer processes the output of the previous layer
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config["num_layers"])
        
        # Output layer to project to vocabulary size
        # This converts the transformer output to logits for each token
        # Shape: (d_model, vocab_size)
        self.fc_out = nn.Linear(config["d_model"], config["vocab_size"])
        
        # Initialize model weights
        # This sets the initial values of the weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize model weights with uniform distribution.
        
        This method sets the initial values of the weights to small random values.
        This helps with training stability and convergence.
        
        The weights are initialized to a uniform distribution between -0.1 and 0.1.
        The biases are initialized to zero.
        """
        with torch.no_grad():  # Don't track gradients for initialization
            # Initialize embedding weights to uniform distribution
            self.embedding.weight.data.uniform_(-0.1, 0.1)
            # Initialize output layer weights to uniform distribution
            self.fc_out.weight.data.uniform_(-0.1, 0.1)
            # Initialize output layer biases to zero
            self.fc_out.bias.data.zero_()
    
    def forward(self, x, mask=None):
        """
        Forward pass through the model.
        
        This method processes the input through the model:
        1. Embed the input tokens
        2. Add positional encoding
        3. Pass through transformer layers
        4. Project to vocabulary size
        
        Args:
            x: Input tensor of shape (batch_size, seq_length)
               This contains token IDs (integers)
            mask: Optional attention mask
                  This prevents the model from attending to certain positions
                  If None, a causal mask is created (can't attend to future tokens)
            
        Returns:
            Output tensor of shape (batch_size, seq_length, vocab_size)
            This contains logits for each token in the vocabulary
        """
        # Get sequence length from input
        seq_len = x.size(1)
        # Move input to the device
        x = x.to(DEVICE)
        
        # Handle sequences longer than max_seq_length
        # If the sequence is longer than the positional encoding, we need to extend it
        if seq_len > self.pos_encoding.size(1):
            # Repeat the positional encoding to cover the longer sequence
            pos_encoding = self.pos_encoding.repeat(1, (seq_len // self.pos_encoding.size(1)) + 1, 1)
            # Trim to the exact sequence length
            pos_encoding = pos_encoding[:, :seq_len, :]
        else:
            # If the sequence is shorter, just use the first seq_len positions
            pos_encoding = self.pos_encoding[:, :seq_len, :]
        
        # Ensure positional encoding is on the correct device
        # This is important for multi-GPU training
        pos_encoding = pos_encoding.to(DEVICE)
        
        # Combine token embeddings with positional encoding
        # This gives the model both token identity and position information
        x = self.embedding(x) + pos_encoding
        
        # Apply layer normalization
        # This normalizes the embeddings to have mean 0 and variance 1
        x = self.embed_norm(x)
        
        # Apply dropout for regularization
        # This randomly drops neurons during training
        x = self.dropout(x)
        
        # Create causal mask if not provided (prevents attending to future tokens)
        # A causal mask ensures that each position can only attend to previous positions
        # This is crucial for autoregressive generation
        if mask is None:
            # Create an upper triangular matrix (including diagonal)
            # This creates a mask where each position can only attend to itself and previous positions
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(DEVICE)
        
        # Pass through transformer and output layer
        # The transformer processes the embeddings through self-attention and feedforward networks
        x = self.transformer(x, mask=mask)
        
        # Project to vocabulary size
        # This converts the transformer output to logits for each token
        return self.fc_out(x)
    
    def generate(self, prompt_ids, max_new_tokens, temperature=0.7, top_k=40, top_p=0.9):
        """
        Generate text from a prompt using various sampling strategies.
        
        This method generates text autoregressively (one token at a time):
        1. Start with the prompt
        2. Generate the next token based on the prompt
        3. Add the generated token to the prompt
        4. Repeat until max_new_tokens is reached or an EOS token is generated
        
        Various sampling strategies are used to control the randomness of generation:
        - Temperature: Controls the sharpness of the probability distribution
        - Top-k: Only consider the top k most likely tokens
        - Top-p (nucleus sampling): Only consider tokens with cumulative probability < p
        
        Args:
            prompt_ids: Input token IDs - the starting point for generation
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
                         Controls how much to scale the logits before softmax
            top_k: Number of highest probability tokens to keep
                   Only consider the top k most likely tokens
            top_p: Cumulative probability threshold for nucleus sampling
                   Only consider tokens with cumulative probability < p
            
        Returns:
            Generated token IDs - the complete sequence including prompt and generated tokens
        """
        # Set model to evaluation mode
        # This disables dropout and other training-specific behaviors
        self.eval()
        
        # Disable gradient computation for generation
        # This saves memory and speeds up generation
        with torch.no_grad():
            # Truncate prompt if it exceeds maximum sequence length
            # The model can only process sequences up to max_seq_length
            if prompt_ids.size(1) > self.config["max_seq_length"]:
                print(f"Warning: Prompt exceeds maximum sequence length ({self.config['max_seq_length']}). Truncating.")
                # Keep only the last max_seq_length tokens
                prompt_ids = prompt_ids[:, -self.config["max_seq_length"]:]
            
            # Move prompt to the same device as the embedding weights
            prompt_ids = prompt_ids.to(DEVICE)
            
            # Initialize generated sequence with the prompt
            generated = prompt_ids.clone()
            
            # Generate tokens one at a time
            for _ in range(max_new_tokens):
                # Get the last max_seq_length tokens as input
                # This ensures we don't exceed the maximum sequence length
                input_ids = generated[:, -self.config["max_seq_length"]:]
                
                # Get model predictions
                # This gives logits for each token in the vocabulary
                output = self(input_ids)
                
                # Get logits for the next token (last position)
                # Apply temperature to control randomness
                # Higher temperature = more random, lower = more deterministic
                next_token_logits = output[:, -1, :] / temperature
                
                # Apply top-k filtering
                # This keeps only the top k most likely tokens
                if top_k > 0:
                    # Find the k-th highest logit value
                    top_k_values, _ = torch.topk(next_token_logits, top_k)
                    # Create a mask for tokens to keep (those with logits >= k-th highest)
                    indices_to_remove = next_token_logits < top_k_values[..., -1, None]
                    # Set logits for tokens to remove to negative infinity
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                # This keeps only tokens with cumulative probability < p
                if top_p < 1.0:
                    # Sort logits in descending order
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    # Calculate cumulative probabilities
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    # Create a mask for tokens to remove (those with cumulative probability > p)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the mask to the right (keep at least one token)
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    # Convert back to original indices
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    # Set logits for tokens to remove to negative infinity
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from the filtered distribution
                # Convert logits to probabilities
                probs = torch.softmax(next_token_logits, dim=-1)
                # Sample one token from the distribution
                next_token = torch.multinomial(probs, num_samples=1)
                # Move to the same device as the generated sequence
                next_token = next_token.to(generated.device)
                
                # Append new token to generated sequence
                # This extends the sequence by one token
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if we generate an EOS token (0)
                # This is a common stopping criterion for text generation
                if next_token.item() == 0:
                    break
            
            # Return the complete generated sequence
            return generated

class TextDataset(torch.utils.data.Dataset):
    """
    Custom dataset class for text data.
    Handles tokenization and sequence creation for training.
    
    This class converts raw text into sequences of tokens that can be used for training.
    It handles:
    1. Tokenizing the text
    2. Creating sequences of fixed length
    3. Providing an interface for PyTorch's DataLoader
    
    The dataset creates overlapping sequences to maximize the use of the text data.
    """
    def __init__(self, text, tokenizer, max_seq_length):
        """
        Initialize the dataset.
        
        This method tokenizes the text and creates sequences of fixed length.
        It handles different types of text input (string, list, etc.).
        
        Args:
            text: Input text (string or list of strings)
                 This is the raw text to be tokenized
            tokenizer: Tokenizer for converting text to tokens
                      This is used to convert text to token IDs
            max_seq_length: Maximum sequence length
                           This determines the length of each sequence
        """
        # Store tokenizer and max sequence length
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        # Handle different types of text input
        # This makes the class more flexible
        if isinstance(text, str):
            # If text is a string, use it directly
            self.text = text
        elif isinstance(text, list):
            # If text is a list of strings, join them with spaces
            self.text = " ".join(text)
        else:
            # For any other type, convert to string
            self.text = str(text)
        
        # Tokenize the text
        # This converts the text to a list of token IDs
        tokens = tokenizer.encode(self.text)
        
        # Initialize list to store sequences
        self.sequences = []
        
        # Create overlapping sequences of max_seq_length
        # This maximizes the use of the text data
        # The step size is max_seq_length // 2 to create 50% overlap
        for i in range(0, len(tokens) - max_seq_length, max_seq_length // 2):
            # Extract a sequence of max_seq_length tokens
            sequence = tokens[i:i + max_seq_length]
            # Only add sequences of the full length
            if len(sequence) == max_seq_length:
                # Convert to tensor and add to sequences
                self.sequences.append(torch.tensor(sequence, dtype=torch.long))
    
    def __len__(self):
        """
        Return the number of sequences in the dataset.
        
        This is required by PyTorch's Dataset interface.
        
        Returns:
            Number of sequences
        """
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Return a sequence at the given index.
        
        This is required by PyTorch's Dataset interface.
        
        Args:
            idx: Index of the sequence to return
            
        Returns:
            Tuple of (input_ids, targets) tensors
            - input_ids: The input sequence (all tokens except the last)
            - targets: The target sequence (all tokens except the first)
        """
        sequence = self.sequences[idx]
        # For next token prediction, input is all tokens except the last
        input_ids = sequence[:-1]
        # Target is all tokens except the first
        targets = sequence[1:]
        return input_ids, targets

# Learning rate scheduler function
def get_lr(step, warmup_steps, total_steps):
    """
    Calculate learning rate with warmup and cosine decay.
    
    Args:
        step: Current step number
        warmup_steps: Number of steps for warmup
        total_steps: Total number of steps
        
    Returns:
        Learning rate for the current step
    """
    if step < warmup_steps:
        # Linear warmup
        return CONFIG["learning_rate"] * (step + 1) / warmup_steps
    else:
        # Cosine decay
        decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
        return CONFIG["learning_rate"] * 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

def save_checkpoint(model, optimizer, epoch, loss, config):
    """
    Save a model checkpoint.
    
    This function saves the model state, optimizer state, and training metadata
    to a file for later loading.
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        epoch: Current epoch number
        loss: Current loss value
        config: Configuration dictionary
    """
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Create checkpoint filename
    checkpoint_path = checkpoint_dir / "best_model.pt"
    
    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }, checkpoint_path)
    
    print(f"Best model saved with loss: {loss:.4f}")

def train_model():
    """Train the transformer model on the dataset"""
    # Setup tokenizer and update vocabulary size
    tokenizer = tiktoken.get_encoding("cl100k_base")
    CONFIG["vocab_size"] = tokenizer.n_vocab
    print(f"Using vocabulary size: {CONFIG['vocab_size']}")
    
    # Load and prepare the dataset
    print("Loading Wikipedia dataset...")
    dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True, trust_remote_code=True)
    print("Wikipedia dataset loaded successfully")
    
    # Process the dataset in chunks to avoid memory issues
    print("Processing Wikipedia text...")
    chunk_size = 2000000  # Process 2 million characters at a time
    all_text = ""
    
    # Get the first few examples to start with
    for i, example in enumerate(dataset.take(2000)):
        all_text += example["text"] + " "
        if len(all_text) >= chunk_size:
            break
    
    print(f"Processed {len(all_text)} characters from Wikipedia")
    
    # Create training dataset
    train_dataset = TextDataset(all_text, tokenizer, CONFIG["max_seq_length"])
    print(f"Created dataset with {len(train_dataset)} training examples")
    
    # Create data loader
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    
    # Initialize model, optimizer, and loss function
    model = SimpleTransformer(CONFIG).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    
    # Initialize mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=CONFIG["use_amp"])
    
    # Training loop
    print("Starting training...")
    model.train()
    
    # Track best loss for checkpointing
    best_loss = float('inf')
    
    # Initialize iteration counter
    iter_num = 0
    
    # Create an infinite data loader
    train_iter = iter(train_loader)
    
    # Training loop based on iterations
    while iter_num < CONFIG["max_iters"]:
        # Determine the learning rate for this iteration
        lr = get_lr(iter_num, CONFIG["warmup_iters"], CONFIG["max_iters"])
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Reset gradients at the beginning of each iteration
        optimizer.zero_grad()
        
        # Accumulate gradients over multiple steps
        for _ in range(CONFIG["gradient_accumulation_steps"]):
            try:
                # Get next batch
                input_ids, targets = next(train_iter)
            except StopIteration:
                # If we run out of data, create a new iterator
                train_iter = iter(train_loader)
                input_ids, targets = next(train_iter)
            
            # Move data to device
            input_ids = input_ids.to(DEVICE)
            targets = targets.to(DEVICE)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=CONFIG["use_amp"]):
                logits = model(input_ids)
                # Calculate loss (cross entropy)
                loss = nn.functional.cross_entropy(logits.view(-1, CONFIG["vocab_size"]), targets.view(-1))
                # Scale loss for gradient accumulation
                loss = loss / CONFIG["gradient_accumulation_steps"]
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
        
        # Clip gradients
        if CONFIG["gradient_clip"] > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip"])
        
        # Update weights
        scaler.step(optimizer)
        scaler.update()
        
        # Print progress
        if iter_num % 100 == 0:
            print(f"Iteration {iter_num}/{CONFIG['max_iters']}, Loss: {loss.item() * CONFIG['gradient_accumulation_steps']:.4f}, LR: {lr:.6f}")
        
        # Evaluate and save model periodically
        if iter_num > 0 and iter_num % CONFIG["eval_interval"] == 0:
            # Calculate average loss over the last eval_interval iterations
            avg_loss = loss.item() * CONFIG["gradient_accumulation_steps"]
            
            # Save checkpoint if loss improved
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(model, optimizer, iter_num, avg_loss, CONFIG)
                print(f"Checkpoint saved at iteration {iter_num} with loss: {avg_loss:.4f}")
        
        # Increment iteration counter
        iter_num += 1
    
    print("Training complete!")
    return model

def generate_text(prompt, max_length=100):
    """
    Generate text from a prompt using the trained model.
    
    This function:
    1. Loads the trained model from a checkpoint
    2. Tokenizes the prompt
    3. Generates text using the model
    4. Decodes the generated tokens back to text
    
    Args:
        prompt: Input text prompt - the starting point for generation
        max_length: Maximum number of tokens to generate
        
    Returns:
        Generated text - the complete text including prompt and generated part
    """
    print(f"Generating text from prompt: '{prompt}'")
    
    # Setup tokenizer and model
    # Create tokenizer for encoding and decoding text
    tokenizer = tiktoken.get_encoding("cl100k_base")
    # Create model and move to appropriate device
    model = SimpleTransformer(CONFIG).to(DEVICE)
    
    # Load best checkpoint if available
    # This loads the trained weights into the model
    checkpoint_dir = Path(CONFIG["checkpoint_dir"])
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        # Find the latest checkpoint
        latest_checkpoint = max(checkpoint_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
        print(f"Loading checkpoint: {latest_checkpoint}")
        
        # Load checkpoint
        checkpoint = torch.load(latest_checkpoint, map_location=DEVICE)
        # Load model weights
        model.load_state_dict(checkpoint['model'])
        # Ensure model is on the correct device
        model = model.to(DEVICE)
    else:
        print("No checkpoint found. Using untrained model.")
    
    # Set model to evaluation mode
    model.eval()
    
    # Tokenize and prepare prompt
    # Convert prompt to token IDs
    prompt_tokens = tokenizer.encode(prompt)
    # Check if prompt is too long
    if len(prompt_tokens) > CONFIG["max_seq_length"]:
        print(f"Warning: Prompt exceeds maximum sequence length ({CONFIG['max_seq_length']}). Truncating.")
        # Truncate prompt to maximum sequence length
        prompt_tokens = prompt_tokens[:CONFIG["max_seq_length"]]
    
    # Convert prompt tokens to tensor
    prompt_ids = torch.tensor(prompt_tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)
    
    # Generate text
    # Disable gradient computation for generation
    with torch.no_grad():
        # Generate tokens
        generated_ids = model.generate(
            prompt_ids, 
            max_new_tokens=max_length,
            temperature=CONFIG["temperature"],
            top_k=CONFIG["top_k"],
            top_p=CONFIG["top_p"]
        )
    
    # Decode generated tokens to text
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    print("\nGenerated text:")
    print(generated_text)
    return generated_text

# Main entry point
if __name__ == "__main__":
    # Parse command line arguments
    # This allows users to specify parameters when running the script
    parser = argparse.ArgumentParser(description="Simple Transformer Example")
    parser.add_argument("--mode", choices=["train", "generate"], default="generate", help="Mode to run the model in")
    parser.add_argument("--prompt", type=str, default="Hello", help="Prompt for text generation")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of generated text")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the appropriate mode
    if args.mode == "train":
        # Train the model
        train_model()
    else:
        # Generate text from prompt
        generate_text(args.prompt, args.max_length) 