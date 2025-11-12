"""
Complete LLM Fine-tuning Script using Hugging Face Transformers
This script fine-tunes GPT-2 on a custom dataset for text generation.

Author: Luis Alejandro Santana Valdez
Date: 2025
"""

import os
import torch
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2Config,
    get_linear_schedule_with_warmup
)
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

class TrainingConfig:
    """Configuration class for training hyperparameters"""
    
    # Model configuration
    MODEL_NAME = "gpt2-medium"  # Options: gpt2, gpt2-medium, gpt2-large, gpt2-xl
    
    # Training hyperparameters
    BATCH_SIZE = 4  # Reduce if out of memory
    GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = BATCH_SIZE * this
    NUM_EPOCHS = 3
    LEARNING_RATE = 5e-5
    WARMUP_STEPS = 500
    MAX_GRAD_NORM = 1.0  # Gradient clipping threshold
    WEIGHT_DECAY = 0.01
    
    # Data configuration
    MAX_LENGTH = 512  # Maximum sequence length
    TRAIN_SPLIT = 0.9  # 90% train, 10% validation
    
    # System configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_MIXED_PRECISION = True  # Use FP16 training
    NUM_WORKERS = 2  # DataLoader workers
    
    # Checkpoint and logging
    OUTPUT_DIR = "./trained_model"
    CHECKPOINT_DIR = "./checkpoints"
    LOG_INTERVAL = 10  # Log every N batches
    SAVE_INTERVAL = 500  # Save checkpoint every N steps
    EVAL_INTERVAL = 100  # Evaluate every N steps
    
    # Random seed for reproducibility
    SEED = 42


# ============================================================================
# DATASET CLASS
# ============================================================================

class TextDataset(Dataset):
    """
    Custom Dataset class for text data.
    Handles tokenization and prepares data for language modeling.
    """
    
    def __init__(self, texts, tokenizer, max_length=512):
        """
        Args:
            texts (list): List of text strings to train on
            tokenizer: HuggingFace tokenizer
            max_length (int): Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        print(f"Tokenizing {len(texts)} examples...")
        for text in tqdm(texts):
            # Tokenize the text
            tokenized = self.tokenizer.encode_plus(
                text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            self.examples.append({
                'input_ids': tokenized['input_ids'].squeeze(),
                'attention_mask': tokenized['attention_mask'].squeeze()
            })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_sample_data():
    """
    Load sample training data.
    Replace this with your own data loading logic.
    
    Returns:
        list: List of text strings for training
    """
    # Example: Simple sample data
    # In practice, load from files, databases, or APIs
    sample_texts = [
        "The quick brown fox jumps over the lazy dog. This is a sample sentence for training.",
        "Machine learning is a subset of artificial intelligence that focuses on data and algorithms.",
        "Natural language processing enables computers to understand and generate human language.",
        "Deep learning uses neural networks with multiple layers to learn complex patterns.",
        "Transfer learning allows models to apply knowledge from one task to another.",
        # Add more training examples here
    ] * 100  # Repeat to create more training data
    
    return sample_texts


def load_data_from_file(file_path):
    """
    Load training data from a text file.
    
    Args:
        file_path (str): Path to the text file
        
    Returns:
        list: List of text strings
    """
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                texts.append(line)
    return texts


def split_data(texts, train_split=0.9):
    """
    Split data into training and validation sets.
    
    Args:
        texts (list): List of text strings
        train_split (float): Proportion of data for training
        
    Returns:
        tuple: (train_texts, val_texts)
    """
    np.random.seed(TrainingConfig.SEED)
    np.random.shuffle(texts)
    
    split_idx = int(len(texts) * train_split)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    print(f"Training examples: {len(train_texts)}")
    print(f"Validation examples: {len(val_texts)}")
    
    return train_texts, val_texts


# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

def initialize_model(model_name, device):
    """
    Initialize the model and tokenizer.
    
    Args:
        model_name (str): Name of the pre-trained model
        device (str): Device to load model on
        
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # GPT-2 doesn't have a padding token by default, so we set it
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Move model to device
    model.to(device)
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model, tokenizer


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, scaler, device, config):
    """
    Train for one epoch.
    
    Args:
        model: The model to train
        dataloader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        scaler: Gradient scaler for mixed precision
        device: Device to train on
        config: Training configuration
        
    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for step, batch in enumerate(progress_bar):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Forward pass with mixed precision
        if config.USE_MIXED_PRECISION:
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids  # For language modeling, labels = inputs
                )
                loss = outputs.loss
                # Normalize loss by gradient accumulation steps
                loss = loss / config.GRADIENT_ACCUMULATION_STEPS
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            loss = outputs.loss / config.GRADIENT_ACCUMULATION_STEPS
        
        # Backward pass
        if config.USE_MIXED_PRECISION:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights after accumulating gradients
        if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
            # Gradient clipping
            if config.USE_MIXED_PRECISION:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            
            # Optimizer step
            if config.USE_MIXED_PRECISION:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
        
        # Track loss
        total_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item() * config.GRADIENT_ACCUMULATION_STEPS,
            'lr': scheduler.get_last_lr()[0]
        })
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader, device, config):
    """
    Evaluate the model on validation data.
    
    Args:
        model: The model to evaluate
        dataloader: Validation data loader
        device: Device to evaluate on
        config: Training configuration
        
    Returns:
        float: Average validation loss
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            if config.USE_MIXED_PRECISION:
                with autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids
                    )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
            
            total_loss += outputs.loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def save_checkpoint(model, tokenizer, optimizer, scheduler, epoch, step, loss, config):
    """
    Save a training checkpoint.
    
    Args:
        model: The model to save
        tokenizer: The tokenizer
        optimizer: The optimizer
        scheduler: The scheduler
        epoch: Current epoch
        step: Current step
        loss: Current loss
        config: Training configuration
    """
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(
        config.CHECKPOINT_DIR,
        f"checkpoint_epoch{epoch}_step{step}.pt"
    )
    
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    
    print(f"Checkpoint saved: {checkpoint_path}")


def save_model(model, tokenizer, config):
    """
    Save the final trained model.
    
    Args:
        model: The trained model
        tokenizer: The tokenizer
        config: Training configuration
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)
    
    print(f"Model saved to: {config.OUTPUT_DIR}")


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train():
    """
    Main training function that orchestrates the entire training process.
    """
    print("=" * 80)
    print("LLM Fine-tuning Script")
    print("=" * 80)
    
    # Set random seeds for reproducibility
    torch.manual_seed(TrainingConfig.SEED)
    np.random.seed(TrainingConfig.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TrainingConfig.SEED)
    
    # Initialize configuration
    config = TrainingConfig()
    
    print(f"\nDevice: {config.DEVICE}")
    print(f"Mixed Precision: {config.USE_MIXED_PRECISION}")
    
    # Load and prepare data
    print("\n" + "=" * 80)
    print("Loading Data")
    print("=" * 80)
    
    # Load your data here
    # Option 1: Use sample data
    # all_texts = load_sample_data()
    
    # Option 2: Load from file
    all_texts = load_data_from_file("codigo_civil_rd_training.txt")
    
    # Split data
    train_texts, val_texts = split_data(all_texts, config.TRAIN_SPLIT)
    
    # Initialize model and tokenizer
    print("\n" + "=" * 80)
    print("Initializing Model")
    print("=" * 80)
    model, tokenizer = initialize_model(config.MODEL_NAME, config.DEVICE)
    
    # Create datasets
    print("\n" + "=" * 80)
    print("Creating Datasets")
    print("=" * 80)
    train_dataset = TextDataset(train_texts, tokenizer, config.MAX_LENGTH)
    val_dataset = TextDataset(val_texts, tokenizer, config.MAX_LENGTH)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == "cuda" else False
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == "cuda" else False
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Calculate total training steps
    total_steps = len(train_dataloader) * config.NUM_EPOCHS // config.GRADIENT_ACCUMULATION_STEPS
    
    # Setup learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler() if config.USE_MIXED_PRECISION else None
    
    # Training loop
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)
    print(f"Total epochs: {config.NUM_EPOCHS}")
    print(f"Total steps: {total_steps}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Gradient accumulation steps: {config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"Effective batch size: {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS}")
    
    best_val_loss = float('inf')
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")
        print('=' * 80)
        
        # Train for one epoch
        train_loss = train_epoch(
            model, train_dataloader, optimizer, scheduler, scaler, config.DEVICE, config
        )
        
        # Evaluate on validation set
        val_loss = evaluate(model, val_dataloader, config.DEVICE, config)
        
        # Log results
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        
        # Save training history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['learning_rates'].append(scheduler.get_last_lr()[0])
        
        # Save checkpoint if best model
        if val_loss < best_val_loss:
            print(f"  New best model! (Previous best: {best_val_loss:.4f})")
            best_val_loss = val_loss
            save_checkpoint(
                model, tokenizer, optimizer, scheduler,
                epoch + 1, 0, val_loss, config
            )
        
        # Save regular checkpoint
        if (epoch + 1) % 1 == 0:  # Save every epoch
            save_checkpoint(
                model, tokenizer, optimizer, scheduler,
                epoch + 1, 0, val_loss, config
            )
    
    # Save final model
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    save_model(model, tokenizer, config)
    
    # Save training history
    history_path = os.path.join(config.OUTPUT_DIR, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"Training history saved to: {history_path}")
    
    return model, tokenizer


# ============================================================================
# INFERENCE FUNCTION
# ============================================================================

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7, device="cuda"):
    """
    Generate text using the trained model.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        prompt (str): Input prompt
        max_length (int): Maximum length of generated text
        temperature (float): Sampling temperature
        device (str): Device to run on
        
    Returns:
        str: Generated text
    """
    model.eval()
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and return
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Train the model
    trained_model, trained_tokenizer = train()
    
    # Test the trained model with some prompts
    print("\n" + "=" * 80)
    print("Testing Trained Model")
    print("=" * 80)
    
    test_prompts = [
        "Artículo 55. Se hará una declaración",
        "El arrendador está obligado por la naturaleza",
        "Artículo 334. El reconocimiento de un hijo natural"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        generated = generate_text(
            trained_model,
            trained_tokenizer,
            prompt,
            max_length=100,
            temperature=0.7,
            device=TrainingConfig.DEVICE
        )
        print(f"Generated: {generated}")
        print("-" * 80)