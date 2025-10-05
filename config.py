"""
Transformer Configuration for Neural Machine Translation

Author: Naresh AI Nexus
Project: Transformer from Scratch for Language Translation
GitHub: https://github.com/nareshAiNexus/transformer-using-numpy

This configuration file contains all the hyperparameters and settings
for training the Transformer model for language translation tasks.
"""

from pathlib import Path

def get_config():
    """
    Get configuration dictionary for Transformer translation model.
    
    Returns:
        dict: Configuration parameters for neural machine translation.
              
    Translation-specific parameters:
        - lang_src: Source language code (e.g., 'en' for English)
        - lang_tgt: Target language code (e.g., 'it' for Italian)
        - hf_dataset: HuggingFace dataset for parallel corpus
        
    Model architecture parameters follow the original "Attention is All You Need" paper.
    """
    return {
        # === Training Parameters ===
        "batch_size": 8,           # Batch size for training (adjust based on GPU memory)
        "num_epochs": 10,          # Number of training epochs
        "lr": 10**-4,              # Learning rate (Adam optimizer)
        "train_split_ratio": 0.9,  # Ratio for train/validation split
        
        # === Model Architecture ===
        "seq_len": 350,            # Maximum sequence length for both source and target
        "d_model": 512,            # Model dimension (embedding size)
        "d_ff": 2048,              # Feed-forward network dimension
        "num_block": 6,            # Number of encoder/decoder blocks (original paper: 6)
        "num_head": 8,             # Number of attention heads (original paper: 8)
        "dropout": 0.1,            # Dropout rate for regularization
        
        # === Translation-Specific Settings ===
        "lang_src": "en",          # Source language (English)
        "lang_tgt": "it",          # Target language (Italian)
        "hf_dataset": "opus_books", # HuggingFace dataset for parallel corpus
        
        # === File Paths and Naming ===
        "model_folder": "weights",                    # Directory to save model weights
        "model_basename": "translator_model_",       # Base name for saved models
        "tokenizer_file": "tokenizer_{0}.json",      # Tokenizer file pattern
        "preload": None,                              # Pre-trained model to load (None for fresh training)
        "experiment_name": "runs/translation_model"   # TensorBoard experiment name
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f'{model_basename}{epoch}.pt'
    return str(Path('.')/model_folder/model_filename)
