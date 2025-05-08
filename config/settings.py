import os
import sys

# Settings that will be configured by main.py
USE_CPU = False
OPTIMIZE_MEMORY = False

# Temporary directory for model and logs
TMP_DIR = "./temp"

# Where to save/load the model
MODEL_DIR = os.path.join(TMP_DIR, "model")

# Where to store training logs
LOG_DIR = os.path.join(TMP_DIR, "logs")

# Cache dir for tokenizer
TOKENIZER_CACHE_DIR = os.path.join(TMP_DIR, "tokenizer_cache")

# Path to your training data
DATA_PATH = os.path.join(TMP_DIR, "data.txt")

# Whether to use local cache for Hugging Face models
# Set to True to store all HF files in the project directory
# Set to False to use the default global HF cache
USE_LOCAL_HF_CACHE = False

# Model configuration
# Get model name from environment variable or use default
MODEL_NAME = os.environ.get("MINI_LLM_MODEL", "EleutherAI/gpt-neo-2.7B")

# fmt: off
MODEL_CONFIG = {
    # Generation parameters
    "max_length": 512,  # Maximum sequence length for input and output
    "temperature": 0.7,  # Controls randomness: lower values make output more focused
    "top_p": 0.9,  # Nucleus sampling: keeps tokens with cumulative probability up to this value
    "top_k": 50,  # Top-k sampling: keeps only the k most likely tokens
    "num_return_sequences": 1,  # Number of different sequences to generate
    "do_sample": True,  # Whether to use sampling; use False for greedy decoding

    # Beam search parameters
    "num_beams": 5,  # Number of beams for beam search
    "early_stopping": True,  # Stop when all beam hypotheses reach the EOS token
    "no_repeat_ngram_size": 3,  # Size of n-grams to prevent from repeating
    "length_penalty": 1.0,  # Penalty for sequence length (1.0 = no penalty)
    "repetition_penalty": 1.2,  # Penalty for repeating tokens (1.0 = no penalty)

    # Additional parameters
    "min_length": 50,  # Minimum length of the sequence to be generated
    "bad_words_ids": None,  # List of token IDs that are not allowed to be generated
}
# fmt: on

# Set Hugging Face home directory if local cache is enabled
if USE_LOCAL_HF_CACHE:
    print("📁 Using local Hugging Face cache")
    os.environ["HF_HOME"] = os.path.join(TMP_DIR, "huggingface")
else:
    print("📁 Using global Hugging Face cache")

# Create necessary directories
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TOKENIZER_CACHE_DIR, exist_ok=True)
