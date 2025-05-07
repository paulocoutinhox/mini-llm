import os

# Path to your training data
DATA_PATH = "data.txt"

# Temporary directory for model and logs
TMP_DIR = "./temp"

# Where to save/load the model
MODEL_DIR = os.path.join(TMP_DIR, "model")

# Where to store training logs
LOG_DIR = os.path.join(TMP_DIR, "logs")

# Cache dir for tokenizer
TOKENIZER_CACHE_DIR = os.path.join(TMP_DIR, "tokenizer_cache")

# Set Hugging Face cache directory
os.environ["TRANSFORMERS_CACHE"] = os.path.join(TMP_DIR, "huggingface")

# Create necessary directories
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TOKENIZER_CACHE_DIR, exist_ok=True)
