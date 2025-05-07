from transformers import GPT2LMHeadModel, GPT2Tokenizer

from config.settings import MODEL_DIR, TOKENIZER_CACHE_DIR


def load_tokenizer():
    """
    Load and configure the GPT-2 tokenizer
    Returns:
        GPT2Tokenizer: Configured tokenizer
    """
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=TOKENIZER_CACHE_DIR)
    tokenizer.pad_token = (
        tokenizer.eos_token
    )  # GPT2 doesn't have pad_token, so we reuse eos_token
    return tokenizer


def load_model(force_train=False):
    """
    Load the GPT-2 model, either from saved state or pre-trained
    Args:
        force_train (bool): Whether to force retraining the model
    Returns:
        GPT2LMHeadModel: The loaded model
    """
    if not force_train:
        print("✅ Using saved model.")
        model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
    else:
        print("🔁 Loading pre-trained model...")
        model = GPT2LMHeadModel.from_pretrained("gpt2")

    return model
