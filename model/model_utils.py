from transformers import AutoModelForCausalLM, AutoTokenizer

from config.settings import MODEL_DIR, MODEL_NAME, TOKENIZER_CACHE_DIR


def load_tokenizer():
    """
    Load and configure the tokenizer
    Returns:
        AutoTokenizer: Configured tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=TOKENIZER_CACHE_DIR,
    )

    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_model(force_train=False, use_original=False):
    """
    Load the model, either from saved state or pre-trained
    Args:
        force_train (bool): Whether to force retraining the model
        use_original (bool): Whether to use the original model without fine-tuning
    Returns:
        AutoModelForCausalLM: The loaded model
    """
    if use_original:
        print("🌍 Using original pre-trained model (without fine-tuning).")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    elif not force_train:
        print("✅ Using saved fine-tuned model.")
        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
    else:
        print("🔁 Loading pre-trained model...")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    return model
