import os

from transformers import AutoModelForCausalLM, AutoTokenizer

from config.settings import MODEL_DIR, MODEL_NAME, TOKENIZER_CACHE_DIR


def load_tokenizer():
    """
    Load and configure the tokenizer
    Returns:
        AutoTokenizer: Configured tokenizer
    """
    # First try to load from the model directory if it exists (for consistency with fine-tuned model)
    if os.path.exists(os.path.join(MODEL_DIR, "tokenizer_config.json")):
        print(f"✅ Loading tokenizer from saved fine-tuned model")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    else:
        # Fall back to the specified model name if no saved tokenizer exists
        print(f"🔄 Loading tokenizer from pre-trained model: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=TOKENIZER_CACHE_DIR,
        )

    # Configure special tokens
    tokenizer.pad_token = tokenizer.eos_token

    # Ensure beginning-of-sequence (BOS) and end-of-sequence (EOS) tokens are configured
    # This is important for separating texts during training, as mentioned
    # in the documentation for models like pierreguillou/gpt2-small-portuguese
    if tokenizer.bos_token is None:
        print("⚠️ BOS token not found, using EOS token as BOS")
        tokenizer.bos_token = tokenizer.eos_token

    print(
        f"🔤 Special tokens configured: BOS={tokenizer.bos_token}, EOS={tokenizer.eos_token}, PAD={tokenizer.pad_token}"
    )

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
    # Determine loading configuration based on memory availability
    config_kwargs = {
        "ignore_mismatched_sizes": True,
        "trust_remote_code": True,
        "torch_dtype": "auto",
    }

    if use_original:
        print(
            f"🌍 Using original pre-trained model: {MODEL_NAME} (without fine-tuning)"
        )
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **config_kwargs)
    elif not force_train:
        print(f"✅ Using saved fine-tuned model")
        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, **config_kwargs)
    else:
        print(f"🔁 Loading pre-trained model: {MODEL_NAME}")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **config_kwargs)

    # Display model parameters count
    model_parameters = sum(p.numel() for p in model.parameters())
    print(f"📊 Model parameters: {model_parameters/1000000:.2f}M")

    return model
