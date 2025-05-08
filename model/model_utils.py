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
