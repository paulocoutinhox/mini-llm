import torch

from config.settings import MODEL_CONFIG


def generate_text(model, tokenizer, prompt: str, device, max_length: int = 100):
    """
    Generate text based on the given prompt
    Args:
        model: The model to use for generation
        tokenizer: The tokenizer to use
        prompt (str): The input prompt
        device: The device to run generation on
        max_length (int): Maximum length of generated text
    Returns:
        str: Generated text
    """
    # Encode the prompt to input IDs and attention mask
    inputs = tokenizer(
        prompt,  # The input text to be tokenized
        return_tensors="pt",  # Return PyTorch tensors instead of lists
        padding=True,  # Pad sequences to the longest sequence in the batch
    )

    # Move all input tensors to the specified device (CPU/GPU)
    # This is necessary because the model and inputs must be on the same device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Set the model to evaluation mode
    # This disables training-specific behaviors like dropout and batch normalization
    # It's important for inference to get consistent results
    model.eval()

    with torch.no_grad():
        # fmt: off
        output = model.generate(
            # Input configuration
            input_ids=inputs["input_ids"],  # Encoded input tokens
            attention_mask=inputs["attention_mask"],  # Mask for real/padding tokens

            # Generation parameters
            max_length=max_length,  # Maximum length of generated sequence
            min_length=MODEL_CONFIG["min_length"],  # Minimum length of generated sequence
            num_return_sequences=MODEL_CONFIG["num_return_sequences"],  # Number of sequences to generate
            pad_token_id=tokenizer.eos_token_id,  # Use eos token as padding

            # Sampling parameters
            temperature=MODEL_CONFIG["temperature"],  # Controls randomness
            top_p=MODEL_CONFIG["top_p"],  # Nucleus sampling
            top_k=MODEL_CONFIG["top_k"],  # Top-k sampling
            do_sample=MODEL_CONFIG["do_sample"],  # Enable/disable sampling

            # Beam search parameters
            num_beams=MODEL_CONFIG["num_beams"],  # Number of beams for beam search
            early_stopping=MODEL_CONFIG["early_stopping"],  # Stop when all beams reach EOS
            length_penalty=MODEL_CONFIG["length_penalty"],  # Penalty for sequence length

            # Repetition control
            repetition_penalty=MODEL_CONFIG["repetition_penalty"],  # Penalty for repeating tokens
            no_repeat_ngram_size=MODEL_CONFIG["no_repeat_ngram_size"],  # Prevent repeating n-grams

            # Additional parameters
            bad_words_ids=MODEL_CONFIG["bad_words_ids"],  # Words to avoid generating
        )
        # fmt: on

    return tokenizer.decode(output[0], skip_special_tokens=True)
