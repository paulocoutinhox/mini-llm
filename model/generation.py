import torch


def generate_text(model, tokenizer, prompt: str, device, max_length: int = 50):
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
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],  # Encoded input tokens
            attention_mask=inputs[
                "attention_mask"
            ],  # Tells model which tokens are real/padding
            max_length=max_length,  # Max length of generated sequence
            num_return_sequences=1,  # Number of completions to return
            pad_token_id=tokenizer.eos_token_id,  # Use eos token as padding (for GPT-2)
            temperature=0.9,  # Controls randomness (lower = more predictable)
            top_p=0.95,  # Nucleus sampling: keeps top X% probability mass
            do_sample=True,  # Enables sampling (instead of greedy decoding)
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)
