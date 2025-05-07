import os
import sys

from config.settings import MODEL_DIR
from model.generation import generate_text
from model.model_utils import load_model, load_tokenizer
from training.trainer import prepare_dataset, train_model
from utils.device import get_device


def main():
    # === CLI Arguments ===
    if len(sys.argv) < 2:
        print('Usage: python3 main.py "your prompt here" [--train]')
        sys.exit(1)

    prompt = sys.argv[1]  # The user input prompt to generate text from
    force_train = "--train" in sys.argv  # Whether to force retraining the model

    # === Setup ===
    device = get_device()
    tokenizer = load_tokenizer()
    model = load_model(force_train=force_train)

    # === Train the model if needed ===
    if not os.path.exists(MODEL_DIR) or force_train:
        print("🔁 Training model...")
        tokenized_dataset, data_collator = prepare_dataset(tokenizer)
        train_model(model, tokenized_dataset, data_collator)

    # Resize token embeddings in case pad_token was added after loading the model
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # === Generate and output text ===
    print("\n📜 Prompt:", prompt)
    print("\n📘 Generated Text:\n")
    print(generate_text(model, tokenizer, prompt, device))


if __name__ == "__main__":
    main()
