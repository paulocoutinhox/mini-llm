import argparse
import sys

from model.generation import generate_text
from model.model_utils import load_model, load_tokenizer
from training.trainer import prepare_dataset, train_model
from utils.device import get_device, show_device_info
from utils.file import clean_temp_dir


def parse_args():
    """
    Parse command line arguments
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Train or generate text with the language model"
    )

    # Create a mutually exclusive group for train/generate modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--train",
        action="store_true",
        help="Train the model (ignores prompt if provided)",
    )
    mode_group.add_argument(
        "--generate", action="store_true", help="Generate text from a prompt"
    )

    # Add option to show device info as a standalone option
    parser.add_argument(
        "--show-device-info",
        action="store_true",
        help="Show detailed device information",
    )

    # Add prompt argument
    parser.add_argument(
        "prompt",
        nargs="?",
        help="The prompt to generate text from (required for --generate mode)",
    )

    # Add clean option for training
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean temporary directory before training (removes all cached data)",
    )

    # Add option to use original model without fine-tuning
    parser.add_argument(
        "--use-original",
        action="store_true",
        help="Use the original model without fine-tuning (for comparison)",
    )

    # Add resource control options
    parser.add_argument(
        "--optimize-memory",
        action="store_true",
        help="Apply device-specific memory optimizations",
    )

    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    # Setup device
    device = get_device(optimize_memory=args.optimize_memory)

    # Show device info if requested
    if args.show_device_info:
        show_device_info()

        # If only showing device info, exit early
        if not args.train and not args.generate:
            return

    # Verify we have a required mode if not just showing device info
    if not args.train and not args.generate and not args.show_device_info:
        print("Error: One of --train, --generate, or --show-device-info is required")
        sys.exit(1)

    # Load model and tokenizer if we're training or generating
    if args.train or args.generate:
        tokenizer = load_tokenizer()
        model = load_model(force_train=args.train, use_original=args.use_original)

        # Resize token embeddings to match the tokenizer's vocabulary size
        # This is necessary because:
        # 1. When we add special tokens (like pad_token), the tokenizer's vocabulary size increases
        # 2. The model's embedding layer needs to match this new vocabulary size
        # 3. Without resizing, we would get an error when trying to use tokens that weren't in the original model
        # 4. This ensures the model can handle all tokens in the tokenizer's vocabulary
        model.resize_token_embeddings(len(tokenizer))

        # Move the model to the specified device (CPU/GPU)
        # This is necessary because:
        # 1. The model and its parameters need to be on the same device as the input data
        # 2. GPU (CUDA/MPS) is much faster for deep learning operations
        # 3. The device is automatically selected based on availability:
        #    - CUDA for NVIDIA GPUs
        #    - MPS for Apple Silicon GPUs
        #    - CPU as fallback
        # 4. All model parameters and buffers are moved to the device
        model.to(device)

        # Train or Generate
        if args.train:
            if args.clean:
                clean_temp_dir()

            print("🔁 Training model...")
            tokenized_dataset, data_collator = prepare_dataset(tokenizer)
            train_model(model, tokenized_dataset, data_collator)
            print("✅ Training completed!")

        elif args.generate:
            if not args.prompt:
                print("Error: Prompt is required for generation mode")
                sys.exit(1)

            model_type = "Original" if args.use_original else "Fine-tuned"
            print(f"\n📜 Prompt: {args.prompt}")
            print(f"\n📘 Generated Text ({model_type} model):\n")
            print(generate_text(model, tokenizer, args.prompt, device))


if __name__ == "__main__":
    main()
