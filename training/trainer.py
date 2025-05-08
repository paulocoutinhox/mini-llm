import os

import torch
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from config.settings import DATA_PATH, LOG_DIR, MODEL_DIR
from utils.device import get_device


def prepare_dataset(tokenizer):
    """
    Prepare the dataset for training
    Args:
        tokenizer: The tokenizer to use
    Returns:
        tuple: (tokenized_dataset, data_collator)
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            "You must create a 'data.txt' file with your training data."
        )

    # Load dataset from plain text file
    dataset = load_dataset("text", data_files={"train": DATA_PATH})

    # Tokenization function for each text entry
    def tokenize_fn(example):
        # Add beginning-of-sequence (BOS) and end-of-sequence (EOS) tokens around the text
        # This follows the procedure mentioned for training GPT-Neo models
        return tokenizer(
            example["text"],  # The text to tokenize
            truncation=True,  # Truncate sequences longer than max_length
            padding="max_length",  # Pad sequences to max_length
            max_length=512,  # Maximum sequence length
            return_special_tokens_mask=True,  # Return mask for special tokens
            add_special_tokens=True,  # Add special tokens (BOS and EOS)
        )

    # Apply tokenization to all dataset examples
    # batched=True: Process multiple examples at once for better performance
    # remove_columns: Remove original columns after tokenization
    tokenized_dataset = dataset["train"].map(
        tokenize_fn, batched=True, remove_columns=dataset["train"].column_names
    )

    # Split dataset into train and validation sets (90% train, 10% validation)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1)

    # Data collator: prepares batches for language modeling
    # mlm=False: We use causal language modeling (predict next token) instead of masked language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal language modeling (predict next token)
    )

    return split_dataset, data_collator


def train_model(model, split_dataset, data_collator):
    """
    Train the model using the provided dataset
    Args:
        model: The model to train
        split_dataset: The tokenized dataset split into train and validation
        data_collator: The data collator for batching
    """
    from config.settings import USE_CPU

    # Detect device type for precision settings
    device = get_device()
    device_type = device.type

    # Check GPU memory if using CUDA
    low_memory_gpu = False
    batch_size = 4
    gradient_accumulation_steps = 4

    if device_type == "cuda" and not USE_CPU:
        try:
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"🧠 GPU Memory: {gpu_mem:.2f} GB")

            # Adjust settings for low memory GPUs
            if gpu_mem < 6:  # Less than 6GB is considered low memory
                low_memory_gpu = True
                batch_size = 1  # Reduce batch size dramatically
                gradient_accumulation_steps = 16  # Accumulate more gradients
                print("⚠️ Low memory GPU detected! Adjusting training parameters...")
        except:
            print("⚠️ Could not detect GPU memory. Using conservative settings.")
            low_memory_gpu = True
            batch_size = 2
            gradient_accumulation_steps = 8

    # Configure precision based on device capability
    precision_config = {}

    # CUDA GPUs support fp16
    if device_type == "cuda" and not USE_CPU:
        precision_config["fp16"] = True
        print("🚀 Using FP16 precision for training (CUDA)")
    # Some CPUs support bfloat16
    elif device_type == "cpu" and hasattr(torch, "bfloat16"):
        precision_config["bf16"] = True
        print("🚀 Using BF16 precision for training (CPU)")
    # MPS and other devices use default precision
    else:
        print("🚀 Using default precision for training")

    # Configure pin_memory based on device (disable for MPS or CPU)
    use_pin_memory = device_type == "cuda" and not USE_CPU

    # If CPU-only, adjust batch size for memory efficiency
    if USE_CPU or device_type == "cpu":
        print("💻 CPU training detected. Adjusting training parameters...")
        batch_size = 2
        gradient_accumulation_steps = 8

    # Training arguments configuration
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,  # Directory where model checkpoints will be saved
        overwrite_output_dir=True,  # Overwrite the content of the output directory
        num_train_epochs=20,  # Total number of training epochs to perform
        per_device_train_batch_size=batch_size,  # Batch size per GPU/TPU core/CPU for training
        per_device_eval_batch_size=batch_size,  # Batch size per GPU/TPU core/CPU for evaluation
        eval_strategy="steps",  # Evaluation is done (and logged) every eval_steps
        eval_steps=50,  # Number of update steps between two evaluations
        save_steps=100,  # Number of updates steps before two checkpoint saves
        save_total_limit=2,  # Limit the total amount of checkpoints
        logging_dir=LOG_DIR,  # Directory where the logs will be written
        logging_steps=10,  # Number of update steps between two logs
        learning_rate=2e-4,  # Initial learning rate for optimizer
        weight_decay=0.01,  # Weight decay to apply to all layers
        warmup_steps=500,  # Number of steps for the warmup phase
        gradient_accumulation_steps=gradient_accumulation_steps,  # Number of updates steps to accumulate before performing a backward pass
        load_best_model_at_end=True,  # Whether to load the best model found during training at the end of training
        metric_for_best_model="eval_loss",  # The metric to use to compare models
        greater_is_better=False,  # Whether the `metric_for_best_model` should be maximized or not
        dataloader_pin_memory=use_pin_memory,  # Whether to pin memory in DataLoader (disabled for MPS)
        optim="adamw_torch",  # Use the PyTorch implementation of AdamW (more memory efficient)
        **precision_config,  # Add precision configuration based on device
    )

    # Enable gradient checkpointing for low memory GPUs or CPU
    if (low_memory_gpu or USE_CPU) and hasattr(model, "gradient_checkpointing_enable"):
        try:
            print("💾 Enabling gradient checkpointing to save memory")
            model.gradient_checkpointing_enable()
            print("✅ Gradient checkpointing enabled successfully")
        except Exception as e:
            print(f"⚠️ Could not enable gradient checkpointing: {str(e)}")
            print("⚠️ Training will continue but might run out of memory")

    # Initialize the Trainer
    # Trainer handles the training loop, evaluation, and model saving
    trainer = Trainer(
        model=model,  # The model to train
        args=training_args,  # Training arguments
        train_dataset=split_dataset["train"],  # Training dataset
        eval_dataset=split_dataset["test"],  # Validation dataset
        data_collator=data_collator,  # Function to create batches
    )

    # Run training
    trainer.train()

    # Save the final model
    trainer.save_model(MODEL_DIR)
