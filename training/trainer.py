import os

from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from config.settings import DATA_PATH, LOG_DIR, MODEL_DIR


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

    # Load dataset from plain text
    dataset = load_dataset("text", data_files={"train": DATA_PATH})

    # Tokenization function for each text entry
    def tokenize_fn(example):
        return tokenizer(
            example["text"], truncation=True, padding="max_length", max_length=128
        )

    # Apply tokenization to all dataset examples
    tokenized_dataset = dataset["train"].map(tokenize_fn, batched=True)

    # Data collator: prepares batches for language modeling (causal, no masking)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # MLM = masked language model. We use causal instead (predict next token)
    )

    return tokenized_dataset, data_collator


def train_model(model, tokenized_dataset, data_collator):
    """
    Train the model using the provided dataset
    Args:
        model: The model to train
        tokenized_dataset: The tokenized dataset
        data_collator: The data collator for batching
    """
    # Training arguments configuration
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,  # Directory to save model checkpoints
        overwrite_output_dir=True,  # Overwrite if model already exists
        num_train_epochs=5,  # Number of training epochs
        per_device_train_batch_size=2,  # Batch size per device (CPU/GPU)
        save_steps=200,  # Save model every 200 steps
        save_total_limit=1,  # Only keep the last checkpoint
        logging_dir=LOG_DIR,  # Directory for training logs
        logging_steps=20,  # How often to log loss/metrics
        prediction_loss_only=True,  # Do not compute eval metrics, only loss
    )

    # Trainer object from Hugging Face
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Run training
    trainer.train()
    trainer.save_model(MODEL_DIR)
