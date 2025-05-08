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

    # Load dataset from plain text file
    dataset = load_dataset("text", data_files={"train": DATA_PATH})

    # Tokenization function for each text entry
    def tokenize_fn(example):
        return tokenizer(
            example["text"],  # The text to tokenize
            truncation=True,  # Truncate sequences longer than max_length
            padding="max_length",  # Pad sequences to max_length
            max_length=512,  # Maximum sequence length
            return_special_tokens_mask=True,  # Return mask for special tokens
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
    # Training arguments configuration
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,  # Directory where model checkpoints will be saved
        overwrite_output_dir=True,  # Overwrite the content of the output directory
        num_train_epochs=10,  # Total number of training epochs to perform
        per_device_train_batch_size=4,  # Batch size per GPU/TPU core/CPU for training
        per_device_eval_batch_size=4,  # Batch size per GPU/TPU core/CPU for evaluation
        evaluation_strategy="steps",  # Evaluation is done (and logged) every X steps
        eval_steps=100,  # Number of update steps between two evaluations
        save_steps=100,  # Number of updates steps before two checkpoint saves
        save_total_limit=2,  # Limit the total amount of checkpoints
        logging_dir=LOG_DIR,  # Directory where the logs will be written
        logging_steps=10,  # Number of update steps between two logs
        learning_rate=5e-5,  # Initial learning rate for optimizer
        weight_decay=0.01,  # Weight decay to apply to all layers
        warmup_steps=500,  # Number of steps for the warmup phase
        gradient_accumulation_steps=4,  # Number of updates steps to accumulate before performing a backward pass
        fp16=True,  # Whether to use fp16 16-bit (mixed) precision training
        load_best_model_at_end=True,  # Whether to load the best model found during training at the end of training
        metric_for_best_model="eval_loss",  # The metric to use to compare models
        greater_is_better=False,  # Whether the `metric_for_best_model` should be maximized or not
    )

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
