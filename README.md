# Mini LLM (SLM)

A simple and lightweight language model implementation using Python and Transformers library. This project helps you fine-tune pre-trained language models (like GPT-Neo and GPT-2) with your own data, allowing you to adapt these models to your specific domain or language while leveraging their existing knowledge.

## What is included

- Simple command-line interface for text generation
- Fine-tuning capabilities with custom data
- Support for multiple pre-trained models (GPT-Neo, GPT-2, etc.)
- Configurable generation parameters
- Easy to use and modify
- Support for Python version from 3.9 or later
- Comparison between original and fine-tuned models

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mini-llm.git
cd mini-llm
```

2. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training Data

The model is trained using the text from `temp/data.txt`. You can download the default training data (Portuguese Bible) from:

```
https://paulo-storage.s3.us-east-1.amazonaws.com/ai/slm/data.txt
```

To download the file, you can use one of these commands:
```bash
# Create temp directory if it doesn't exist
mkdir -p temp

# Using curl
curl -o temp/data.txt https://paulo-storage.s3.us-east-1.amazonaws.com/ai/slm/data.txt

# Or using wget
wget -O temp/data.txt https://paulo-storage.s3.us-east-1.amazonaws.com/ai/slm/data.txt
```

You can also modify this file with any text you want to train the model on. After modifying `temp/data.txt`, you need to retrain the model using the `--train` flag.

### Text Generation

To generate text based on a prompt:

```bash
# Generate text using your fine-tuned model (saved in temp/model/)
python3 main.py --generate "your prompt here"

# Generate text using the original pre-trained model (not fine-tuned)
python3 main.py --generate "your prompt here" --use-original
```

By default (without the `--use-original` flag), the system will use your fine-tuned model from the `temp/model/` directory. If you haven't trained a model yet, or want to compare results with the original pre-trained model, use the `--use-original` flag.

For example:
```bash
# Using your fine-tuned model
python3 main.py --generate "jesus disse"

# Using the original pre-trained model
python3 main.py --generate "jesus disse" --use-original
```

#### Comparing Original vs Fine-tuned Models

To compare the output between the original model and your fine-tuned version:

```bash
# Generate with your fine-tuned model
python3 main.py --generate "your prompt here"

# Generate with the original pre-trained model
python3 main.py --generate "your prompt here" --use-original
```

This is useful for seeing how your training data has influenced the model's output.

### Training

To train the model with new data:

```bash
python3 main.py --train
```

For a completely fresh training (clears all cached data):

```bash
python3 main.py --train --clean
```

Note: You must use the `--train` flag whenever you modify the `temp/data.txt` file to ensure the model learns from the new content.

### Understanding Training Logs

During model training, you'll see various numbers and metrics displayed in the terminal. Think of these as the "vital signs" of your model's learning process. Even if you're new to machine learning, these metrics can help you understand if your model is learning well.

#### What You'll See in the Terminal

The output will look something like this:
```
{'loss': 2.5481, 'grad_norm': 2.6401376724243164, 'learning_rate': 1.538135593220339e-05, 'epoch': 7.01}
```

#### Simple Explanations of These Numbers

**The Important Training Numbers (shown frequently):**
- **loss**: Think of this as an "error score" - the lower, the better. It shows how well the model is learning from your data. For language models, values that start high (around 4-5) and gradually decrease to 2-3 indicate good learning progress.
- **epoch**: Shows how many times the model has gone through your entire dataset. For example, 7.01 means "just started the 7th round of training."

**Advanced Metrics (for those who want to know more):**
- **grad_norm**: Shows how dramatically the model is changing its internal knowledge with each update. Consistent values without sudden jumps are good.
- **learning_rate**: Controls how big of adjustments the model makes when learning. This number typically gets smaller as training progresses.

**Evaluation Metrics (shown occasionally):**
```
{'eval_loss': 3.1315, 'eval_runtime': 11.1366, 'eval_samples_per_second': 290.663, 'eval_steps_per_second': 72.733, 'epoch': 6.97}
```

- **eval_loss**: Similar to regular loss, but measured on data the model hasn't seen during training. This helps verify if the model is truly learning useful patterns rather than just memorizing.

#### What Success Looks Like

You're on the right track if:

- **The loss numbers are getting smaller** over time - this is the most important indicator that training is working.
- **The eval_loss is also decreasing** - if only the training loss decreases but eval_loss doesn't, the model might just be memorizing your data.
- **The loss eventually stabilizes** - at some point, the numbers will stop decreasing significantly, which usually indicates the model has learned what it can.

Don't worry too much about the exact values - what matters is the trend. If the loss is generally decreasing, your model is learning!

The system will automatically save the best version of your model (the one with the lowest eval_loss) to the `temp/model/` directory.

### Configuration

The model and generation parameters can be configured in `config/settings.py`:

```python
MODEL_CONFIG = {
    "max_length": 512,  # Maximum sequence length
    "temperature": 0.7,  # Controls randomness
    "top_p": 0.9,  # Nucleus sampling
    "top_k": 50,  # Top-k sampling
    # ... other parameters
}
```

#### Changing the Model

You can change the base model in two ways:

1. **Using environment variables**:
   ```bash
   # Linux/macOS
   export MINI_LLM_MODEL="pierreguillou/gpt2-small-portuguese"
   python3 main.py --generate "your prompt"

   # Windows (cmd)
   set MINI_LLM_MODEL=pierreguillou/gpt2-small-portuguese
   python main.py --generate "your prompt"

   # Windows (PowerShell)
   $env:MINI_LLM_MODEL="pierreguillou/gpt2-small-portuguese"
   python main.py --generate "your prompt"
   ```

2. **Editing the settings file**:
   Edit the `config/settings.py` file directly to change the default model:
   ```python
   MODEL_NAME = os.environ.get("MINI_LLM_MODEL", "EleutherAI/gpt-neo-2.7B")
   ```

The default model is `EleutherAI/gpt-neo-2.7B` if no environment variable is set.

#### Configuring Training Epochs

You can customize the number of training epochs using an environment variable:

```bash
# Linux/macOS
export MINI_LLM_EPOCHS=5
python3 main.py --train

# Windows (cmd)
set MINI_LLM_EPOCHS=5
python main.py --train

# Windows (PowerShell)
$env:MINI_LLM_EPOCHS=5
python main.py --train
```

The default is 20 epochs if not specified. For models already pre-trained in your target language, fewer epochs (1-5) may be sufficient.

## Project Structure

```
.
├── config/
│   └── settings.py        # Configuration settings
├── model/
│   ├── generation.py      # Text generation logic
│   └── model_utils.py     # Model loading utilities
├── training/
│   └── trainer.py         # Training logic
├── utils/
│   ├── device.py          # Device utilities
│   └── file.py            # File operations utilities
├── temp/                  # Temporary files directory
│   ├── data.txt           # Training data
│   ├── model/             # Saved model
│   ├── logs/              # Training logs
│   ├── tokenizer_cache/   # Tokenizer cache
│   └── huggingface/       # Hugging Face cache
├── main.py                # Main application file
├── backup_to_s3.py        # S3 backup utility script
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## Dependencies

The project relies on the following Python packages:

- **torch**: PyTorch deep learning framework that provides tensor computation and neural networks
- **transformers**: Hugging Face library that provides pre-trained models and utilities for natural language processing
- **accelerate**: Library that enables easy use of distributed training on multiple GPUs/TPUs
- **scipy**: Scientific computing library used for various mathematical operations
- **datasets**: Hugging Face library for managing and processing training datasets
- **black**: Code formatter used for maintaining consistent code style
- **packaging**: Utilities for parsing and comparing version numbers
- **psutil**: Cross-platform library for retrieving system information (memory, CPU)
- **boto3**: AWS SDK for Python, used by the backup script to interact with S3

## Available Models

The project supports a wide range of models from Hugging Face. Here are some popular options. Each model is listed with its Hugging Face identifier followed by its parameter count in parentheses (e.g., `model-name` (size)):

### Small Models (Mobile/CPU)
- `distilgpt2` (334M) – Fastest, works well on mobile devices
- `gpt2` (124M) – Good balance of size and performance
- `microsoft/phi-1_5` (1.3B) – Microsoft's efficient model
- `TinyLlama/TinyLlama-1.1B` (1.1B) – Efficient Llama variant
- `facebook/opt-125m` (125M) – Meta's efficient model
- `stabilityai/stablelm-2-1_6b` (1.6B) – StableLM 2.0
- `Qwen/Qwen-1_5B` (1.5B) – Alibaba's efficient model
- `google/gemma-2b` (2B) – Google's lightweight model
- `microsoft/phi-3-mini` (1.8B) – Microsoft's small Phi-3 model
- `mistralai/Mistral-2B-v0.1` (2B) – Mistral's small model

### Medium Models (GPU Recommended)
- `microsoft/phi-2` (2.7B) – Microsoft's Phi model
- `stabilityai/stablelm-2-3b` (3B) – StableLM 2.0
- `Qwen/Qwen-4B` (4B) – Alibaba's medium model
- `mistralai/Mistral-7B-v0.1` (7B) – High performance
- `meta-llama/Llama-2-7b` (7B) – Meta's Llama 2
- `tiiuae/falcon-7b` (7B) – TII's Falcon
- `google/gemma-7b` (7B) – Google's Gemma model
- `meta-llama/Llama-3-8b` (8B) – Meta's Llama 3
- `microsoft/phi-3` (3B) – Microsoft's next-gen Phi

### Large Models (High-End GPU Required)
- `mistralai/Mixtral-8x7B-v0.1` (47B) – Mixture of Experts
- `meta-llama/Llama-2-13b` (13B) – Meta's larger Llama 2
- `meta-llama/Llama-2-70b` (70B) – Meta's largest Llama 2
- `Qwen/Qwen-7B` (7B) – Alibaba's large model
- `Qwen/Qwen-14B` (14B) – Alibaba's larger model
- `stabilityai/stablelm-2-12b` (12B) – StableLM 2.0
- `meta-llama/Llama-3-70b` (70B) – Meta's largest Llama 3
- `Qwen/Qwen-20B` (20B) – Alibaba's extra large model
- `mistralai/Mixtral-v2` (55B) – Mistral's next-gen MoE

### Specialized Models
- `bigscience/bloom-560m` (560M) – Multilingual
- `bigscience/bloom-1b7` (1.7B) – Multilingual
- `THUDM/chatglm2-6b` (6B) – Chinese-optimized
- `fnlp/moss-moon-003-sft` (7B) – Chinese-optimized
- `Qwen/Qwen-7B-Chat` (7B) – Chat-optimized
- `stabilityai/stablelm-2-12b-chat` (12B) – Chat-optimized
- `mistralai/Mistral-8B-Instruct-v0.2` (8B) – Mistral's instruct model
- `meta-llama/Llama-3-8b-chat` (8B) – Meta's Llama 3 chat
- `google/gemma-7b-instruct` (7B) – Google's instruction model
- `microsoft/phi-2-vision` (2.7B) – Microsoft's vision-language model
- `Anthropic/Claude-Next-7B` (7B) – Anthropic's Claude Next

### Best Models for iPhone/Mobile
1. `distilgpt2` (334M) – Best for iPhone/mobile
2. `microsoft/phi-1_5` (1.3B) – Excellent quality-to-size ratio
3. `TinyLlama/TinyLlama-1.1B` (1.1B) – Optimized for mobile
4. `google/gemma-2b` (2B) – Good performance on mobile
5. `stabilityai/stablelm-2-1_6b` (1.6B) – Modern architecture

### Portuguese (Brazilian - PTBR) Models
- `pierreguillou/gpt2-small-portuguese` (124M) – GPT-2 trained specifically for Portuguese
- `unicamp-dl/gpt-neox-pt-small` (125M) – Brazilian model from Unicamp
- `pierreguillou/gpt2-small-portuguese-blog` (124M) – Optimized for blog content
- `neuralmind/bert-large-portuguese-cased` (354M) – BERT for Portuguese
- `jotape/bert-pt-br` (110M) – BERT model for Brazilian Portuguese
- `rufimelo/portuguese-gpt2-large` (774M) – Larger Portuguese model
- `nauvu/brazilian-legal-bert` (110M) – Specialized for Brazilian legal texts

Note: Model availability and compatibility may change. Check the model's documentation on Hugging Face for specific requirements and limitations.

## Training on Limited Hardware

When training on hardware with limited GPU memory (less than 2GB VRAM), you may encounter "CUDA out of memory" errors. The project provides two solutions:

### 1. Using CPU Instead of GPU

The most reliable solution is to use the `--use-cpu` flag which forces the model to run on CPU instead of GPU:

```bash
python main.py --train --use-cpu
```

This approach:
- Bypasses GPU memory limitations completely
- Uses system RAM instead of VRAM
- Works with any model size
- Is significantly slower than GPU training

### 2. Using Smaller Models

Instead of using the default 2.7B parameter model, you can switch to a smaller model that fits in your GPU's memory:

```bash
# Set a smaller model (124M parameters)
export MINI_LLM_MODEL="gpt2"

# Then train as usual
python main.py --train
```

Recommended models for low-memory GPUs (2-4GB VRAM):
- `gpt2` (124M)
- `distilgpt2` (82M)
- `facebook/opt-125m` (125M)
- `bigscience/bloom-560m` (560M)

## Backing Up to S3

The project includes a Python script (`backup_to_s3.py`) to easily back up your model, training data, and configuration to Amazon S3. This is useful for preserving your work and transferring it between different environments.

### Prerequisites

- Python 3
- Boto3 package installed (`pip install boto3`)
- AWS credentials configured
- Access to an S3 bucket

### Installing Dependencies

Install the necessary dependencies for the backup script:

```bash
pip install boto3
```

### Environment Variables

The backup script requires the following environment variables:

```bash
# AWS Credentials (Required)
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_DEFAULT_REGION="your_region"
export S3_BUCKET_NAME="your-bucket-name"

# Backup Configuration (Optional - shown with default values)
export S3_BUCKET_PATH="mini-llm-backup/"  # Path within the bucket (defaults to "mini-llm-backup/")
export FOLDER_TO_BACKUP="./temp"          # Local folder to backup (defaults to "./temp")
```

You only need to set these optional variables if you want to override the default values.

### Running the Backup

Once you've set the environment variables, run the backup script:

```bash
python3 backup_to_s3.py
```

The script will:
1. Create a timestamped tar.gz archive of your specified folder
2. Upload it to your S3 bucket with public read access
3. Clean up temporary files
4. Output a public URL that can be shared or used for downloads

> **Note**: The backup file will be publicly accessible via the generated URL. Make sure you don't include sensitive information if you're backing up to a public-facing bucket.

### Accessing Public Backups

After running the backup script, you'll receive a public URL that looks like:
```
https://your-bucket-name.s3.amazonaws.com/mini-llm-backup/backup_2023-05-08_12-34-56.tar.gz
```

You can share this URL with others or use it to download your backup from any machine without AWS credentials.

### Restoring from Backup

To restore your data from an S3 backup, you have two options:

#### Option 1: Using the public URL (no AWS credentials needed)
```bash
# Download the backup using the public URL
wget https://your-bucket-name.s3.amazonaws.com/mini-llm-backup/your-backup-file.tar.gz
# or
curl -O https://your-bucket-name.s3.amazonaws.com/mini-llm-backup/your-backup-file.tar.gz

# Extract the archive
rm -rf temp/
tar -xzvf your-backup-file.tar.gz -C .
```

#### Option 2: Using AWS CLI
```bash
# Download the backup archive using AWS CLI
aws s3 cp s3://your-bucket-name/mini-llm-backup/your-backup-file.tar.gz .

# Extract the archive
rm -rf temp/
tar -xzvf your-backup-file.tar.gz -C .
```

This will restore your model, training data, and other files to the specified destination path.

## License

[MIT](http://opensource.org/licenses/MIT)

Copyright (c) 2025, Paulo Coutinho
