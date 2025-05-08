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
python3 main.py --generate "your prompt here"
```

For example:
```bash
python3 main.py --generate "jesus disse"
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
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## Dependencies

The project uses the following main dependencies:
- torch
- transformers
- accelerate
- scipy
- datasets

## Available Models

The project supports a wide range of models from Hugging Face. Here are some popular options:

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

## License

[MIT](http://opensource.org/licenses/MIT)

Copyright (c) 2025, Paulo Coutinho
