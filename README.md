# Mini LLM (SLM)

A simple and lightweight language model implementation using Python and Transformers library.

## What is included

- Simple command-line interface for text generation
- Training capabilities with custom data
- Support for multiple models (GPT-Neo, GPT-2, etc.)
- Configurable generation parameters
- Easy to use and modify
- Support for Python version from 3.9 or later

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

### Training

To train the model with new data:

```bash
python3 main.py --train
```

Note: You must use the `--train` flag whenever you modify the `temp/data.txt` file to ensure the model learns from the new content.

### Configuration

The model and generation parameters can be configured in `config/settings.py`:

```python
# Model configuration
MODEL_NAME = "EleutherAI/gpt-neo-2.7B"  # Change to any Hugging Face model
MODEL_CONFIG = {
    "max_length": 512,  # Maximum sequence length
    "temperature": 0.7,  # Controls randomness
    "top_p": 0.9,  # Nucleus sampling
    "top_k": 50,  # Top-k sampling
    # ... other parameters
}
```

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
│   └── device.py          # Device utilities
├── temp/                  # Temporary files directory
│   ├── data.txt           # Training data
│   ├── model/             # Saved model
│   ├── logs/              # Training logs
│   ├── tokenizer_cache/   # Tokenizer cache
│   └── huggingface/       # Hugging Face cache
├── main.py                # Main application file
└── requirements.txt       # Project dependencies
```

## Dependencies

The project uses the following main dependencies:
- torch
- transformers
- accelerate
- scipy
- datasets

## License

[MIT](http://opensource.org/licenses/MIT)

Copyright (c) 2025, Paulo Coutinho
