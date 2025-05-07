# Mini LLM (SLM)

A simple and lightweight language model implementation using Python and Transformers library.

## What is included

- Simple command-line interface for text generation
- Training capabilities with custom data
- Support for GPT-2 tokenizer
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

The model is trained using the text from `data.txt`. You can download the default training data (Portuguese Bible) from:

```
https://paulo-storage.s3.us-east-1.amazonaws.com/ai/slm/data.txt
```

To download the file, you can use one of these commands:
```bash
# Using curl
curl -o data.txt https://paulo-storage.s3.us-east-1.amazonaws.com/ai/slm/data.txt

# Or using wget
wget -O data.txt https://paulo-storage.s3.us-east-1.amazonaws.com/ai/slm/data.txt
```

You can also modify this file with any text you want to train the model on. After modifying `data.txt`, you need to retrain the model using the `--train` flag.

### Text Generation

To generate text based on a prompt:

```bash
python3 main.py "your prompt here"
```

For example:
```bash
python3 main.py "jesus disse"
```

### Training

To train the model with new data and overwrite the existing training:

```bash
python3 main.py "your prompt here" --train
```

For example:
```bash
python3 main.py "jesus disse" --train
```

Note: You must use the `--train` flag whenever you modify the `data.txt` file to ensure the model learns from the new content.

## Dependencies

The project uses the following main dependencies:
- torch
- transformers
- accelerate
- scipy
- datasets

## Project Structure

- `main.py`: Main application file
- `data.txt`: Training data file (modify this file with your custom text)
- `requirements.txt`: Project dependencies
- `.venv/`: Virtual environment directory

## License

[MIT](http://opensource.org/licenses/MIT)

Copyright (c) 2025, Paulo Coutinho
