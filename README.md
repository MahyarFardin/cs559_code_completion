# CS559 Code Completion Project

(Add your names here guys)
22401352 - Mayasah Lami

Code completion model using transformer architecture trained on Python code from the py150 dataset.

## Project Overview

This project implements a transformer-based code completion model for Python. The model uses a causal self-attention mechanism to predict the next token in a sequence of code tokens.

## Project Structure

```
cs559_code_completion/
├── model.py                 # Transformer model architecture
├── preprocess.py            # Data preprocessing script (tokenization)
├── download_and_extract.sh  # Script to download py150 dataset
├── literals.json            # Common string/number literals for tokenization
├── py150_files/             # Dataset directory (Python source files)
│   ├── data/                # Python source files
│   ├── python100k_train.txt # Training file paths
│   ├── python50k_eval.txt   # Evaluation file paths
│   └── ...
└── token_completion/        # Preprocessed tokenized datasets
    ├── train.txt            # 95,000 training examples
    ├── dev.txt              # 5,000 development examples
    └── test.txt             # 50,000 test examples
```

## Setup

1. **Download and extract the dataset:**
   ```bash
   bash download_and_extract.sh
   ```

2. **Preprocess the data:**
   ```bash
   python preprocess.py --base_dir py150_files --output_dir token_completion
   ```

## Model Architecture

The model (`model.py`) implements:
- **CodeCompletionTransformer**: A GPT-style decoder-only transformer
- **Configuration**:
  - Vocabulary size: 32,000
  - Model dimension: 512
  - Number of layers: 6
  - Number of attention heads: 8
  - Feed-forward dimension: 2,048
  - Maximum sequence length: 256
  - Dropout: 0.1

## Data Preprocessing

The `preprocess.py` script:
- Tokenizes Python source files using Python's `tokenize` module
- Replaces string and number literals with special tokens (`<STR_LIT>`, `<NUM_LIT>`)
- Adds `<EOL>` markers for line breaks
- Splits data into train (95k), dev (5k), and test (50k) sets
- Outputs tokenized sequences with `<s>` and `</s>` markers

### To Do

- **Mahyar**
  - Creating the model architecture
  - Testing the model architecture
- **Mayasah**
  - Data loading
  - Data preprocessing
