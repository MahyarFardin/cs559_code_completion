# CS559 Code Completion Project

(Add your names here guys)
22401352 - Mayasah Lami

Code completion model using transformer architecture trained on Python code from the py150 dataset.

## Project Overview

This project implements a transformer-based code completion model for Python. The model uses a causal self-attention mechanism to predict the next token in a sequence of code tokens.

## Project Structure

```
cs559_code_completion/
├── model.py                      # Transformer model architecture
├── preprocess.py                 # Data preprocessing script (tokenization)
├── create_completion_datasets.py # Create token/line-level completion datasets
├── train.py                      # Training script
├── inference.py                  # Inference script
├── requirements.txt              # Python dependencies
├── download_and_extract.sh       # Script to download py150 dataset
├── literals.json                 # Common string/number literals for tokenization
├── py150_files/                  # Dataset directory (Python source files)
│   ├── data/                     # Python source files
│   ├── python100k_train.txt      # Training file paths
│   ├── python50k_eval.txt        # Evaluation file paths
│   └── ...
├── token_completion/             # Preprocessed tokenized datasets
│   ├── train.txt                 # 95,000 training examples
│   ├── dev.txt                   # 5,000 development examples
│   └── test.txt                  # 50,000 test examples
└── completion_datasets/          # Code completion datasets (generated)
    ├── token_level/              # Next token prediction datasets
    └── line_level/               # Line completion datasets
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install individually:
   ```bash
   pip install torch tqdm numpy
   ```

2. **Download and extract the dataset:**
   ```bash
   bash download_and_extract.sh
   ```

3. **Preprocess the data:**
   ```bash
   python preprocess.py --base_dir py150_files --output_dir token_completion
   ```

4. **Create completion datasets:**
   ```bash
   python create_completion_datasets.py \
       --input_dir token_completion \
       --output_dir completion_datasets
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

The `create_completion_datasets.py` script creates task-specific datasets:
- **Token-level**: Next token prediction (context → target token)
- **Line-level**: Line completion (previous lines + prefix → suffix)

## Training

### Quick Start

1. **Create completion datasets** (if not already done):
   ```bash
   python create_completion_datasets.py \
       --input_dir token_completion \
       --output_dir completion_datasets
   ```

2. **Train token-level model**:
   ```bash
   python train.py \
       --task token \
       --dataset_dir completion_datasets \
       --tokenized_dir token_completion \
       --batch_size 32 \
       --num_epochs 10 \
       --max_length 256 \
       --device cuda
   ```

3. **Train line-level model**:
   ```bash
   python train.py \
       --task line \
       --dataset_dir completion_datasets \
       --tokenized_dir token_completion \
       --batch_size 32 \
       --num_epochs 10 \
       --max_length 256 \
       --device cuda
   ```

### Training Options

- `--task`: `token` or `line` (default: `token`)
- `--dataset_dir`: Directory with completion datasets (default: `completion_datasets`)
- `--tokenized_dir`: Directory with tokenized files for vocabulary (default: `token_completion`)
- `--batch_size`: Batch size (default: 32)
- `--num_epochs`: Number of training epochs (default: 10)
- `--max_length`: Maximum sequence length (default: 256)
- `--vocab_min_freq`: Minimum token frequency for vocabulary (default: 2)
- `--device`: `cuda` or `cpu` (auto-detected)

### Output Files

Training creates:
- `vocab.json` - Vocabulary mapping (tokens ↔ indices)
- `best_model_token_level.pt` or `best_model_line_level.pt` - Best model checkpoint

## Inference

### Token-Level Inference

Predict the next token:

```bash
python inference.py \
    --model_path best_model_token_level.pt \
    --vocab_path vocab.json \
    --task token \
    --context "from bootstrap import" \
    --top_k 5
```

### Line-Level Inference

Complete a line:

```bash
python inference.py \
    --model_path best_model_line_level.pt \
    --vocab_path vocab.json \
    --task line \
    --context "def hello(name):" \
    --device cuda
```


### Data Flow

1. **Tokenized files** (`token_completion/*.txt`)
   - Already tokenized with special tokens (`<EOL>`, `<STR_LIT>`, etc.)

2. **Completion datasets** (`completion_datasets/*/`)
   - JSONL format with context/target pairs
   - Created by `create_completion_datasets.py`

3. **Training** (`train.py`)
   - Builds vocabulary from tokenized files
   - Converts tokens to indices
   - Trains model with PyTorch DataLoader

4. **Inference** (`inference.py`)
   - Loads trained model and vocabulary
   - Converts input tokens to indices
   - Generates predictions

### Example Usage in Code

```python
from model import CodeCompletionTransformer, ModelConfig
from train import Vocabulary
import torch
import json

# Load vocabulary
vocab = Vocabulary()
with open('vocab.json', 'r') as f:
    vocab_data = json.load(f)
    vocab.token_to_idx = vocab_data['token_to_idx']
    vocab.idx_to_token = {int(k): v for k, v in vocab_data['idx_to_token'].items()}

# Create model
config = ModelConfig()
config.vocab_size = len(vocab.token_to_idx)
model = CodeCompletionTransformer(config)
model.load_state_dict(torch.load('best_model_token_level.pt'))
model.eval()

# Predict
context = "from bootstrap import".split()
context_ids = vocab.encode(context, max_length=256, pad=True)
input_ids = torch.tensor([context_ids])

with torch.no_grad():
    logits = model(input_ids)
    next_token_logits = logits[0, -1, :]
    predicted_token_idx = torch.argmax(next_token_logits).item()
    predicted_token = vocab.idx_to_token[predicted_token_idx]
    print(f"Next token: {predicted_token}")
```

## To Do

- **Mahyar**
  - Creating the model architecture
  - Testing the model architecture
- **Mayasah**
  - Data loading
  - Data preprocessing
