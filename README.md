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
├── train.py                      # Original training script (token + line level)
├── train_v2.py                   # Recommended training script (cleaner, with gradient accumulation)
├── evaluate.py                   # Evaluation script
├── inference.py                  # Inference script
├── PARAMETER_GUIDE.md            # Guide for choosing training parameters
├── EXPERIMENT_GUIDE.md           # Guide for systematic hyperparameter experiments
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
└── runs/                         # Training run directories (generated)
    └── run_*/                    # Individual run directories named by parameters
        ├── best_model_*.pt       # Model checkpoint
        ├── vocab.json            # Vocabulary file
        ├── training_params.json  # Training parameters
        └── test_results.json    # Evaluation results (after running evaluate.py)
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
   
   **Note:** For GPU support, install PyTorch with CUDA:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   Then install other dependencies:
   ```bash
   pip install tqdm numpy
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

### Recommended: Using train_v2.py

`train_v2.py` is the recommended training script with improved features:
- Gradient accumulation support (reduces GPU memory usage)
- Configurable learning rate and weight decay
- Validation accuracy reporting
- Better code organization
- Reproducibility (random seed support)

### Quick Start

1. **Create completion datasets** (if not already done):
   ```bash
   python create_completion_datasets.py \
       --input_dir token_completion \
       --output_dir completion_datasets
   ```

2. **Train token-level model** (recommended):
   ```bash
   python train_v2.py \
       --task token \
       --vocab_min_freq 25 \
       --batch_size 32 \
       --num_epochs 15 \
       --max_length 256 \
       --max_train_examples 200000 \
       --device cuda
   ```

3. **Train line-level model**:
   ```bash
   python train_v2.py \
       --task line \
       --vocab_min_freq 25 \
       --batch_size 32 \
       --num_epochs 15 \
       --max_length 256 \
       --max_train_examples 200000 \
       --device cuda
   ```

4. **With gradient accumulation** (if GPU memory is limited):
   ```bash
   python train_v2.py \
       --task token \
       --batch_size 16 \
       --accumulation_steps 4 \
       --num_epochs 15 \
       --max_length 256 \
       --device cuda
   ```
   This simulates batch_size=64 (16×4) with less GPU memory.

### Training Options (train_v2.py)

**Data arguments:**
- `--task`: `token` or `line` (default: `token`)
- `--dataset_dir`: Directory with completion datasets (default: `completion_datasets`)
- `--tokenized_dir`: Directory with tokenized files for vocabulary (default: `token_completion`)

**Training hyperparameters:**
- `--batch_size`: Batch size (default: 32)
- `--num_epochs`: Number of training epochs (default: 10)
- `--max_length`: Maximum sequence length (default: 256)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--weight_decay`: Weight decay / L2 regularization (default: 0.01)
- `--accumulation_steps`: Gradient accumulation steps (default: 1, use >1 to reduce GPU memory)

**Vocabulary arguments:**
- `--vocab_min_freq`: Minimum token frequency for vocabulary (default: 10, higher = smaller vocab)
- `--vocab_sample_lines`: Sample N lines for vocabulary building (default: 50000)

**Data loading:**
- `--max_train_examples`: Limit number of training examples (None = all)
- `--max_val_examples`: Limit number of validation examples (default: 10000)
- `--lazy_load`: Use lazy loading for datasets (saves memory)
- `--num_workers`: Number of data loading workers (default: 4)

**System:**
- `--device`: `cuda` or `cpu` (auto-detected)
- `--seed`: Random seed for reproducibility (default: 42)

### Alternative: Using train.py

The original `train.py` script is also available and supports both token and line-level training. See `train.py --help` for options.

### Parameter Selection Guide

For guidance on choosing parameters (vocab size, batch size, etc.), see **`PARAMETER_GUIDE.md`**.

For systematic hyperparameter experiments (learning rate, dropout, etc.), see **`EXPERIMENT_GUIDE.md`**.

### Output Files

Training creates a run directory in `runs/` named after the training parameters:
- **train.py format**: `run_{task}_bs{batch_size}_ep{epochs}_len{max_length}_vocab{min_freq}_{timestamp}`
- **train_v2.py format**: `run_{task}_v2_bs{batch_size}_ep{epochs}_len{max_length}_vocab{min_freq}_{timestamp}`
- Example: `runs/run_token_v2_bs32_ep15_len256_vocab25_20240101_120000/`

Each run directory contains:
- `best_model.pt` (train_v2.py) or `best_model_token_level.pt`/`best_model_line_level.pt` (train.py) - Best model checkpoint
- `vocab.json` - Vocabulary mapping (tokens ↔ indices)
- `training_params.json` - All training parameters used for this run
- `test_results.json` - Evaluation results (created after running `evaluate.py`)

## Inference

### Token-Level Inference

Predict the next token:

```bash
python inference.py \
    --model_path runs/run_token_bs32_ep10_len256_vocab10_20240101_120000/best_model_token_level.pt \
    --vocab_path runs/run_token_bs32_ep10_len256_vocab10_20240101_120000/vocab.json \
    --task token \
    --context "from bootstrap import" \
    --top_k 5
```

### Line-Level Inference

Complete a line:

```bash
python inference.py \
    --model_path runs/run_line_bs32_ep10_len256_vocab10_20240101_120000/best_model_line_level.pt \
    --vocab_path runs/run_line_bs32_ep10_len256_vocab10_20240101_120000/vocab.json \
    --task line \
    --context "def hello(name):" \
    --device cuda
```

## Evaluation

Evaluate a trained model on the test set. The script automatically detects vocabulary and training parameters from the run directory:

```bash
# Simple evaluation (auto-detects vocab and max_length from run directory)
python evaluate.py \
    --model_path runs/run_token_v2_bs32_ep15_len256_vocab25_20240101_120000/best_model.pt \
    --task token \
    --device cuda
```

### Evaluation with Options

```bash
# Limit test examples for faster evaluation
python evaluate.py \
    --model_path runs/run_token_v2_bs32_ep15_len256_vocab25_20240101_120000/best_model.pt \
    --task token \
    --max_test_examples 50000 \
    --num_workers 0 \
    --device cuda
```

### Evaluation Options

- `--model_path`: Path to trained model checkpoint (required)
- `--vocab_path`: Path to vocabulary file (auto-detected from model directory if not specified)
- `--task`: `token` or `line` (default: `token`)
- `--dataset_dir`: Directory containing test datasets (default: `completion_datasets`)
- `--max_length`: Maximum sequence length (auto-detected from training_params.json if available)
- `--batch_size`: Batch size for evaluation (default: 32)
- `--max_test_examples`: Limit number of test examples (None = all, recommended for large test sets)
- `--num_workers`: Number of data loading workers (default: 4, use 0 if experiencing hangs)
- `--lazy_load`: Use lazy loading (default: True, saves memory)
- `--device`: `cuda` or `cpu` (auto-detected, falls back to CPU if CUDA unavailable)

### Auto-Detection Features

The evaluation script automatically:
- **Detects vocabulary** from the same directory as the model (if `vocab.json` exists there)
- **Loads training parameters** from `training_params.json` to set `max_length`
- **Saves results** to the same run directory as the model

This means you typically only need to specify `--model_path` and `--task`!


### Data Flow

1. **Tokenized files** (`token_completion/*.txt`)
   - Already tokenized with special tokens (`<EOL>`, `<STR_LIT>`, etc.)

2. **Completion datasets** (`completion_datasets/*/`)
   - JSONL format with context/target pairs
   - Created by `create_completion_datasets.py`

3. **Training** (`train.py` or `train_v2.py`)
   - Builds vocabulary from tokenized files
   - Converts tokens to indices
   - Trains model with PyTorch DataLoader
   - Creates run directory with model, vocabulary, and training parameters
   - Supports gradient accumulation (train_v2.py) for memory efficiency

4. **Evaluation** (`evaluate.py`)
   - Auto-detects vocabulary and training parameters from run directory
   - Loads trained model
   - Evaluates on test set
   - Saves results to run directory

5. **Inference** (`inference.py`)
   - Auto-detects vocabulary and training parameters from run directory
   - Loads trained model
   - Converts input tokens to indices
   - Generates predictions

### Example Usage in Code

```python
from model import CodeCompletionTransformer, ModelConfig
from train import Vocabulary
import torch
import json

# Load vocabulary from run directory
run_dir = 'runs/run_token_bs32_ep10_len256_vocab10_20240101_120000'
vocab = Vocabulary()
with open(f'{run_dir}/vocab.json', 'r') as f:
    vocab_data = json.load(f)
    vocab.token_to_idx = vocab_data['token_to_idx']
    vocab.idx_to_token = {int(k): v for k, v in vocab_data['idx_to_token'].items()}

# Create model
config = ModelConfig()
config.vocab_size = len(vocab.token_to_idx)
model = CodeCompletionTransformer(config)
model.load_state_dict(torch.load(f'{run_dir}/best_model_token_level.pt'))
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

## Troubleshooting

### Memory Issues (Process Killed)

If training starts but gets killed due to memory issues:

1. **Reduce batch size**:
   ```bash
   python train.py --batch_size 8  # or even 4
   ```

2. **Use lazy loading** (enabled by default):
   ```bash
   python train.py --lazy_load  # Already default
   ```

3. **Sample vocabulary building**:
   ```bash
   python train.py --vocab_sample_lines 100000  # Only use 100k lines for vocab
   ```

4. **Reduce sequence length**:
   ```bash
   python train.py --max_length 128  # Instead of 256
   ```

5. **Process smaller dataset first**:
   - Test with a subset of data to verify it works
   - Use `--limit` in `create_completion_datasets.py` to create smaller datasets

### Other Issues

- **Out of memory**: 
  - Reduce `--batch_size` or `--max_length`
  - Use gradient accumulation: `--accumulation_steps 4` (train_v2.py)
  - Use `--vocab_min_freq 50` or higher to reduce vocabulary size
  
- **Slow training**: 
  - Use GPU (`--device cuda`)
  - Reduce batch size or use gradient accumulation
  - Use `--lazy_load` to save memory
  
- **Poor predictions**: 
  - Train longer (more epochs)
  - Check data quality
  - Adjust learning rate (see `EXPERIMENT_GUIDE.md`)
  - Try different vocabulary sizes (see `PARAMETER_GUIDE.md`)
  
- **Process killed during vocabulary building**: 
  - Use `--vocab_sample_lines` to limit vocabulary building
  
- **Evaluation hangs or is slow**:
  - Use `--num_workers 0` to disable multiprocessing
  - Use `--max_test_examples` to limit test set size
  - Use `--lazy_load` (default: True) to avoid loading all examples into memory
  
- **Vocabulary mismatch errors**:
  - Always use the `vocab.json` from the same run directory as the model
  - The evaluation script auto-detects this, but you can specify `--vocab_path` explicitly

## Additional Resources

- **`PARAMETER_GUIDE.md`**: Comprehensive guide for choosing training parameters (vocab size, batch size, etc.)
- **`EXPERIMENT_GUIDE.md`**: Guide for systematic hyperparameter experiments (learning rate, dropout, regularization, etc.)
- **`diagnose_accuracy_issue.py`**: Diagnostic tool to identify vocabulary mismatches and configuration issues

## To Do

- **Mahyar**
  - Creating the model architecture
  - Testing the model architecture
- **Mayasah**
  - Data loading
  - Data preprocessing
