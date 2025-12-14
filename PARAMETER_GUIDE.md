# Parameter Guide for Code Completion Training

## Model Size Analysis

The model size is primarily determined by:
1. **Vocabulary size** (biggest factor) - controlled by `--vocab_min_freq`
2. **Model architecture** (d_model, n_layer, d_ff) - currently hardcoded in `model.py`
3. **Sequence length** - affects memory usage, not model size

### Approximate Model Size Formula

```
Model Parameters ≈ vocab_size × d_model × 2 + (n_layer × (4×d_model² + 2×d_model×d_ff)) + max_len×d_model
```

With current defaults (d_model=512, n_layer=6, d_ff=2048):
- **vocab_size=10,000**: ~15M parameters
- **vocab_size=20,000**: ~25M parameters  
- **vocab_size=50,000**: ~62M parameters
- **vocab_size=100,000**: ~125M parameters

## Recommended Parameter Sets

### Small Model (10-20M parameters) - Good for testing/development

**Token-Level:**
```bash
python train.py \
    --task token \
    --vocab_min_freq 50 \
    --vocab_sample_lines 100000 \
    --batch_size 16 \
    --num_epochs 10 \
    --max_length 128 \
    --max_train_examples 50000 \
    --max_val_examples 5000 \
    --device cuda
```

**Line-Level:**
```bash
python train.py \
    --task line \
    --vocab_min_freq 50 \
    --vocab_sample_lines 100000 \
    --batch_size 16 \
    --num_epochs 10 \
    --max_length 128 \
    --max_train_examples 50000 \
    --max_val_examples 5000 \
    --device cuda
```

**Expected vocab size:** ~8,000-12,000 tokens  
**Expected model size:** ~12-18M parameters

---

### Medium Model (20-35M parameters) - Balanced performance/size

**Token-Level:**
```bash
python train.py \
    --task token \
    --vocab_min_freq 25 \
    --vocab_sample_lines 200000 \
    --batch_size 32 \
    --num_epochs 15 \
    --max_length 256 \
    --max_train_examples 200000 \
    --max_val_examples 10000 \
    --device cuda
```

**Line-Level:**
```bash
python train.py \
    --task line \
    --vocab_min_freq 25 \
    --vocab_sample_lines 200000 \
    --batch_size 32 \
    --num_epochs 15 \
    --max_length 256 \
    --max_train_examples 200000 \
    --max_val_examples 10000 \
    --device cuda
```

**Expected vocab size:** ~15,000-20,000 tokens  
**Expected model size:** ~25-30M parameters

---

### Large Model (35-50M parameters) - Best performance if you have GPU memory

**Token-Level:**
```bash
python train.py \
    --task token \
    --vocab_min_freq 15 \
    --vocab_sample_lines 500000 \
    --batch_size 32 \
    --num_epochs 20 \
    --max_length 256 \
    --max_train_examples 500000 \
    --max_val_examples 10000 \
    --device cuda
```

**Line-Level:**
```bash
python train.py \
    --task line \
    --vocab_min_freq 15 \
    --vocab_sample_lines 500000 \
    --batch_size 32 \
    --num_epochs 20 \
    --max_length 256 \
    --max_train_examples 500000 \
    --max_val_examples 10000 \
    --device cuda
```

**Expected vocab size:** ~25,000-35,000 tokens  
**Expected model size:** ~40-50M parameters

---

## Key Parameters Explained

### Vocabulary Control (Most Important!)

- **`--vocab_min_freq`**: Minimum token frequency to include in vocabulary
  - **Higher = smaller vocab = smaller model**
  - Recommended range: 15-50
  - Start with 25-30 for balanced results
  - Use 50+ if you need a very small model

- **`--vocab_sample_lines`**: Number of lines to sample for vocabulary building
  - Limits memory during vocab construction
  - Doesn't affect final vocab size much if `vocab_min_freq` is set
  - Recommended: 100k-500k

### Training Data Control

- **`--max_train_examples`**: Limit training examples
  - Use this to train faster and test configurations
  - Start small (50k) to test, then scale up
  - None = use all data

- **`--max_val_examples`**: Limit validation examples
  - Default 10k is usually sufficient
  - Can reduce to 5k for faster validation

### Model Architecture (Hardcoded - would need to modify model.py)

Currently fixed in `model.py`:
- `d_model = 512` (model dimension)
- `n_layer = 6` (transformer layers)
- `d_ff = 2048` (feed-forward dimension)
- `n_head = 8` (attention heads)

To reduce model size further, you'd need to modify these in `model.py`:
- Smaller `d_model` (e.g., 256 or 384)
- Fewer layers (e.g., 4 instead of 6)
- Smaller `d_ff` (e.g., 1024 instead of 2048)

### Sequence Length

- **`--max_length`**: Maximum sequence length
  - Affects **memory usage** during training, not model size
  - Smaller = less memory, but shorter context
  - Recommended: 128-256
  - Use 128 if you have memory issues

### Batch Size

- **`--batch_size`**: Examples per batch
  - Affects **memory usage** and training speed
  - Smaller = less memory, slower training
  - Recommended: 16-32
  - Use 8-16 if you have memory issues

## Quick Start Recommendations

### For First Experiments (Small & Fast)
```bash
# Token-level, small model
python train.py --task token --vocab_min_freq 50 --batch_size 16 --max_length 128 --max_train_examples 50000

# Line-level, small model  
python train.py --task line --vocab_min_freq 50 --batch_size 16 --max_length 128 --max_train_examples 50000
```

### For Final Training (Balanced)
```bash
# Token-level, medium model
python train.py --task token --vocab_min_freq 25 --batch_size 32 --max_length 256 --max_train_examples 200000

# Line-level, medium model
python train.py --task line --vocab_min_freq 25 --batch_size 32 --max_length 256 --max_train_examples 200000
```

## Memory Troubleshooting

If you still get out-of-memory errors:

1. **Reduce batch size**: `--batch_size 8` or even `--batch_size 4`
2. **Reduce sequence length**: `--max_length 128` or `--max_length 64`
3. **Use lazy loading**: `--lazy_load` (already default)
4. **Reduce training examples**: `--max_train_examples 100000`
5. **Reduce vocab sample**: `--vocab_sample_lines 50000`

## Parameter Sweep Suggestions

For systematic experimentation, try these combinations:

### Vocabulary Size Sweep (Token-Level)
```bash
# Very small vocab
python train.py --task token --vocab_min_freq 100 --max_train_examples 100000

# Small vocab
python train.py --task token --vocab_min_freq 50 --max_train_examples 100000

# Medium vocab
python train.py --task token --vocab_min_freq 25 --max_train_examples 100000

# Large vocab (if you have memory)
python train.py --task token --vocab_min_freq 15 --max_train_examples 100000
```

### Sequence Length Sweep
```bash
# Short context
python train.py --task token --vocab_min_freq 25 --max_length 128

# Medium context
python train.py --task token --vocab_min_freq 25 --max_length 256

# Long context (if you have memory)
python train.py --task token --vocab_min_freq 25 --max_length 512
```

## Notes

- **Vocabulary size is the biggest factor** - focus on `--vocab_min_freq` first
- Start with small models to test your setup, then scale up
- Use `--max_train_examples` to limit data for faster iteration
- Monitor vocabulary size during training - it's printed at the start
- If vocab > 50k, your model will be > 60M parameters

