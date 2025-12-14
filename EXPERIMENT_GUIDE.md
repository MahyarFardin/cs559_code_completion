# Systematic Experiment Guide for Code Completion Model

This guide outlines meaningful experiments to run with your transformer model. Experiments are ranked by expected impact and ease of implementation.

## Current Hardcoded Values

**Training hyperparameters:**
- Learning rate: `1e-4` (0.0001)
- Weight decay: `0.01`
- Optimizer: `AdamW`
- Gradient clipping: `1.0`
- No learning rate schedule

**Model architecture:**
- Dropout: `0.1`
- d_model: `512`
- n_layer: `6`
- d_ff: `2048`
- n_head: `8`

---

## High-Impact Experiments (Do These First)

### 1. Learning Rate Sweep ⭐⭐⭐
**Impact: HIGH | Effort: MEDIUM (requires code changes)**

Learning rate is one of the most important hyperparameters. Try:
- `1e-5` (0.00001) - Conservative, slower but more stable
- `5e-5` (0.00005) - Slightly lower than current
- `1e-4` (0.0001) - **Current default**
- `2e-4` (0.0002) - Higher, faster convergence
- `5e-4` (0.0005) - Aggressive, may be unstable

**Expected outcome:** 2-5% accuracy difference between best and worst
**Recommendation:** Start with `[1e-5, 5e-5, 1e-4, 2e-4]` for token-level, then narrow down

---

### 2. Weight Decay (L2 Regularization) Sweep ⭐⭐⭐
**Impact: HIGH | Effort: MEDIUM (requires code changes)**

Weight decay prevents overfitting. Current `0.01` is moderate. Try:
- `0.0` - No regularization (baseline)
- `0.001` - Light regularization
- `0.01` - **Current default**
- `0.1` - Strong regularization
- `0.5` - Very strong (may underfit)

**Expected outcome:** 1-3% accuracy difference, affects train/val gap
**Recommendation:** Test `[0.0, 0.001, 0.01, 0.1]` - especially important if you see overfitting

---

### 3. Dropout Sweep ⭐⭐
**Impact: MEDIUM-HIGH | Effort: MEDIUM (requires model.py changes)**

Dropout prevents overfitting. Current `0.1` is standard. Try:
- `0.0` - No dropout (baseline)
- `0.1` - **Current default**
- `0.2` - Moderate dropout
- `0.3` - Strong dropout
- `0.5` - Very strong (may hurt performance)

**Expected outcome:** 1-2% accuracy difference, affects generalization
**Recommendation:** Test `[0.0, 0.1, 0.2, 0.3]` - especially if validation loss plateaus while train keeps decreasing

---

### 4. Number of Epochs / Early Stopping ⭐⭐
**Impact: MEDIUM | Effort: LOW (just run longer)**

Current default is 10 epochs. Try:
- `5` epochs - Quick test
- `10` epochs - **Current default**
- `15` epochs - More training
- `20` epochs - Extended training
- `30` epochs - Long training (with patience/early stopping)

**Expected outcome:** Diminishing returns after ~15-20 epochs typically
**Recommendation:** Monitor train/val curves - stop when val loss plateaus or starts increasing

---

## Medium-Impact Experiments

### 5. Batch Size Sweep ⭐⭐
**Impact: MEDIUM | Effort: LOW (already parameterized)**

Batch size affects training dynamics and memory. Try:
- `8` - Small batches, more gradient noise
- `16` - Smaller batches
- `32` - **Current default**
- `64` - Larger batches, smoother gradients
- `128` - Large batches (if memory allows)

**Expected outcome:** 0.5-2% accuracy difference, affects convergence speed
**Recommendation:** Test `[16, 32, 64]` - larger batches often help but require more memory

---

### 6. Gradient Clipping Threshold ⭐
**Impact: MEDIUM | Effort: MEDIUM (requires code changes)**

Current `1.0` is standard. Try:
- `0.5` - Tighter clipping
- `1.0` - **Current default**
- `2.0` - Looser clipping
- `5.0` - Very loose clipping
- `None` - No clipping

**Expected outcome:** Affects training stability, especially with higher learning rates
**Recommendation:** Test `[0.5, 1.0, 2.0]` - important if you see training instability

---

### 7. Learning Rate Schedule ⭐⭐
**Impact: MEDIUM | Effort: MEDIUM (requires code changes)**

Currently no schedule (constant LR). Try:
- **Cosine annealing**: Decay LR smoothly to 0
- **Step decay**: Reduce LR by 0.5x every N epochs
- **Warmup + decay**: Start low, ramp up, then decay
- **Constant** (current): No schedule

**Expected outcome:** 1-3% accuracy improvement, better convergence
**Recommendation:** Cosine annealing with warmup is often best - try warmup_steps=1000, then cosine to 0

---

## Lower-Impact Experiments (But Still Worthwhile)

### 8. Optimizer Comparison ⭐
**Impact: LOW-MEDIUM | Effort: MEDIUM (requires code changes)**

Current `AdamW` is good. Could try:
- `AdamW` - **Current** (usually best)
- `Adam` - Similar but different weight decay handling
- `SGD` with momentum - Different optimization dynamics

**Expected outcome:** Usually <1% difference, AdamW typically best
**Recommendation:** Only if you have time - AdamW is usually optimal

---

### 9. Sequence Length (Context Window) ⭐
**Impact: MEDIUM (for line-level) | Effort: LOW (already parameterized)**

Current `256` tokens. Try:
- `128` - Short context
- `256` - **Current default**
- `512` - Long context (if memory allows)

**Expected outcome:** More impact on line-level than token-level
**Recommendation:** Test `[128, 256, 512]` especially for line-level completion

---

## Architecture Experiments (Requires model.py Changes)

### 10. Model Dimension (d_model) ⭐⭐
**Impact: HIGH | Effort: HIGH (requires model.py changes)**

Current `512`. Try:
- `256` - Smaller model (~4x fewer params)
- `384` - Medium-small
- `512` - **Current default**
- `768` - Larger model

**Expected outcome:** Significant accuracy difference, but also affects model size
**Recommendation:** If you want smaller models, try `[256, 384, 512]`

---

### 11. Number of Layers (n_layer) ⭐⭐
**Impact: HIGH | Effort: HIGH (requires model.py changes)**

Current `6` layers. Try:
- `3` - Shallow model
- `4` - Fewer layers
- `6` - **Current default**
- `8` - Deeper model
- `12` - Very deep (may need more regularization)

**Expected outcome:** Deeper = better but diminishing returns, more overfitting risk
**Recommendation:** Test `[4, 6, 8]` - 6 is usually a good balance

---

### 12. Feed-Forward Dimension (d_ff) ⭐
**Impact: MEDIUM | Effort: HIGH (requires model.py changes)**

Current `2048` (4x d_model). Try:
- `1024` - Smaller FF (2x d_model)
- `2048` - **Current default** (4x d_model)
- `3072` - Larger FF (6x d_model)

**Expected outcome:** Moderate impact, affects model capacity
**Recommendation:** Less critical - 4x d_model is standard

---

## Recommended Experiment Order

### Phase 1: Quick Wins (No Code Changes)
1. ✅ **Vocabulary size** (already covered in PARAMETER_GUIDE.md)
2. ✅ **Number of epochs** - Run for 15-20 epochs, monitor curves
3. ✅ **Batch size** - Try 16, 32, 64
4. ✅ **Sequence length** - Try 128, 256, 512 (especially line-level)

### Phase 2: Training Hyperparameters (Requires Code Changes)
5. **Learning rate** - Most important! Test `[1e-5, 5e-5, 1e-4, 2e-4]`
6. **Weight decay** - Test `[0.0, 0.001, 0.01, 0.1]`
7. **Dropout** - Test `[0.0, 0.1, 0.2, 0.3]`
8. **Learning rate schedule** - Add cosine annealing with warmup

### Phase 3: Advanced (If Time Permits)
9. **Gradient clipping** - Test `[0.5, 1.0, 2.0]`
10. **Architecture** - If you want to modify model.py, try different d_model or n_layer

---

## Experiment Design Best Practices

### 1. Control Variables
- Keep most parameters fixed, vary one at a time
- Use same random seed for reproducibility
- Use same train/val/test splits

### 2. Baseline First
- Establish a baseline with current defaults
- Then compare all experiments to baseline

### 3. Systematic Sweeps
For each hyperparameter:
- Start with wide range (e.g., `[1e-5, 1e-4, 1e-3]` for LR)
- Narrow down based on results
- Fine-tune in promising region

### 4. Monitor Key Metrics
- **Training loss** - Should decrease smoothly
- **Validation loss** - Should track training loss (gap = overfitting)
- **Test accuracy** - Final metric
- **Training time** - Practical consideration

### 5. Early Stopping
- Stop if validation loss plateaus for 3+ epochs
- Stop if validation loss increases (overfitting)
- Save best model based on validation loss

---

## Expected Results Summary

| Experiment | Expected Impact | Typical Range |
|------------|----------------|---------------|
| Learning Rate | High | ±2-5% accuracy |
| Weight Decay | High | ±1-3% accuracy |
| Dropout | Medium-High | ±1-2% accuracy |
| Batch Size | Medium | ±0.5-2% accuracy |
| LR Schedule | Medium | +1-3% accuracy |
| Gradient Clipping | Medium | Stability improvement |
| Sequence Length | Medium (line-level) | ±1-2% accuracy |
| Architecture (d_model) | High | Significant impact |
| Architecture (n_layer) | High | Significant impact |

---

## Quick Reference: Most Important Experiments

**Top 3 must-do experiments:**
1. **Learning rate sweep** - `[1e-5, 5e-5, 1e-4, 2e-4]`
2. **Weight decay sweep** - `[0.0, 0.001, 0.01, 0.1]`
3. **Dropout sweep** - `[0.0, 0.1, 0.2, 0.3]`

**If you have time:**
4. Learning rate schedule (cosine annealing)
5. Batch size `[16, 32, 64]`
6. More epochs (15-20) with early stopping

**If you want to modify architecture:**
7. d_model `[256, 384, 512]`
8. n_layer `[4, 6, 8]`

---

## Notes

- **Dropout and weight decay are complementary** - if you increase one, you might decrease the other
- **Learning rate and batch size interact** - larger batches often allow higher learning rates
- **More regularization needed for:**
  - Larger models
  - More training data
  - Longer training
- **Less regularization needed for:**
  - Smaller models
  - Limited data
  - Shorter training

Good luck with your experiments! 🚀

