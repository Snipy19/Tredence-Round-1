# Self-Pruning Neural Network on CIFAR-10

> **Tredence Analytics — AI Engineering Intern Case Study**
> *"The Self-Pruning Neural Network"*

A feed-forward neural network that learns to prune its own weights **during training** via learnable gate parameters and L1 sparsity regularisation — no post-training pruning required.

---

## How It Works

Each weight `w_ij` is paired with a learnable scalar `gate_score_ij`. During the forward pass:

```
gate      = sigmoid(gate_score)          # squash to (0, 1)
eff_weight = weight × gate              # element-wise gating
output    = x @ eff_weight.T + bias     # standard linear op
```

The total training loss is:

```
Loss = CrossEntropy(logits, labels) + λ × Σ_all_gates gate_value
```

The L1 term (sum of all gate values) applies a **constant gradient pressure** toward zero, driving unimportant connections to `gate ≈ 0` (effectively pruned) while letting important connections keep `gate ≈ 1`.

---

## Project Structure

```
self_pruning_nn/
├── train.py           # All code: PrunableLinear, SelfPruningNet, training loop, evaluation
├── requirements.txt   # Python dependencies
├── REPORT.md          # Analysis, results table, gate distribution explanation
└── README.md          # This file
```

---

## Setup

### Prerequisites

- Python 3.9+
- pip

### Install dependencies

```bash
git clone <your-repo-url>
cd self_pruning_nn
pip install -r requirements.txt
```

---

## Usage

### Default run (3 lambda values, 60 epochs each)

```bash
python train.py
```

This trains three separate models with `λ ∈ {1e-5, 1e-4, 1e-3}`, saves the best checkpoint for each, and generates:

- `results/gate_dist_lambda<λ>.png` — gate value distribution histogram
- `results/curves_lambda<λ>.png` — accuracy, loss, and sparsity curves over training
- `results/gate_distributions_combined.png` — side-by-side comparison across all λ values
- `results/best_model_lambda<λ>.pt` — best model checkpoint

### Custom options

```bash
# Custom lambdas and epochs
python train.py --lambdas 1e-5 5e-5 1e-4 5e-4 1e-3 --epochs 80

# Faster run on CPU (fewer epochs)
python train.py --device cpu --epochs 20

# GPU
python train.py --device cuda --epochs 60

# Custom output directory
python train.py --save_dir my_results
```

### Full CLI reference

```
--epochs INT         Training epochs per experiment    (default: 60)
--lambdas FLOAT ...  Sparsity penalty values to sweep  (default: 1e-5 1e-4 1e-3)
--save_dir STR       Output directory                  (default: results/)
--device STR         auto | cpu | cuda | mps           (default: auto)
```

---

## Expected Results

| Lambda (λ) | Test Accuracy | Sparsity Level |
|:----------:|:-------------:|:--------------:|
| 1e-5 (Low)    | ~58–60%       | ~15–25%        |
| 1e-4 (Medium) | ~54–57%       | ~40–60%        |
| 1e-3 (High)   | ~45–50%       | ~70–85%        |

Higher λ → more aggressive pruning → lower accuracy. The medium λ gives the best sparsity–accuracy trade-off.

---

## Architecture

```
Input (3072) → PrunableLinear(1024) → BN → ReLU → Dropout(0.3)
             → PrunableLinear(512)  → BN → ReLU → Dropout(0.3)
             → PrunableLinear(256)  → BN → ReLU → Dropout(0.2)
             → PrunableLinear(128)  → BN → ReLU
             → PrunableLinear(10)   → logits
```

All five linear layers are `PrunableLinear` — every connection is learnable-gated.

---

## Key Implementation Details

| Design Choice | Reason |
|---------------|--------|
| L1 penalty on gates | Constant gradient → drives values to exactly 0 (sparsity) |
| Sigmoid on gate_scores | Keeps gates in (0,1); gradients flow to both weight and gate |
| Separate LR for gates (3×) | Gates need to move faster to their on/off state |
| Cosine Annealing LR | Stable convergence; gates settle cleanly at end of training |
| BatchNorm + Dropout | Prevent overfitting on CIFAR-10 with a pure MLP |
| Gradient clipping | Stability when combined loss adds extra gradient signal |

---

## Why L1 Encourages Sparsity (Quick Summary)

- **L1 gradient:** `∂|gate|/∂gate = sign(gate)` — constant magnitude regardless of gate size.
- **L2 gradient:** `∂gate²/∂gate = 2·gate` — shrinks to 0 as gate → 0, so L2 never fully prunes.
- L1 keeps pushing even when a gate is already tiny → drives it all the way to 0.
- This is the same principle behind LASSO regression (Tibshirani, 1996).

---

## License

MIT
