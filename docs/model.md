# Model Design

## Problem Statement

Given a Tension Board problem — a set of holds with assigned roles (start, hand, finish, foot) and a board angle — predict its V-grade difficulty.

---

## Input Representation

### Why not a flat vector?

The original model encoded holds as a 1008-dimensional flat binary vector (504 positions × 2 channels for hands/feet). This has a fundamental flaw: it has no concept of spatial proximity. Position 400 and position 401 are treated as completely unrelated despite being physically adjacent on the board. A sequence of two close holds looks identical to two distant holds to an MLP.

### The grid encoding

Each problem is encoded as a **(5, 35, 33) float tensor** — 5 channels stacked over the board grid:

| Channel | Contents |
|---------|----------|
| 0 | Start holds (r5) |
| 1 | Hand holds (r6) |
| 2 | Finish holds (r7) |
| 3 | Foot-only holds (r8) |
| 4 | Plastic indicator: 1.0 if hold is plastic, 0.0 if wood |

The grid is **35 rows × 33 columns**, matching the physical board spacing (4-inch grid, x: -64 to 64, y: 4 to 140). A 1 at position (row, col) in channels 0–3 means a hold of that role is present. Row 0 = top of the board, row 34 = bottom.

**Planned expansion to 8 channels** once `hold_labels.json` annotation is complete:

| Channel | Contents | Normalization |
|---------|----------|--------------|
| 5 | Incut (edge angle) | / 90 → [-0.33, 1.0] |
| 6 | Depth (finger pads) | / 3 → [0, 1] |
| 7 | Pinchability | already [0, 1] |

### Why the plastic channel matters

Wood and plastic holds on the Tension Board have fundamentally different properties — plastic holds are generally smaller, less positive, and more angle-sensitive than wood holds. Adding a plastic channel gave a meaningful accuracy improvement because the model can now learn interactions like "plastic hold cluster at the top at 40° = crux sequence."

### Why foot holds get their own channel

Feet are never open on the Tension Board — every foot hold is explicitly marked r8. Hand vs. foot use is a fundamental part of what makes a problem hard or easy. Giving feet their own channel lets the model learn things like "foot cluster at the bottom + hand holds spread wide = powerful move."

### Angle

The board angle (0°, 5°, 10°, ..., 65°) is encoded as a **learned embedding** (14 discrete values → 16-dimensional vector) injected after the convolutional layers. This is more expressive than a normalized scalar — each of the 14 angles gets its own learned representation, and the model can learn that 40° and 45° are similar while 20° is qualitatively different.

A normalized scalar was used originally but gave insufficient signal relative to the 1024 conv features it was concatenated with.

### is_nomatch

`is_nomatch` from the climb JSON encodes whether the problem has a matchable hold (a hold that can be grabbed with both hands simultaneously). `is_nomatch: false` means matching is possible, which generally makes a problem easier. Encoded as a scalar: `1.0` = can match, `0.0` = no match. Injected alongside the angle embedding.

---

## Architecture: BoardCNN

```
Input: grid (batch, 5, 35, 33) + angle_idx (batch,) + nomatch (batch,)

Conv Block 1:  Conv2d(5→16, 3×3, pad=1) → BatchNorm → ReLU → MaxPool2d(2)
               Output: (batch, 16, 17, 16)

Conv Block 2:  Conv2d(16→32, 3×3, pad=1) → BatchNorm → ReLU → MaxPool2d(2)
               Output: (batch, 32, 8, 8)

Conv Block 3:  Conv2d(32→64, 3×3, pad=1) → BatchNorm → ReLU
               Output: (batch, 64, 8, 8)

Spatial Pool:  AdaptiveAvgPool2d(4, 4)
               Output: (batch, 64, 4, 4) → flatten → (batch, 1024)

Angle embed:   Embedding(14, 16) → (batch, 16)
Concat:        [conv_features, angle_vec, nomatch] → (batch, 1041)

FC Head:       Linear(1041→256) → ReLU → Dropout(0.4)
               Linear(256→64) → ReLU
               Linear(64→11)

Output: (batch, 11) logits — one per class (≤V1, V2–V10, V11+)
```

### Why convolutional layers?

CNNs apply learned filters that slide across the entire grid. A filter that recognizes "two holds close together" will fire wherever it sees that pattern. This is **translation equivariance** — the network doesn't need to learn the same pattern separately for every board region.

Three blocks of increasing depth (16→32→64) let later layers detect increasingly abstract patterns: first block sees local hold relationships, second block sees move sequences, third block sees overall problem structure.

---

## Output: Classification

### Class scheme

11 output classes:

| Class | Label | V-Grades | Notes |
|-------|-------|----------|-------|
| 0 | ≤V1 | V0, V1 | Collapsed — hard to distinguish on a board, high beta variance |
| 1 | V2 | V2 | |
| 2 | V3 | V3 | |
| 3 | V4 | V4 | |
| 4 | V5 | V5 | |
| 5 | V6 | V6 | |
| 6 | V7 | V7 | |
| 7 | V8 | V8 | |
| 8 | V9 | V9 | |
| 9 | V10 | V10 | |
| 10 | V11+ | V11, V12, V13 | Collapsed — tiny samples, community grade consensus unclear |

### Why classification instead of regression?

Early versions regressed to `difficulty_average` using MSE. Problems:
1. MSE penalizes a 2-unit error the same at 4A as at 7C+, but physically these are very different gaps.
2. The end user wants a V-grade, not a float. Predictions of 5.7 and 6.3 are both correct V6 but MSE treats them differently.

Classification with the combined EMD + CrossEntropy loss directly optimizes for correct V-grade bucket prediction.

---

## Loss Function

### Combined EMD + CrossEntropy

```
loss = 0.7 * EMD + 0.3 * CrossEntropy
```

**Earth Mover's Distance (EMD)** treats the output as a probability distribution and measures the cost of moving probability mass from the predicted distribution to the true label. Predicting one grade off costs 1 unit; predicting five grades off costs 5 units. This directly penalizes large errors more than small ones, which CrossEntropy alone does not.

**CrossEntropy** keeps predictions sharp — without it, EMD alone can cause the model to spread probability broadly across adjacent grades rather than committing to a single prediction.

EMD is computed via L1 distance between predicted and true CDFs:
```python
cum_pred = cumsum(softmax(logits))[:, :-1]
cum_true = cumsum(one_hot(target))[:, :-1]
emd = |cum_pred - cum_true|.mean(dim=1)
```

### Class weights

Inverse frequency weighting with exponent 0.6 (tunable):
```python
weight = (N / (num_classes * count_per_class)) ** 0.6
```

Lower exponent (0.5 = sqrt) is more conservative. Higher (0.75, 1.0) gives harder grades more influence. The exponent is a training hyperparameter defined as `CLASS_WEIGHT_POWER` in `model.py`.

### Per-sample ascent weights

Problems with more ascents have more reliable community grades. Weight = `min(log(1 + ascents) / median_log_ascents_for_class, 2.0)`, normalized within each V-grade class so hard grades (few ascents by nature) are not penalized relative to easy grades.

The total per-sample loss:
```python
loss = (combined_loss(fine_logits, vgrade, ce_fn) * ascent_weight).mean()
```

---

## Data Augmentation: Horizontal Flip

The TB2 board is perfectly symmetric — every hold at (x, y) has a mirror at (-x, y). Every problem is duplicated with its grid flipped along the column axis (`grid.flip(2)`), doubling the dataset from ~9K to ~18K samples at zero label-noise cost.

---

## Training Setup

| Hyperparameter | Value | Reason |
|---------------|-------|--------|
| Optimizer | Adam, lr=1e-3 | Adaptive learning rate, robust default |
| LR Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) | Halves LR when val loss plateaus |
| Loss | 0.7×EMD + 0.3×CE with class weights | Ordinal penalties + sharp predictions |
| Batch size | 64 | Balances gradient noise and memory |
| Epochs | 50 | Sufficient for convergence with scheduler |
| Validation | K-Folds (k=5) | Every sample validated exactly once |
| Model selection | Best val MAE across all folds | MAE is stable; loss can spike |

---

## Metrics

| Metric | Description |
|--------|-------------|
| Loss | Combined EMD+CE — used for training and LR scheduling |
| MAE | Mean absolute error in class indices (≈ V-grades off on average) |
| % Correct | Fraction of val set where predicted class exactly matches true class — primary metric |

**On expected performance:** ~45% exact class accuracy is a reasonable result given the task difficulty. A human examining a hold layout photo without climbing the problem would likely perform similarly. The more informative metric is **within-1-class accuracy** (roughly 75–85% based on MAE ~0.75), which measures whether the model is within one V-grade of the correct answer.

---

## Separate Models per Layout

TB2 Mirror (layout 10) and TB2 Spray (layout 11) use different physical hold sets at the same grid positions. Training a single model on both creates contradictory signal — the same grid position means a different physical hold depending on which board you're on. Separate models are trained for each layout.

---

## What Was Tried and Reverted

### Multi-task coarse head

An auxiliary head was added to predict overlapping grade buckets (Easy/Easy-Mid/Mid/Mid-Hard/Hard) alongside the fine V-grade head, sharing a common trunk. The intuition was that coarse grade structure would regularize the shared features. In practice this slightly hurt performance, likely because:
- The coarse bucket boundaries introduced conflicting gradient signal near boundary grades
- The shared trunk was already learning grade-range structure via class weights and EMD loss

Reverted to a single-head model.
