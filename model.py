import argparse
import json
import os
import random
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import TB_util as tb_util
from TB_util import ClimbDataset, vgrade_to_label, difficulty_to_french


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class BoardCNN(nn.Module):
    """
    Small CNN that predicts climb difficulty from a 4-channel board grid + angle.

    Input:
        grid      — (batch, 4, 35, 33) float tensor; ch0=start, ch1=hand, ch2=finish, ch3=foot
        angle_idx — (batch,) long, integer angle index 0–13 (degrees // 5)
        nomatch   — (batch,) float, 1.0 if the problem has a matchable hold, 0.0 if not

    Output:
        (batch, 17) logits — one per V-grade class (V0–V16)
    """

    # 14 discrete angles (0°, 5°, ..., 65°) each embedded into 16 dims.
    # A learned embedding gives the model more capacity to represent angle
    # than a single normalized scalar, and respects that angle is categorical.
    NUM_ANGLES = 14
    ANGLE_EMB_DIM = 16

    def __init__(self, dropout=0.4):
        super().__init__()
        self.angle_emb = nn.Embedding(self.NUM_ANGLES, self.ANGLE_EMB_DIM)

        # --- Convolutional backbone ---
        # Each Conv2d slides a small filter across the whole board.
        # The filter sees every local neighbourhood of holds, learning patterns
        # like "two adjacent starts" or "big gap before finish".
        # BatchNorm stabilises training by normalising activations per batch.

        self.conv = nn.Sequential(
            # Block 1: 4 input channels → 16 feature maps
            # The kernel sees a 3×3 patch of board positions at a time.
            nn.Conv2d(tb_util.NUM_CHANNELS, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),          # (16, 35, 33) → (16, 17, 16)

            # Block 2: 16 → 32 feature maps
            # Deeper filters detect patterns-of-patterns, e.g. a hard move
            # sequence leading to a big dynamic finish.
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),          # (32, 17, 16) → (32, 8, 8)

            # Block 3: 32 → 64 feature maps
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # AdaptiveAvgPool collapses the spatial dims to a fixed 4×4,
            # so FC layer size doesn't depend on exact input resolution.
            nn.AdaptiveAvgPool2d((4, 4)),   # (64, 4, 4) = 1024 values
        )

        # --- Fully connected head ---
        # Conv features (1024) + angle embedding (16) + nomatch scalar (1) = 1041 inputs.
        self.fc = nn.Sequential(
            nn.Linear(1024 + self.ANGLE_EMB_DIM + 1, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, tb_util.NUM_CLASSES),  # ≤V2, V3–V11, V12+ = 11 classes
        )

    def forward(self, grid, angle_idx, nomatch):
        x = self.conv(grid)                              # (batch, 64, 4, 4)
        x = x.flatten(1)                                # (batch, 1024)
        angle_vec = self.angle_emb(angle_idx)           # (batch, 16)
        x = torch.cat([x, angle_vec, nomatch.unsqueeze(1)], dim=1)  # (batch, 1041)
        return self.fc(x)                               # (batch, 14) logits


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def emd_loss(logits, targets):
    """
    Earth Mover's Distance loss for ordinal classification.

    Measures the L1 distance between the predicted CDF and the true CDF
    (a step function that jumps to 1 at the target class).

    Predicting one grade off costs 1 unit; predicting five grades off costs 5.
    Returns per-sample losses (shape: batch,) for compatibility with sample weighting.
    """
    probs = torch.softmax(logits, dim=1)                     # (batch, C)
    cum_pred = torch.cumsum(probs, dim=1)[:, :-1]            # (batch, C-1)

    # True CDF: cumsum of one-hot = step function jumping to 1 at target index
    targets_oh = torch.zeros_like(probs)
    targets_oh.scatter_(1, targets.unsqueeze(1), 1.0)
    cum_true = torch.cumsum(targets_oh, dim=1)[:, :-1]       # (batch, C-1)

    return (cum_pred - cum_true).abs().mean(dim=1)           # (batch,)


def combined_loss(logits, targets, ce_fn, alpha=0.7):
    """
    alpha * EMD + (1 - alpha) * CrossEntropy.
    EMD enforces ordinal penalties; CE keeps predictions sharp.
    Returns per-sample losses for sample weighting.
    """
    return alpha * emd_loss(logits, targets) + (1 - alpha) * ce_fn(logits, targets)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for grid, angle_idx, nomatch, vgrade, sample_w in loader:
        grid, angle_idx, nomatch, vgrade, sample_w = (
            grid.to(device), angle_idx.to(device), nomatch.to(device),
            vgrade.to(device), sample_w.to(device))
        optimizer.zero_grad()
        logits = model(grid, angle_idx, nomatch)
        loss = (combined_loss(logits, vgrade, loss_fn) * sample_w).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(vgrade)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_correct = 0
    for grid, angle_idx, nomatch, vgrade, _ in loader:
        grid, angle_idx, nomatch, vgrade = (
            grid.to(device), angle_idx.to(device), nomatch.to(device), vgrade.to(device))
        logits = model(grid, angle_idx, nomatch)
        pred_classes  = logits.argmax(dim=1)
        total_loss    += combined_loss(logits, vgrade, loss_fn).mean().item() * len(vgrade)
        total_mae     += (pred_classes - vgrade).abs().float().sum().item()
        total_correct += (pred_classes == vgrade).sum().item()
    n = len(loader.dataset)
    return total_loss / n, total_mae / n, total_correct / n * 100


# ---------------------------------------------------------------------------
# Full training (production model)
# ---------------------------------------------------------------------------

def run_full_train(climbs, test_climbs, args, device, loss_fn):
    """Train on all non-test data, evaluate on the held-out test set.

    Holds back 5% of climbs as a scheduler signal set for ReduceLROnPlateau
    and checkpointing by val MAE — matching the k-folds setup. This set is
    not used for reporting; only the held-out test set is used for that.
    """
    random.shuffle(climbs)
    n_sched_val = max(1, int(0.05 * len(climbs)))
    sched_val_climbs = climbs[:n_sched_val]
    train_climbs_raw = climbs[n_sched_val:]

    train_climbs = tb_util.augment_with_flips(train_climbs_raw)
    train_weights = tb_util.compute_ascent_weights(train_climbs, cap=2.0)
    train_dataset = ClimbDataset(train_climbs, train_weights)
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    sched_val_dataset = ClimbDataset(sched_val_climbs)
    sched_val_loader  = DataLoader(sched_val_dataset, batch_size=args.batch_size)

    print(f"  Scheduler signal set: {len(sched_val_climbs)} climbs | Training: {len(train_climbs_raw)} climbs")

    model     = BoardCNN(dropout=0.4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_mae = float('inf')
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        sched_loss, sched_mae, _ = eval_epoch(model, sched_val_loader, loss_fn, device)
        scheduler.step(sched_loss)
        lr = optimizer.param_groups[0]['lr']
        marker = ' *' if sched_mae < best_val_mae else ''
        print(f"  Epoch {epoch:3d} | train={train_loss:.3f} | val={sched_loss:.3f} MAE={sched_mae:.2f} | lr={lr:.2e}{marker}")
        if sched_mae < best_val_mae:
            best_val_mae = sched_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    torch.save(best_state, args.save)
    print(f"\nBest val MAE: {best_val_mae:.3f} — model saved to {args.save}")

    # --- Test set evaluation ---
    if not test_climbs:
        print("No test climbs loaded — skipping test evaluation.")
        return

    test_dataset = ClimbDataset(test_climbs)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size)

    model.eval()
    grade_correct = defaultdict(int)
    grade_total   = defaultdict(int)

    with torch.no_grad():
        for grid, angle_idx, nomatch, vgrade, _ in test_loader:
            grid, angle_idx, nomatch = grid.to(device), angle_idx.to(device), nomatch.to(device)
            preds = model(grid, angle_idx, nomatch).cpu().argmax(dim=1)
            for true_cls, pred_cls in zip(vgrade.tolist(), preds.tolist()):
                grade_total[true_cls] += 1
                if pred_cls == true_cls:
                    grade_correct[true_cls] += 1

    total   = sum(grade_total.values())
    correct = sum(grade_correct.values())
    print(f"\nTest set accuracy: {correct}/{total} = {100*correct/total:.1f}%")
    print(f"  {'Grade':>5}  {'Correct':>7}  {'Total':>5}  {'Pct':>5}")
    for cls in range(tb_util.NUM_CLASSES):
        t = grade_total[cls]
        c = grade_correct[cls]
        pct = 100 * c / t if t else 0
        print(f"  {vgrade_to_label(cls):>5}  {c:>7}  {t:>5}  {pct:>4.0f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_fold(fold, train_idx, val_idx, climbs, args, device, loss_fn):
    """Train one fold, return best val MAE and the model state dict.

    Augmentation (horizontal flip) is applied to the training split only.
    Splitting first, then augmenting, prevents a problem and its mirror from
    appearing on opposite sides of the train/val boundary.
    """
    train_climbs = tb_util.augment_with_flips([climbs[i] for i in train_idx])
    train_weights = tb_util.compute_ascent_weights(train_climbs, cap=2.0)
    train_dataset = ClimbDataset(train_climbs, train_weights)

    val_climbs = [climbs[i] for i in val_idx]
    val_dataset = ClimbDataset(val_climbs)   # ascent weights unused during eval

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size)

    model = BoardCNN(dropout=0.4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    best_val_mae = float('inf')
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_mae, val_pct = eval_epoch(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)
        print(f"  Fold {fold} | Epoch {epoch:3d} | "
              f"train={train_loss:.3f}  val={val_loss:.3f}  MAE={val_mae:.2f}  correct={val_pct:.1f}%")

        # Save by MAE — more stable and directly interpretable than weighted loss
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_val_mae, best_val_loss, best_state, val_idx


def main(args):
    device = torch.device('mps' if torch.backends.mps.is_available()
                          else 'cuda' if torch.cuda.is_available()
                          else 'cpu')
    print(f"Device: {device}")

    # --- Load data ---
    pos_map = tb_util.load_position_map(args.pos_map)
    material_map = tb_util.load_material_map(args.placements)

    exclude_uuids = set()
    if args.test_uuids and os.path.exists(args.test_uuids):
        with open(args.test_uuids) as f:
            test_map = json.load(f)
        layout_id = str(10 if 'mirror' in args.dir.lower() else 11)
        exclude_uuids = set(test_map.get(layout_id, []))
        print(f"Excluding {len(exclude_uuids)} test UUIDs (layout {layout_id})")

    climbs = tb_util.load_climbs(args.dir, pos_map, material_map, exclude_uuids=exclude_uuids)
    print(f"Loaded {len(climbs)} climbs after quality/ascent filters")

    # Class weights: inverse frequency raised to CLASS_WEIGHT_POWER.
    # pow=0.5 (sqrt) is conservative; pow=0.75 gives harder grades more influence;
    # pow=1.0 is full inverse frequency (very aggressive).
    # V13 / V0 weight ratio: ~5× at 0.5, ~13× at 0.75, ~28× at 1.0.
    CLASS_WEIGHT_POWER = 0.3
    num_classes = tb_util.NUM_CLASSES
    counts = torch.zeros(num_classes)
    for c in climbs:
        counts[c['difficulty']] += 1
    weights = torch.where(
        counts > 0,
        (len(climbs) / (num_classes * counts)).pow(CLASS_WEIGHT_POWER),
        torch.zeros(num_classes)
    ).to(device)
    print(f"Class weights (power={CLASS_WEIGHT_POWER}):")
    for v in range(num_classes):
        print(f"  {vgrade_to_label(v):>5}: count={int(counts[v]):>5}  weight={weights[v].item():.2f}")

    # reduction='none' so we can apply per-sample ascent weights in train_epoch.
    # eval_epoch reduces manually without sample weights (uniform evaluation).
    loss_fn = nn.CrossEntropyLoss(weight=weights, reduction='none')

    # --- Full train (production mode) ---
    if args.full_train:
        test_climbs = []
        if exclude_uuids:
            test_climbs = tb_util.load_climbs(
                args.dir, pos_map, material_map,
                include_uuids=exclude_uuids   # exclude_uuids == the test UUID set
            )
            print(f"Loaded {len(test_climbs)} test climb instances for evaluation")
        run_full_train(climbs, test_climbs, args, device, loss_fn)
        return

    # --- K-Folds ---
    # Each fold uses a different 1/k chunk as the validation set.
    # This means every climb gets validated exactly once, giving a more
    # reliable picture of generalisation than a single 80/20 split.
    # We also save the single best model across all folds.
    kf = KFold(n_splits=args.k, shuffle=True, random_state=42)
    indices = list(range(len(climbs)))

    fold_maes = []
    fold_pcts = []
    best_overall_loss = float('inf')
    best_overall_state = None
    best_val_idx = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices), start=1):
        print(f"\n=== Fold {fold}/{args.k} | train={len(train_idx)} val={len(val_idx)} ===")
        mae, val_loss, state, v_idx = run_fold(
            fold, train_idx, val_idx, climbs, args, device, loss_fn
        )
        # Compute % correct at best checkpoint
        best_model = BoardCNN(dropout=0.4).to(device)
        best_model.load_state_dict({k: v.to(device) for k, v in state.items()})
        fold_val_dataset = ClimbDataset([climbs[i] for i in v_idx])
        val_loader_tmp = DataLoader(fold_val_dataset, batch_size=args.batch_size)
        _, fold_mae, fold_pct = eval_epoch(best_model, val_loader_tmp, loss_fn, device)
        fold_maes.append(fold_mae)
        fold_pcts.append(fold_pct)
        print(f"  Fold {fold} best — MAE: {fold_mae:.2f} V-grades  correct: {fold_pct:.1f}%")

        if fold_mae < best_overall_loss:
            best_overall_loss = fold_mae
            best_overall_state = state
            best_val_idx = v_idx

    # --- Summary ---
    mean_mae = sum(fold_maes) / len(fold_maes)
    mean_pct = sum(fold_pcts) / len(fold_pcts)
    print(f"\n{'='*50}")
    print(f"K-Folds complete ({args.k} folds)")
    print(f"Per-fold MAE:     {[f'{m:.2f}' for m in fold_maes]}")
    print(f"Per-fold correct: {[f'{p:.1f}%' for p in fold_pcts]}")
    print(f"Mean MAE: {mean_mae:.2f} V-grades  |  Mean correct: {mean_pct:.1f}%")

    # Save the best model (lowest val MAE across all folds)
    torch.save(best_overall_state, args.save)
    print(f"Best model saved to {args.save}")

    # --- Per-grade accuracy breakdown from best fold's val set ---
    model = BoardCNN(dropout=0.4).to(device)
    model.load_state_dict({k: v.to(device) for k, v in best_overall_state.items()})
    model.eval()

    val_climbs = [climbs[i] for i in best_val_idx]
    val_dataset = ClimbDataset(val_climbs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    grade_correct = defaultdict(int)
    grade_total   = defaultdict(int)

    with torch.no_grad():
        for grid, angle_idx, nomatch, vgrade, _ in val_loader:
            grid, angle_idx, nomatch = grid.to(device), angle_idx.to(device), nomatch.to(device)
            preds = model(grid, angle_idx, nomatch).cpu().argmax(dim=1)
            for true_cls, pred_cls in zip(vgrade.tolist(), preds.tolist()):
                grade_total[true_cls] += 1
                if pred_cls == true_cls:
                    grade_correct[true_cls] += 1

    num_classes = tb_util.NUM_CLASSES
    print(f"\nPer-grade accuracy (best fold val set):")
    print(f"  {'Grade':>5}  {'Correct':>7}  {'Total':>5}  {'Pct':>5}")
    for cls in range(num_classes):
        total   = grade_total[cls]
        correct = grade_correct[cls]
        pct     = 100 * correct / total if total else 0
        label   = vgrade_to_label(cls)
        print(f"  {label:>5}  {correct:>7}  {total:>5}  {pct:>4.0f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CNN difficulty predictor')
    parser.add_argument('--dir', default='TB2_Mirror_Problems',
                        help='Parsed climb JSON directory')
    parser.add_argument('--pos-map', default='position_map.json',
                        help='Position map JSON (default: position_map.json)')
    parser.add_argument('--placements', default='placements.json',
                        help='Placements JSON for wood/plastic material map (default: placements.json)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--save', default='best_model.pt',
                        help='Path to save best model weights')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of folds for cross-validation (default: 10)')
    parser.add_argument('--test-uuids', default='test_uuids.json',
                        help='Path to test UUID file generated by create_test_split.py (default: test_uuids.json)')
    parser.add_argument('--full-train', action='store_true',
                        help='Train on all non-test data and evaluate on the test set. '
                             'Use this to produce the production model after K-Folds tuning is done.')
    args = parser.parse_args()
    main(args)
