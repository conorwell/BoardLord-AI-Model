# Board Lord AI

A CNN-based difficulty grader for Tension Board 2 problems. Given a problem's hold layout and board angle, predicts its V-grade.

---

## Repo Structure

```
board_lord.ai/
├── docs/
│   ├── annotation.md       # Hold annotation guide and workflow
│   ├── api.md              # Tension Board API reference
│   ├── data.md             # Local data formats and schemas
│   └── model.md            # Model architecture and design decisions
│
├── sync_files/             # Raw paginated API responses (fetch_001.json, ...)
├── holes.json              # All 1177 physical hole positions (fetched once)
├── placements.json         # All 1299 placements, joins position_id → hole (fetched once)
├── position_map.json       # position_id → (x, y) for TB2 Mirror (layout 10)
├── position_map_spray.json # position_id → (x, y) for TB2 Spray (layout 11)
├── mirror_map.json         # position_id → mirrored_position_id for TB2 Mirror
├── hold_labels.json        # Per-hold annotations: incut, depth, pinchability (in progress)
│
├── TB2_Mirror_Problems/    # ~17K parsed climb JSONs for layout 10
├── TB2_Spray_Problems/     # ~13.6K parsed climb JSONs for layout 11
│
├── fetch_all.py            # Paginates the API and saves raw sync files
├── parse_climbs.py         # Parses sync files into per-climb JSONs
├── find_layout.py          # CLI tool: search problems by name, view difficulty
├── visualize_holes.py      # Renders hole_map.png — board grid verification tool
├── build_mirror_map.py     # Builds mirror_map.json (position_id → mirrored_position_id)
├── expand_annotations.py   # Auto-fills mirrored holds in hold_labels.json
│
├── TB_util.py              # Data loading, encoding, dataset class
├── model.py                # CNN architecture and training script
│
├── best_model.pt           # Saved weights from best training run
├── hole_map.png            # Visual verification of TB2 Mirror hold positions
└── hole_map_spray.png      # Visual verification of TB2 Spray hold positions
```

---

## End-to-End Pipeline

### Step 1 — Fetch raw data from the API

```bash
python fetch_all.py
```

Paginates the `/sync` endpoint in batches of up to 2000 climbs, saving each batch to `sync_files/fetch_NNN.json`. See `docs/api.md` for full API details.

**Note:** The token in `fetch_all.py` is a session token tied to a specific account. To re-fetch, update `TOKEN` at the top of the file with a valid token from a captured app request.

### Step 2 — Fetch static board data (one-time)

`holes.json` and `placements.json` are already committed and rarely need to be re-fetched. If you ever need to regenerate them, set their timestamps to epoch in the POST body.

### Step 3 — Parse climbs into per-problem JSONs

```bash
# TB2 Mirror (layout 10)
python parse_climbs.py --layout-id 10 --output-dir TB2_Mirror_Problems

# TB2 Spray (layout 11)
python parse_climbs.py --layout-id 11 --output-dir TB2_Spray_Problems
```

### Step 4 — Verify board mapping (optional)

```bash
python visualize_holes.py
```

Renders `hole_map.png` — a labeled plot of all hold positions on TB2 Mirror.

### Step 5 — Search problems by name

```bash
python find_layout.py "jade"
python find_layout.py "sleepless" --dir TB2_Spray_Problems
```

### Step 6 — Annotate holds (in progress)

```bash
# Build mirror map (one-time, already committed)
python build_mirror_map.py

# After annotating ~250 holds in hold_labels.json:
python expand_annotations.py   # auto-fills mirrored positions
```

See `docs/annotation.md` for the full annotation guide.

### Step 7 — Train the model

```bash
# TB2 Mirror
python model.py --dir TB2_Mirror_Problems --k 5 --epochs 50

# TB2 Spray (separate model)
python model.py --dir TB2_Spray_Problems --pos-map position_map_spray.json --save best_model_spray.pt --k 5 --epochs 50
```

Trains with 5-fold cross-validation. Saves the best checkpoint (by val MAE) to `best_model.pt`.

---

## Key Concepts

### position_id is a placement ID, not a hole ID

The `frames` string in every climb JSON uses `position_id` values that refer to **placement IDs** (rows in `placements.json`), not hole IDs. The correct lookup chain is:

```
position_id → placements[id] → hole_id → holes[id] → (x, y)
```

Querying `holes.json` directly by position_id gives completely wrong coordinates.

### Mirror and Spray are separate models

TB2 Mirror (layout 10) and TB2 Spray (layout 11) share the same grid coordinate space but have different physical hold sets. The same grid position means a different hold on each board.

### Grade scale

The model outputs 11 classes:

| Class | Label | Notes |
|-------|-------|-------|
| 0 | ≤V1 | V0+V1 collapsed — hard to distinguish on a board |
| 1–9 | V2–V10 | One class per V-grade |
| 10 | V11+ | V11/V12/V13 collapsed — tiny samples, unclear community consensus |

`difficulty_average` from the API is a continuous float (1 unit = 1 French grade, 4A=10 through 8C+=33). See `TB_util.py` for the full piecewise V-grade mapping.

---

## Historical Notes

### Why we rewrote the data pipeline

The original model read from hand-crafted `.tb` files with named hold positions (e.g. `"A5"`). This format had no difficulty scores, a bug where foot holds were always zeroed, and was unscalable. The new pipeline fetches directly from the Tension Board API, giving ~17K graded Mirror problems and ~13K Spray problems with community-verified difficulty scores.

### Why we use spatial (x, y) coordinates

The original model treated the 504 board positions as a flat unordered list. An MLP has no concept of physical adjacency. Spatial encoding as a 2D grid lets the CNN detect local hold patterns which are exactly the features that determine difficulty.

### Why classification instead of regression

The first working model regressed to `difficulty_average` using MSE. MSE treats errors uniformly across the grade scale and doesn't know that 5.7 and 6.3 are both correct V6 predictions. Switching to CrossEntropyLoss (and later combined EMD+CE) directly optimizes for correct V-grade prediction.

### Why the combined EMD + CrossEntropy loss

CrossEntropy treats predicting V10 for a V5 as equally wrong as predicting V6 for a V5. Earth Mover's Distance adds ordinal awareness — being further from the correct grade costs more. The combination (0.7×EMD + 0.3×CE) enforces ordinal penalties while keeping predictions sharp. This yielded measurable gains over CE alone.

### Why horizontal flip augmentation

The TB2 board is perfectly symmetric — every hold at (x, y) has a mirror at (-x, y). Flipping problems doubles the dataset for free with zero label noise.

### Why grade-dependent ascent thresholds

A blanket minimum-ascent filter was silently deleting most hard training data. Hard problems have fewer ascents by definition — fewer people can climb them. The current filter relaxes to 1 ascent for 7C+ and above.

### Why grade-normalized ascent weights

Problems with more ascents have more reliable community grades. But weighting by raw ascent count would downweight hard grades (which have fewer ascents by nature). The fix: normalize ascent counts within each grade class using `log(1 + ascents) / median_log_ascents_for_class`, capped at 2×. This measures grade reliability independently of grade difficulty.

### Why the wood/plastic channel

Wood and plastic holds on the Tension Board have fundamentally different properties. The same grid position with a wood jug vs. a plastic crimp produces completely different difficulty. Adding a 5th channel from `set_id` in `placements.json` gave a meaningful accuracy improvement at zero annotation cost.

### Why we collapsed V0/V1 and V11+

V0 and V1 on a tension board are genuinely hard to distinguish — problems at that level have many holds and high beta variance, making community grades noisy. V11, V12, and V13 have very few ascents and the climbing community itself lacks consensus on what these grades mean on a tension board. Collapsing both ends reduces label noise and avoids wasting model capacity on indistinguishable classes.

### Why we use a learned angle embedding instead of a scalar

The board angle (0–65° in 5° steps) is discrete, not continuous. A single normalized scalar in [0, 1] may be drowned out by the 1024 conv features it's concatenated with. A learned 16-dimensional embedding gives the model richer capacity to represent each angle and gave measurable improvement over the scalar approach.

### Why the multi-task coarse head was reverted

An auxiliary output head for coarse grade buckets (Easy/Easy-Mid/Mid/Mid-Hard/Hard) with overlapping ranges was added to help the backbone learn grade-range structure. In practice it slightly hurt performance — the class weights and EMD loss were already teaching the model this structure, and the coarse head's gradient signal conflicted near bucket boundaries.

### Why separate models for Mirror and Spray

Early in the project the assumption was that both layouts could share a model since they share a coordinate grid. However, the same grid position on Mirror and Spray corresponds to a different physical hold. Training together creates contradictory signal. Separate models sidestep this entirely.
