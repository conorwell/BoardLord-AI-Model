# Hold Annotation Guide

## Overview

Each physical hold on TB2 Mirror can be annotated with three continuous properties that the model cannot infer from position alone. These are stored in `hold_labels.json` and encoded as additional grid channels during training.

---

## Properties

### Incut (degrees)
The angle of the hold's edge — how "positive" it is to grab.

| Value | Description |
|-------|-------------|
| -10 or below | Sloping away — friction dependent, very sensitive to board angle |
| 0 | Flat — neither incut nor sloping |
| 10–20 | Slightly incut — a comfortable edge |
| 30–50 | Clearly incut crimp — positive, secure |
| 60+ | Very incut jug-like edge |

### Depth (finger pads)
How much of a finger fits on the hold.

| Value | Description |
|-------|-------------|
| 0.5 | Tiny crimp — just the very tip of one pad |
| 1.0 | Pad crimp — one pad, no skin above first knuckle |
| 1.5 | Medium crimp — comfortable single-pad crimp |
| 2.0 | Last two digits fit |
| 3.0 | Full finger — whole finger on the hold (jug territory) |

### Pinchability (0–1)
How engaged the thumb can be on the hold.

| Value | Description |
|-------|-------------|
| 0.0 | No thumb involvement possible — pure edge or sloper |
| 0.3 | Slight thumb contact but not load-bearing |
| 0.5 | Thumb engaged but weaker than main edge |
| 1.0 | Full pinch — thumb as positive as the fingers |

---

## Annotation File Format

`hold_labels.json` — keyed by **position_id** (304–801 for TB2 Mirror):

```json
{
  "304": { "incut": 20,  "depth": 1.5, "pinchability": 0.1 },
  "305": { "incut": -5,  "depth": 2.5, "pinchability": 0.0 },
  "400": { "incut": 45,  "depth": 1.0, "pinchability": 0.0 }
}
```

Position IDs match those in `position_map.json` and are visible on the physical board chart.

---

## Annotation Workflow

### Step 1 — Build the mirror map (one-time, already done)

```bash
python build_mirror_map.py
```

Outputs `mirror_map.json`: position_id → mirrored_position_id for all 498 TB2 Mirror holds. The board is perfectly symmetric, so mirrored holds are identical physical shapes with identical properties.

### Step 2 — Annotate one side of the board

Using your physical board chart, annotate the ~250 holds on one side of the board. Save as `hold_labels.json`. You do not need to cover both sides — the expand script handles that.

Foot-only holds (set_id wood positions used only as feet) can be annotated or skipped. Skipped positions receive neutral defaults at load time.

### Step 3 — Expand to mirrored positions

```bash
python expand_annotations.py
```

Reads `hold_labels.json` and `mirror_map.json`. For every annotated position with an unannotated mirror, copies the annotation across. Overwrites `hold_labels.json` in place with the full ~498-entry set.

To preview without overwriting:

```bash
python expand_annotations.py --output hold_labels_full.json
```

---

## Default Values for Unannotated Holds

If a position_id is not present in `hold_labels.json`, the model uses these neutral defaults:

| Property | Default | Rationale |
|----------|---------|-----------|
| incut | 10 | Slightly positive — average board hold |
| depth | 1.5 | Medium crimp — most common hold depth |
| pinchability | 0.1 | Minimal thumb — most edges have no real pinch |

---

## Model Encoding

Each property is normalized to approximately [0, 1] before being placed in the grid tensor:

| Property | Normalization | Channel |
|----------|--------------|---------|
| incut | / 90 → [-0.33, 1.0] | 5 |
| depth | / 3 → [0, 1] | 6 |
| pinchability | already [0, 1] | 7 |

Combined with the existing 5 channels (4 roles + plastic), the full grid tensor will be **(8, 35, 33)** once hold labels are integrated.

---

## Notes

- **Identical holds**: Many position IDs share the same physical hold shape. The mirror expansion handles bilateral pairs automatically. For non-mirrored duplicates (same hold shape at different positions on the same side), annotate each separately — they will have the same values but there is no automated deduplication.
- **Spray wall**: A separate `hold_labels_spray.json` will be needed for layout 11 when that model is developed. The position ID range for Spray is 802–1299.
- **Directionality**: Considered but deferred. The idea was to encode the arc of usable grab angles as a (lo, hi) degree tuple. Encoding this spatially in the CNN is non-trivial and the annotation effort is high. Revisit after the three primary properties are validated.
