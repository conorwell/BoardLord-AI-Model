import json
import os
from collections import defaultdict
import torch
import numpy as np
from torch.utils.data import Dataset

# Grid dimensions (TB2 Mirror & Spray share the same coordinate space)
# x: -64 to 64, step 4 → 33 cols
# y: 4 to 140, step 4 → 35 rows
GRID_H = 35
GRID_W = 33

# Hold role → channel index
ROLE_TO_CHANNEL = {5: 0, 6: 1, 7: 2, 8: 3}  # start, hand, finish, foot
# Channel 4: plastic hold indicator (1.0=plastic, 0.0=wood) at each active hold position
PLASTIC_CHANNEL = 4
NUM_CHANNELS = 5

PLASTIC_SET_ID = 13  # set_id=12 is wood, set_id=13 is plastic

ANGLE_MAX = 65.0
MIN_ASCENTS = 2
MIN_QUALITY = 1.5
MAX_VGRADE    = 11  # V11+ is a single class: V11/V12/V13 collapsed
LOW_COLLAPSE  = 2   # V0, V1, and V2 collapsed into a single ≤V2 class
NUM_CLASSES   = MAX_VGRADE - LOW_COLLAPSE + 1  # 11 classes: ≤V1, V2, ..., V10, V11+

# French grade labels; index = round(difficulty_average - 10), so 4A=10, 8C+=33
FRENCH_GRADES = ['4A','4B','4C','5A','5B','5C', '6A', '6A+', '6B', '6B+', '6C', '6C+',
                 '7A', '7A+', '7B', '7B+', '7C', '7C+', '8A', '8A+','8B','8B+','8C','8C+']


def load_material_map(path='placements.json'):
    """Return dict {position_id (int) -> 1.0 if plastic, 0.0 if wood}."""
    with open(path) as f:
        placements = json.load(f)
    return {p['id']: 1.0 if p['set_id'] == PLASTIC_SET_ID else 0.0
            for p in placements}


def load_position_map(path='position_map.json'):
    """Return dict {position_id (int) -> (row, col)} for a layout."""
    with open(path) as f:
        raw = json.load(f)
    pos_map = {}
    for pid_str, (x, y) in raw.items():
        col = (x + 64) // 4        # 0-32, left→right
        row = (140 - y) // 4       # 0-34, top→bottom
        pos_map[int(pid_str)] = (row, col)
    return pos_map


def encode_climb(holds, pos_map, material_map):
    """Return (5, GRID_H, GRID_W) float tensor encoding a climb's holds.
    Channels 0-3: hold roles (start/hand/finish/foot).
    Channel 4: 1.0 where the hold is plastic, 0.0 where it is wood.
    """
    grid = torch.zeros(NUM_CHANNELS, GRID_H, GRID_W, dtype=torch.float32)
    for h in holds:
        pid = h['position_id']
        ch = ROLE_TO_CHANNEL.get(h['role'])
        if ch is None or pid not in pos_map:
            continue
        row, col = pos_map[pid]
        grid[ch, row, col] = 1.0
        grid[PLASTIC_CHANNEL, row, col] = material_map.get(pid, 0.0)
    return grid


def load_climbs(directory, pos_map, material_map, min_ascents=MIN_ASCENTS, min_quality=MIN_QUALITY):
    """
    Load all climb JSONs from directory.
    Returns list of dicts: {grid, angle (normalized), difficulty, name}
    """
    climbs = []
    for fname in os.listdir(directory):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(directory, fname)) as f:
            data = json.load(f)

        diff = data.get('difficulty_average')
        ascents = data.get('ascensionist_count') or 0
        quality = data.get('quality_average') or 0
        if diff is None or quality < min_quality:
            continue
        # Hard problems naturally get fewer ascents — relax threshold at top grades
        if diff >= 27:      # 7C+ and harder: 1 ascent is enough
            required = 1
        elif diff >= 23:    # 7A+ through 7C: require 2
            required = 2
        else:               # everything else: require 3
            required = min_ascents
        if ascents < required:
            continue

        grid = encode_climb(data['holds'], pos_map, material_map)
        angle_idx = int(float(data.get('angle') or 0) // 5)  # 0–13, one per 5° step
        nomatch = 0.0 if data.get('is_nomatch') else 1.0     # 1.0 = can match (easier)
        vgrade = vgrade_to_class(round(difficulty_to_vgrade(diff)))
        name = data.get('name', '')
        log_ascents = float(np.log1p(ascents))               # compress extreme outliers

        entry = {'grid': grid, 'angle_idx': angle_idx, 'nomatch': nomatch,
                 'difficulty': vgrade, 'name': name, 'log_ascents': log_ascents}
        climbs.append(entry)

    return climbs


def augment_with_flips(climbs):
    """Return a new list: originals followed by horizontally flipped copies.

    Call this on the training split only, never on the validation split.
    Applying it before the train/val split creates leakage — every problem's
    flip would appear in training even when the original is in validation.
    """
    return list(climbs) + [{**c, 'grid': c['grid'].flip(2)} for c in climbs]


class ClimbDataset(Dataset):
    def __init__(self, climbs, ascent_weights=None):
        self.climbs = climbs
        # Default to uniform weights if not provided
        if ascent_weights is None:
            ascent_weights = torch.ones(len(climbs), dtype=torch.float32)
        self.ascent_weights = ascent_weights

    def __len__(self):
        return len(self.climbs)

    def __getitem__(self, idx):
        c = self.climbs[idx]
        return (c['grid'],
                torch.tensor(c['angle_idx'], dtype=torch.long),
                torch.tensor(c['nomatch'], dtype=torch.float32),
                torch.tensor(c['difficulty'], dtype=torch.long),
                self.ascent_weights[idx])


def compute_ascent_weights(climbs, cap=2.0):
    """
    Per-sample reliability weights based on ascent count, normalized within each
    V-grade class so hard grades (few ascents by nature) aren't penalized.

    weight = min(log(1+ascents) / median_log_ascents_for_class, cap)

    Returns a float tensor of shape (len(climbs),).
    """
    # Group log_ascents by vgrade
    by_grade = defaultdict(list)
    for c in climbs:
        by_grade[c['difficulty']].append(c['log_ascents'])

    # Median log-ascents per grade (robust to outliers)
    grade_median = {g: float(np.median(vals)) for g, vals in by_grade.items()}

    weights = []
    for c in climbs:
        median = grade_median[c['difficulty']]
        # Avoid div-by-zero for grades where median log_ascents rounds to 0
        w = c['log_ascents'] / median if median > 0 else 1.0
        weights.append(min(w, cap))

    return torch.tensor(weights, dtype=torch.float32)


def difficulty_to_french(diff):
    idx = max(0, min(round(diff - 10), len(FRENCH_GRADES) - 1))
    return FRENCH_GRADES[idx]


# Piecewise linear difficulty_average → V-grade mapping.
# Centers are the midpoint of each V-grade's French grade range.
# V0={4A,4B,4C}, V1={5A,5B}, V2={5C}, V3={6A,6A+}, V4={6B,6B+},
# V5={6C,6C+}, V6={7A}, V7={7A+}, V8={7B,7B+}, V9={7C}, V10={7C+},
# V11={8A}, V12={8A+}, V13={8B}, V14={8B+}, V15={8C}, V16={8C+}
_DIFF_CENTERS = [11.0, 13.5, 15.0, 16.5, 18.5, 20.5,
                 22.0, 23.0, 24.5, 26.0, 27.0,
                 28.0, 29.0, 30.0, 31.0, 32.0, 33.0]
_VGRADE_VALS  = [ 0.0,  1.0,  2.0,  3.0,  4.0,  5.0,
                  6.0,  7.0,  8.0,  9.0, 10.0,
                 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]


def difficulty_to_vgrade(diff):
    """Convert continuous difficulty_average to continuous V-grade float."""
    return float(np.interp(diff, _DIFF_CENTERS, _VGRADE_VALS))


def vgrade_to_class(v):
    """Map raw V-grade integer (0–16) to model class index (0 to NUM_CLASSES-1).
    V0/V1 → class 0 (≤V1), V2 → class 1, ..., V10 → class 9, V11+ → class 10.
    """
    v = min(v, MAX_VGRADE)          # cap top end (V11+)
    if v <= LOW_COLLAPSE:
        return 0                    # collapse V0 and V1 into class 0
    return v - LOW_COLLAPSE         # shift remaining grades down by 1


def vgrade_to_label(cls):
    """Convert model class index to display string."""
    cls = max(0, min(NUM_CLASSES - 1, round(cls)))
    if cls == 0:
        return f'≤V{LOW_COLLAPSE}'
    v = cls + LOW_COLLAPSE
    return 'V11+' if v >= MAX_VGRADE else f'V{v}'
