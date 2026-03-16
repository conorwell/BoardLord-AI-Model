#!/usr/bin/env python3
"""
create_test_split.py — Randomly assign a stratified 5% test set for each layout.

Re-run any time to generate a fresh random split. The previous test_uuids.json
will be overwritten. Test problems are excluded from all training and K-Folds runs.

Quality bar for the test set is intentionally higher than training:
  - quality_average >= 1.5  (same as training)
  - ascensionist_count >= 3 (all grades — stricter than training)

Output: test_uuids.json — {"10": [uuid, ...], "11": [uuid, ...]}
"""

import json
import os
import random
from collections import defaultdict

import TB_util as tb_util

TEST_FRACTION  = 0.05
MIN_ASCENTS    = 3
MIN_QUALITY    = 1.5
OUTPUT_PATH    = 'test_uuids.json'

LAYOUTS = [
    {'layout_id': 10, 'dir': 'TB2_Mirror_Problems',  'pos_map': 'position_map.json'},
    {'layout_id': 11, 'dir': 'TB2_Spray_Problems',   'pos_map': 'position_map_spray.json'},
]


def collect_uuids_by_grade(directory):
    """
    Scan problem JSONs and return {grade_class: [uuid, ...]} for problems that
    pass the test-set quality bar. Each UUID is counted once regardless of how
    many angle variants exist.
    """
    uuid_to_grade = {}

    for fname in os.listdir(directory):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(directory, fname)) as f:
            data = json.load(f)

        uuid    = data.get('uuid')
        diff    = data.get('difficulty_average')
        ascents = data.get('ascensionist_count') or 0
        quality = data.get('quality_average') or 0

        if uuid is None or diff is None:
            continue
        if quality < MIN_QUALITY or ascents < MIN_ASCENTS:
            continue
        if uuid in uuid_to_grade:
            continue  # already counted from another angle file

        grade = tb_util.vgrade_to_class(round(tb_util.difficulty_to_vgrade(diff)))
        uuid_to_grade[uuid] = grade

    by_grade = defaultdict(list)
    for uuid, grade in uuid_to_grade.items():
        by_grade[grade].append(uuid)

    return by_grade


def sample_test_uuids(by_grade, fraction, seed):
    """Stratified sample: take `fraction` of each grade class."""
    rng = random.Random(seed)
    test_uuids = []

    for grade in sorted(by_grade):
        uuids = by_grade[grade]
        n = max(1, round(len(uuids) * fraction))
        sampled = rng.sample(uuids, min(n, len(uuids)))
        test_uuids.extend(sampled)
        label = tb_util.vgrade_to_label(grade)
        print(f"  {label:>5}: {len(sampled):>4} test / {len(uuids):>5} eligible")

    return test_uuids


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate a stratified test split.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (default: random each run)')
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    print(f"Random seed: {seed}")

    result = {}

    for layout in LAYOUTS:
        layout_id = layout['layout_id']
        directory = layout['dir']

        if not os.path.isdir(directory):
            print(f"\nLayout {layout_id}: {directory}/ not found, skipping.")
            continue

        print(f"\nLayout {layout_id} ({directory}/):")
        by_grade = collect_uuids_by_grade(directory)
        total_eligible = sum(len(v) for v in by_grade.values())

        test_uuids = sample_test_uuids(by_grade, TEST_FRACTION, seed)
        result[str(layout_id)] = test_uuids

        print(f"  Total eligible: {total_eligible}  →  test set: {len(test_uuids)} UUIDs "
              f"({100 * len(test_uuids) / total_eligible:.1f}%)")

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(result, f, indent=2)

    total = sum(len(v) for v in result.values())
    print(f"\nWrote {total} test UUIDs to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
