"""
build_mirror_map.py

Builds mirror_map.json: position_id -> mirrored_position_id for TB2 Mirror (layout 10).

Chain:
  position_id -> hole_id          (placements.json)
  hole_id -> mirrored_hole_id     (holes.json)
  mirrored_hole_id -> position_id (reverse lookup in placements.json)

Run once:
  python build_mirror_map.py
"""

import json

LAYOUT_ID = 10  # TB2 Mirror

with open('placements.json') as f:
    placements = json.load(f)

with open('holes.json') as f:
    holes = json.load(f)

# Build lookup tables
hole_to_mirror = {h['id']: h['mirrored_hole_id'] for h in holes}
hole_to_position = {}   # hole_id -> position_id, for this layout only
position_to_hole = {}   # position_id -> hole_id, for this layout only

for p in placements:
    if p['layout_id'] == LAYOUT_ID:
        position_to_hole[p['id']] = p['hole_id']
        hole_to_position[p['hole_id']] = p['id']

# Build position_id -> mirrored_position_id
mirror_map = {}
missing = []
for pos_id, hole_id in position_to_hole.items():
    mirror_hole = hole_to_mirror.get(hole_id)
    mirror_pos = hole_to_position.get(mirror_hole)
    if mirror_pos is not None:
        mirror_map[str(pos_id)] = mirror_pos
    else:
        missing.append(pos_id)

with open('mirror_map.json', 'w') as f:
    json.dump(mirror_map, f, indent=2)

print(f"Built mirror_map.json: {len(mirror_map)} pairs")
if missing:
    print(f"No mirror found for {len(missing)} positions: {missing}")
