import argparse
import json
import glob
import os
import re

parser = argparse.ArgumentParser(description='Parse Tension Board sync files into per-climb JSON.')
parser.add_argument('--layout-id', type=int, default=10, help='Layout ID to filter (default: 10)')
parser.add_argument('--output-dir', type=str, default='TB2_Mirror_Problems', help='Output directory name (default: TB2_Mirror_Problems)')
parser.add_argument('--fetch-file', type=str, default=None,
                    help='Incremental mode: process only this fetch file, sourcing climb '
                         'definitions from existing problem files. If omitted, does a full '
                         'rebuild from all sync_files/.')
args = parser.parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, args.output_dir)
os.makedirs(output_dir, exist_ok=True)


def parse_frames(frames_str):
    holds = []
    for match in re.finditer(r'p(\d+)r(\d+)', frames_str):
        holds.append({
            'position_id': int(match.group(1)),
            'role': int(match.group(2))
        })
    return holds


def sanitize_name(name):
    name = name.encode('ascii', 'ignore').decode('ascii')
    name = re.sub(r"[.,!?'\"()]", '', name)
    name = name.replace(' ', '_')
    name = re.sub(r'[^\w]', '', name)
    name = re.sub(r'_+', '_', name).strip('_')
    return name


def write_problem(output_dir, filename, uuid, climb, stat):
    output = {
        'uuid': uuid,
        'name': climb['name'],
        'frames': climb['frames'],
        'holds': parse_frames(climb['frames']),
        'original_angle': climb.get('original_angle') or climb.get('angle'),
        'is_nomatch': climb['is_nomatch'],
        'layout_id': climb['layout_id'],
        'angle': stat['angle'],
        'difficulty_average': stat.get('difficulty_average'),
        'quality_average': stat.get('quality_average'),
        'ascensionist_count': stat.get('ascensionist_count'),
        'benchmark_difficulty': stat.get('benchmark_difficulty'),
    }
    with open(os.path.join(output_dir, f"{filename}.json"), 'w') as f:
        json.dump(output, f, indent=2)


if args.fetch_file:
    # -------------------------------------------------------------------------
    # Incremental mode
    #
    # Climb definitions come from existing problem files in output_dir — no
    # sync files needed. Only the stats (and any brand-new climb definitions)
    # in the specified fetch file are processed. Existing problem files are
    # overwritten in-place when their stats update; new ones are created.
    # -------------------------------------------------------------------------

    # Build lookup tables from existing problem files
    all_climbs = {}          # uuid -> climb dict
    uuid_angle_to_file = {}  # (uuid, angle) -> filename stem (no .json)
    seen_filenames = set()   # all existing stems, for collision avoidance on new files

    for filepath in glob.glob(os.path.join(output_dir, '*.json')):
        stem = os.path.splitext(os.path.basename(filepath))[0]
        seen_filenames.add(stem)
        with open(filepath) as f:
            data = json.load(f)
        if data.get('layout_id') == args.layout_id:
            uuid_angle_to_file[(data['uuid'], data['angle'])] = stem
            if data['uuid'] not in all_climbs:
                all_climbs[data['uuid']] = data  # existing problem files have all needed fields

    # Load new data from the fetch file
    with open(args.fetch_file) as f:
        fetch = json.load(f)

    for climb in fetch.get('climbs', []):
        if climb.get('layout_id') == args.layout_id:
            all_climbs[climb['uuid']] = climb  # new climb definitions override

    new_stats = {}
    for stat in fetch.get('climb_stats', []):
        new_stats.setdefault(stat['climb_uuid'], []).append(stat)

    written = 0
    skipped = 0
    for uuid, stats_list in new_stats.items():
        climb = all_climbs.get(uuid)
        if not climb:
            skipped += 1
            continue  # stat for a different layout or a climb we've never seen

        sanitized = sanitize_name(climb['name']) or uuid[:8]

        for stat in stats_list:
            angle = stat['angle']
            existing = uuid_angle_to_file.get((uuid, angle))
            if existing:
                filename = existing  # overwrite in-place, preserving the original filename
            else:
                base = f"{sanitized}_{angle}"
                filename = base if base not in seen_filenames else f"{base}_{uuid[:8]}"
                seen_filenames.add(filename)
                uuid_angle_to_file[(uuid, angle)] = filename

            write_problem(output_dir, filename, uuid, climb, stat)
            written += 1

    print(f"Done. {written} files written/updated in {output_dir}/")
    print(f"  New climbs in fetch: {sum(1 for c in fetch.get('climbs', []) if c.get('layout_id') == args.layout_id)}")
    print(f"  Stats processed: {sum(len(v) for v in new_stats.values())}")
    print(f"  Stats skipped (unknown layout): {skipped}")

else:
    # -------------------------------------------------------------------------
    # Full rebuild mode
    #
    # Reads all sync_files/fetch*.json from scratch. Use this when setting up
    # a fresh clone that has the sync files available.
    # -------------------------------------------------------------------------

    all_climbs = {}
    all_stats = {}

    for filepath in sorted(glob.glob(os.path.join(script_dir, 'sync_files/fetch*.json'))):
        with open(filepath) as f:
            data = json.load(f)
        for climb in data.get('climbs', []):
            if climb.get('layout_id') == args.layout_id:
                all_climbs[climb['uuid']] = climb
        for stat in data.get('climb_stats', []):
            all_stats.setdefault(stat['climb_uuid'], []).append(stat)

    seen_filenames = set()
    written = 0

    for uuid, climb in all_climbs.items():
        stats_list = all_stats.get(uuid, [])
        if not stats_list:
            continue

        sanitized = sanitize_name(climb['name']) or uuid[:8]

        for stat in stats_list:
            angle = stat['angle']
            base = f"{sanitized}_{angle}"
            filename = base if base not in seen_filenames else f"{base}_{uuid[:8]}"
            seen_filenames.add(filename)
            write_problem(output_dir, filename, uuid, climb, stat)
            written += 1

    print(f"Done. {written} files written to {output_dir}/")
    print(f"  Climbs found (layout {args.layout_id}): {len(all_climbs)}")
    print(f"  Climbs skipped (no stats): {len(all_climbs) - sum(1 for u in all_climbs if u in all_stats)}")
