#!/usr/bin/env python3
"""
update.py — Fetch new Tension Board data since the last sync and rebuild training files.

Steps:
  1. Read sync_state.json for the last-synced timestamps and next file number
  2. Paginate the API forward from those timestamps, accumulating into one new fetch file
  3. Write updated sync_state.json
  4. Re-run parse_climbs.py for both layouts (Mirror layout 10, Spray layout 11)

sync_state.json is the source of truth for sync progress. The raw fetch files in
sync_files/ are archival and can be deleted without breaking future updates.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta

TOKEN = 'f9a85d71e24eb0b6862601a72ba1fe4ecc938543'

SYNC_STATE_PATH = 'sync_state.json'
EPOCH = '1970-01-01 00:00:00.000000'

STATIC_PARAMS = {
    'products':                    '2024-06-22 23:01:25.255758',
    'product_sizes':               '2026-02-28 16:49:15.426179',
    'holes':                       '2026-02-28 16:49:15.401075',
    'leds':                        '2026-02-28 16:49:18.035293',
    'products_angles':             '2024-06-22 23:01:25.255758',
    'layouts':                     '2025-08-22 00:38:52.971578',
    'product_sizes_layouts_sets':  '2026-02-28 16:49:18.048272',
    'placements':                  '2024-06-22 23:01:25.255758',
    'sets':                        '2024-06-22 23:01:25.255758',
    'placement_roles':             '2025-08-23 05:22:16.042123',
    'beta_links':                  '2026-01-09 02:29:20.891517',
    'attempts':                    '2024-06-22 23:01:25.255758',
    'kits':                        '2026-01-22 20:08:48.978201',
    'users':                       '2025-07-30 03:39:41.085853',
    'walls':                       '1970-01-01 00:00:00.000000',
    'draft_climbs':                '1970-01-01 00:00:00.000000',
    'ascents':                     '1970-01-01 00:00:00.000000',
    'bids':                        '1970-01-01 00:00:00.000000',
    'tags':                        '1970-01-01 00:00:00.000000',
    'circuits':                    '1970-01-01 00:00:00.000000',
}


def load_sync_state():
    """Return (climbs_ts, climb_stats_ts, next_file_num) from sync_state.json."""
    if os.path.exists(SYNC_STATE_PATH):
        with open(SYNC_STATE_PATH) as f:
            state = json.load(f)
        return state['climbs_ts'], state['climb_stats_ts'], state['next_file_num']
    return EPOCH, EPOCH, 1


def save_sync_state(climbs_ts, climb_stats_ts, next_file_num):
    with open(SYNC_STATE_PATH, 'w') as f:
        json.dump({
            'climbs_ts':     climbs_ts,
            'climb_stats_ts': climb_stats_ts,
            'next_file_num': next_file_num,
        }, f, indent=2)


def encode_date(d):
    return d.replace(' ', '+').replace(':', '%3A')


def build_body(climbs_ts, climb_stats_ts):
    parts = [f"{k}={encode_date(v)}" for k, v in STATIC_PARAMS.items()]
    parts.append(f"climbs={encode_date(climbs_ts)}")
    parts.append(f"climb_stats={encode_date(climb_stats_ts)}")
    return '&'.join(parts)


def run_curl(body):
    cmd = [
        'curl', '-s',
        '-H', 'Accept: application/json',
        '-H', 'Content-Type: application/x-www-form-urlencoded',
        '-H', 'Connection: keep-alive',
        '-H', f'Cookie: token={TOKEN}',
        '-H', 'Accept-Language: en-US,en;q=0.9',
        '--compressed',
        '-H', 'User-Agent: Tension%20Board/335 CFNetwork/3826.500.131 Darwin/24.5.0',
        '-X', 'POST',
        'https://tensionboardapp2.com/sync',
        '-d', body,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout


def fetch_new(climbs_ts, climb_stats_ts, next_file_num):
    """Paginate API from given timestamps, accumulate into one file.

    Returns (True, final_climbs_ts, final_climb_stats_ts) if new data was written,
    (False, climbs_ts, climb_stats_ts) if nothing was new.
    """
    os.makedirs('sync_files', exist_ok=True)

    all_climbs = []
    all_climb_stats = []
    page = 1

    while True:
        print(f"  Page {page} | climbs >= {climbs_ts} | climb_stats >= {climb_stats_ts}")
        body = build_body(climbs_ts, climb_stats_ts)
        raw = run_curl(body)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            print("Failed to parse response, stopping.")
            print(raw[:500])
            break

        climbs      = data.get('climbs', [])
        climb_stats = data.get('climb_stats', [])

        if not climbs and not climb_stats:
            print("  No new data — up to date.")
            break

        print(f"    +{len(climbs)} climbs, +{len(climb_stats)} climb_stats")
        all_climbs.extend(climbs)
        all_climb_stats.extend(climb_stats)

        climbs_done = False
        if climbs:
            latest = max(c['created_at'] for c in climbs)
            if latest > climbs_ts:
                climbs_ts = latest
            elif len(climbs) == 2000:
                nudged = datetime.strptime(climbs_ts, '%Y-%m-%d %H:%M:%S.%f') + timedelta(days=1)
                climbs_ts = nudged.strftime('%Y-%m-%d %H:%M:%S.%f')
                print(f"  [stuck] nudging climbs timestamp to {climbs_ts}")
            else:
                climbs_done = True

        stats_done = False
        if climb_stats:
            latest = max(s['created_at'] for s in climb_stats)
            if latest > climb_stats_ts:
                climb_stats_ts = latest
            elif len(climb_stats) == 2000:
                nudged = datetime.strptime(climb_stats_ts, '%Y-%m-%d %H:%M:%S.%f') + timedelta(days=1)
                climb_stats_ts = nudged.strftime('%Y-%m-%d %H:%M:%S.%f')
                print(f"  [stuck] nudging climb_stats timestamp to {climb_stats_ts}")
            else:
                stats_done = True

        if climbs_done and stats_done:
            print("  Timestamps stopped advancing — done.")
            break

        page += 1

    if not all_climbs and not all_climb_stats:
        return False, climbs_ts, climb_stats_ts

    filepath = f"sync_files/fetch_{next_file_num:03d}.json"
    with open(filepath, 'w') as f:
        json.dump({'climbs': all_climbs, 'climb_stats': all_climb_stats}, f)
    print(f"\n  Wrote {len(all_climbs)} climbs + {len(all_climb_stats)} climb_stats -> {filepath}")
    return True, climbs_ts, climb_stats_ts


def run_parse(layout_id, output_dir, fetch_file):
    print(f"Parsing layout {layout_id} -> {output_dir}/")
    subprocess.run(
        [sys.executable, 'parse_climbs.py',
         '--layout-id', str(layout_id),
         '--output-dir', output_dir,
         '--fetch-file', fetch_file],
        check=True
    )


def main():
    climbs_ts, climb_stats_ts, next_file_num = load_sync_state()
    print(f"Last synced timestamps:")
    print(f"  climbs:      {climbs_ts}")
    print(f"  climb_stats: {climb_stats_ts}")
    print(f"  Next file:   fetch_{next_file_num:03d}.json\n")

    fetched, climbs_ts, climb_stats_ts = fetch_new(climbs_ts, climb_stats_ts, next_file_num)

    if not fetched:
        print("Nothing to parse.")
        return

    fetch_file = f"sync_files/fetch_{next_file_num:03d}.json"
    print("Rebuilding training data...")
    run_parse(10, 'TB2_Mirror_Problems', fetch_file)
    run_parse(11, 'TB2_Spray_Problems', fetch_file)

    save_sync_state(climbs_ts, climb_stats_ts, next_file_num + 1)
    print("\nsync_state.json updated.")
    print("Done.")


if __name__ == '__main__':
    main()
