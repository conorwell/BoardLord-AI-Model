import json
import subprocess
import os

TOKEN = 'f9a85d71e24eb0b6862601a72ba1fe4ecc938543'

# Static params (keep at their current timestamps)
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

def encode_date(d):
    return d.replace(' ', '+').replace(':', '%3A')

def build_body(climbs_ts, climb_stats_ts):
    parts = []
    for k, v in STATIC_PARAMS.items():
        parts.append(f"{k}={encode_date(v)}")
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

os.makedirs('sync_files', exist_ok=True)

climbs_ts     = '2010-01-01 00:00:00.000000'
climb_stats_ts = '2010-01-01 00:00:00.000000'
batch = 1

while True:
    print(f"Batch {batch:03d} | climbs >= {climbs_ts} | climb_stats >= {climb_stats_ts}")
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
        print("No new data returned, done!")
        break

    filepath = f"sync_files/fetch_{batch:03d}.json"
    with open(filepath, 'w') as f:
        json.dump(data, f)

    print(f"  Got {len(climbs)} climbs, {len(climb_stats)} climb_stats -> {filepath}")

    if climbs:
        latest = max(c['created_at'] for c in climbs)
        if latest > climbs_ts:
            climbs_ts = latest
        elif len(climbs) == 2000:
            # Stuck on a bulk-import timestamp — nudge forward 1 day
            from datetime import datetime, timedelta
            nudged = datetime.strptime(climbs_ts, '%Y-%m-%d %H:%M:%S.%f') + timedelta(days=1)
            climbs_ts = nudged.strftime('%Y-%m-%d %H:%M:%S.%f')
            print(f"  [stuck] nudging climbs timestamp to {climbs_ts}")

    if climb_stats:
        latest = max(s['created_at'] for s in climb_stats)
        if latest > climb_stats_ts:
            climb_stats_ts = latest
        elif len(climb_stats) == 2000:
            from datetime import datetime, timedelta
            nudged = datetime.strptime(climb_stats_ts, '%Y-%m-%d %H:%M:%S.%f') + timedelta(days=1)
            climb_stats_ts = nudged.strftime('%Y-%m-%d %H:%M:%S.%f')
            print(f"  [stuck] nudging climb_stats timestamp to {climb_stats_ts}")

    batch += 1
