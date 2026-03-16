import json
import glob
import sys
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('query', nargs='*', help='Name search terms')
parser.add_argument('--dir', default='TB2_Mirror_Problems',
                    help='Parsed problems directory (default: TB2_Mirror_Problems)')
args = parser.parse_args()

query = ' '.join(args.query).lower()

for fname in sorted(os.listdir(args.dir)):
    if not fname.endswith('.json'):
        continue
    with open(os.path.join(args.dir, fname)) as f:
        data = json.load(f)
    if query in data.get('name', '').lower():
        diff = data.get('difficulty_average')
        angle = data.get('angle')
        ascents = data.get('ascensionist_count')
        print(f"{data['name']!r:40s}  diff={diff}  angle={angle}  ascents={ascents}")
