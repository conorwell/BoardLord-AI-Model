import json
import matplotlib.pyplot as plt

with open('placements.json') as f:
    placements = json.load(f)
with open('holes.json') as f:
    holes = json.load(f)

hole_lookup = {h['id']: h for h in holes}
layout10 = [p for p in placements if p['layout_id'] == 10]

# position_id -> (x, y, hole_id)
positions = []
for p in layout10:
    hole = hole_lookup[p['hole_id']]
    positions.append({'id': p['id'], 'x': hole['x'], 'y': hole['y']})

CORNERS = {679: 'TL', 672: 'BL', 794: 'BR', 801: 'TR'}

fig, ax = plt.subplots(figsize=(22, 22))
ax.set_facecolor('#555555')
fig.patch.set_facecolor('#555555')

for pos in positions:
    px, py = pos['x'], pos['y']
    pid = pos['id']

    if pid <= 545:
        hold_color = '#C8A86B'   # tan - wood
        text_color = '#3B2000'
    else:
        hold_color = '#333333'   # dark - plastic
        text_color = '#FF3333'

    ms = 16 if pid in CORNERS else 10
    ax.plot(px, py, 'o', color=hold_color, markersize=ms, zorder=1)

    label = f"{pid}\n{CORNERS[pid]}" if pid in CORNERS else str(pid)
    ax.text(px, py, label, color=text_color, fontsize=5.5,
            ha='center', va='center', zorder=2, fontweight='bold')

ax.set_xlim(-70, 70)
ax.set_ylim(-2, 146)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.savefig('hole_map.png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
print("Saved hole_map.png")
