# Tension Board API

## Overview

The Tension Board app syncs data from a single POST endpoint:

```
POST https://tensionboardapp2.com/sync
```

There is no public API documentation. All knowledge here was reverse-engineered from app traffic.

---

## Authentication

Authentication is via a session token passed as a cookie:

```
Cookie: token=<your_token>
```

To get your token, log into the Tension Board app and capture a sync request (e.g. via Charles Proxy or similar). The token is a 40-character hex string. It appears to be long-lived (does not expire quickly).

---

## Request Format

**Method:** POST
**Content-Type:** `application/x-www-form-urlencoded`

The body is a set of key=value pairs, one per data table. Each value is a timestamp — the server returns only records created/updated **after** that timestamp. Setting a timestamp to the epoch (`1970-01-01 00:00:00.000000`) fetches all records for that table.

Timestamps must be URL-encoded:
- spaces → `+`
- colons → `%3A`

### Required Headers

```
Accept: application/json
Content-Type: application/x-www-form-urlencoded
Connection: keep-alive
Cookie: token=<your_token>
Accept-Language: en-US,en;q=0.9
User-Agent: Tension%20Board/335 CFNetwork/3826.500.131 Darwin/24.5.0
```

The `--compressed` flag should be passed to curl (response is gzip-compressed).

### Example curl

```bash
curl -s \
  -H 'Accept: application/json' \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -H 'Connection: keep-alive' \
  -H 'Cookie: token=YOUR_TOKEN' \
  -H 'Accept-Language: en-US,en;q=0.9' \
  -H 'User-Agent: Tension%20Board/335 CFNetwork/3826.500.131 Darwin/24.5.0' \
  --compressed \
  -X POST \
  'https://tensionboardapp2.com/sync' \
  -d 'climbs=2024-01-01+00%3A00%3A00.000000&climb_stats=2024-01-01+00%3A00%3A00.000000&...'
```

---

## Data Tables

Each key in the POST body corresponds to a table. The value is the "since" timestamp.

### Static tables (fetch once with epoch timestamp)

| Table | Description |
|-------|-------------|
| `holes` | All physical hole positions on all boards (id, x, y, product_id) |
| `placements` | Maps placement_id → hole_id per layout. This is the critical join table. |
| `layouts` | Board layout definitions |
| `products` | Board product definitions |
| `product_sizes` | Board size variants |
| `product_sizes_layouts_sets` | Join table for layouts and hold sets |
| `sets` | Hold set definitions (e.g. wood set, plastic set) |
| `placement_roles` | Default role assignments for placements |
| `leds` | LED position mappings |
| `beta_links` | Video beta links for problems |
| `kits` | Kit definitions |
| `users` | User records |
| `products_angles` | Supported angles per product |

### Dynamic tables (paginate with timestamps)

| Table | Description |
|-------|-------------|
| `climbs` | Problem definitions (name, frames, layout_id, angle, uuid, etc.) |
| `climb_stats` | Per-problem statistics (difficulty_average, quality_average, ascensionist_count) |
| `ascents` | Individual ascent records |
| `attempts` | Attempt records |
| `circuits` | User-created circuit collections |
| `walls` | Wall configurations |
| `draft_climbs` | Draft/unpublished problems |
| `bids` | Bid records |
| `tags` | Tag assignments |

---

## Pagination

The API returns at most **2000 records** per request for `climbs` and `climb_stats`. To paginate:

1. Send request with timestamp T
2. Receive up to 2000 records
3. Set new timestamp = `max(created_at)` across returned records
4. Repeat until 0 records returned

**Edge case:** If all 2000 records share the same `created_at` (bulk import), the timestamp won't advance. Solution: nudge the timestamp forward by 1 day.

```python
if len(climbs) == 2000 and max_ts == prev_ts:
    nudge forward by timedelta(days=1)
```

See `fetch_all.py` for the complete paginated fetcher implementation.

---

## Response Format

JSON object with one key per requested table, each containing a list of records:

```json
{
  "climbs": [...],
  "climb_stats": [...],
  "holes": [...],
  ...
}
```

Tables with no new data are either absent or empty lists.

---

## Key Data Relationships

```
climb.frames  →  position_id (= placement.id)
placement.id  →  placement.hole_id
placement.hole_id  →  hole.id  →  hole.x, hole.y
```

**Critical:** The `position_id` values in climb `frames` strings are **placement IDs**, NOT hole IDs. You must join through the `placements` table to get physical coordinates. Querying `holes` directly by position_id will give wrong results.

---

## Layout IDs

| ID | Name |
|----|------|
| 9 | Tension Board 1 |
| 10 | TB2 Mirror |
| 11 | TB2 Spray |

---

## Frames String Format

Climb hold layouts are encoded as a compact string in the `frames` field:

```
p{position_id}r{role_id}p{position_id}r{role_id}...
```

Example: `p338r8p394r5p412r6p801r7`

| Role ID | Meaning |
|---------|---------|
| 5 | Start hold |
| 6 | Hand hold (mid) |
| 7 | Finish hold |
| 8 | Foot-only hold |

Feet are **never open** on the Tension Board — foot-only holds are always explicitly marked r8. Any hold not listed is unused.
