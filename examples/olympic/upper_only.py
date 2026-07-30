#!/usr/bin/env python3
"""Strip the *lower* (sideline) can from the two-can `can_duo` scene, leaving
the upper (near-basket) can only.

The duo base places two cans per frame (each = one shadow draw + one lit draw):
an upper can near the hoop (farther from camera, more-negative view-space z) and
a lower can on the sideline (closer to camera, larger z) that overlaps a player.
For the "reveal_scan" videos the lower can must disappear partway through, so we
render an upper-only pass and composite it over the tail (see composite_bases.py).

Rule: within each 4-draw frame keep the two draws with the *smaller* (more
negative) translation-z (the upper can + its shadow); drop the two with the
larger z (the lower can + its shadow). Empty frames (both cans already gone in
the base animation) pass through unchanged.

Usage: upper_only.py <can_duo.jsonl> <out_upper.jsonl>
"""
import json
import sys


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    out = []
    for i, line in enumerate(open(src)):
        row = json.loads(line)
        draws = row.get("draws", [])
        if len(draws) == 4:
            # keep the two most-distant (upper can + shadow) by view-space z
            order = sorted(range(4), key=lambda k: draws[k]["model"][14])
            keep = sorted(order[:2])
            row = dict(row, draws=[draws[k] for k in keep])
        elif len(draws) not in (0,):
            raise SystemExit(f"frame {i}: expected 4 or 0 draws, got {len(draws)}")
        out.append(row)
    with open(dst, "w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")
    kept = sum(1 for r in out if r.get("draws"))
    print(f"upper_only: wrote {dst} ({len(out)} frames, {kept} with the upper can)")


if __name__ == "__main__":
    main()
