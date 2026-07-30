#!/usr/bin/env python3
"""Before/after divider intro. The split rule holds on EVERY frame:
   left of the white line = ORIGINAL clean plate, right = a RENDER.
Which render fills the right side depends on the leg.

Sources (all WxH):
  orig   clean original plate
  normal PBR render (clean cans)
  gizmo  PBR render + placement quad (grid + aabb + axes) + can aabb/axes

Timeline (line x-position; right-side source):
  leg1  x: 0 -> W      right = NORMAL   L->R reveal, ends fully ORIGINAL
  hold  x: W           right = NORMAL   beat on the clean original
  legA  x: W -> 0      right = GIZMO    R->L, reveals gizmo render (ends full gizmo)
  legB  x: 0 -> W      right = GIZMO    L->R, wipes back to original
  legC  x: W -> off    right = NORMAL   R->L, reveals clean render, ends fully
                                        NORMAL -> flows into the video

Adjacent legs share their end/start fill, so the whole intro is one continuous wipe.
legs A/B/C each take n_seg frames (go-back phase = 3*n_seg).

Usage: make_scan_intro2.py <orig> <normal> <gizmo> <out_dir>
                           [n_scan] [n_hold] [n_seg] [line_w]
"""
import sys, os
from PIL import Image, ImageDraw

orig_p, norm_p, giz_p, out_dir = sys.argv[1:5]
n_scan = int(sys.argv[5]) if len(sys.argv) > 5 else 60
n_hold = int(sys.argv[6]) if len(sys.argv) > 6 else 8
n_seg  = int(sys.argv[7]) if len(sys.argv) > 7 else 36   # frames per go-back leg (36 = 1.5s -> 4.5s)
line_w = int(sys.argv[8]) if len(sys.argv) > 8 else 6

os.makedirs(out_dir, exist_ok=True)
orig = Image.open(orig_p).convert("RGB")
norm = Image.open(norm_p).convert("RGB")
giz  = Image.open(giz_p).convert("RGB")
W, H = orig.size
if norm.size != (W, H): norm = norm.resize((W, H))
if giz.size  != (W, H): giz  = giz.resize((W, H))

hw = line_w // 2
WHITE = (255, 255, 255)

def smooth(t):
    return t * t * (3 - 2 * t)

def frame_at(x, right):
    """left of x = original, right of x = `right` render; white divider at x."""
    im = right.copy()
    xi = int(round(x))
    if xi > 0:
        im.paste(orig.crop((0, 0, min(xi, W), H)), (0, 0))
    if -hw <= x <= W + hw:
        ImageDraw.Draw(im).rectangle([x - hw, 0, x + hw, H], fill=WHITE)
    return im

seq = []                                       # (x, right_img)
for j in range(n_scan):                        # leg1: 0 -> W, right = normal
    seq.append((W * smooth((j + 1) / n_scan), norm))
seq += [(W, norm)] * n_hold                     # hold on clean original

def leg(a, b, right):
    return [(a + (b - a) * smooth((j + 1) / n_seg), right) for j in range(n_seg)]

seq += leg(W, 0, giz)                            # legA: R->L, right = gizmo
seq += leg(0, W, giz)                            # legB: L->R, right = gizmo
seq += leg(W, -(hw + 2), norm)                   # legC: R->L off, right = normal (clean)

for n, (x, right) in enumerate(seq):
    frame_at(x, right).save(os.path.join(out_dir, f"f{n:04d}.png"))

print(f"wrote {len(seq)} frames ({W}x{H})")
