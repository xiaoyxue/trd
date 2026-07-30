#!/usr/bin/env python3
"""Composite the two "base" tails for the reveal_scan videos so the lower
(sideline) can disappears at a fixed frame and STAYS gone.

Inputs are two already-rendered frame sequences of the same length N:
  clean_dir/c%04d.png   the duo base   (both cans)
  upper_dir/u%04d.png   the upper-only base (lower can removed)

Two outputs are produced, each N frames:
  fade_dir/f%04d.png    0..CUT-2 clean | CUT-1..CUT+? crossfade | rest upper
  cut_dir/f%04d.png     0..CUT-1 clean |                           rest upper

The lower can vanishes at frame CUT (default 91 == 3.8 s into the 12 s base,
i.e. 12 + 3.8 = 15.8 s into the final video). The fade base ramps the lower-can
opacity 1 -> 0 across the FADE frames immediately before CUT; the cut base is a
hard cut. Pass-through frames are hard-linked (fast, no re-encode); only the
handful of crossfade frames are blended pixel-wise.

Usage: composite_bases.py <clean_dir> <upper_dir> <fade_dir> <cut_dir>
                          [N] [CUT] [FADE]
"""
import os
import sys

from PIL import Image
import numpy as np

clean_d, upper_d, fade_d, cut_d = sys.argv[1:5]
N = int(sys.argv[5]) if len(sys.argv) > 5 else 288
CUT = int(sys.argv[6]) if len(sys.argv) > 6 else 91   # first frame with lower can gone
FADE = int(sys.argv[7]) if len(sys.argv) > 7 else 6   # crossfade length before CUT
os.makedirs(fade_d, exist_ok=True)
os.makedirs(cut_d, exist_ok=True)


def link(src: str, dst: str) -> None:
    if os.path.exists(dst):
        os.remove(dst)
    os.link(src, dst)


fade_start = CUT - FADE  # first crossfaded frame
blended = 0
for i in range(N):
    c = f"{clean_d}/c{i:04d}.png"
    u = f"{upper_d}/u{i:04d}.png"
    fout = f"{fade_d}/f{i:04d}.png"
    cout = f"{cut_d}/f{i:04d}.png"
    # ---- cut base: clean before CUT, upper from CUT ----
    link(c if i < CUT else u, cout)
    # ---- fade base: clean, crossfade window, then upper ----
    if i < fade_start:
        link(c, fout)
    elif i < CUT:
        alpha = (CUT - i) / (FADE + 1.0)  # lower-can opacity, 1 -> 0
        out = alpha * np.asarray(Image.open(c).convert("RGB"), np.float32) \
            + (1.0 - alpha) * np.asarray(Image.open(u).convert("RGB"), np.float32)
        Image.fromarray(np.clip(out, 0, 255).astype("uint8")).save(fout)
        blended += 1
    else:
        link(u, fout)
print(f"composite_bases: {N} frames, cut@{CUT}, {blended} crossfade frames "
      f"-> fade({fade_d}) + cut({cut_d})")
