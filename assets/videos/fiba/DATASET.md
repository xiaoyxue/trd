<!--
trd vendoring note (issue #110)
===============================
Only `per_frame_KVP_cube_best.parquet` + `per_frame_KVP_cube_best_schema.json` are
vendored into this repo — pure numeric camera calibration (K, floor quads,
homographies) for the two most-accurate methods (2VP + 1circle), no imagery. They
are all the FIBA court AR demo needs (see the repo README "FIBA court AR demo" +
`scripts/fiba_perception_to_arrow.py`).

NOT vendored (copyrighted broadcast footage / bulky / not needed by the demo):
`shot_0001.mp4`, `2024-Olympic-Basketball-1.mp4`, `inputs/`, the other
`per_frame_KVP_cube*.parquet` variants, `ba_focals.parquet`, and the `gen_*.py`
reproduction scripts. The demo extracts background frames from your own local copy
of `shot_0001.mp4` at render time into the gitignored `output/`. The full file
list below documents the *upstream* dataset for reference.
-->

# FIBA-shot1 — per-frame camera K / VP / cube dataset

Per-frame camera calibration + AR-cube geometry for a single continuous
broadcast shot of the **2024 Paris Olympic basketball final (France vs USA)**.
Reproduces the [`nba-short`](../nba-short/README.md) methodology
(issue `Hong-Xiang/VideoAnalysis#1133`) with a full **multi-method K** +
**held-out accuracy ranking** + **circle-IAC** on top.

> **Camera regime.** Rotation + zoom only (negligible translation, `t≈0`), so
> one global homography aligns the whole static scene between frames. Verified:
> per-frame homography residual p50 ≈ 0.49 px; the back-projection contract
> `H⁻¹·vp[p]` is constant across frames to ~1e-15 (see §5 of the methodology).

---

## Source

| | |
|---|---|
| clip | `shot_0001.mp4` — 1920×1080, **24 fps**, h264, **288 frames** (12.0 s). Transcoded from a 4K/60 original. |
| event | 2024 Paris Olympics, basketball gold-medal game (France vs USA) |
| shot | single continuous shot (no scene cuts) |
| `present_index` | 0-based frame index into `shot_0001.mp4` (time = `present_index / 24` s) |

`meta_shot0001.parquet`: `{shot, W, H, start_pi=0, end_pi=287, n_frames=288, start_sec, end_sec}`.

---

## The camera model — K has one unknown

All methods assume square pixels, zero skew, principal point at the frame centre:

```
K = | f  0  cx |     cx = W/2 = 960
    | 0  f  cy |     cy = H/2 = 540
    | 0  0  1  |
```

So "estimate K" ≡ "estimate one scalar `f`". `f` is **constant per (method)**
across the whole shot (the camera zooms, but per-frame zoom is not reliably
observable — see Caveats). Every **per-frame** quantity (`vp_*`, `ad_quad`,
`cube_verts`) is the fixed reference geometry transported by the tracked
homography `h_ref_to_i`.

---

## Focal-estimation methods (stored side-by-side)

Each method pins `f` from an **independent** geometric constraint; results are
published together in the `method` column (no fusion). Shared across all
methods: the two orthogonal court directions from the **paint (key) rectangle**
edges → the cube's rotation basis `r1, r2, r3`.

| method | f (px) | FOV | principle | constraint |
|---|---|---|---|---|
| `2VP_4510`     | **4510** | 24.0° | two orthogonal paint-edge VPs        | `v₁ᵀ ω v₂ = 0` |
| `1circle_4252` | **4252** | 25.4° | imaged court circle (BEV semicircle) | `I'ᵀ ω I' = 0` (circular points) |
| `Zhang_5257`   | 5257 | 20.7° | known-size paint rectangle (4.9×5.8 m) | Zhang 2-constraint |
| `quad_4510`    | 4510 | 24.0° | paint self-focal (= 2VP)             | same as 2VP (sanity label) |
| `BA_2397`      | 2397 | 43.7° | motion rot+zoom self-calibration     | `H = K R K⁻¹` reprojection |

`ω = K⁻ᵀK⁻¹` is the image of the absolute conic; with pp fixed its only unknown
is `f`. See [`../K-ESTIMATION-METHODOLOGY.md`](../K-ESTIMATION-METHODOLOGY.md)
for the full derivation (incl. why a circle ⇒ `r₁ᵀr₁=r₂ᵀr₂, r₁ᵀr₂=0`, the same
two constraints as a known rectangle but **without** needing the true size).

### Which K is most accurate? (held-out cross-validation)

No ground-truth focal exists. Each K is judged by **held-out geometric
consistency** (`gen_K_eval.py`): metrically rectify the court with that K's
floor normal and measure errors on features **not used** to derive it.

| rank | method | f | avg held-out rank | key held-out evidence |
|---|---|---|---|---|
| 🥇 **1** | **2VP**     | **4510** | 1.00 | rectified free-throw circle roundness err **6.8 %** (best) |
| 🥈 2 | **1circle** | 4252 | 1.50 | paint-edge orthogonality err **1.6°** (best) |
| 🥉 3 | Zhang       | 5257 | 2.00 | — |
| 4 | BA          | 2397 | 3.00 | roundness err **64 %**, orth err 16° (worst — zoom-degraded) |

**Verdict:** the trustworthy K is **2VP (f=4510) ≈ 1circle (f=4252)** — three
independent principles (lines / conic / rectangle) converge on **f ≈ 4250–5250**.
`BA` is the outlier because zoom breaks the single-K motion model (reprojection
residual ≈ 48 px). This is the **inverse of nba-short**, where BA was trusted
and 2VP was degenerate (NBA sidelines VP at infinity) — a clean demonstration of
why multiple methods are stored: the best method flips with the footage.

---

## Files

| File | Frames | Methods | What |
|---|---|---|---|
| `per_frame_KVP_cube.parquet` | 288 (all) | BA/2VP/Zhang/quad | base multi-method dataset |
| `per_frame_KVP_cube_trimmed.parquet` | 222 | same | tracked frames only |
| `per_frame_KVP_cube_circle.parquet` | 288 (all) | + `1circle` | base + circle method |
| `per_frame_KVP_cube_circle_trimmed.parquet` | 222 | + `1circle` | circle + tracked only |
| **`per_frame_KVP_cube_best.parquet`** | **288 (all)** | **2VP + 1circle** | **the two most-accurate K's; invalid frames keep the frame id only (geometry = null)** |
| `ba_focals.parquet` | — | BA | pooled motion-BA focal (nba-short schema) |
| `*_schema.json` | — | — | machine-readable column dictionaries (aligned 1:1 with each parquet) |

**"trimmed" refers to FRAMES, not to the K.** Every parquet stores multiple
methods in the `method` column; the *most accurate K* is a **method filter**
(`method == '2VP_4510'`), orthogonal to the frame trimming.

### `per_frame_KVP_cube_best.parquet` (recommended)

- **576 rows = 2 methods (2VP + 1circle) × 288 frames.**
- **Tracked frames (222, `tracked=True`)**: full geometry (K, VPs, ad_quad,
  cube_verts, H).
- **Invalid frames (66, `tracked=False`)**: **only the frame identity is kept**
  (`shot, method, present_index, W, H, cx, cy, tracked, visibility`); all
  geometry columns (`f_ref_px, f_frame_px, tilt_deg, K, vp_*, has_ad_quad,
  n_corners_*, ad_quad, cube_verts, h_ref_to_i`) are **null**.

---

## Schema (grain: one row per `method` × `present_index`)

| column | type | meaning |
|---|---|---|
| `shot` | int | shot id (1) |
| `method` | str | focal method + px, e.g. `2VP_4510` / `1circle_4252` |
| `present_index` | int | 0-based frame index into `shot_0001.mp4` |
| `f_ref_px` | double | method focal px (const per method) — *null on invalid frames* |
| `f_frame_px` | double | == `f_ref_px` (schema stability) |
| `tilt_deg` | double | floor-normal tilt |
| `cx, cy` | double | principal point = (W/2, H/2) = (960, 540) |
| `W, H` | int | 1920, 1080 |
| `K` | double[9] | 3×3 row-major `[f,0,cx, 0,f,cy, 0,0,1]` |
| `vp_len` | double[3] | cube X-axis VP (court length `r1`), homogeneous `[x,y,w]` this frame |
| `vp_wid` | double[3] | cube Y-axis VP (court width `r2`), homogeneous |
| `vp_up`  | double[3] | cube Z-axis VP (floor normal `r3`), homogeneous (`w≈0` ⇒ at ∞) |
| `has_ad_quad` | bool | ad_quad valid (True on tracked frames) |
| `n_corners_in` | int | quad corners inside the frame (0..4) |
| `n_corners_out` | int | `4 - n_corners_in` |
| `tracked` | bool | tracking valid (≥3 corners in-frame). False ⇒ ended (≥2 corners out) |
| `visibility` | str | `tracked` or `ended` |
| `ad_quad` | double[8] | cube base ring UL,UR,LR,LL (the tracked ad surface), px |
| `cube_verts` | double[16] | 8 verts `[b0..b3, t0..t3]`; base = ad_quad, top extruded up |
| `h_ref_to_i` | double[9] | 3×3 row-major ref→frame homography (hybrid direct/chained) |

**VP homogeneous divide:** `x,y = vp[:2]/vp[2]` only when `|vp[2]| ≫ 0`.

**Vertex order:** `b0=UL, b1=UR, b2=LR, b3=LL`; `t_k` = `b_k` extruded along the
floor normal. Edges: bottom `0-1-2-3-0`, top `4-5-6-7-4`, verticals `0-4,1-5,2-6,3-7`.

---

## Tracking + visibility policy

- **Reference frame** = the middle frame (144): minimises the max camera
  rotation/zoom to any frame ⇒ most accurate homographies.
- **Homography** = RAFT dense flow + forward/backward consistency + `USAC_MAGSAC`.
  **Hybrid**: direct ref→i where strong (231 frames), else **chained**
  `H(ref→i) = ∏ H(i-1→i)` (57 frames) so corners keep tracking off-screen —
  this is court/mosaic stitching (t≈0 ⇒ one homography for the whole scene).
- **Stop rule**: tracking is valid only while **≥3 quad corners are in-frame**;
  once **2 corners leave**, the track **ends** (`tracked=False`) and drawing
  stops. Frames 0–221 track; 222–287 ended.

---

## Loading

```python
import pandas as pd, numpy as np
df = pd.read_parquet("per_frame_KVP_cube_best.parquet")

# most-accurate single K (2VP), tracked frames with full geometry:
best = df[(df.method == "2VP_4510") & df.tracked]
row  = best.iloc[0]
K     = np.array(row.K).reshape(3, 3)
verts = np.array(row.cube_verts).reshape(8, 2)   # [b0..b3, t0..t3]
adq   = np.array(row.ad_quad).reshape(4, 2)       # UL,UR,LR,LL
H     = np.array(row.h_ref_to_i).reshape(3, 3)    # ref → this frame

# cross-validation band (2VP + 1circle both kept):
band = df[df.tracked]

# full timeline incl. invalid frames (geometry null) for continuous playback:
all_2vp = df[df.method == "2VP_4510"]             # 288 rows
```

---

## Reproduction scripts (`fiba-shot1/`)

| script | stage |
|---|---|
| `gen_homography_chained.py` | per-frame hybrid direct/chained homography |
| `gen_track.py` | H-propagate the paint quad → `final.parquet` |
| `quad_drawer.py --role paint\|adquad` | annotate the paint / ad quad (browser canvas) |
| `gen_K.py` | 3-method focal + confidence vote (per-shot probe) |
| `gen_bev_circle.py` | build BEV, detect/annotate circle, circle-IAC focal |
| `circle_drawer.py --image bev.png` | annotate a circle on the clean top-down BEV |
| `gen_dataset.py --out-name …` | assemble the multi-method parquet + schema + visibility |
| `gen_K_eval.py` | held-out geometric accuracy ranking → `K_eval.json` |
| `restream_labeled.py` | rerun viz (5 cubes, frame#, visibility gate, frame-clipped) |
| `S18 cube math` | reused from `~/data/case-nba/work/floor_cal/s18_rrd_separate.py` |

---

## Viewing (Rerun .rrd)

```bash
# from a VideoAnalysis checkout:
nix run .#rrd-viewer inputs/shot_0001/labeled_shot_0001.rrd
```

App `fiba-labeled`, recording `shot_0001`: select the `frame` timeline, scrub.
Colour → K: 🟡 1circle(4252) / 🟢 2VP(4510) / 🔵 Zhang(5257) / 🔴 BA(2397, outlier).
Cube edges are clipped to the frame (constant view size); cubes disappear when
the ad surface leaves the frame (video stays continuous).

---

## Caveats

1. **K is per-method constant across frames** — the calibrated focal. Genuine
   per-frame variation is carried by `vp_*`, `cube_verts`, `ad_quad`,
   `h_ref_to_i` (all via H). A real zoom curve would need a per-frame focal
   (not included).
2. **`aspect` metric is soft** — the paint quad was hand-annotated (not pixel-
   exact), so the 4.9:5.8 target is approximate; the **roundness** metric (a
   circle is a circle regardless of annotation precision) is the stronger
   held-out signal.
3. **BA is zoom-degraded here** — do not use `BA_2397`; prefer `2VP_4510` /
   `1circle_4252`.
4. **Circle from a semicircle** — the free-throw / restricted-area arc is ~180°;
   geometrically sufficient to fit the conic, but treat 1circle as a strong
   *corroborating* vote, not an independent ground truth.
