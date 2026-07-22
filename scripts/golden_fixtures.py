#!/usr/bin/env python3
"""Generate the small, self-contained Arrow IPC fixtures for the golden e2e
render test (issue #88).

Both fixtures are *derived from* the committed two-stage cornellbox placement
demo (``examples/frames.cornellbox.stage{1,2}.jsonl``, #77) but reduced to be
tiny, deterministic, and fully self-contained so the golden test needs **no**
external assets (no extracted video frames):

* **stage1** — the reconstructed placement quad only: mesh table ``[quad]`` +
  a few frames of the authored CV ``k`` + per-draw ``model`` (a wireframe quad).
* **stage2** — the mesh anchored on that quad: mesh table ``[bunny, quad]`` +
  the bunny texture + a few frames (bunny draw + wireframe quad draw).

Reductions vs. the full 125-frame demo:

* only a few representative frames are kept (``FRAME_INDICES``);
* the CV intrinsics ``k`` (baked for the 960x540 demo) are **rescaled** to the
  small golden resolution ``WIDTH x HEIGHT`` (``k`` is resolution-specific: fx,
  fy, cx, cy scale linearly with the render size);
* the ``0.0.5`` ``frame_path`` background reference is **dropped** — the golden
  test renders on a black background (``run_stream`` with no frame resolver),
  which is exactly what the CLI does without ``--frames-base``;
* the bound texture is downscaled (``TEXTURE_MAX_SIZE``).

The fixtures embed the mesh + texture + params, so ``run_stream`` reads them off
one byte stream with no file I/O. Regenerate with::

    python3 scripts/golden_fixtures.py

(needs ``uv`` on PATH — the pyarrow producers run via ``uv run --with``). After
regenerating the ``.arrow`` inputs, refresh the golden PNGs on a GPU box::

    TRD_UPDATE_GOLDENS=1 cargo test -p trd-core --test golden_render -- --ignored
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "crates" / "trd-core" / "tests" / "golden"

# Small, 16:9 golden resolution (960x540 / 3). Keep the aspect ratio of the
# authored demo so the rescaled CV `k` stays isotropic.
WIDTH = 320
HEIGHT = 180
SRC_WIDTH = 960
SRC_HEIGHT = 540

# A few representative frames out of the demo's 125 (first / middle / last).
FRAME_INDICES = [0, 62, 124]

# Downscale the bound albedo so the fixture stays small (longest side). The
# golden renders at 320x180, so a 128px albedo carries plenty of detail.
TEXTURE_MAX_SIZE = 128

BUNNY_OBJ = ROOT / "assets" / "meshes" / "bunny_with_texture" / "bunny.obj"
BUNNY_TEX = ROOT / "assets" / "meshes" / "bunny_with_texture" / "bunny_uv_map1.jpg"
STAGE1_JSONL = ROOT / "examples" / "frames.cornellbox.stage1.jsonl"
STAGE2_JSONL = ROOT / "examples" / "frames.cornellbox.stage2.jsonl"

# The canonical placement-quad overlay `render.sh --placement-quad` appends:
# origin-centred, extent 2 (corners +-1), cyan vertex colour baked in so the
# wireframe outline is cyan.
QUAD_OBJ_TEXT = """\
# canonical placement-quad overlay (golden fixture): centred, extent 2, corners +-1.
v -1 -1 0 0 1 1
v 1 -1 0 0 1 1
v 1 1 0 0 1 1
v -1 1 0 0 1 1
f 1 2 3
f 1 3 4
"""


def rescale_k(k: list[float]) -> list[float]:
    """Rescale CV intrinsics `k` (row-major [fx,0,0, 0,fy,0, cx,cy,1], baked for
    SRC_WIDTH x SRC_HEIGHT) to the golden WIDTH x HEIGHT."""
    sx = WIDTH / SRC_WIDTH
    sy = HEIGHT / SRC_HEIGHT
    k = list(k)
    k[0] *= sx  # fx
    k[4] *= sy  # fy
    k[6] *= sx  # cx
    k[7] *= sy  # cy
    return k


def reduced_jsonl(src: Path) -> str:
    """Subset to FRAME_INDICES, rescale `k`, drop the `frame_path` background ref."""
    lines = src.read_text().splitlines()
    out = []
    for i in FRAME_INDICES:
        row = json.loads(lines[i])
        if "k" in row:
            row["k"] = rescale_k(row["k"])
        row.pop("frame_path", None)
        row.pop("frame_url", None)
        out.append(json.dumps(row))
    return "\n".join(out) + "\n"


def run(cmd: list[str]) -> bytes:
    """Run a producer, returning its stdout bytes (Arrow IPC stream)."""
    proc = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, check=True)
    return proc.stdout


def obj_stream(objs: list[Path]) -> bytes:
    return run(
        ["uv", "run", "--with", "pyarrow", str(ROOT / "scripts" / "obj_to_arrow.py")]
        + [str(o) for o in objs]
    )


def texture_stream(img: Path) -> bytes:
    return run(
        [
            "uv", "run", "--with", "pyarrow", "--with", "pillow", "--with", "numpy",
            str(ROOT / "scripts" / "texture_to_arrow.py"),
            str(img), "--max-size", str(TEXTURE_MAX_SIZE),
        ]
    )


def params_stream(jsonl_text: str) -> bytes:
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tmp:
        tmp.write(jsonl_text)
        tmp_path = tmp.name
    try:
        return run(
            ["uv", "run", "--with", "pyarrow",
             str(ROOT / "scripts" / "jsonl_to_arrow.py"), tmp_path]
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def build_stage1(quad_obj: Path) -> bytes:
    # mesh table [quad] ++ params (a single wireframe quad draw per frame).
    mesh = obj_stream([quad_obj])
    params = params_stream(reduced_jsonl(STAGE1_JSONL))
    return mesh + params


def build_stage2(quad_obj: Path) -> bytes:
    # mesh table [bunny, quad] ++ texture ++ params (bunny draw + wireframe quad).
    mesh = obj_stream([BUNNY_OBJ, quad_obj])
    texture = texture_stream(BUNNY_TEX)
    params = params_stream(reduced_jsonl(STAGE2_JSONL))
    return mesh + texture + params


def main() -> None:
    for path in (BUNNY_OBJ, BUNNY_TEX, STAGE1_JSONL, STAGE2_JSONL):
        if not path.exists():
            sys.exit(f"missing input asset: {path}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        quad_obj = Path(td) / "placement_quad.obj"
        quad_obj.write_text(QUAD_OBJ_TEXT)

        stage1 = build_stage1(quad_obj)
        stage2 = build_stage2(quad_obj)

    (OUT_DIR / "stage1.arrow").write_bytes(stage1)
    (OUT_DIR / "stage2.arrow").write_bytes(stage2)
    print(f"wrote {OUT_DIR / 'stage1.arrow'} ({len(stage1)} bytes)")
    print(f"wrote {OUT_DIR / 'stage2.arrow'} ({len(stage2)} bytes)")


if __name__ == "__main__":
    main()
