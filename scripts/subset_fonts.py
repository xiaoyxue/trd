#!/usr/bin/env python3
"""Cut the UI fonts down to the glyphs trd actually draws — and prove it.

`eframe`/`egui`'s ``default_fonts`` embeds four whole font files — 1,414,018
bytes in the shipped wasm, 57% of its data section (#359). This writes the
subsets ``crates/trd-gui/src/fonts.rs`` embeds instead, ~50 KB for the three.

Three, not two, because the glyphs are not interchangeable:

* **Ubuntu-Light** — proportional UI text.
* **Hack** — monospace, and the fallback for arrows Ubuntu lacks (``→`` ``⇒``).
* **emoji-icon-font** — cut to almost nothing, but it is the *only* one of the
  four with ``⏴⏵⏶⏷``, which egui draws in every ``CollapsingHeader`` and
  ``DragValue``. Dropping it outright turns those into empty boxes.

The sources are the files ``epaint_default_fonts`` ships, so the rendering stays
egui's own — only the glyph coverage changes. Point ``--source`` at that crate in
the cargo registry::

    uv run --with fonttools scripts/subset_fonts.py \
        --source ~/.cargo/registry/src/*/epaint_default_fonts-0.36.1/fonts

``--check`` re-reads the generated subsets and asserts that every non-ASCII
character in the GUI sources is present in at least one of them. A dropped glyph
fails **nothing** at runtime — it renders as an empty box on someone's screen —
so this check is the only thing between a narrowed range and a broken UI.

Licences travel with the files: ``assets/fonts/*.LICENSE.txt``. Re-run after an
egui upgrade so the subsets track the fonts egui itself would have used.
"""

import argparse
import collections
import pathlib
import re
import subprocess
import sys

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = REPOSITORY_ROOT / "assets/fonts"

# Latin-1, punctuation, arrows, maths (`≥` `≈` `−`), and the geometric shapes
# egui's widgets draw. Emoji are deliberately absent: they are most of the weight
# and this UI never draws one.
TEXT_RANGES = (
    "U+0000-00FF,U+2000-206F,U+2190-21FF,U+2200-22FF,"
    "U+2300-23FF,U+2500-257F,U+25A0-25FF,U+2600-26FF,U+2B00-2BFF"
)
# Only the block holding egui's widget arrows. This font is 324,132 bytes whole.
ICON_RANGES = "U+2300-23FF"

FONTS = (
    ("Ubuntu-Light.ttf", "Ubuntu-Light.subset.ttf", TEXT_RANGES),
    ("Hack-Regular.ttf", "Hack-Regular.subset.ttf", TEXT_RANGES),
    ("emoji-icon-font.ttf", "emoji-icon-font.subset.ttf", ICON_RANGES),
)

STRING = re.compile(r'"((?:[^"\\]|\\.)*)"')
# Sources whose UI strings must be renderable. egui's own crate is not scanned:
# most of its non-ASCII lives in its demo app, which trd never builds.
DEFAULT_SCAN = ("crates/trd-gui/src", "crates/trd-wasm/src", "native")


def subset(source, target, unicodes):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "fontTools.subset",
            str(source),
            f"--output-file={target}",
            f"--unicodes={unicodes}",
            "--layout-features=",
            "--no-hinting",
            "--desubroutinize",
        ],
        check=True,
        capture_output=True,
    )


def coverage(out_dir):
    """The codepoints the generated subsets can actually draw."""
    from fontTools.ttLib import TTFont

    points = set()
    for _, target_name, _ in FONTS:
        path = out_dir / target_name
        if not path.is_file():
            raise SystemExit(f"{path} missing — run without --check first")
        points |= set(TTFont(path).getBestCmap())
    return points


def check(out_dir, roots):
    points = coverage(out_dir)
    missing = collections.Counter()
    for root in roots:
        for path in (REPOSITORY_ROOT / root).rglob("*.rs"):
            for match in STRING.finditer(path.read_text(encoding="utf-8")):
                for char in match.group(1):
                    if ord(char) >= 128 and ord(char) not in points:
                        missing[char] += 1
    if missing:
        print("These characters appear in UI strings but no subset can draw them:")
        for char, count in missing.most_common():
            print(f"  U+{ord(char):04X} {char!r} x{count}")
        return 1
    total = sum(path.stat().st_size for path in out_dir.glob("*.subset.ttf"))
    print(f"every non-ASCII UI character is covered; subsets total {total:,} bytes")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=pathlib.Path,
        help="the epaint_default_fonts `fonts/` directory in the cargo registry",
    )
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed subsets cover the GUI sources; write nothing",
    )
    parser.add_argument("--scan", nargs="*", default=DEFAULT_SCAN)
    args = parser.parse_args()

    if args.check:
        return check(args.out_dir, args.scan)

    if not args.source:
        parser.error("--source is required unless --check is passed")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, target_name, unicodes in FONTS:
        source = args.source / name
        if not source.is_file():
            raise SystemExit(f"{source} not found — is --source the crate's fonts/ dir?")
        target = args.out_dir / target_name
        subset(source, target, unicodes)
        before, after = source.stat().st_size, target.stat().st_size
        print(f"{name}: {before:,} -> {after:,} bytes ({after / before - 1:+.0%})")
    return check(args.out_dir, args.scan)


if __name__ == "__main__":
    raise SystemExit(main())
