#!/usr/bin/env python3
"""Measure the comment burden, so the doctrine's targets are checked rather than asserted.

The counting rule is deliberately simple and stated here, because the point of
this script is that two people get the *same* number (an earlier hand audit and a
review disagreed 31 vs 27 on the same tree):

  * A **comment line** is a line whose stripped form starts with ``//`` (so
    ``//``, ``///`` and ``//!`` all count) or that lies inside a ``/* ... */``
    block.
  * A trailing comment on a line of code is **not** counted -- it costs a reader
    nothing extra.
  * A **block** is a run of consecutive comment lines. Blocks are reported per
    kind, because a 20-line ``//!`` module header and a 20-line ``///`` item doc
    are different problems.

No attempt is made to exclude ``//`` inside string literals; a line that *starts*
with one is vanishingly rare in this tree, and the alternative (a real parser)
would make the number harder to reproduce than the thing it measures.

Usage::

    python3 scripts/comment_audit.py                    # every area
    python3 scripts/comment_audit.py --scope front-end  # the areas the doctrine binds
    python3 scripts/comment_audit.py --scope front-end --check
    python3 scripts/comment_audit.py --files            # per-file, worst first
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUFFIXES = {".rs", ".ts", ".wgsl"}
SKIP_PARTS = {"node_modules", "pkg", "dist", "target", ".worktree"}

# The doctrine binds `front-end`; `core` is exempt pending its own pass. Keeping
# both here is what makes that exemption a number instead of an omission.
#
# Each area is a `src` tree: `tests/` is excluded, because an integration test is
# a separate crate with its own audience, and because #332's targets were stated
# against `src` -- a script whose numbers cannot be compared with the issue that
# asked for it would be worse than no script.
AREAS = {
    "crates/trd-gui/src": "front-end",
    "crates/trd-wasm/src": "front-end",
    "web/gui-video-editing/src": "front-end",
    "native": "front-end",
    "crates/trd-core/src": "core",
    "crates/trd-placement/src": "core",
    "crates/trd-cli/src": "core",
}

# Budgets are per scope: (max comment share, max blocks over --max-block).
# `front-end` starts at today's measurement so the number can only go down; the
# doctrine's target is the second column of the report.
BUDGETS = {"front-end": (0.12, 0)}


def classify(path: Path) -> list[bool]:
    """Return one flag per line: True when the line is a comment line."""
    flags: list[bool] = []
    in_block = False
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if in_block:
            flags.append(True)
            if "*/" in line:
                in_block = False
            continue
        if line.startswith("/*"):
            flags.append(True)
            if "*/" not in line[2:]:
                in_block = True
            continue
        flags.append(line.startswith("//"))
    return flags


def kind_of(line: str) -> str:
    line = line.strip()
    if line.startswith("//!"):
        return "//!"
    if line.startswith("///"):
        return "///"
    if line.startswith("//"):
        return "//"
    return "/*"


def blocks(path: Path, flags: list[bool]) -> list[tuple[int, int, str]]:
    """Runs of consecutive comment lines as (length, start_line, kind)."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    out, run, start = [], 0, 0
    for i, flag in enumerate(flags):
        if flag:
            if run == 0:
                start = i
            run += 1
        elif run:
            out.append((run, start + 1, kind_of(lines[start])))
            run = 0
    if run:
        out.append((run, start + 1, kind_of(lines[start])))
    return out


def walk(area: str):
    base = ROOT / area
    if not base.exists():
        return
    for path in sorted(base.rglob("*")):
        if path.suffix not in SUFFIXES or not path.is_file():
            continue
        if SKIP_PARTS & set(path.relative_to(ROOT).parts):
            continue
        yield path


def self_test() -> int:
    """Pin the counting rule, since the whole point is that it is reproducible."""
    import tempfile

    cases = [
        ("// a\n// b\ncode();\n", 2, [(2, "//")]),
        ("let x = 1; // trailing\n", 0, []),  # trailing comments cost nothing extra
        ("/* a\n b */\ncode();\n", 2, [(2, "/*")]),
        ("/* one line */\ncode();\n", 1, [(1, "/*")]),
        ("/// doc\npub fn f() {}\n/// doc2\n", 2, [(1, "///"), (1, "///")]),
        ("//! header\n//! more\n\n// sep\n", 3, [(2, "//!"), (1, "//")]),
    ]
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for i, (source, want_lines, want_blocks) in enumerate(cases):
            path = Path(tmp) / f"case{i}.rs"
            path.write_text(source, encoding="utf-8")
            flags = classify(path)
            got_lines = sum(flags)
            got_blocks = [(length, kind) for length, _, kind in blocks(path, flags)]
            if got_lines != want_lines or got_blocks != want_blocks:
                failures += 1
                print(f"case {i}: lines {got_lines} (want {want_lines}), "
                      f"blocks {got_blocks} (want {want_blocks})")
    if failures:
        print(f"{failures} self-test failure(s)")
        return 1
    print(f"self-test ok ({len(cases)} cases)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--scope",
        default="all",
        choices=sorted({"all", *AREAS.values()}),
        help="which group of areas to measure (default: all)",
    )
    ap.add_argument("--max-block", type=int, default=10, help="longest allowed comment block")
    ap.add_argument("--files", action="store_true", help="list per-file totals, worst first")
    ap.add_argument("--check", action="store_true", help="exit 1 when the scope is over budget")
    ap.add_argument("--self-test", action="store_true", help="check the counting rule itself")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    areas = [a for a, group in AREAS.items() if args.scope in ("all", group)]
    rows, files, long_blocks = [], [], []
    for area in areas:
        total = comments = 0
        over = over_lines = 0
        for path in walk(area):
            flags = classify(path)
            n = sum(flags)
            total += len(flags)
            comments += n
            for length, start, kind in blocks(path, flags):
                if length > args.max_block:
                    over += 1
                    over_lines += length
                    long_blocks.append((length, path.relative_to(ROOT).as_posix(), start, kind))
            files.append((n, len(flags), path.relative_to(ROOT).as_posix()))
        rows.append((area, AREAS[area], total, comments, over, over_lines))

    width = max(len(r[0]) for r in rows) + 2
    print(f"{'area':<{width}}{'group':<11}{'lines':>8}{'comments':>10}{'share':>8}"
          f"{'>' + str(args.max_block):>7}{'in them':>9}")
    for area, group, total, comments, over, over_lines in sorted(rows, key=lambda r: -r[3]):
        share = 100.0 * comments / total if total else 0.0
        print(f"{area:<{width}}{group:<11}{total:>8}{comments:>10}{share:>7.1f}%{over:>7}{over_lines:>9}")

    tot = sum(r[2] for r in rows)
    com = sum(r[3] for r in rows)
    over = sum(r[4] for r in rows)
    over_lines = sum(r[5] for r in rows)
    share = 100.0 * com / tot if tot else 0.0
    print(f"{'TOTAL':<{width}}{args.scope:<11}{tot:>8}{com:>10}{share:>7.1f}%{over:>7}{over_lines:>9}")

    if args.files:
        print(f"\nper-file, worst first (comment lines, share):")
        for n, total, name in sorted(files, reverse=True)[:20]:
            pct = 100.0 * n / total if total else 0.0
            print(f"  {n:>5}{pct:>7.1f}%  of {total:>5} lines  {name}")
        print(f"\nlongest comment blocks:")
        for length, name, start, kind in sorted(long_blocks, reverse=True)[:20]:
            print(f"  {length:>4} lines  {kind:<4} {name}:{start}")

    if args.check:
        budget = BUDGETS.get(args.scope)
        if budget is None:
            print(f"\nno budget defined for scope '{args.scope}' -- reporting only")
            return 0
        max_share, max_over = budget
        failures = []
        if share > max_share * 100:
            failures.append(f"comment share {share:.1f}% exceeds {max_share * 100:.0f}%")
        if over > max_over:
            failures.append(f"{over} block(s) over {args.max_block} lines, budget {max_over}")
        if failures:
            print("\nOVER BUDGET:")
            for failure in failures:
                print(f"  - {failure}")
            return 1
        print("\nwithin budget")
    return 0


if __name__ == "__main__":
    sys.exit(main())
