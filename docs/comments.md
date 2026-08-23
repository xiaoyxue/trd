# Comments

Comments should say **why**, not what, and stay short enough that a reader can
see what they attach to. That is the whole of it; the rest of this page is the
reasoning and a script for when a file feels out of hand.

This is guidance, not a gate. Nothing enforces it, and it is not worth an
argument in review — but a comment block running past ~10 lines is usually a
sign the explanation belongs in `docs/` instead.

## What to cut, when you are already in a file

- **Prose that restates the code.** `/// Vertical field of view, radians.` above
  `pub fovy: f32` costs a line and says nothing.
- **An issue's content.** A bare `(#322)` is the whole citation — a reader who
  wants the argument can open it, and one who doesn't shouldn't have to skip it.
  `PendingSeek::settled_by` once carried 23 doc lines over a one-line body,
  eleven of them replaying a review thread.
- **History.** "This used to…", "X replaced Y". Git has it.
- **Architecture essays.** `crates/trd-gui/src/lib.rs` opened with a 28-line
  header — an ASCII pipeline diagram and a module-layout essay — over 10 lines
  of `pub mod`. [`docs/architecture.md`](architecture.md) and
  [`docs/gui-design.md`](gui-design.md) are its home.
- **A rationale repeated on five fields.** Say it once, on the type.
- **A paraphrase of what a test asserts.** Name the test instead.

## What to keep

- Units, ranges, clamps, and sign or coordinate conventions.
- `None` / `0` / empty semantics that are not obvious — `None` meaning "not
  checked" rather than "checked, zero" is a real example in this tree.
- **Anything a caller can violate**: ordering requirements, ownership and
  lifetime rules, preconditions that would panic. These are load-bearing however
  old they are. *Past tense is not the test; whether the reader can act on it is.*
- **API contracts.** `FrameReader`'s "the caller owns the frame and must
  `close()` it" stays — a caller shouldn't have to read a test to learn a rule
  they can break from outside the crate.

## Two things that are not comments

- **A `clap` doc comment is `--help` output.** `native/trd-app/src/cli.rs`
  measures as one of the highest comment shares in the tree, and nearly all of
  it is text a user reads. Deleting it removes a feature.
- Anything else that is *generated output* rather than a note to a maintainer.

## Why length in particular

Long blocks are where documentation goes wrong silently. In
`video_editing_renderer.rs`, 25 `///` lines with no blank line before
`pub struct QuadOverlay` attached the description of a *free function* to the
struct — publishing `QuadOverlay` as *"Authors the three layers of an editor
frame"* and leaving `placement_scenes` undocumented.

No gate can catch that. #308's rustdoc check looks for **broken links**, and
every link in those 25 lines resolved; it was correctly-linked prose on the wrong
item. And a `//` comment naming a `crate::web_renderer` that exists nowhere
survived because rustdoc never parses `//` at all.

Two adjacent 12-line blocks are exactly how a missing blank line becomes
invisible — which is the case for keeping blocks short, rather than an aesthetic
preference.

## The script, for when you want a number

`scripts/comment_audit.py` counts comment lines and long blocks per area. Run it
when a file feels heavy or after a cleanup; there is no need to run it otherwise.

```sh
python3 scripts/comment_audit.py --scope front-end --files   # worst files, longest blocks
python3 scripts/comment_audit.py --scope front-end --check   # non-zero if over the soft budget
python3 scripts/comment_audit.py --self-test                 # pins the counting rule
```

A comment line is one whose stripped form starts with `//` or lies inside
`/* */`; a trailing comment on a line of code doesn't count. The rule is fixed
and self-tested because two good-faith hand counts of this tree once disagreed
31 vs 27, and because it should give the same answer on every machine — this
tree measures **1,797** on both Windows and Linux.

**It measures volume, not value.** A file can sit inside the budget and still be
full of useless one-liners, and the script is blind to the `QuadOverlay` defect
above — it counted those 25 lines as 25 ordinary comment lines and said nothing.
An automated cleanup later re-created that same defect, and the script again said
nothing; a person reading the diff caught it. Treat the number as a ruler, never
as a verdict.

## Where it stands

| Scope | Comment lines | Share | Blocks > 10 |
|---|---:|---:|---:|
| `crates/trd-gui`, `crates/trd-wasm`, `web/gui-video-editing`, `native/**` | **1,797** | 11.8% | 0 |
| `crates/trd-core` — not swept | 5,372 | 20.0% | 96 |

The front-end areas came down from **3,171 lines / 19.2% / 32 long blocks**. The
render core was left alone deliberately — goldens and blast radius — and holds
most of what remains; the number is recorded so a later pass starts from a fact
rather than an argument.
