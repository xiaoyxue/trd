# Comment doctrine

**The `front-end` scope now sits inside its budget**, after a comment-only sweep
that removed ~1,780 lines. Measured by `scripts/comment_audit.py`:

| Scope | Lines | Comment lines | Share | Blocks > 10 |
|---|---:|---:|---:|---:|
| **`front-end`** — bound by these rules | 15,182 | **1,797** | **11.8%** | **0** |
| `core` — exempt, see below | 26,874 | 5,372 | 20.0% | 96 |

It started at **3,171 lines / 19.2% / 32 long blocks** — one line in five.

Reproduce any number here with:

```sh
python3 scripts/comment_audit.py                       # every area
python3 scripts/comment_audit.py --scope front-end --files
python3 scripts/comment_audit.py --scope front-end --check   # exits 1 over budget
python3 scripts/comment_audit.py --self-test                 # pins the counting rule
```

### What the script cannot do

It measures **volume, not value**, and the gap matters enough to state up front:

| It cannot | Consequence |
|---|---|
| **Tell whether a comment is any good** | A file can sit comfortably inside the budget while being full of useless one-liners. The rules below are the judgement; the script only sizes the surface they apply to. |
| **See a doc attached to the wrong item** | This is the defect that motivated the whole doctrine — 25 `///` lines landing on `QuadOverlay` instead of `placement_scenes`. The script counted them as 25 ordinary comment lines and said nothing. **An automated sweep re-created that exact defect while applying these rules, and the script again said nothing**; a human reading the diff caught it. |
| **Distinguish a `clap` doc from a real comment** | `native/trd-app/src/cli.rs` measures at ~44% for text that is `--help` output. The exception below is prose, and nothing enforces it. |
| **Ignore `//` inside a string literal** | A deliberate trade-off, stated in the script: a real parser would be harder to reproduce than the thing it measures. |

So the number is a *ruler*, not a verdict. What it buys is that "too many
comments" stops being an argument and becomes a figure two machines reproduce
exactly — this sweep measured **1,797 on both Windows and Linux**, and anyone can
re-run it and contradict it.

## Why volume is the problem, not just style

**1,797 comment lines are 1,797 lines no compiler checks.** The repository has a
gate that keeps *code* references honest and nothing that keeps *prose* honest,
and the two defects that prove it were both real — and are both fixed by the
sweep that produced the numbers above:

- `crates/trd-gui/src/video_editing_renderer.rs:579` — 25 `///` lines with no
  blank line before `pub struct QuadOverlay`, so rustdoc attaches all of them to
  the struct. `QuadOverlay`, a six-field struct, is published as *"Authors the
  three layers of an editor frame"* and *"Free function rather than a method"* —
  and `placement_scenes`, the function actually described, has **no**
  documentation at all. #308's rustdoc gate cannot see this: every link in those
  25 lines resolves. It is correctly-linked prose on the wrong item.
- `crates/trd-gui/src/renderer.rs:26-28` — a `//` comment citing a
  `crate::web_renderer` that exists nowhere in the tree. rustdoc never parses
  `//` at all, so nothing could have caught it.

Two adjacent 12-line blocks are precisely how a missing blank line becomes
invisible. **That is why the rules below cap block length**: the cap is not
aesthetic, it is what keeps a doc block short enough that its attachment is
obvious.

## Scope — and the exemption, with its number

These rules bind **`crates/trd-gui`, `crates/trd-wasm`,
`web/gui-video-editing/src` and `native/**`** (the `front-end` scope above).

**`crates/trd-core` and `crates/trd-placement` are exempt pending their own
pass**, and the exemption is stated with its size deliberately: the exempt scope
holds **5,372 comment lines and 96 blocks over 10 lines — 63% of the
repository's comment volume and 75% of its long blocks**. The two longest blocks
in the repository are both there (a **72-line** module header at
`render/renderer.rs:1`, a **41-line** doc at `render/scene.rs:260`).

Excluding the render core from the *work* is deliberate — goldens, the render
path and the blast radius all argue for it. But a rule with 96 silent exceptions
is one people learn to read past, so the exception is written down and counted
rather than left implicit. `--scope core` measures it with the same tool, so a
future pass starts from a number rather than a fresh argument.

## The rules

### 1. Say why, not what

A comment that restates the code is a second copy that can rot. Delete it.

```rust
// Bad -- the signature already says this.
/// Sets the material.
pub fn set_material(&mut self, material: DisneyMaterial)
```

### 2. No comment block over 10 lines

**Every block in scope now satisfies this.** If a rationale needs more than 10
lines it is design, and design belongs in `docs/` with a link from the code —
one line pointing at a page beats twenty lines in a header.

**An automated sweep re-created the `QuadOverlay` defect above** while applying
this very doctrine: a worker merged the free function's description back onto the
struct, and neither the rustdoc gate nor `comment_audit.py` could see it. Two
adjacent blocks and a missing blank line is a trap that catches careful readers
too, which is the case for the cap rather than an argument against it.

### 3. No doc comment longer than the item it documents

A 23-line doc over a one-line function inverts the reader's cost. Shorten the
comment or, if the explanation is genuinely that large, the item is under-named.

### 4. Cite the issue number; never reproduce its content

**A bare `(#322)` is the whole citation.** Do not carry the issue's argument, its
alternatives, or what the code used to do — a reader who wants that can open the
issue, and a reader who does not shouldn't have to skip it.

`PendingSeek::settled_by` (`crates/trd-gui/src/video_editing/mod.rs`) carried 23
doc lines over a one-line body, eleven of them replaying the #322 review
argument — *"This used to compare the delivered frame's timestamp against the
requested instant… the two readers miss in opposite directions…"*.

That paragraph is not badly written. **It is a review argument that got absorbed
into the source**: the right thing to write where a reader is deciding whether a
change is correct, and the wrong thing to leave where every future reader pays
for a decision already made.

**The limit of this rule.** *Past tense* is not the test; **whether the reader can
act on it** is. The comment explaining why `dispatched_seek` is set inside
`take_seek_frame` is also history and is load-bearing — it names an invariant a
caller can violate. Keep that. Rule 1 already tells them apart.

### 5. No architecture essays in source

`crates/trd-gui/src/lib.rs` opens with a 28-line header — an ASCII pipeline
diagram and a "Module layout" essay — over 10 lines of `pub mod`.
[`docs/architecture.md`](architecture.md) and [`docs/gui-design.md`](gui-design.md)
already exist and are its documented home. Target: three lines plus a pointer.

### 6. State a rationale once, on the type — not on each of five fields

If five fields share a reason, the reason belongs on the struct.

### 7. Do not restate what a test enforces

Name the test instead: *"pinned by `wasm_bindgen_containment.rs`"* beats
paraphrasing what it asserts, because the paraphrase can drift and the test
cannot.

**The limit of this rule is important.** It applies to **internal invariants**,
not **API contracts**. `FrameReader`'s *"the caller owns the frame and must
`close()` it"* stays: a caller should not have to read a test to learn a contract
they can violate from outside the crate.

### 8. No changelog in source

What the code used to be belongs in git history and the issue. `renderer.rs:26-28`
is what happens when it does not: a comment describing a module that no longer
exists, in a form no gate can check.

### 9. Prefer deleting to rewording

If you cannot say why a reader needs a comment, the comment is the thing to
remove — not the thing to polish.

## What is not a comment

**A `clap` doc comment is `--help` output.** `native/trd-app/src/cli.rs` measures
as one of the highest comment shares in the tree, and almost all of it is the
text a user sees when they run `--help`. Deleting it removes a user-facing
feature, not a maintenance cost, so that file's number is expected to stay high
and rules 1 and 3 do not apply to a `#[derive(Parser)]` field.

The same reasoning covers any doc that is *generated output* rather than a note
to a maintainer.

## Budgets

`--check` fails the `front-end` scope when it exceeds them:

| Metric | Before the sweep | Now | Budget |
|---|---:|---:|---:|
| comment lines | 3,171 | **1,797** | ≤ 1,800 |
| comment share | 19.2% | **11.8%** | ≤ 12% |
| blocks over 10 lines | 32 | **0** | 0 |

**`--check` is not wired into `nix flake check`** — it is run by hand, like every
other gate in this repository. What the script buys is not automation but a
*shared* verdict: the doctrine is scored by one tool instead of a fresh hand
count, and two good-faith hand counts of this tree previously disagreed 31 vs 27.
Run it when you touch the front-end scope, and put the numbers in the PR.
