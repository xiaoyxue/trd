# Comment doctrine

**One line in five in this repository is a comment**, and the weight is not in
useful one-liners: measured by `scripts/comment_audit.py` at `d2b5f86`,

| Scope | Lines | Comment lines | Share | Blocks > 10 | Lines in them |
|---|---:|---:|---:|---:|---:|
| **`front-end`** — bound by these rules | 16,554 | **3,171** | 19.2% | **32** | 493 |
| `core` — exempt, see below | 26,874 | 5,372 | 20.0% | 96 | 1,735 |

Reproduce any number here with:

```sh
python3 scripts/comment_audit.py                       # every area
python3 scripts/comment_audit.py --scope front-end --files
python3 scripts/comment_audit.py --scope front-end --check   # exits 1 over budget
```

## Why volume is the problem, not just style

**3,171 comment lines are 3,171 lines no tool checks.** The repository has a gate
that keeps *code* references honest and nothing that keeps *prose* honest, and
the two defects that prove it are both real:

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

**32 blocks in scope violate this today; the target is 0.** If a rationale needs
more than 10 lines it is design, and design belongs in `docs/` with a link from
the code — one line pointing at a page beats twenty lines in a header.

### 3. No doc comment longer than the item it documents

A 23-line doc over a one-line function inverts the reader's cost. Shorten the
comment or, if the explanation is genuinely that large, the item is under-named.

### 4. Cite the issue; do not re-litigate it

`PendingSeek::settled_by` (`crates/trd-gui/src/video_editing/mod.rs:2056`) carries
23 doc lines over a one-line body, eleven of which replay the #322 review
argument — *"This used to compare the delivered frame's timestamp against the
requested instant… the two readers miss in opposite directions…"*.

That paragraph is not badly written. **It is a review argument that got absorbed
into the source**: the right thing to write where a reader is deciding whether a
change is correct, and the wrong thing to leave where every future reader pays
for a decision already made. Target: three lines — what it decides, why `>=`, and
`(#322)`.

**The limit of this rule.** *Past tense* is not the test; **whether the reader can
act on it** is. The comment explaining why `dispatched_seek` is set inside
`take_seek_frame` is also history, and it is load-bearing — it names an invariant
a caller can violate. Keep that. Rule 1 already tells them apart.

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

## Budgets

`--check` fails the `front-end` scope when it exceeds them:

| Metric | Today | Target |
|---|---:|---:|
| comment lines | 3,171 | ≤ 1,800 |
| comment share | 19.2% | ≤ 12%, and no file over 20% |
| blocks over 10 lines | 32 | **0** |

`--check` is **advisory today and deliberately not wired into
`nix flake check`**: a gate that fails on day one for 32 pre-existing violations
teaches people to bypass it, which is the failure this doctrine exists to prevent.
It is armed once the reduction slices bring the scope inside budget — and it is
shipped now, with the rules, so every later slice is scored by the same tool
instead of a fresh hand count. (Two good-faith hand counts of this tree
previously disagreed 31 vs 27; that is what the script removes.)
