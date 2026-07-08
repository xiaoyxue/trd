# AGENTS.md

Guidance for agents working in this repository.

## Architecture

- **Rust + wgpu is the single unified rendering core** (`crates/trd-core`).
  The same core renders natively (CLI) and in the browser (compiled to wasm).
- **JS/TS is a thin bootstrap wrapper only.** Do not call the WebGPU API
  directly from JavaScript; all rendering logic lives in Rust.
- **Vertical slicing.** Each increment threads the whole stack and is
  independently end-to-end verifiable.
- Major input data is columnar (Apache Arrow tables) with simple glue logic.

## Toolchain

- Enter the dev environment with `nix develop` (provides the pinned Rust
  toolchain via rust-overlay, `bun`, `wasm-bindgen-cli`, and Vulkan).
- Build/test/debug with plain `cargo` inside the dev shell.
- The `web/` folder is bun-managed; run bun from inside `nix develop`.
- We always work on GPU machines. GPU-dependent tests are marked `#[ignore]`
  and run locally; simple CI skips them.

## PR Workflow

- **pr_first: true** — push work on a feature branch and open a **draft PR**
  as early as practical; use the PR as the working surface.
- **auto_merge: small** — small, low-risk PRs may be squash-merged once CI is
  green. Risky PRs (public API, schemas, migrations, auth, infra) require human
  review.
- **branch naming:** `feat/<topic>`, `fix/<topic>`, etc.
- **merge strategy:** squash.
- PRs that resolve an issue must include a `Closes #nn` keyword.
