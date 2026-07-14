# AGENTS.md

Guidance for agents working in this repository.

## Architecture

- **Rust + wgpu is the single unified rendering core** (`crates/trd-core`).
  The same core renders natively (headless CLI and an interactive window) and
  in the browser (compiled to wasm).
- **JS/TS is a thin bootstrap wrapper only.** Do not call the WebGPU API
  directly from JavaScript; all rendering logic lives in Rust.
- **Vertical slicing.** Each increment threads the whole stack and is
  independently end-to-end verifiable.
- Major input data is columnar (Apache Arrow tables) with simple glue logic.

## Toolchain

- Enter the dev environment with `nix develop` (provides the pinned Rust
  toolchain via rust-overlay, `bun`, `wasm-bindgen-cli`, `biome`, `typescript`,
  and Vulkan) for local iteration and GPU work.
- The flake is the build system, not just a dev shell. Prefer the real outputs:
  - `nix build .#trd-cli` — native CLI binary (`trd`), wrapped with the Vulkan/
    GL runtime libs. `nix run .#trd -- --width 256 --height 256` runs the Arrow
    stream filter (frames on stdin -> images on stdout).
  - `nix build .#trd-wasm` — the `wasm-bindgen` JS/TS library package (built with
    `wasm-bindgen-cli` + `wasm-opt`, replacing `wasm-pack` in the nix build).
  - `nix build .#web` (also `.#`) — the bun-bundled, HTTP-servable `dist/`.
    `nix run .#web` serves it (`PORT` overridable, defaults to 8080).
  - `nix flake check` — every quality gate: `cargo fmt`, clippy (native + wasm32),
    `cargo test`, `tsc --noEmit`, and Biome (format + lint). No GPU required.
  - `nix fmt` — formats nix files (`nixfmt`).
- Local `cargo` inside `nix develop` still works for fast iteration.
- The `web/` folder is bun-managed; its lint/format gate is **Biome**
  (`web/biome.json`). Run `bun run check` / `bun run format` / `bun run typecheck`
  from inside `nix develop`, or directly on Windows — `@biomejs/biome` and
  `apache-arrow` are now declared in `web/package.json`, so `bun install` +
  `bun run check` / `typecheck` / `build:web` work without Nix.
- **The Nix web build (`nix build .#web`, `nix flake check`) still needs a bun2nix
  step to install `web/`'s external npm deps (`apache-arrow`) reproducibly.** Until
  that lands, run the web gates with plain `bun` (above); the native gates
  (`cargo fmt`/clippy/test) are unaffected.
- **`nix build`/`nix flake check` only see git-tracked files.** `git add` new
  files (e.g. a new `biome.json` or source file) before building, or the sandbox
  won't include them.
- We always work on GPU machines. GPU-dependent tests are marked `#[ignore]`
  and run locally; CI skips them.
- **WSL2 GPU:** NVIDIA ships no native Linux Vulkan ICD for WSL, so the Vulkan
  backend falls back to software (llvmpipe) and Mesa's `dzn` (Vulkan-on-D3D12)
  crashes at device creation. Use `WGPU_BACKEND=gl` for real GPU rendering via
  Mesa's D3D12 OpenGL driver; the dev shell auto-configures this on WSL.

## PR Workflow

- **pr_first: true** — push work on a feature branch and open a **draft PR**
  as early as practical; use the PR as the working surface.
- **auto_merge: small** — small, low-risk PRs may be squash-merged once CI is
  green. Risky PRs (public API, schemas, migrations, auth, infra) require human
  review.
- **branch naming:** `feat/<topic>`, `fix/<topic>`, etc.
- **merge strategy:** squash.
- PRs that resolve an issue must include a `Closes #nn` keyword.
- **Worktrees:** keep the git root checkout on `main` at all times. Do all
  branch/PR work in a git worktree under the root's `.worktree/` folder, e.g.:
  ```sh
  git worktree add .worktree/<topic> -b feat/<topic>
  cd .worktree/<topic>
  ```
  Never check out a feature branch in the root itself. `.worktree/` is
  gitignored. Remove the worktree after the PR merges
  (`git worktree remove .worktree/<topic>`).
