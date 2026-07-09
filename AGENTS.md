# AGENTS.md

Guidance for AI coding agents (and humans) working in this repository.

## Project

**trd** — a tile (relational) oriented renderer prototype.

The renderer is written in **Rust** on top of **[wgpu](https://wgpu.rs/)** and compiled to
**WebAssembly** for the browser via **wasm-pack / wasm-bindgen**. It also builds natively for
fast local iteration and testing.

## Architecture & core principles

- **Rust + wgpu is the single, unified rendering core.** All rendering logic lives in Rust and is
  compiled to wasm for the browser. Do **not** call the WebGPU API directly from JavaScript/TypeScript.
- **JavaScript/TypeScript is only a thin bootstrap wrapper.** Its job is to load the wasm module,
  hand it a canvas/surface, and forward events. Keep it minimal — no rendering logic in JS/TS.
- One code path, two targets: the same Rust core runs both natively (via wgpu on the host GPU
  backend) and in the browser (via wgpu's WebGPU backend in wasm). Prefer target-agnostic code and
  isolate platform specifics behind `#[cfg(target_arch = "wasm32")]`.

## Toolchain

- **Rust** (stable) with the `wasm32-unknown-unknown` target.
- **wasm-pack** for building the browser-consumable package.
- **wasm-bindgen** for the JS/wasm interop layer.
- Optional: **Node.js** for the thin JS/TS wrapper and any web dev server.

> **Windows native builds must use the MSVC toolchain** (`stable-x86_64-pc-windows-msvc`),
> not `-gnu`. The `-gnu` toolchain needs `dlltool` for the `raw-dylib` import libs that
> wgpu's transitive deps (`windows-sys 0.61`) require, and substituting `llvm-dlltool`
> produces broken import stubs that crash at runtime (`0xC0000005`, faulting module
> "unknown"). Build from a VS "x64 Native Tools" prompt (or after running `vcvars64.bat`)
> so `link.exe` and the Windows SDK are on `PATH`/`LIB`/`INCLUDE`.

Install the wasm target and tools locally:

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
```

## Build, test & lint commands

Native (fast iteration):

```bash
cargo build
cargo test
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
```

WebAssembly / browser:

```bash
# Build the npm-consumable wasm package (outputs to ./pkg by default)
wasm-pack build --target web

# Run wasm-targeted tests in a headless browser
wasm-pack test --headless --chrome   # or --firefox
```

> Note: This is an early-stage prototype. If a `Cargo.toml` does not yet exist, scaffold the crate
> with `cargo init` (add `crate-type = ["cdylib", "rlib"]` under `[lib]` for wasm output) before
> running the commands above.

## Conventions

- Format with `cargo fmt` and keep `cargo clippy` warning-free before committing.
- Keep the JS/TS wrapper as small as possible; push logic into the Rust core.
- Gate platform-specific code with `#[cfg(target_arch = "wasm32")]` / `#[cfg(not(...))]` rather than
  forking modules.
- Add tests alongside the code they cover; use `wasm-pack test` for anything that must run in-browser.

## Repository layout

- `README.md` — short project description.
- `AGENTS.md` — this file.
- `.github/workflows/copilot-setup-steps.yml` — preinstalls the Rust + wasm toolchain for the
  Copilot cloud agent environment.
- `src/` — Rust rendering core (to be added).
- `pkg/` — generated wasm-pack output (do not edit by hand; not committed).
