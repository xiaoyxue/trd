# trd
A tile (relational) oriented renderer prototype.

The renderer is written in **Rust** on top of **[wgpu](https://wgpu.rs/)** and runs both
natively and in the browser (compiled to WebAssembly via wasm-pack). The JavaScript side is
only a thin bootstrap wrapper — all rendering logic lives in the Rust core.

## Prerequisites

Install the Rust toolchain (this provides `rustc`, `cargo`, and `rustup`) via **[rustup](https://rustup.rs/)**:

- **Windows**: download and run [`rustup-init.exe`](https://win.rustup.rs/x86_64), or:
  ```powershell
  winget install Rustlang.Rustup
  ```
  On Windows you must use the **MSVC** toolchain (the default), which requires the
  **Visual Studio C++ Build Tools + Windows SDK**. If you don't have them:
  ```powershell
  winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
  ```
  Then make MSVC the default toolchain and build from a *"x64 Native Tools Command Prompt for VS"*
  (or after running `vcvars64.bat`) so `link.exe` is on `PATH`:
  ```powershell
  rustup default stable-x86_64-pc-windows-msvc
  ```
  > Do **not** use the `-gnu` toolchain here: wgpu's dependencies need `dlltool` for their
  > import libraries, and non-MSVC substitutes produce stubs that crash at runtime.

- **macOS / Linux**:
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```
  On Linux, also install the usual native GPU/windowing dev headers (X11/Wayland, `libudev`),
  e.g. on Debian/Ubuntu see `.github/workflows/copilot-setup-steps.yml` for the exact list.

Verify the install:

```bash
rustc --version
cargo --version
```

## Hello triangle demo

The current prototype renders a classic multi-colored "hello triangle".

### Run natively

```bash
cargo run
```

Press `Esc` or close the window to exit.

### Run in the browser (WebAssembly)

Requires the wasm target and [`wasm-pack`](https://rustwasm.github.io/wasm-pack/):

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
```

Build the wasm package (outputs to `./pkg`) and serve the folder:

```bash
wasm-pack build --target web
python -m http.server 8080      # or any static file server
```

Then open <http://localhost:8080/> in a WebGPU-capable browser (e.g. recent Chrome/Edge).
If WebGPU is unavailable, the demo automatically falls back to WebGL2.

## Project layout

- `src/lib.rs` — the unified Rust + wgpu rendering core (native + wasm).
- `src/shader.wgsl` — the triangle's WGSL vertex/fragment shaders.
- `src/main.rs` — native entry point.
- `index.html` / `bootstrap.js` — thin browser wrapper that loads and starts the wasm core.
- `AGENTS.md` — guidance for AI coding agents and contributors.
