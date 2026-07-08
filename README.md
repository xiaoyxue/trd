# trd

A tile (relational) oriented renderer prototype, built on Rust + wgpu.

The Rust/wgpu core (`trd-core`) is the single rendering core; it runs natively
(headless CLI) and in the browser (compiled to wasm). JavaScript/TypeScript is a
thin bootstrap wrapper only — no WebGPU API is called from JS.

## Layout

- `crates/trd-core` — platform-agnostic wgpu render core (shared by all targets)
- `crates/trd-cli` — native headless CLI; renders to a PNG
- `crates/trd-wasm` — `wasm-bindgen` entry point for the browser
- `web/` — bun-managed thin TypeScript wrapper + bundler

## Development

Everything runs inside the Nix dev shell (pinned Rust toolchain, `bun`,
`wasm-bindgen-cli`, and Vulkan):

```sh
nix develop
```

### Native CLI (headless render to PNG)

```sh
cargo run -p trd-cli -- --width 512 --height 512 --output triangle.png
```

### Tests

```sh
cargo test --workspace            # fast, no GPU
cargo test --workspace -- --ignored   # GPU-gated render tests (needs a GPU)
```

### Web (wasm)

```sh
cd web
bun install
bun run build      # -> web/dist (index.html + JS bundle + trd_wasm_bg.wasm)
bun run dev        # dev server; open the printed URL in a WebGPU browser
```
