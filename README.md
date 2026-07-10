# trd

A tile (relational) oriented renderer prototype, built on Rust + wgpu.

The Rust/wgpu core (`trd-core`) is the single rendering core; it runs natively
(headless CLI) and in the browser (compiled to wasm). JavaScript/TypeScript is a
thin bootstrap wrapper only — no WebGPU API is called from JS.

## Layout

- `crates/trd-core` — platform-agnostic wgpu render core (shared by all targets)
- `crates/trd-cli` — native headless CLI; renders to a PNG
- `crates/trd-wasm` — `wasm-bindgen` entry point; packaged as the `trd-wasm`
  npm library via `wasm-pack` (output in `crates/trd-wasm/pkg`, gitignored)
- `web/` — bun-managed thin TypeScript wrapper that consumes the `trd-wasm`
  package

## Development

The flake is the build system. Real, reproducible outputs:

```sh
nix build .#trd-cli   # native headless CLI binary (wrapped with Vulkan/GL libs)
nix build .#trd-wasm  # wasm-bindgen JS/TS library package
nix build .#web       # bun-bundled, HTTP-servable dist/ (also `nix build`)
nix run   .#trd -- --output triangle.png   # render a PNG natively
nix run   .#web                            # serve dist/ (PORT defaults to 8080)
nix flake check       # all gates: fmt, clippy (native+wasm32), test, tsc, biome
```

For fast local iteration, enter the dev shell (pinned Rust toolchain, `bun`,
`wasm-bindgen` / `wasm-pack`, `biome`, `typescript`, and Vulkan) and use plain
`cargo` / `bun`:

```sh
nix develop
```

> `nix build` and `nix flake check` only see git-tracked files — `git add` new
> files before building.

### Native CLI (headless render to PNG)

```sh
cargo run -p trd-cli -- --width 512 --height 512 --output triangle.png
```

The renderer honours `WGPU_BACKEND` (e.g. `vulkan`, `gl`) and logs which adapter
it selected. Set `RUST_LOG=info` for more detail.

#### GPU selection

- **Native Linux / NVIDIA:** the default (Vulkan) backend uses the GPU directly.
- **WSL2:** there is no native Linux Vulkan driver, so the default Vulkan
  backend falls back to software (llvmpipe). For real GPU rendering, use the GL
  backend over Mesa's D3D12 driver:
  ```sh
  WGPU_BACKEND=gl cargo run -p trd-cli -- --output triangle.png
  ```
  The dev shell auto-detects WSL2 (`/dev/dxg`) and sets `GALLIUM_DRIVER=d3d12`
  plus the Windows GPU library path, so only `WGPU_BACKEND=gl` is needed.

### Tests

```sh
cargo test --workspace            # fast, no GPU
cargo test --workspace -- --ignored   # GPU-gated render tests (needs a GPU)
```

### Web (wasm)

For production/CI, build the servable bundle reproducibly with Nix:

```sh
nix build .#web    # Rust core -> wasm-bindgen library -> bun dist/ (in ./result)
nix run   .#web    # serve the built dist/ over HTTP (PORT defaults to 8080)
```

The wasm core is a standard, TypeScript-typed npm package. In the nix build it
is produced by `wasm-bindgen-cli` + `wasm-opt` (`nix build .#trd-wasm`); `web/`
consumes it as the `trd-wasm` dependency and imports it like any other library:

```ts
import init, { start } from "trd-wasm"; // fully typed
```

For local iteration inside `nix develop`, the `web/` scripts use `wasm-pack`
(which emits the same package shape) plus bun:

```sh
cd web
bun run build      # wasm-pack -> pkg, then bun bundles -> web/dist
bun run dev        # dev server; open the printed URL in a WebGPU browser
bun run check      # Biome format-check + lint
bun run format     # Biome auto-format
bun run typecheck  # tsc --noEmit
```

The wasm-pack/wasm-bindgen `web` target is used because bun does not instantiate
the `bundler` target's ESM-imported wasm.
