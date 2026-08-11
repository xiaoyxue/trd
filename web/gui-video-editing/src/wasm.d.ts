// Importing a *.wasm file with `{ type: "file" }` yields a URL string that the
// bundler (bun) emits and serves as an asset.
//
// Declared here rather than relied on from the generated package: the nix build
// strips wasm-bindgen's `trd_wasm_bg.wasm.d.ts` (see flake.nix), so every web
// package carries this shim.
declare module "*.wasm" {
  const url: string;
  export default url;
}
