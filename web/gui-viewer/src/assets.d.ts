// The HDR probe imported with `{ type: "file" }` yields a URL string the bundler
// emits as an asset — the same shim shape as `wasm.d.ts`.
declare module "*.hdr" {
  const url: string;
  export default url;
}
