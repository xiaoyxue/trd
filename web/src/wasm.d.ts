// Importing a *.wasm file with `{ type: "file" }` yields a URL string that the
// bundler (bun) emits and serves as an asset.
declare module "*.wasm" {
  const url: string;
  export default url;
}
