// Importing a *.jsonl file with { type: "file" } yields a URL string that the
// bundler (bun) emits and serves as an asset, fetched at runtime.
declare module "*.jsonl" {
  const url: string;
  export default url;
}
