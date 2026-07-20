// Importing a *.jpg file with { type: "file" } yields a URL string that the
// bundler (bun) emits and serves as an asset, fetched + decoded at runtime.
declare module "*.jpg" {
  const url: string;
  export default url;
}
