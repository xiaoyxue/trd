// Importing a *.obj file with { type: "file" } yields a URL string that the
// bundler (bun) emits and serves as an asset, fetched + parsed at runtime.
declare module "*.obj" {
  const url: string;
  export default url;
}
