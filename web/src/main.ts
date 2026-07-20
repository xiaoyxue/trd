const query = new URLSearchParams(window.location.search);

if (query.has("arrow-smoke")) {
  await import("./arrow-renderer-smoke");
} else if (query.has("textured")) {
  await import("./textured-demo");
} else {
  await import("./canvas-demo");
}

export {};
