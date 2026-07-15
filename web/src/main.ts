const query = new URLSearchParams(window.location.search);

if (query.has("arrow-smoke")) {
  await import("./arrow-renderer-smoke");
} else {
  await import("./canvas-demo");
}

export {};
