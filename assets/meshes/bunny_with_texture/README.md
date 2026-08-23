# bunny_with_texture

A UV-unwrapped Stanford Bunny for the textured-rendering slice (#20).

- `bunny.obj` — UV-mapped mesh (`v`/`vt`/`vn`, `f v/vt/vn`; ~34k vertices, ~33.5k
  texcoords, ~65.6k faces). Loaded via `Mesh::from_obj`, which reads the `vt`
  texture coordinates into the `Vertex.uv` attribute (v flipped bottom-up →
  top-left texel origin).
- `bunny_uv_map1.jpg`, `bunny_uv_map2.png`, `bunny_uv_map3.jpg` — candidate 3072²
  albedo maps for the `uv` layout. Pick one as the bound texture; note the CLI
  batch renderer caps textures at the adapter limit, and the demo may downscale
  to ≤2048² depending on the GPU.

Source: Blender-exported "UVUnwrapped Ceramic Stanford Bunny".
