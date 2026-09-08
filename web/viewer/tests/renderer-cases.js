import init, { ArrowSceneDocument, CanvasRenderer, OffscreenRenderer } from "/pkg/trd_wasm.js";

const width = 320;
const height = 180;
const results = [];

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function rejects(work, pattern) {
  let failure;
  try {
    await work();
  } catch (error) {
    failure = String(error);
  }
  assert(failure && pattern.test(failure), `expected ${pattern}, got ${failure ?? "success"}`);
}

async function bytes(path) {
  const response = await fetch(path);
  assert(response.ok, `${path}: HTTP ${response.status}`);
  return new Uint8Array(await response.arrayBuffer());
}

async function image(path) {
  const bitmap = await createImageBitmap(new Blob([await bytes(path)]));
  const canvas = new OffscreenCanvas(width, height);
  const context = canvas.getContext("2d");
  context.drawImage(bitmap, 0, 0, width, height);
  bitmap.close();
  return new Uint8Array(context.getImageData(0, 0, width, height).data);
}

async function hash(rgba) {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", rgba));
  return Array.from(digest, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function compare(actual, expected, name, tolerance = 1, allowedFraction = 0) {
  assert(actual.length === expected.length, `${name}: pixel buffer length`);
  let max = 0;
  let outside = 0;
  let pixelsOutside = 0;
  let pixelOutside = false;
  for (let i = 0; i < actual.length; i++) {
    const delta = Math.abs(actual[i] - expected[i]);
    max = Math.max(max, delta);
    if (delta > tolerance) {
      outside++;
      pixelOutside = true;
    }
    if (i % 4 === 3) {
      if (pixelOutside) pixelsOutside++;
      pixelOutside = false;
    }
  }
  const fraction = pixelsOutside / (actual.length / 4);
  results.push({
    name,
    maxDelta: max,
    outsideTolerance: outside,
    pixelsOutside,
    fraction,
    tolerance,
    allowedFraction,
  });
  if (fraction > allowedFraction) {
    globalThis.rendererFailure = {
      name,
      actual: base64(actual),
      expected: base64(expected),
      results,
    };
    throw new Error(
      `${name}: ${pixelsOutside} pixels exceed ${tolerance}; fraction ${fraction}, max ${max}`,
    );
  }
}

async function target(kind) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  document.body.append(canvas);
  const renderer =
    kind === "canvas"
      ? await CanvasRenderer.create(canvas)
      : await OffscreenRenderer.create(width, height);
  return {
    renderer,
    async render(row) {
      if (kind === "offscreen") return renderer.renderIndex(row);
      return new Promise((resolve, reject) => {
        requestAnimationFrame(() => {
          try {
            // Copy in the same display tick, before the canvas texture expires.
            renderer.renderIndex(row);
            const copy = new OffscreenCanvas(width, height);
            const context = copy.getContext("2d");
            context.drawImage(canvas, 0, 0);
            resolve(new Uint8Array(context.getImageData(0, 0, width, height).data));
          } catch (error) {
            reject(error);
          }
        });
      });
    },
    dispose() {
      renderer.free();
      canvas.remove();
    },
  };
}

function material(renderer, operator) {
  renderer.setPbr(true);
  renderer.setPbrMaterial(0, 0.35, 0.6, 0, 0.9, 0.45, 0.03, operator);
}

function base64(bytes) {
  let binary = "";
  for (let i = 0; i < bytes.length; i += 8192) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 8192));
  }
  return btoa(binary);
}

export async function run(goldenLimits) {
  await init();
  const stage = await bytes("/stage2.arrow");
  const params = await bytes("/params.arrow");
  const oldParams = await bytes("/old-params.arrow");
  const oldMesh = await bytes("/old-mesh.arrow");
  const override = await bytes("/tonemap.arrow");
  const env = await bytes("/env.hdr");
  const mesh = await bytes("/mesh.glb");
  const backgrounds = await Promise.all(
    [0, 1, 2].map((row) => image(`/frames/stage2-resource-${row}.png`)),
  );
  const goldens = await Promise.all([0, 1, 2].map((row) => image(`/stage2_frame_${row}.png`)));
  const byTarget = {};
  const ipcHashes = [];
  const ipcChunks = [];

  for (const kind of ["offscreen", "canvas"]) {
    const t = await target(kind);
    const r = t.renderer;
    const document = ArrowSceneDocument.fromArrow(stage);
    try {
      assert(r.frameCount() === 0 && r.meshResourceCount() === 0, `${kind}: initial counts`);
      assert(
        r.pushIpc === undefined && r.resolveGltf === undefined,
        `${kind}: retired entrypoints`,
      );
      await rejects(() => t.render(0), /load a params\/GLB document first/);
      await rejects(() => r.loadIpc(oldParams), /0\.0\.6/);
      await rejects(() => r.loadIpc(oldMesh), /0\.0\.6/);
      await rejects(() => r.loadIpc(stage.subarray(0, 32)), /.+/);
      assert(r.loadIpc(params) === 3 && r.meshResourceCount() === 0, `${kind}: params-only cube`);
      const cube = await t.render(0);
      assert(
        cube.some((value, index) => index % 4 !== 3 && value !== 0),
        `${kind}: visible cube; alpha=${cube[3]}, max=${cube.reduce((a, b) => Math.max(a, b), 0)}`,
      );

      assert(r.loadSceneDocument(document) === 3, `${kind}: document load`);
      assert(r.meshResourceCount() === 2 && document.objectCount(0) === 2, `${kind}: two GLBs`);
      await rejects(() => r.frameRef(3), /out of range/);
      await rejects(() => t.render(3), /out of range/);
      await rejects(() => r.updateFrameTextureRgba(new Uint8Array(3), 1, 1), /width\*height\*4/);
      r.setTextured(true);
      r.setShowAabb(true);
      r.setShowLocalAxes(true);
      r.setCompositeFrame(true);
      byTarget[kind] = [];

      // Non-monotonic row order exercises seeking without a second decoder/cache.
      for (const row of [2, 0, 1, 0]) {
        assert(r.frameRef(row).endsWith(`stage2-resource-${row}.png`), `${kind}: frame reference`);
        r.updateFrameTextureRgba(backgrounds[row], width, height);
        const rgba = await t.render(row);
        compare(
          rgba,
          goldens[row],
          `${kind}: golden row ${row}`,
          goldenLimits.channelEps,
          goldenLimits.maxDiffFraction,
        );
        byTarget[kind].push(rgba);
        if (kind === "offscreen" && ipcHashes.length < 3) {
          ipcHashes.push(await hash(rgba));
          r.updateFrameTextureRgba(backgrounds[row], width, height);
          ipcChunks.push(await r.renderIpc(row));
        }
      }
      compare(byTarget[kind][1], byTarget[kind][3], `${kind}: seek back`, 0);

      const withoutUpload = await t.render(0);
      r.setCompositeFrame(false);
      compare(await t.render(0), withoutUpload, `${kind}: no stale background`, 0);
      assert(
        (await hash(withoutUpload)) !== (await hash(byTarget[kind][1])),
        `${kind}: background used`,
      );

      // Editing the shared document stays visible after seeking and a fresh decoder.
      for (let row = 0; row < document.frameCount(); row++) {
        const model = document.getModel(row, 0);
        model[12] += 0.2;
        document.setModel(row, 0, model);
      }
      const edited = [];
      for (const row of [0, 2, 1]) edited.push(await t.render(row));
      assert(
        (await hash(edited[0])) !== (await hash(withoutUpload)),
        `${kind}: edit changes pixels`,
      );
      const reopened = ArrowSceneDocument.fromArrow(document.exportArrow());
      try {
        r.loadSceneDocument(reopened);
        for (const [index, row] of [0, 2, 1].entries()) {
          compare(await t.render(row), edited[index], `${kind}: edited reload row ${row}`, 0);
        }
      } finally {
        reopened.free();
      }

      const inherited = ArrowSceneDocument.fromArrowWithGlb(params, mesh);
      r.loadSceneDocument(inherited);
      inherited.free();
      const wireframeBaseline = await t.render(0);
      r.setWireframe(true);
      assert(
        (await hash(await t.render(0))) !== (await hash(wireframeBaseline)),
        `${kind}: wireframe`,
      );
      r.setShowAxes(true);
      r.setEnvMapHdr(env);
      material(r, "reinhard");
      r.setEnvBackground(true, 0.2);
      r.loadIpc(override);
      const stagedBeforeLoad = await t.render(0);
      // The document's ACES override must apply to the sky as well as mesh materials.
      material(r, "aces");
      compare(await t.render(0), stagedBeforeLoad, `${kind}: loaded sky tonemap`, 0);
      material(r, "reinhard");
      r.loadIpc(params);
      const withoutOverride = await t.render(0);
      material(r, "reinhard");
      compare(await t.render(0), withoutOverride, `${kind}: cleared sky tonemap`, 0);
      assert(
        (await hash(withoutOverride)) !== (await hash(stagedBeforeLoad)),
        `${kind}: tone curves`,
      );

      const ending = r.finish();
      if (kind === "offscreen") ipcChunks.push(ending);
      await rejects(() => r.finish(), /finished/);
      await rejects(() => r.loadIpc(stage), /finished/);
      await rejects(() => t.render(0), /finished/);
      await rejects(() => r.updateFrameTextureRgba(backgrounds[0], width, height), /finished/);
      results.push({ name: `${kind}: lifecycle, loading, edits, background, PBR`, passed: true });
    } finally {
      document.free();
      t.dispose();
    }
  }
  for (let i = 0; i < byTarget.canvas.length; i++) {
    compare(byTarget.canvas[i], byTarget.offscreen[i], `canvas/offscreen ${i}`);
  }
  const ipc = new Uint8Array(ipcChunks.reduce((total, chunk) => total + chunk.length, 0));
  let offset = 0;
  for (const chunk of ipcChunks) {
    ipc.set(chunk, offset);
    offset += chunk.length;
  }
  return { results, ipc: base64(ipc), ipcHashes };
}
