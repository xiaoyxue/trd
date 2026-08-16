/// Random access to a video's bytes, so nothing above this layer needs to know
/// whether it is reading a local file or a URL.
///
/// This is what makes a multi-gigabyte video openable at all, and it is not new
/// work: a `<video>` element seeks by issuing exactly these range requests
/// internally — which is why a server that answers `200` instead of `206` lets
/// the element play straight through but never scrub. WebCodecs unbundles the
/// decoder from the container (#282), so the byte fetching becomes ours to do.
/// The same bytes move; only the control does.
export interface ByteSource {
  /// Total length in bytes.
  readonly size: number;
  /// Where the bytes come from, for diagnostics.
  readonly label: string;
  /// Bytes actually transferred so far — the number that shows an open or a
  /// seek stayed cheap instead of quietly pulling the whole file.
  readonly bytesRead: number;
  /// Reads up to `length` bytes at `offset`. A short read at the end of the
  /// source is not an error; a zero-length result means there is nothing there.
  read(offset: number, length: number): Promise<ArrayBuffer>;
}

/// Largest source this will pull in one piece when a server refuses ranges.
/// Well under a browser's `ArrayBuffer` ceiling and far under the videos this
/// editor is meant to open — past it, refusing with an explanation beats
/// silently downloading gigabytes.
const WHOLE_FILE_LIMIT = 256 * 1024 * 1024;

class BlobByteSource implements ByteSource {
  readonly label: string;
  readonly #blob: Blob;
  #bytesRead = 0;

  constructor(blob: Blob, label: string) {
    this.#blob = blob;
    this.label = label;
  }

  get size(): number {
    return this.#blob.size;
  }

  get bytesRead(): number {
    return this.#bytesRead;
  }

  async read(offset: number, length: number): Promise<ArrayBuffer> {
    const end = Math.min(offset + length, this.#blob.size);
    if (end <= offset) {
      return new ArrayBuffer(0);
    }
    const buffer = await this.#blob.slice(offset, end).arrayBuffer();
    this.#bytesRead += buffer.byteLength;
    return buffer;
  }
}

class HttpByteSource implements ByteSource {
  readonly label: string;
  readonly size: number;
  readonly #url: string;
  #bytesRead = 0;

  constructor(url: string, size: number) {
    this.#url = url;
    this.size = size;
    this.label = url;
  }

  get bytesRead(): number {
    return this.#bytesRead;
  }

  async read(offset: number, length: number): Promise<ArrayBuffer> {
    const last = Math.min(offset + length, this.size) - 1;
    if (last < offset) {
      return new ArrayBuffer(0);
    }
    const response = await fetch(this.#url, { headers: { Range: `bytes=${offset}-${last}` } });
    // A `200` here means the server ignored the range and is sending the whole
    // file. Accepting it would turn one seek into a multi-gigabyte download, so
    // it is an error rather than a slow path.
    if (response.status !== 206) {
      throw new Error(
        `range request returned ${response.status} ${response.statusText}, expected 206`,
      );
    }
    const buffer = await response.arrayBuffer();
    this.#bytesRead += buffer.byteLength;
    return buffer;
  }
}

/// Reads the total length out of a `Content-Range: bytes 0-0/12345` header.
function sizeFromContentRange(header: string | null): number | undefined {
  const total = header?.split("/")[1];
  if (!total || total === "*") {
    return undefined;
  }
  const size = Number(total);
  return Number.isFinite(size) && size > 0 ? size : undefined;
}

export function fileByteSource(file: File): ByteSource {
  return new BlobByteSource(file, file.name);
}

/// Opens a URL for random access, preferring ranges and falling back to a whole
/// download only for something small enough to justify it.
export async function urlByteSource(url: string): Promise<ByteSource> {
  let probe: Response;
  try {
    probe = await fetch(url, { headers: { Range: "bytes=0-0" } });
  } catch (error) {
    // A `fetch` rejection is the browser refusing the request rather than the
    // server answering, so there is no status to report — almost always a
    // missing CORS header or nothing listening.
    throw new Error(`${String(error)} — a cross-origin video needs Access-Control-Allow-Origin`);
  }
  if (!probe.ok && probe.status !== 206) {
    throw new Error(`failed to open video URL: ${probe.status} ${probe.statusText}`);
  }
  const servesRanges = probe.status === 206;
  let size = servesRanges
    ? sizeFromContentRange(probe.headers.get("content-range"))
    : Number(probe.headers.get("content-length") ?? Number.NaN);
  // Nothing below needs this response's body, and for a `200` it is the entire
  // file — cancel it rather than let it stream.
  await probe.body?.cancel();

  if (servesRanges && size === undefined) {
    // `Content-Range` is not a CORS-safelisted response header, so a
    // cross-origin server that serves ranges can still hide the length unless
    // it names the header in `Access-Control-Expose-Headers`. `Content-Length`
    // on a `HEAD` is safelisted, so ask that way instead of giving up.
    const head = await fetch(url, { method: "HEAD" });
    size = Number(head.headers.get("content-length") ?? Number.NaN);
  }
  if (size === undefined || !Number.isFinite(size) || size <= 0) {
    throw new Error(
      `cannot determine the length of "${url}" — the server must expose Content-Range or Content-Length`,
    );
  }
  if (servesRanges) {
    return new HttpByteSource(url, size);
  }
  if (size > WHOLE_FILE_LIMIT) {
    throw new Error(
      `"${url}" is ${(size / 1_048_576).toFixed(0)} MiB and the server does not support Range ` +
        "requests, so it could only be opened by downloading all of it — serve it with byte ranges",
    );
  }
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`failed to fetch video: ${response.status} ${response.statusText}`);
  }
  return new BlobByteSource(await response.blob(), url);
}
