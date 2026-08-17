#!/usr/bin/env python3
"""Write the Parquet fixtures the video-editing document tests read.

Both containers decode through one code path, so the tests need the *same*
document in both. The Arrow timeline is generated from an external MP4 by
``fiba_video_editing_bundle.py`` and is gitignored; this converts it, plus one
copy per compression codec, into the directory the tests look in
(``TRD_DOC_DIR``).

The codec copies drive ``unsupported_compression_says_so_clearly``: ``snappy``
and uncompressed must read, while ``zstd``/``gzip`` must be refused, because
those two are C shims left out so the crate keeps cross-compiling to wasm32.
A codec this pyarrow cannot write is reported and skipped rather than aborting
the run.
"""

import argparse
from pathlib import Path
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARROW = REPOSITORY_ROOT / "web/gui-video-editing/data/fiba-shot1.arrow"

# `none` and `snappy` must read; `zstd` and `gzip` must be refused.
CODECS = ("snappy", "none", "zstd", "gzip")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arrow",
        type=Path,
        default=DEFAULT_ARROW,
        help="the generated Arrow document to convert (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=Path(tempfile.gettempdir()),
        help="where the tests look, i.e. TRD_DOC_DIR (default: %(default)s)",
    )
    args = parser.parse_args()

    if not args.arrow.exists():
        parser.error(
            f"{args.arrow} does not exist; generate it first with "
            "scripts/fiba_video_editing_bundle.py"
        )

    with pa.ipc.open_stream(pa.memory_map(str(args.arrow))) as reader:
        table = reader.read_all()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"read {table.num_rows} rows from {args.arrow}")

    # The parity test reads this one; the codec test reads the rest.
    written = [args.out_dir / "fiba-shot1.parquet"]
    pq.write_table(table, written[0])
    for codec in CODECS:
        path = args.out_dir / f"fiba-{codec}.parquet"
        try:
            pq.write_table(table, path, compression=codec)
        except Exception as error:
            print(f"  {codec}: unavailable in this pyarrow, skipped ({error})")
            continue
        written.append(path)

    for path in written:
        print(f"  wrote {path.name} ({path.stat().st_size} bytes)")
    print(f"now run: TRD_DOC_DIR={args.out_dir} cargo test -p trd-core --lib video_editing -- --ignored")


if __name__ == "__main__":
    main()
