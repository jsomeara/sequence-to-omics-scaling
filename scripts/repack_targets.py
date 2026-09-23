from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import queue
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np


def _copy_attrs(src_obj, dst_obj) -> None:
    for key, value in src_obj.attrs.items():
        try:
            dst_obj.attrs[key] = value
        except Exception:
            pass


def _aligned_ranges(
    n: int,
    nshards: int,
    source_chunk_examples: int,
) -> list[tuple[int, int]]:
    step = math.ceil(n / nshards)
    align = max(1, int(source_chunk_examples))

    ranges: list[tuple[int, int]] = []
    start = 0

    while start < n:
        raw_end = min(n, start + step)

        if raw_end < n:
            end = min(
                n,
                ((raw_end + align - 1) // align) * align,
            )
        else:
            end = n

        if end <= start:
            end = min(n, start + align)

        ranges.append((start, end))
        start = end

    return ranges


def _worker_repack(
    src_path: str,
    target_key: str,
    shard_path: str,
    shard_id: int,
    start: int,
    end: int,
    block_examples: int,
    compression: str,
    progress_queue,
) -> tuple[str, int, int]:

    src_path_p = Path(src_path)
    shard_path_p = Path(shard_path)

    with h5py.File(src_path_p, "r") as src:
        src_ds = src[target_key]

        bins = int(src_ds.shape[1])
        tracks = int(src_ds.shape[2])

        compression_arg = (
            None
            if compression == "none"
            else compression
        )

        with h5py.File(
            shard_path_p,
            "w",
            libver="latest",
        ) as dst:
            out = dst.create_dataset(
                target_key,
                shape=(end - start, bins, tracks),
                dtype=src_ds.dtype,
                chunks=(1, bins, tracks),
                compression=compression_arg,
            )

            _copy_attrs(src_ds, out)

            local = 0

            for g0 in range(start, end, block_examples):
                g1 = min(end, g0 + block_examples)

                # Contiguous source read.
                block = np.asarray(src_ds[g0:g1])

                # Contiguous destination write.
                out[local : local + (g1 - g0)] = block

                count = g1 - g0
                local += count

                progress_queue.put(
                    (
                        "progress",
                        shard_id,
                        count,
                        local,
                        end - start,
                    )
                )

            dst.flush()

    progress_queue.put(
        (
            "done",
            shard_id,
            0,
            end - start,
            end - start,
        )
    )

    return str(shard_path_p), start, end


def _make_vds(
    src_path: Path,
    output_path: Path,
    shard_ranges: list[tuple[Path, int, int]],
    target_key: str,
) -> None:

    with h5py.File(src_path, "r") as src:
        src_ds = src[target_key]

        shape = tuple(
            int(x)
            for x in src_ds.shape
        )

        layout = h5py.VirtualLayout(
            shape=shape,
            dtype=src_ds.dtype,
        )

        for shard_path, start, end in shard_ranges:
            relative = os.path.relpath(
                shard_path,
                output_path.parent,
            )

            source = h5py.VirtualSource(
                relative,
                target_key,
                shape=(
                    end - start,
                    shape[1],
                    shape[2],
                ),
            )

            layout[start:end, :, :] = source

        with h5py.File(
            output_path,
            "w",
            libver="latest",
        ) as dst:

            vds = dst.create_virtual_dataset(
                target_key,
                layout,
                fillvalue=0,
            )

            _copy_attrs(src_ds, vds)
            _copy_attrs(src, dst)

            dst.attrs["repacked_from"] = str(src_path)
            dst.attrs[
                "storage"
            ] = "HDF5 virtual dataset over optimized shards"

            dst.attrs[
                "optimized_chunking"
            ] = "(1, bins, tracks)"


def _format_seconds(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "--"

    seconds = max(0, int(seconds))

    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)

    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"

    return f"{minutes:02d}:{secs:02d}"


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Parallel Enformer HDF5 repacker with live progress."
        )
    )

    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)

    parser.add_argument(
        "--target-key",
        default="targets",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=min(
            8,
            os.cpu_count() or 1,
        ),
    )

    parser.add_argument(
        "--block-examples",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--compression",
        choices=("none", "lzf"),
        default="none",
    )

    parser.add_argument(
        "--keep-existing-shards",
        action="store_true",
    )

    parser.add_argument(
        "--progress-interval",
        type=float,
        default=1.0,
        help="Seconds between progress display updates.",
    )

    args = parser.parse_args()

    args.src = args.src.resolve()
    args.dst = args.dst.resolve()

    if args.workers < 1:
        raise ValueError(
            "--workers must be >= 1"
        )

    if args.block_examples < 1:
        raise ValueError(
            "--block-examples must be >= 1"
        )

    if not args.src.exists():
        raise FileNotFoundError(args.src)

    if args.src == args.dst:
        raise ValueError(
            "Source and destination must differ."
        )

    args.dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shard_dir = (
        args.dst.parent
        / f".{args.dst.stem}_shards"
    )

    if (
        shard_dir.exists()
        and not args.keep_existing_shards
    ):
        print(
            f"Removing old shard directory: "
            f"{shard_dir}"
        )

        shutil.rmtree(shard_dir)

    shard_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.dst.exists():
        print(
            f"Removing old destination: "
            f"{args.dst}"
        )

        args.dst.unlink()

    with h5py.File(
        args.src,
        "r",
    ) as src:

        if args.target_key not in src:
            raise KeyError(
                f"{args.target_key!r} not found. "
                f"Keys: {list(src.keys())}"
            )

        ds = src[args.target_key]

        if ds.ndim != 3:
            raise ValueError(
                f"Expected 3D targets dataset, "
                f"got {ds.shape}"
            )

        n = int(ds.shape[0])

        source_chunk_examples = (
            int(ds.chunks[0])
            if ds.chunks is not None
            else 1
        )

        print("Source:")
        print(f"  file:        {args.src}")
        print(f"  shape:       {ds.shape}")
        print(f"  dtype:       {ds.dtype}")
        print(f"  chunks:      {ds.chunks}")
        print(f"  compression: {ds.compression}")
        print()

    ranges = _aligned_ranges(
        n,
        args.workers,
        source_chunk_examples,
    )

    jobs = []

    for shard_id, (start, end) in enumerate(ranges):
        shard_path = (
            shard_dir
            / f"part-{shard_id:03d}.h5"
        )

        jobs.append(
            (
                shard_id,
                shard_path,
                start,
                end,
            )
        )

    print(
        f"Repacking {n:,} examples "
        f"with {len(jobs)} workers"
    )
    print(
        f"Block size: {args.block_examples}"
    )
    print(
        f"Output compression: {args.compression}"
    )
    print()

    ctx = mp.get_context("spawn")

    with mp.Manager() as manager:

        progress_queue = manager.Queue()

        completed_ranges: list[
            tuple[Path, int, int]
        ] = []

        shard_progress = {
            shard_id: 0
            for shard_id, _, _, _ in jobs
        }

        shard_totals = {
            shard_id: end - start
            for shard_id, _, start, end in jobs
        }

        already_done = 0

        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=ctx,
        ) as pool:

            futures = []

            for (
                shard_id,
                shard_path,
                start,
                end,
            ) in jobs:

                if (
                    args.keep_existing_shards
                    and shard_path.exists()
                ):
                    try:
                        with h5py.File(
                            shard_path,
                            "r",
                        ) as f:
                            if (
                                args.target_key in f
                                and
                                f[args.target_key].shape[0]
                                == end - start
                            ):
                                shard_progress[
                                    shard_id
                                ] = end - start

                                already_done += end - start

                                completed_ranges.append(
                                    (
                                        shard_path,
                                        start,
                                        end,
                                    )
                                )

                                continue
                    except Exception:
                        pass

                future = pool.submit(
                    _worker_repack,
                    str(args.src),
                    args.target_key,
                    str(shard_path),
                    shard_id,
                    start,
                    end,
                    args.block_examples,
                    args.compression,
                    progress_queue,
                )

                futures.append(
                    (
                        future,
                        shard_id,
                        shard_path,
                        start,
                        end,
                    )
                )

            start_time = time.monotonic()

            last_print = 0.0
            total_completed = already_done

            unfinished = {
                id(future): (
                    future,
                    shard_id,
                    shard_path,
                    start,
                    end,
                )
                for (
                    future,
                    shard_id,
                    shard_path,
                    start,
                    end,
                ) in futures
            }

            while unfinished:

                # Drain all currently available progress events.
                while True:
                    try:
                        (
                            event,
                            shard_id,
                            increment,
                            local_done,
                            shard_total,
                        ) = progress_queue.get_nowait()

                    except queue.Empty:
                        break

                    if event == "progress":
                        total_completed += increment
                        shard_progress[
                            shard_id
                        ] = local_done

                    elif event == "done":
                        shard_progress[
                            shard_id
                        ] = shard_total

                # Pick up completed futures / surface errors.
                finished_keys = []

                for key, (
                    future,
                    shard_id,
                    shard_path,
                    start,
                    end,
                ) in unfinished.items():

                    if future.done():
                        path_str, rs, re = (
                            future.result()
                        )

                        completed_ranges.append(
                            (
                                Path(path_str),
                                rs,
                                re,
                            )
                        )

                        finished_keys.append(key)

                for key in finished_keys:
                    del unfinished[key]

                now = time.monotonic()

                if (
                    now - last_print
                    >= args.progress_interval
                    or not unfinished
                ):
                    elapsed = max(
                        now - start_time,
                        1e-9,
                    )

                    rate = (
                        (total_completed - already_done)
                        / elapsed
                    )

                    percent = (
                        100.0
                        * total_completed
                        / n
                    )

                    remaining = (
                        n - total_completed
                    )

                    eta = (
                        remaining / rate
                        if rate > 0
                        else float("inf")
                    )

                    shard_text = " ".join(
                        (
                            f"{sid}:"
                            f"{100.0 * shard_progress[sid] / shard_totals[sid]:.0f}%"
                        )
                        for sid in sorted(
                            shard_progress
                        )
                    )

                    print(
                        "\r"
                        f"{percent:6.2f}%  "
                        f"{total_completed:,}/{n:,}  "
                        f"{rate:,.1f} ex/s  "
                        f"ETA {_format_seconds(eta)}  "
                        f"[{shard_text}]",
                        end="",
                        flush=True,
                    )

                    last_print = now

                time.sleep(0.05)

            # Final drain in case events arrived just before completion.
            while True:
                try:
                    (
                        event,
                        shard_id,
                        increment,
                        local_done,
                        shard_total,
                    ) = progress_queue.get_nowait()

                except queue.Empty:
                    break

                if event == "progress":
                    total_completed += increment
                    shard_progress[
                        shard_id
                    ] = local_done

            print()

        completed_ranges.sort(
            key=lambda x: x[1]
        )

    print()
    print(
        "Creating final HDF5 virtual dataset..."
    )

    _make_vds(
        args.src,
        args.dst,
        completed_ranges,
        args.target_key,
    )

    print()
    print("Done.")
    print(
        f"Training HDF5: {args.dst}"
    )
    print(
        f"Shard directory: {shard_dir}"
    )


if __name__ == "__main__":
    main()
