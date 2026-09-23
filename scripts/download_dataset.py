from __future__ import annotations

import argparse
import gzip
import shutil
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download
from tqdm import tqdm


DEFAULT_REPO_ID = "yangyz1230/space"
GENOME_URLS = {
    "hg38": "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz",
    "mm10": "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the human and mouse Enformer data and, optionally, "
            "the test sets and genome assemblies."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./DATASET"),
        help="Directory in which the downloaded data will be stored.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face dataset repository to download from.",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Also download the human and mouse test H5/BED files.",
    )
    parser.add_argument(
        "--species",
        nargs="+",
        choices=("human", "mouse"),
        default=["human", "mouse"],
        help="Species to download; defaults to both human and mouse.",
    )
    parser.add_argument(
        "--download-hg38",
        action="store_true",
        help="Download and decompress the hg38 FASTA into output-dir/genome.",
    )
    parser.add_argument(
        "--download-mm10",
        action="store_true",
        help="Download and decompress the mm10 FASTA into output-dir/genome.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token for a gated or private dataset.",
    )
    parser.add_argument(
        "--keep-train-parts",
        action="store_true",
        help="Keep split train files after reconstructing the train H5 files.",
    )
    parser.add_argument(
        "--keep-genome-archive",
        action="store_true",
        help="Keep compressed genome archives after decompressing the FASTA files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload or replace files that already exist in the output directory.",
    )
    return parser.parse_args()


def species_patterns(species: str, include_test: bool) -> list[str]:
    prefix = species
    patterns = [
        f"{prefix}_train.bed",
        f"{prefix}_train.h5",
        f"{prefix}_train_part_*",
        f"{prefix}_valid.bed",
        f"{prefix}_valid.h5",
        f"targets_{prefix}_sorted.txt",
    ]
    if include_test:
        patterns.extend([f"{prefix}_test.bed", f"{prefix}_test.h5"])
    return patterns


def reconstruct_species_train(
    output_dir: Path,
    *,
    species: str,
    keep_parts: bool,
    force: bool,
) -> Path:
    destination = output_dir / f"{species}_train.h5"
    parts = sorted(output_dir.glob(f"{species}_train_part_*"))

    if destination.exists() and not force:
        print(f"Using existing {destination}")
        return destination

    if not parts:
        if destination.exists():
            return destination
        raise FileNotFoundError(
            f"No {species}_train_part_* files were downloaded and {destination.name} "
            "does not exist."
        )

    temporary_destination = destination.with_name(destination.name + ".partial")
    if temporary_destination.exists():
        temporary_destination.unlink()

    print(f"Reconstructing {destination} from {len(parts)} split files")
    with temporary_destination.open("wb") as destination_file:
        for part in parts:
            print(f"  appending {part.name}")
            with part.open("rb") as part_file:
                shutil.copyfileobj(part_file, destination_file, length=16 * 1024 * 1024)

    temporary_destination.replace(destination)

    if not keep_parts:
        for part in parts:
            part.unlink()

    return destination


def download_file(url: str, destination: Path, *, force: bool) -> None:
    if destination.exists() and not force:
        print(f"Using existing {destination}")
        return

    temporary_destination = destination.with_name(destination.name + ".partial")
    if temporary_destination.exists():
        temporary_destination.unlink()

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "enformer-training-guide/0.1"},
    )
    print(f"Downloading {url}")
    with urllib.request.urlopen(request) as response, temporary_destination.open("wb") as output:
        total_size = int(response.headers.get("Content-Length", 0))
        progress = tqdm(
            total=total_size or None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=destination.name,
        )
        try:
            while True:
                chunk = response.read(16 * 1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                progress.update(len(chunk))
        finally:
            progress.close()

    temporary_destination.replace(destination)


def download_genome(
    output_dir: Path,
    *,
    assembly: str,
    keep_archive: bool,
    force: bool,
) -> Path:
    genome_dir = output_dir / "genome"
    genome_dir.mkdir(parents=True, exist_ok=True)
    archive_path = genome_dir / f"{assembly}.fa.gz"
    fasta_path = genome_dir / f"{assembly}.fa"

    if fasta_path.exists() and not force:
        print(f"Using existing {fasta_path}")
        return fasta_path

    download_file(GENOME_URLS[assembly], archive_path, force=force)

    if not fasta_path.exists() or force:
        temporary_fasta = fasta_path.with_name(fasta_path.name + ".partial")
        if temporary_fasta.exists():
            temporary_fasta.unlink()
        print(f"Decompressing {archive_path} to {fasta_path}")
        with gzip.open(archive_path, "rb") as compressed, temporary_fasta.open("wb") as fasta:
            shutil.copyfileobj(compressed, fasta, length=16 * 1024 * 1024)
        temporary_fasta.replace(fasta_path)

    if not keep_archive:
        archive_path.unlink()

    return fasta_path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {', '.join(args.species)} data to {output_dir}")
    for species in args.species:
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=str(output_dir),
            allow_patterns=species_patterns(species, args.include_test),
            token=args.token,
            force_download=args.force,
        )
        reconstruct_species_train(
            output_dir,
            species=species,
            keep_parts=args.keep_train_parts,
            force=args.force,
        )

    assemblies = []
    if args.download_hg38:
        assemblies.append("hg38")
    if args.download_mm10:
        assemblies.append("mm10")

    for assembly in assemblies:
        genome_path = download_genome(
            output_dir,
            assembly=assembly,
            keep_archive=args.keep_genome_archive,
            force=args.force,
        )
        print(f"{assembly} FASTA available at {genome_path}")

    if not assemblies:
        print("Genome downloads skipped; pass --download-hg38 and/or --download-mm10.")

    if args.include_test:
        print(f"Test sets included for: {', '.join(args.species)}.")
    else:
        print("Test sets skipped; pass --include-test to add them.")


if __name__ == "__main__":
    main()
