"""Reproducible track subsets without consuming the training RNG stream."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path


def select_tracks(total: int, fraction: float, seed: int, species: str,
                  count: int | None = None) -> list[int]:
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("--track-fraction must be in (0, 1].")
    if count is not None and fraction != 1.0:
        raise ValueError("Use track counts or --track-fraction, not both.")
    count = max(1, round(total * fraction)) if count is None else count
    if not 1 <= count <= total:
        raise ValueError(f"{species} track count must be between 1 and {total}.")
    if count == total:
        return list(range(total))
    indices = list(range(total))
    random.Random(f"{seed}:{species}").shuffle(indices)
    return sorted(indices[:count])


def save_manifest(directory: Path, manifest: dict) -> None:
    """Refuse to relabel an existing experiment with a different selection."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "track_selection.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError(f"Track selection differs from {path}; use a new output directory.")
        return
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def validate_resume(directory: Path, manifest: dict, *, subset: bool) -> None:
    path = directory / "track_selection.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError(f"Track selection does not match checkpoint manifest: {path}")
    elif subset:
        raise ValueError(f"Cannot resume a track subset without {path}.")
