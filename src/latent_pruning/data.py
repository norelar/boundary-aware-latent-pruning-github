"""Image-folder scanning, leakage-safe splitting, and materialization."""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class ImageSample:
    path: Path
    label: str
    class_index: int

    def to_dict(self) -> dict[str, str | int]:
        return {"path": str(self.path), "label": self.label, "class_index": self.class_index}


@dataclass(frozen=True)
class DatasetIndex:
    root: Path
    classes: tuple[str, ...]
    samples: tuple[ImageSample, ...]

    @property
    def class_to_index(self) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.classes)}

    def counts(self) -> dict[str, int]:
        counter = Counter(sample.label for sample in self.samples)
        return {label: int(counter.get(label, 0)) for label in self.classes}


def scan_image_folder(
    root: str | Path,
    extensions: Sequence[str],
    expected_classes: Sequence[str] | None = None,
) -> DatasetIndex:
    base = Path(root)
    if not base.is_dir():
        raise FileNotFoundError(f"Image-folder root not found: {base}")
    allowed = {extension.lower() if extension.startswith(".") else f".{extension.lower()}" for extension in extensions}
    classes = tuple(sorted(path.name for path in base.iterdir() if path.is_dir()))
    if not classes:
        raise ValueError(f"No class directories found under {base}")
    if expected_classes is not None and set(classes) != set(expected_classes):
        missing = sorted(set(expected_classes) - set(classes))
        extra = sorted(set(classes) - set(expected_classes))
        raise ValueError(f"Class mismatch for {base}: missing={missing}, extra={extra}")

    class_to_index = {label: index for index, label in enumerate(classes)}
    samples: list[ImageSample] = []
    for label in classes:
        paths = sorted(
            path for path in (base / label).rglob("*") if path.is_file() and path.suffix.lower() in allowed
        )
        if not paths:
            raise ValueError(f"Class {label!r} contains no supported images under {base}")
        samples.extend(ImageSample(path.resolve(), label, class_to_index[label]) for path in paths)
    return DatasetIndex(base.resolve(), classes, tuple(samples))


def check_compatible_classes(reference: DatasetIndex, other: DatasetIndex) -> None:
    if reference.classes != other.classes:
        raise ValueError(
            f"Dataset classes differ: {reference.root} has {reference.classes}, "
            f"but {other.root} has {other.classes}"
        )


def stratified_split(
    samples: Sequence[ImageSample], validation_fraction: float, seed: int
) -> tuple[list[ImageSample], list[ImageSample]]:
    if validation_fraction == 0.0:
        return list(samples), []
    labels = np.asarray([sample.class_index for sample in samples])
    counts = Counter(labels.tolist())
    too_small = [label for label, count in counts.items() if count < 2]
    if too_small:
        raise ValueError(f"Cannot create a stratified split; class indices {too_small} have <2 images")
    indices = np.arange(len(samples))
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=seed,
        stratify=labels,
    )
    return (
        [samples[int(index)] for index in sorted(train_indices)],
        [samples[int(index)] for index in sorted(validation_indices)],
    )


def write_sample_manifest(samples: Iterable[ImageSample], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "label", "class_index"])
        writer.writeheader()
        for sample in samples:
            writer.writerow(sample.to_dict())
    return destination


def read_sample_manifest(path: str | Path) -> list[ImageSample]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return [
            ImageSample(Path(row["path"]), row["label"], int(row["class_index"]))
            for row in reader
        ]


def _unique_destination(directory: Path, source: Path) -> Path:
    candidate = directory / source.name
    if not candidate.exists():
        return candidate
    digest = hashlib.sha1(str(source).encode("utf-8"), usedforsecurity=False).hexdigest()[:10]
    return directory / f"{source.stem}_{digest}{source.suffix.lower()}"


def materialize_samples(
    samples: Sequence[ImageSample],
    destination: str | Path,
    mode: str = "copy",
    overwrite: bool = False,
) -> Path:
    """Create a complete ImageFolder tree without flattening class semantics."""
    if mode == "manifest":
        return write_sample_manifest(samples, Path(destination) / "samples.csv")
    if mode not in {"copy", "symlink", "hardlink"}:
        raise ValueError(f"Unsupported materialization mode: {mode}")

    target = Path(destination)
    if target.exists() and any(target.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Destination is not empty: {target}; pass overwrite=True")
        resolved = target.resolve()
        if len(resolved.parts) < 4:
            raise ValueError(f"Refusing to replace an unsafe destination: {resolved}")
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        class_dir = target / sample.label
        class_dir.mkdir(parents=True, exist_ok=True)
        output = _unique_destination(class_dir, sample.path)
        if mode == "copy":
            shutil.copy2(sample.path, output)
        elif mode == "symlink":
            output.symlink_to(sample.path)
        else:
            os.link(sample.path, output)
    write_sample_manifest(samples, target / "source_manifest.csv")
    return target


def summarize_samples(samples: Sequence[ImageSample]) -> dict[str, int]:
    counts = Counter(sample.label for sample in samples)
    return dict(sorted((str(label), int(count)) for label, count in counts.items()))
