"""Leakage-safe classifier training and held-out evaluation."""

from __future__ import annotations

import copy
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from latent_pruning.config import ExperimentConfig, TrainingConfig
from latent_pruning.data import ImageSample, stratified_split, summarize_samples, write_sample_manifest
from latent_pruning.losses import FocalLoss
from latent_pruning.metrics import (
    classification_metrics,
    prediction_table,
    save_confusion_matrix_plot,
)
from latent_pruning.models.classifiers import create_classifier
from latent_pruning.utils import environment_metadata, resolve_device, set_reproducibility, write_json

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageSampleDataset(Dataset):
    def __init__(self, samples: Sequence[ImageSample], transform):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, sample.class_index, str(sample.path)


@dataclass
class TrainingArtifact:
    output_dir: Path
    best_epoch: int
    validation_metrics: dict[str, float | None]
    test_metrics: dict[str, float | None]
    history: pd.DataFrame


def _transforms(image_size: int):
    # The supplied notebooks used resize + ImageNet normalization without
    # augmentation. Keeping this default makes the modular code comparable.
    common = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return common, common


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def _make_loader(
    samples: Sequence[ImageSample],
    transform,
    config: TrainingConfig,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ImageSampleDataset(samples, transform),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=config.num_workers > 0,
    )


def _class_weights(samples: Sequence[ImageSample], class_count: int, device: str) -> torch.Tensor:
    counts = np.bincount([sample.class_index for sample in samples], minlength=class_count).astype(float)
    weights = counts.sum() / (class_count * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _criterion(
    config: TrainingConfig,
    samples: Sequence[ImageSample],
    class_count: int,
    device: str,
) -> nn.Module:
    weights = _class_weights(samples, class_count, device) if config.use_class_weights else None
    if config.loss == "cross_entropy":
        return nn.CrossEntropyLoss(weight=weights)
    if config.loss == "focal":
        return FocalLoss(gamma=2.0, weight=weights)
    raise ValueError(f"Unsupported loss: {config.loss}")


def _optimizer(model: nn.Module, config: TrainingConfig):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if config.optimizer == "adam":
        return torch.optim.Adam(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.optimizer == "adamw":
        return torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    raise ValueError(config.optimizer)


def _scheduler(optimizer, config: TrainingConfig):
    if config.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma
        )
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    if config.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=config.scheduler_gamma, patience=2
        )
    if config.scheduler == "none":
        return None
    raise ValueError(config.scheduler)


def _autocast(device: str, enabled: bool):
    device_type = device.split(":", 1)[0]
    return torch.autocast(device_type=device_type, enabled=enabled and device_type == "cuda")


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    optimizer=None,
    scaler=None,
    mixed_precision: bool = False,
) -> tuple[float, list[int], np.ndarray, list[str]]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    sample_count = 0
    truths: list[int] = []
    probability_batches: list[np.ndarray] = []
    paths: list[str] = []

    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for inputs, labels, batch_paths in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with _autocast(device, mixed_precision):
                logits = model(inputs)
                loss = criterion(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            loss_sum += float(loss.item()) * inputs.size(0)
            sample_count += inputs.size(0)
            truths.extend(labels.detach().cpu().tolist())
            probability_batches.append(torch.softmax(logits.detach().float(), dim=1).cpu().numpy())
            paths.extend(batch_paths)
    return loss_sum / max(sample_count, 1), truths, np.concatenate(probability_batches), paths


def _class_names(samples: Sequence[ImageSample]) -> list[str]:
    mapping = {sample.class_index: sample.label for sample in samples}
    expected = list(range(len(mapping)))
    if sorted(mapping) != expected:
        raise ValueError(f"Class indices must be contiguous from zero, got {sorted(mapping)}")
    return [mapping[index] for index in expected]


def _validate_samples(
    train_samples: Sequence[ImageSample],
    validation_samples: Sequence[ImageSample],
    test_samples: Sequence[ImageSample],
) -> list[str]:
    class_names = _class_names([*train_samples, *validation_samples, *test_samples])
    for name, split in (
        ("train", train_samples),
        ("validation", validation_samples),
        ("test", test_samples),
    ):
        present = {sample.class_index for sample in split}
        missing = sorted(set(range(len(class_names))) - present)
        if missing:
            raise ValueError(f"{name} split has no samples for class indices {missing}")
    train_paths = {sample.path for sample in train_samples}
    validation_paths = {sample.path for sample in validation_samples}
    test_paths = {sample.path for sample in test_samples}
    if train_paths & validation_paths or train_paths & test_paths or validation_paths & test_paths:
        raise ValueError("Train, validation, and test splits contain overlapping paths")
    return class_names


def train_classifier(
    train_samples: Sequence[ImageSample],
    validation_samples: Sequence[ImageSample],
    test_samples: Sequence[ImageSample],
    config: ExperimentConfig,
    seed: int,
    output_dir: str | Path,
    device: str = "auto",
) -> TrainingArtifact:
    set_reproducibility(seed)
    resolved_device = resolve_device(device)
    class_names = _validate_samples(train_samples, validation_samples, test_samples)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_transform, evaluation_transform = _transforms(config.training.image_size)
    train_loader = _make_loader(train_samples, train_transform, config.training, seed, shuffle=True)
    validation_loader = _make_loader(
        validation_samples, evaluation_transform, config.training, seed, shuffle=False
    )
    test_loader = _make_loader(test_samples, evaluation_transform, config.training, seed, shuffle=False)

    model = create_classifier(
        config.training.model,
        len(class_names),
        pretrained=config.training.pretrained,
        hidden_dim=config.training.hidden_dim,
        dropout=config.training.dropout,
    ).to(resolved_device)
    criterion = _criterion(
        config.training, train_samples, len(class_names), resolved_device
    )
    optimizer = _optimizer(model, config.training)
    scheduler = _scheduler(optimizer, config.training)
    amp_enabled = config.training.mixed_precision and resolved_device.startswith("cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_epoch = -1
    best_score = -math.inf
    best_state = None
    epochs_without_improvement = 0
    history_rows: list[dict[str, float | int | None]] = []
    started = time.perf_counter()

    for epoch in range(config.training.epochs):
        train_loss, train_truth, train_probability, _ = _run_epoch(
            model,
            train_loader,
            criterion,
            resolved_device,
            optimizer=optimizer,
            scaler=scaler,
            mixed_precision=amp_enabled,
        )
        validation_loss, validation_truth, validation_probability, _ = _run_epoch(
            model,
            validation_loader,
            criterion,
            resolved_device,
            mixed_precision=amp_enabled,
        )
        train_metrics, _, _ = classification_metrics(train_truth, train_probability, class_names)
        validation_metrics, _, _ = classification_metrics(
            validation_truth, validation_probability, class_names
        )
        history_rows.append(
            {
                "epoch": epoch + 1,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_loss": train_loss,
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "validation_loss": validation_loss,
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_macro_f1": validation_metrics["macro_f1"],
                "validation_weighted_f1": validation_metrics["weighted_f1"],
            }
        )
        score = float(validation_metrics["macro_f1"] or 0.0)
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if scheduler is not None:
            if config.training.scheduler == "plateau":
                scheduler.step(score)
            else:
                scheduler.step()
        if epochs_without_improvement >= config.training.early_stopping_patience:
            break

    if best_state is None:
        raise RuntimeError("Training completed without a model checkpoint")
    model.load_state_dict(best_state)
    checkpoint_path = output / "best_model.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "model_name": config.training.model,
            "class_names": class_names,
            "best_epoch": best_epoch,
            "seed": seed,
            "resolved_config": config.to_dict(),
        },
        checkpoint_path,
    )

    _, validation_truth, validation_probability, validation_paths = _run_epoch(
        model, validation_loader, criterion, resolved_device, mixed_precision=amp_enabled
    )
    test_loss, test_truth, test_probability, test_paths = _run_epoch(
        model, test_loader, criterion, resolved_device, mixed_precision=amp_enabled
    )
    validation_metrics, _, _ = classification_metrics(
        validation_truth, validation_probability, class_names
    )
    test_metrics, per_class, matrix = classification_metrics(test_truth, test_probability, class_names)
    test_metrics["loss"] = test_loss
    test_metrics["best_epoch"] = best_epoch
    test_metrics["training_minutes"] = (time.perf_counter() - started) / 60.0
    history = pd.DataFrame(history_rows)
    history.to_csv(output / "history.csv", index=False)
    per_class.to_csv(output / "per_class_metrics.csv")
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(
        output / "confusion_matrix.csv"
    )
    save_confusion_matrix_plot(
        matrix,
        class_names,
        output / "confusion_matrix.png",
        f"{config.dataset.name} | {config.training.model} | seed={seed}",
    )
    prediction_table(test_paths, test_truth, test_probability, class_names).to_csv(
        output / "test_predictions.csv", index=False
    )
    prediction_table(
        validation_paths, validation_truth, validation_probability, class_names
    ).to_csv(output / "validation_predictions.csv", index=False)
    write_sample_manifest(train_samples, output / "train_manifest.csv")
    write_sample_manifest(validation_samples, output / "validation_manifest.csv")
    write_sample_manifest(test_samples, output / "test_manifest.csv")
    write_json(output / "validation_metrics.json", validation_metrics)
    write_json(output / "test_metrics.json", test_metrics)
    write_json(
        output / "run_metadata.json",
        {
            "dataset": config.dataset.name,
            "model": config.training.model,
            "seed": seed,
            "device": resolved_device,
            "class_names": class_names,
            "split_counts": {
                "train": summarize_samples(train_samples),
                "validation": summarize_samples(validation_samples),
                "test": summarize_samples(test_samples),
            },
            "environment": environment_metadata(),
        },
    )
    return TrainingArtifact(output, best_epoch, validation_metrics, test_metrics, history)


def split_and_train_classifier(
    full_train_samples: Sequence[ImageSample],
    test_samples: Sequence[ImageSample],
    config: ExperimentConfig,
    seed: int,
    output_dir: str | Path,
    validation_samples: Sequence[ImageSample] | None = None,
    device: str = "auto",
) -> TrainingArtifact:
    if validation_samples is None:
        train_samples, validation = stratified_split(
            full_train_samples,
            config.training.validation_fraction,
            seed,
        )
    else:
        train_samples, validation = list(full_train_samples), list(validation_samples)
    if not validation:
        raise ValueError("A non-empty validation set is required for checkpoint selection")
    return train_classifier(
        train_samples,
        validation,
        test_samples,
        config,
        seed,
        output_dir,
        device=device,
    )
