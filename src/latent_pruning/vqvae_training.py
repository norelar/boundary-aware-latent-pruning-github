"""VQ-VAE training with resumable checkpoints and validation loss."""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from latent_pruning.config import ExperimentConfig
from latent_pruning.data import ImageSample, stratified_split
from latent_pruning.models.vqvae import create_vqvae
from latent_pruning.utils import resolve_device, set_reproducibility, write_json


class ReconstructionDataset(Dataset):
    def __init__(self, samples: Sequence[ImageSample], image_size: int):
        self.samples = list(samples)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        import numpy as np

        with Image.open(self.samples[index].path) as image:
            image = image.convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _reconstruction_epoch(model, loader, device: str, optimizer=None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "Reconstruction_Loss": 0.0, "VQ_Loss": 0.0}
    count = 0
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for inputs in loader:
            inputs = inputs.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(inputs)
            losses = model.loss_function(*output)
            if training:
                losses["loss"].backward()
                optimizer.step()
            batch_size = inputs.size(0)
            count += batch_size
            for name in totals:
                totals[name] += float(losses[name].item()) * batch_size
    return {name: value / max(1, count) for name, value in totals.items()}


def train_vqvae(
    samples: Sequence[ImageSample],
    config: ExperimentConfig,
    output_dir: str | Path,
    seed: int = 42,
    epochs: int = 300,
    learning_rate: float = 1e-4,
    validation_fraction: float = 0.10,
    device: str = "auto",
    resume: str | Path | None = None,
) -> Path:
    set_reproducibility(seed)
    resolved_device = resolve_device(device)
    train_samples, validation_samples = stratified_split(samples, validation_fraction, seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        ReconstructionDataset(train_samples, config.vqvae.image_size),
        batch_size=config.vqvae.batch_size,
        shuffle=True,
        num_workers=config.vqvae.num_workers,
        pin_memory=resolved_device.startswith("cuda"),
        generator=generator,
    )
    validation_loader = DataLoader(
        ReconstructionDataset(validation_samples, config.vqvae.image_size),
        batch_size=config.vqvae.batch_size,
        shuffle=False,
        num_workers=config.vqvae.num_workers,
        pin_memory=resolved_device.startswith("cuda"),
    )
    model = create_vqvae(config.vqvae).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start_epoch = 0
    if resume is not None:
        checkpoint = torch.load(resume, map_location=resolved_device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1

    best_loss = float("inf")
    best_state = None
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        train_metrics = _reconstruction_epoch(model, train_loader, resolved_device, optimizer)
        validation_metrics = _reconstruction_epoch(model, validation_loader, resolved_device)
        history.append(
            {
                "epoch": epoch + 1,
                **{f"train_{name}": value for name, value in train_metrics.items()},
                **{f"validation_{name}": value for name, value in validation_metrics.items()},
            }
        )
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.to_dict(),
            "seed": seed,
        }
        torch.save(checkpoint, output / "last_checkpoint.pt")
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, output / "best_vqvae.pth")
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)

    if best_state is None:
        raise RuntimeError("VQ-VAE training did not produce a checkpoint")
    write_json(
        output / "training_summary.json",
        {
            "best_validation_loss": best_loss,
            "epochs_completed": len(history),
            "training_minutes": (time.perf_counter() - started) / 60.0,
            "seed": seed,
            "device": resolved_device,
            "train_size": len(train_samples),
            "validation_size": len(validation_samples),
        },
    )
    return output / "best_vqvae.pth"
