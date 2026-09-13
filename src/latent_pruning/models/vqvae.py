"""Checkpoint-compatible VQ-VAE from the supplied experiment notebooks."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from latent_pruning.config import VQVAEConfig


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        super().__init__()
        self.K = int(num_embeddings)
        self.D = int(embedding_dim)
        self.beta = float(beta)
        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # [B,D,H,W] -> [B,H,W,D]
        channels_last = latents.permute(0, 2, 3, 1).contiguous()
        latent_shape = channels_last.shape
        flattened = channels_last.view(-1, self.D)
        distances = (
            flattened.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * flattened @ self.embedding.weight.t()
        )
        encoding_indices = distances.argmin(dim=1, keepdim=True)
        quantized = F.embedding(encoding_indices.squeeze(1), self.embedding.weight)
        quantized = quantized.view(latent_shape)

        commitment_loss = F.mse_loss(quantized.detach(), channels_last)
        embedding_loss = F.mse_loss(quantized, channels_last.detach())
        vq_loss = self.beta * commitment_loss + embedding_loss
        quantized = channels_last + (quantized - channels_last).detach()
        return quantized.permute(0, 3, 1, 2).contiguous(), vq_loss, encoding_indices

    def reconstruct_vectors(self, encoding_indices: Tensor, latent_shape: Sequence[int]) -> Tensor:
        quantized = F.embedding(encoding_indices.squeeze(-1), self.embedding.weight)
        quantized = quantized.view(tuple(latent_shape))
        return quantized.permute(0, 3, 1, 2).contiguous()


class ResidualLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.resblock(inputs)


class VQVAE(nn.Module):
    """VQ-VAE architecture preserving the original checkpoint key layout."""

    def __init__(
        self,
        in_channels: int,
        embedding_dim: int,
        num_embeddings: int,
        hidden_dims: Sequence[int] | None = None,
        beta: float = 0.25,
        img_size: int = 64,
        **_: Any,
    ):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.num_embeddings = int(num_embeddings)
        self.img_size = int(img_size)
        self.beta = float(beta)
        encoder_dims = list(hidden_dims or [32, 64, 128, 256, 512])

        modules: list[nn.Module] = []
        current_channels = int(in_channels)
        for output_channels in encoder_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        current_channels,
                        output_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.LeakyReLU(),
                )
            )
            current_channels = output_channels
        modules.append(
            nn.Sequential(
                nn.Conv2d(current_channels, current_channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
            )
        )
        for _ in range(6):
            modules.extend([ResidualLayer(current_channels, current_channels), nn.LeakyReLU()])
        modules.append(
            nn.Sequential(
                nn.Conv2d(current_channels, self.embedding_dim, kernel_size=1, stride=1),
                nn.LeakyReLU(),
            )
        )
        self.encoder = nn.Sequential(*modules)
        self.vq_layer = VectorQuantizer(self.num_embeddings, self.embedding_dim, self.beta)

        decoder_modules: list[nn.Module] = [
            nn.Sequential(
                nn.Conv2d(self.embedding_dim, encoder_dims[-1], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
            )
        ]
        decoder_modules.extend(ResidualLayer(encoder_dims[-1], encoder_dims[-1]) for _ in range(6))
        decoder_modules.append(nn.LeakyReLU())
        decoder_dims = list(reversed(encoder_dims))
        for index in range(len(decoder_dims) - 1):
            decoder_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        decoder_dims[index],
                        decoder_dims[index + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.LeakyReLU(),
                )
            )
        decoder_modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    decoder_dims[-1],
                    out_channels=3,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.Tanh(),
            )
        )
        self.decoder = nn.Sequential(*decoder_modules)

    def encode(self, inputs: Tensor) -> list[Tensor]:
        return [self.encoder(inputs)]

    def quantize(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.vq_layer(self.encoder(inputs))

    def decode(self, latent: Tensor) -> Tensor:
        return self.decoder(latent)

    def forward(self, inputs: Tensor, **_: Any) -> list[Tensor]:
        quantized, vq_loss, encoding_indices = self.quantize(inputs)
        return [self.decode(quantized), inputs, vq_loss, quantized, encoding_indices]

    @staticmethod
    def loss_function(*args: Tensor, **_: Any) -> dict[str, Tensor]:
        reconstruction, inputs, vq_loss = args[:3]
        reconstruction_loss = F.mse_loss(reconstruction, inputs)
        return {
            "loss": reconstruction_loss + vq_loss,
            "Reconstruction_Loss": reconstruction_loss,
            "VQ_Loss": vq_loss,
        }

    def generate(self, inputs: Tensor, **_: Any) -> Tensor:
        return self.forward(inputs)[0]

    def representation(self, inputs: Tensor, kind: str = "histogram") -> Tensor:
        quantized, _, indices = self.quantize(inputs)
        batch_size = inputs.shape[0]
        if kind == "encoding_indices":
            return indices.view(batch_size, -1).float()
        if kind == "histogram":
            flat = indices.view(batch_size, -1)
            histogram = torch.zeros(
                batch_size,
                self.num_embeddings,
                dtype=torch.float32,
                device=inputs.device,
            )
            histogram.scatter_add_(1, flat, torch.ones_like(flat, dtype=torch.float32))
            return histogram / histogram.sum(dim=1, keepdim=True).clamp_min(1.0)
        if kind == "quantized":
            return quantized.flatten(start_dim=1)
        if kind == "pooled":
            return quantized.mean(dim=(2, 3))
        raise ValueError(f"Unknown VQ-VAE representation: {kind}")


def _extract_state_dict(checkpoint: Any) -> dict[str, Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("The checkpoint must contain a state-dict mapping")
    for key in ("model_state_dict", "state_dict", "model"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            checkpoint = candidate
            break
    state_dict = dict(checkpoint)
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def create_vqvae(config: VQVAEConfig) -> VQVAE:
    return VQVAE(
        in_channels=config.in_channels,
        embedding_dim=config.embedding_dim,
        num_embeddings=config.num_embeddings,
        hidden_dims=config.hidden_dims,
        beta=config.beta,
        img_size=config.image_size,
    )


def load_vqvae_checkpoint(
    config: VQVAEConfig,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> VQVAE:
    checkpoint_path = Path(config.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"VQ-VAE checkpoint not found: {checkpoint_path}")
    model = create_vqvae(config)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    incompatible = model.load_state_dict(_extract_state_dict(checkpoint), strict=strict)
    if not strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        print(
            "Checkpoint loaded non-strictly; "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    model.to(device)
    model.eval()
    return model
