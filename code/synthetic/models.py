"""synthetic/models.py — compositional model F(X) = g_phi(f_theta(X)).

Provides:
    - SpectralReduction: per-pixel linear projection S -> K (the f_theta block).
    - SpatialMLP:        per-pixel MLP head K -> n_classes (one g_phi option).
    - SpatialCNN:        2-layer 3x3 CNN head K -> n_classes (other g_phi option).
    - CompositionModel:  wires spectral and spatial together, exposing the
                         submodule names `.spectral` and `.spatial` that
                         downstream EGR / Hessian tooling depends on.

Shape contract (channels-LAST in the public API, no exceptions):
    X      : (B, H, W, S)         float32
    Z      : (B, H, W, K)         float32  =  spectral(X)
    logits : (B, H, W, n_classes) float32  =  spatial(Z) = CompositionModel(X)

Downstream code computes
    F.cross_entropy(logits.reshape(-1, n_classes), y.reshape(-1))
so any axis mix-up here silently mis-attributes pixels. SpatialCNN must permute
in -> conv -> permute out.

Activations are ReLU(inplace=False) everywhere because the Hessian power
iteration calls torch.autograd.grad with retain_graph=True and inplace ops
destroy graph nodes it needs.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SpectralReduction(nn.Module):
    """Per-pixel linear projection S -> K, implemented as nn.Linear(S, K).

    Default bias=False because the paper theory assumes a purely linear
    f_theta; adding a bias changes both the Hessian block size and the EGR
    norm denominator. Weights are initialised with a small isotropic Gaussian
    (std = 1/sqrt(S)) so the spectral pathway does not dominate at init.
    """

    def __init__(self, S: int, K: int, bias: bool = False) -> None:
        super().__init__()
        self.S = S
        self.K = K
        self.proj = nn.Linear(S, K, bias=bias)
        nn.init.normal_(self.proj.weight, std=1.0 / math.sqrt(S))
        if bias and self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # nn.Linear acts on the last dim, so (B, H, W, S) -> (B, H, W, K)
        # without any reshape gymnastics.
        return self.proj(X)


class SpatialMLP(nn.Module):
    """Per-pixel MLP head with one hidden layer of width `width`.

    Operates independently per pixel (acts on the last axis only), so it has
    zero spatial receptive field — useful as the "no spatial context" baseline
    contrast against SpatialCNN.
    """

    def __init__(self, K: int, n_classes: int, width: int) -> None:
        super().__init__()
        self.K = K
        self.n_classes = n_classes
        self.width = width
        self.fc1 = nn.Linear(K, width)
        self.act = nn.ReLU(inplace=False)  # see module docstring
        self.fc2 = nn.Linear(width, n_classes)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        # Z: (B, H, W, K) -> logits: (B, H, W, n_classes)
        return self.fc2(self.act(self.fc1(Z)))


class SpatialCNN(nn.Module):
    """Two 3x3 conv layers with ReLU between, channels-LAST public I/O.

    Internally we permute (B, H, W, K) -> (B, K, H, W) for Conv2d, run the
    conv stack, then permute back to (B, H, W, n_classes). .contiguous() after
    each permute keeps strides sane for downstream view/reshape ops.
    """

    def __init__(
        self,
        K: int,
        n_classes: int,
        width: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.K = K
        self.n_classes = n_classes
        self.width = width
        self.kernel_size = kernel_size
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(K, width, kernel_size, padding=pad)
        self.act = nn.ReLU(inplace=False)  # see module docstring
        self.conv2 = nn.Conv2d(width, n_classes, kernel_size, padding=pad)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        # (B, H, W, K) -> (B, K, H, W) for conv
        Z = Z.permute(0, 3, 1, 2).contiguous()
        h = self.act(self.conv1(Z))
        logits = self.conv2(h)
        # (B, n_classes, H, W) -> (B, H, W, n_classes) for the public contract
        return logits.permute(0, 2, 3, 1).contiguous()


class CompositionModel(nn.Module):
    """F(X) = spatial(spectral(X)).

    The submodule names `.spectral` and `.spatial` are part of the public API:
    EGRLogger and the Hessian top-eigenvalue routine iterate parameters via
    `model.spectral.parameters()` / `model.spatial.parameters()`. Renaming
    these breaks both tools, so don't.
    """

    def __init__(self, spectral: nn.Module, spatial: nn.Module) -> None:
        super().__init__()
        self.spectral = spectral  # theta block — name used by EGR + Hessian
        self.spatial = spatial    # phi   block — name used by EGR + Hessian

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.spectral(X))


def _count_params(module: nn.Module) -> int:
    """Total number of trainable parameters in `module`."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Sanity-check shapes and param counts. Mirrors what the train loop sees:
    # X is channels-LAST, logits are channels-LAST.
    B, H, W, S = 2, 8, 8, 16
    K = 4
    n_classes = 3
    width = 32

    spectral = SpectralReduction(S=S, K=K)
    mlp_head = SpatialMLP(K=K, n_classes=n_classes, width=width)
    cnn_head = SpatialCNN(K=K, n_classes=n_classes, width=width)

    model_mlp = CompositionModel(spectral, mlp_head)
    model_cnn = CompositionModel(SpectralReduction(S=S, K=K), cnn_head)

    X = torch.randn(B, H, W, S)
    Z = spectral(X)
    logits_mlp = model_mlp(X)
    logits_cnn = model_cnn(X)

    assert Z.shape == (B, H, W, K), Z.shape
    assert logits_mlp.shape == (B, H, W, n_classes), logits_mlp.shape
    assert logits_cnn.shape == (B, H, W, n_classes), logits_cnn.shape

    print(f"SpectralReduction(S={S}, K={K}) params: {_count_params(spectral)}")
    print(f"SpatialMLP(K={K}, n_classes={n_classes}, width={width}) params: "
          f"{_count_params(mlp_head)}")
    print(f"SpatialCNN(K={K}, n_classes={n_classes}, width={width}) params: "
          f"{_count_params(cnn_head)}")
    print(f"CompositionModel (MLP head) params: {_count_params(model_mlp)}")
    print(f"CompositionModel (CNN head) params: {_count_params(model_cnn)}")
    print(f"Z shape: {tuple(Z.shape)}")
    print(f"logits (MLP) shape: {tuple(logits_mlp.shape)}")
    print(f"logits (CNN) shape: {tuple(logits_cnn.shape)}")
