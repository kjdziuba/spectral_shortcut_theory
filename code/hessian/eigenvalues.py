"""Hessian top-eigenvalue estimation via Hessian-vector-product power iteration.

Used to track sharpness of the spectral vs spatial parameter blocks during training
(gradient starvation diagnostic). Operates on one batch at a time and differentiates
against either ``model.spectral`` or ``model.spatial`` parameters.
"""
from __future__ import annotations

import math
from typing import List, Literal, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def count_params(module: nn.Module) -> int:
    """Sum of ``numel()`` over parameters of ``module`` that require grad."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _global_l2_norm(vs: List[torch.Tensor]) -> torch.Tensor:
    """Global L2 norm over a list-of-tensors (flattened concatenation)."""
    sq = torch.zeros((), device=vs[0].device, dtype=vs[0].dtype)
    for v in vs:
        sq = sq + (v * v).sum()
    return torch.sqrt(sq)


def _normalize_inplace(vs: List[torch.Tensor]) -> List[torch.Tensor]:
    """Return a list of tensors normalized to unit global L2 norm."""
    with torch.no_grad():
        n = _global_l2_norm(vs)
        n = torch.clamp(n, min=1e-12)
        return [v / n for v in vs]


def top_eigenvalue_block(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    block: Literal["spectral", "spatial"],
    n_iter: int = 20,
    device: str = "cuda",
    tol: float = 1e-6,
    return_history: bool = False,
) -> Union[float, Tuple[float, List[float]]]:
    """Top eigenvalue of H_{block,block} = d^2 L / d block_params^2 via power iteration.

    See module docstring / spec for full semantics. Caller passes ONE batch already
    on ``device``; second-order autograd uses ~3-4x forward memory.
    """
    # --- argument validation -----------------------------------------------
    if block not in ("spectral", "spatial"):
        raise ValueError(f"block must be 'spectral' or 'spatial', got {block!r}")
    if not isinstance(X, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("X and y must be torch.Tensors (one batch), not DataLoaders.")
    if y.dtype != torch.long:
        raise TypeError(f"y must be torch.long, got {y.dtype}")

    sub = getattr(model, block)
    params = [p for p in sub.parameters() if p.requires_grad]
    if not params:
        raise ValueError(f"block '{block}' has no trainable parameters")

    # Sanity check: tensors live on the same device as the params.
    pdev = params[0].device
    if X.device != pdev or y.device != pdev:
        raise RuntimeError(
            f"device mismatch: params on {pdev}, X on {X.device}, y on {y.device}"
        )

    # --- forward + first-order grad (with create_graph!) -------------------
    logits = model(X)
    C = logits.shape[-1]
    loss = F.cross_entropy(logits.reshape(-1, C), y.reshape(-1), reduction="mean")

    g = torch.autograd.grad(loss, params, create_graph=True)
    assert g[0].grad_fn is not None, (
        "First-order grad has no graph — create_graph=True was lost. "
        "Hv would be zero. Check model.eval() / inplace ops."
    )

    # --- power iteration ---------------------------------------------------
    v = [torch.randn_like(p) for p in params]
    v = _normalize_inplace(v)

    history: List[float] = []
    eig_old = float("inf")
    eig: float = 0.0

    for _ in range(n_iter):
        gv = sum((gi * vi).sum() for gi, vi in zip(g, v))
        Hv = torch.autograd.grad(gv, params, retain_graph=True)

        # Rayleigh quotient: v has unit norm here, so <v, Hv> is the eigenvalue est.
        with torch.no_grad():
            ray = torch.zeros((), device=pdev, dtype=Hv[0].dtype)
            for hi, vi in zip(Hv, v):
                ray = ray + (hi * vi).sum()
            eig = float(ray.item())
            history.append(eig)

        # Renormalize for next iteration; detach to prevent graph growth.
        v = _normalize_inplace([hi.detach() for hi in Hv])

        if abs(eig - eig_old) / max(abs(eig), 1e-12) < tol:
            break
        eig_old = eig

    if return_history:
        return eig, history
    return eig


if __name__ == "__main__":
    # Tiny smoke test: random toy model with .spectral / .spatial blocks.
    torch.manual_seed(0)

    class Toy(nn.Module):
        def __init__(self, S=8, C=3):
            super().__init__()
            self.spectral = nn.Linear(S, 16)
            self.spatial = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=False))
            self.head = nn.Linear(16, C)

        def forward(self, X):
            B, H, W, S = X.shape
            z = self.spectral(X)                     # (B, H, W, 16)
            z = z.permute(0, 3, 1, 2).contiguous()   # (B, 16, H, W)
            z = self.spatial(z)
            z = z.permute(0, 2, 3, 1).contiguous()   # (B, H, W, 16)
            return self.head(z)                      # (B, H, W, C)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Toy().to(dev).eval()
    X = torch.randn(2, 4, 4, 8, device=dev)
    y = torch.randint(0, 3, (2, 4, 4), device=dev, dtype=torch.long)

    eig_sp, hist_sp = top_eigenvalue_block(model, X, y, "spectral", n_iter=15, return_history=True)
    eig_sx, hist_sx = top_eigenvalue_block(model, X, y, "spatial", n_iter=15, return_history=True)
    print(f"spectral block: eig={eig_sp:.6f} params={count_params(model.spectral)}")
    print(f"spatial  block: eig={eig_sx:.6f} params={count_params(model.spatial)}")
    print(f"spectral history (last 3): {hist_sp[-3:]}")
    print(f"spatial  history (last 3): {hist_sx[-3:]}")
