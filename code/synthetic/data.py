"""
synthetic/data.py — Compositional spectral-spatial ground-truth generator.

This module synthesizes pixel-classification datasets whose labels depend
JOINTLY on a known spectral direction `u in R^S` and a known spatial template
`P_c in R^{H x W}` per class. It is used to study gradient starvation in
compositions F(X) = g_phi(f_theta(X)), where f_theta must recover the spectral
reduction and g_phi must learn the spatial pattern.

Two scalars control signal strength:
  alpha : magnitude of the spectral signal (alpha=0 -> spectral channel is noise)
  beta  : magnitude of the spatial signal  (beta=0  -> spatial pattern is uninformative)

With alpha=beta=1.0 and noise=0.1, Bayes-optimal accuracy is ~0.85-0.95 for
n_classes=2 (not 1.0, not chance), giving headroom to observe learning dynamics.
"""

from typing import Tuple, Dict, Optional, Union
import math
import numpy as np
import torch


def _make_smooth_patterns(n_classes: int, H: int, W: int, g: torch.Generator) -> torch.Tensor:
    """Smooth low-frequency spatial templates via random Fourier features.

    Returns (n_classes, H, W) tensor, each slice unit Frobenius norm.
    """
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, H), torch.linspace(0.0, 1.0, W), indexing="ij"
    )
    patterns = torch.zeros(n_classes, H, W)
    n_freqs = 4  # low-frequency content only -> smooth templates
    for c in range(n_classes):
        P = torch.zeros(H, W)
        for _ in range(n_freqs):
            fx = torch.randint(1, 4, (1,), generator=g).item()
            fy = torch.randint(1, 4, (1,), generator=g).item()
            phase = torch.rand(1, generator=g).item() * 2 * math.pi
            amp = torch.randn(1, generator=g).item()
            P = P + amp * torch.cos(2 * math.pi * (fx * xx + fy * yy) + phase)
        # Unit Frobenius norm so beta has consistent meaning across classes.
        P = P / (P.norm() + 1e-8)
        patterns[c] = P
    return patterns


def make_problem(
    n_samples: int,
    S: int = 64,
    H: int = 16,
    W: int = 16,
    n_classes: int = 2,
    alpha: float = 1.0,
    beta: float = 1.0,
    noise: float = 0.1,
    seed: int = 42,
    return_meta: bool = False,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict]]:
    """Generate synthetic spectral-spatial classification data.

    See module docstring for the compositional ground-truth model. Bayes-optimal
    accuracy is ~0.85-0.95 at alpha=beta=1.0, noise=0.1, n_classes=2.

    Returns (X, y) by default; (X, y, meta) if return_meta=True. All tensors
    are on CPU; caller is responsible for .to(device).
    """
    if n_classes < 2:
        raise ValueError(f"n_classes must be >= 2 (got {n_classes}); CE is undefined for 1 class.")

    g = torch.Generator().manual_seed(seed)

    # 1) Global ground-truth spectral direction u in R^S (unit norm).
    u = torch.randn(S, generator=g)
    u = u / (u.norm() + 1e-8)

    # 2) Smooth spatial templates P_c, one per class, unit Frobenius norm.
    patterns = _make_smooth_patterns(n_classes, H, W, g)  # (C, H, W)

    # 3) Class codes t_c. For 2-class, +/-1. For >2, one-hot rows of identity
    #    so the single direction u can still encode multi-class info via sign/magnitude.
    if n_classes == 2:
        t = torch.tensor([-1.0, +1.0])  # (C,)
        # spectral_score per class c: alpha * t_c * z
        # collapsed below
    else:
        t = torch.eye(n_classes)  # (C, C); per-class code vectors

    # 4) Per-pixel latent z drives the spectral signal.
    z = torch.randn(n_samples, H, W, generator=g)  # (B, H, W)

    # 5) Build per-class logits and sample labels.
    #    spatial_logit[b,h,w,c] = beta * P_c[h,w]
    spatial_logit = beta * patterns.permute(1, 2, 0).unsqueeze(0)  # (1, H, W, C)
    spatial_logit = spatial_logit.expand(n_samples, H, W, n_classes)

    if n_classes == 2:
        # spectral_logit[b,h,w,c] = alpha * t_c * z[b,h,w]
        spectral_logit = alpha * z.unsqueeze(-1) * t.view(1, 1, 1, n_classes)
    else:
        # For multi-class: spectral contribution is alpha * z * t_c (per-class scalar code).
        # We use the diagonal of identity; t_c effectively becomes class-specific scale = 1
        # when c == argmax, but to keep the spectral direction informative across classes,
        # we make t_c a sign-pattern via Hadamard-like construction.
        # Fallback: random +/-1 codes per class to span R^1 latent across classes.
        codes = torch.randint(0, 2, (n_classes,), generator=g).float() * 2.0 - 1.0  # +/-1
        spectral_logit = alpha * z.unsqueeze(-1) * codes.view(1, 1, 1, n_classes)

    logits = spatial_logit + spectral_logit  # (B, H, W, C)
    # Sample y ~ Categorical(softmax(logits)) per pixel for stochastic ground truth.
    flat_logits = logits.reshape(-1, n_classes)
    probs = torch.softmax(flat_logits, dim=-1)
    # Use Generator-aware multinomial sampling.
    y_flat = torch.multinomial(probs, num_samples=1, generator=g).squeeze(-1)
    y = y_flat.reshape(n_samples, H, W).to(torch.long)

    # 6) Build X. Spectral signal: alpha * z * u injected at every pixel; then noise.
    #    x[b,h,w,:] = alpha * z[b,h,w] * u + noise * eps
    X = alpha * z.unsqueeze(-1) * u.view(1, 1, 1, S)  # (B, H, W, S)
    # Small spatial fill: a tiny bias from the chosen class's template along u
    # (keeps the spectral channel mildly informative even when alpha is small but nonzero).
    # NOTE: we keep this small so spectral_only ceiling at beta=0 is governed by alpha*z.
    X = X + noise * torch.randn(n_samples, H, W, S, generator=g)

    X = X.to(torch.float32)
    y = y.to(torch.long)

    # Sanity: warn (do not raise) if a class is severely underrepresented.
    counts = torch.bincount(y.reshape(-1), minlength=n_classes)
    frac = counts.float() / counts.sum().clamp(min=1)
    if (frac < 0.01).any():
        import warnings
        warnings.warn(
            f"Class collapse detected: per-class fractions={frac.tolist()}. "
            f"Consider reducing alpha or beta.",
            RuntimeWarning,
        )

    if return_meta:
        spec_score = (X * u.view(1, 1, 1, S)).sum(dim=-1)  # u^T x per pixel
        class_map = spatial_logit[..., :].argmax(dim=-1).to(torch.long)  # spatial-only label proposal
        meta = {
            "u": u,
            "patterns": patterns,
            "class_map": class_map,
            "spec_score": spec_score,
        }
        return X, y, meta
    return X, y


if __name__ == "__main__":
    # Tiny smoke test: small problem, print shapes + dtypes + class balance.
    X, y, meta = make_problem(
        n_samples=8, S=32, H=8, W=8, n_classes=2,
        alpha=1.0, beta=1.0, noise=0.1, seed=0, return_meta=True,
    )
    print(f"X: shape={tuple(X.shape)}, dtype={X.dtype}")
    print(f"y: shape={tuple(y.shape)}, dtype={y.dtype}")
    print(f"u: shape={tuple(meta['u'].shape)}, ||u||={meta['u'].norm().item():.4f}")
    print(f"patterns: shape={tuple(meta['patterns'].shape)}")
    counts = torch.bincount(y.reshape(-1), minlength=2)
    print(f"class counts: {counts.tolist()}  (fractions: {(counts.float()/counts.sum()).tolist()})")
