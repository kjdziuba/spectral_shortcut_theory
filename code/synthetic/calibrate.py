"""Calibration routines for the equal-information experiment.

Three modes operationalize "spectral and spatial features carry equal
information about y":

  1. 'bayes':   tune (alpha, beta) such that a logistic regression
                trained on spectral-only data (beta=0) and one trained
                on spatial-only data (alpha=0) achieve equal accuracy.
  2. 'ntk':     tune so the leading Gram-matrix eigenvalue of the
                spectral pathway equals that of the spatial pathway.
                Operationalizes the "speed of learning" notion from
                Pezeshki's NTK framework.
  3. 'margin':  tune so the L2 margins of logistic-regression hyperplanes
                fit on each modality alone are equal.

For each mode we return the (alpha, beta) pair that satisfies the
condition at a chosen target level. The returned scalars are then used
when building the dataset for the main experiment.
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

CODE_DIR = Path("/home/u37314kd/Projects/spectral_shortcut_theory/code")
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from synthetic.data import make_problem  # noqa: E402


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _spectral_only_features(n_samples: int, alpha: float, S: int = 64,
                             H: int = 16, W: int = 16,
                             noise: float = 0.1,
                             seed: int = 12345) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a dataset with spectral signal only (beta = 0)."""
    X, y = make_problem(
        n_samples=n_samples, S=S, H=H, W=W, n_classes=2,
        alpha=alpha, beta=0.0, noise=noise, seed=seed,
    )
    return X, y


def _spatial_only_features(n_samples: int, beta: float, S: int = 64,
                            H: int = 16, W: int = 16,
                            noise: float = 0.1,
                            seed: int = 12345) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a dataset with spatial signal only (alpha = 0)."""
    X, y = make_problem(
        n_samples=n_samples, S=S, H=H, W=W, n_classes=2,
        alpha=0.0, beta=beta, noise=noise, seed=seed,
    )
    return X, y


def _train_logreg(X: torch.Tensor, y: torch.Tensor,
                  epochs: int = 200, lr: float = 0.1,
                  weight_decay: float = 0.01) -> tuple[float, float]:
    """Train a single-pixel logistic regression on the flattened input.

    X is (B, H, W, S). We reshape to (B*H*W, S) and treat each pixel as
    its own example. The classifier weight w in R^S is L2-regularized.

    Returns (accuracy, margin) where margin = 1 / ||w||.
    """
    X = X.reshape(-1, X.shape[-1])
    y = y.reshape(-1)
    N, D = X.shape
    w = torch.zeros(D, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.SGD([w, b], lr=lr, momentum=0.9, weight_decay=weight_decay)
    y_signed = 2.0 * y.float() - 1.0  # {0,1} -> {-1,+1}
    for _ in range(epochs):
        opt.zero_grad()
        z = X @ w + b
        # Logistic loss
        loss = torch.log1p(torch.exp(-y_signed * z)).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        z = X @ w + b
        pred = (z > 0).long()
        acc = (pred == y).float().mean().item()
        margin = (1.0 / (w.norm().item() + 1e-12))
    return float(acc), float(margin)


def _position_majority_acc(y: torch.Tensor) -> float:
    """Upper bound on accuracy achievable by a position-only classifier.

    For each position (h,w), look at the labels seen there across all
    samples, predict the majority class. Return the average accuracy.
    This is what a CNN can in principle match by memorizing per-position
    label biases (the spatial shortcut).
    """
    # y: (B, H, W) long
    B, H, W = y.shape
    # Per-position majority class is the argmax over class counts
    # Compute fraction of majority class per position
    one_hot = torch.zeros(B, H, W, 2)
    one_hot.scatter_(3, y.unsqueeze(-1).clamp_min(0).clamp_max(1), 1.0)
    # counts per position per class: (H, W, 2)
    counts = one_hot.sum(dim=0)
    majority_count = counts.max(dim=-1).values  # (H, W)
    total = counts.sum(dim=-1)  # (H, W)
    pos_acc = majority_count / total.clamp(min=1)
    return float(pos_acc.mean().item())


def _top_eigval_gram(X: torch.Tensor) -> float:
    """Top eigenvalue of X^T X / N where X is flattened to (N, D)."""
    X = X.reshape(-1, X.shape[-1])
    N = X.shape[0]
    G = X.T @ X / N
    return float(torch.linalg.eigvalsh(G)[-1].item())


# ------------------------------------------------------------------ #
# Calibration modes
# ------------------------------------------------------------------ #

@dataclass
class CalibResult:
    mode: str
    alpha: float
    beta: float
    spectral_only_acc: float
    spatial_only_acc: float
    spectral_only_margin: float
    spatial_only_margin: float
    spectral_top_eig: float
    spatial_top_eig: float
    target_value: float
    note: str = ""


def _bisect_to_target(metric_fn: Callable[[float], float],
                       target: float,
                       lo: float = 0.05, hi: float = 5.0,
                       tol: float = 0.005, max_iter: int = 25) -> float:
    """Find x such that metric_fn(x) approx target."""
    f_lo = metric_fn(lo); f_hi = metric_fn(hi)
    # Make sure we bracket the target
    if not (min(f_lo, f_hi) <= target <= max(f_lo, f_hi)):
        # extend hi if needed
        hi2 = hi
        for _ in range(8):
            hi2 *= 2
            f_hi2 = metric_fn(hi2)
            if min(f_lo, f_hi2) <= target <= max(f_lo, f_hi2):
                hi = hi2; f_hi = f_hi2
                break
    increasing = f_hi >= f_lo
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = metric_fn(mid)
        if abs(f_mid - target) < tol:
            return mid
        if (increasing and f_mid < target) or (not increasing and f_mid > target):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def calibrate_bayes(target_acc: float = 0.75,
                    n_samples: int = 2048,
                    noise: float = 0.1,
                    S: int = 64, H: int = 16, W: int = 16,
                    seed: int = 12345) -> CalibResult:
    """Find (alpha, beta) such that:
        - per-pixel logreg on spectral-only data hits target_acc, and
        - position-majority classifier on spatial-only data hits target_acc.
    These are the two natural "best-possible-modality-only" baselines.
    """
    def spec_acc(alpha):
        X, y = _spectral_only_features(n_samples, alpha, S, H, W, noise, seed)
        a, _ = _train_logreg(X, y)
        return a

    def spat_acc(beta):
        _, y = _spatial_only_features(n_samples, beta, S, H, W, noise, seed)
        return _position_majority_acc(y)

    # Note: position-majority acc saturates at 1.0 if beta is large enough,
    # so we may need a higher search range. Use wider bracket for spatial.
    alpha = _bisect_to_target(spec_acc, target_acc)
    beta = _bisect_to_target(spat_acc, target_acc, lo=0.05, hi=20.0)

    Xs, ys = _spectral_only_features(n_samples, alpha, S, H, W, noise, seed)
    Xl, yl = _spatial_only_features(n_samples, beta, S, H, W, noise, seed)
    a_s, m_s = _train_logreg(Xs, ys)
    a_l = _position_majority_acc(yl)
    _, m_l = _train_logreg(Xl, yl)

    return CalibResult(
        mode="bayes",
        alpha=alpha, beta=beta,
        spectral_only_acc=a_s, spatial_only_acc=a_l,
        spectral_only_margin=m_s, spatial_only_margin=m_l,
        spectral_top_eig=_top_eigval_gram(Xs),
        spatial_top_eig=_top_eigval_gram(Xl),
        target_value=target_acc,
        note=f"target_acc={target_acc}",
    )


def calibrate_ntk(target_eig: float = 1.0,
                  n_samples: int = 1024,
                  noise: float = 0.1,
                  S: int = 64, H: int = 16, W: int = 16,
                  seed: int = 12345) -> CalibResult:
    """Find (alpha, beta) such that top eigenvalues of the per-modality Gram matrices match `target_eig`."""
    def spec_eig(alpha):
        X, _ = _spectral_only_features(n_samples, alpha, S, H, W, noise, seed)
        return _top_eigval_gram(X)

    def spat_eig(beta):
        X, _ = _spatial_only_features(n_samples, beta, S, H, W, noise, seed)
        return _top_eigval_gram(X)

    alpha = _bisect_to_target(spec_eig, target_eig)
    beta = _bisect_to_target(spat_eig, target_eig)

    Xs, ys = _spectral_only_features(n_samples, alpha, S, H, W, noise, seed)
    Xl, yl = _spatial_only_features(n_samples, beta, S, H, W, noise, seed)
    a_s, m_s = _train_logreg(Xs, ys)
    a_l, m_l = _train_logreg(Xl, yl)

    return CalibResult(
        mode="ntk",
        alpha=alpha, beta=beta,
        spectral_only_acc=a_s, spatial_only_acc=a_l,
        spectral_only_margin=m_s, spatial_only_margin=m_l,
        spectral_top_eig=_top_eigval_gram(Xs),
        spatial_top_eig=_top_eigval_gram(Xl),
        target_value=target_eig,
        note=f"target_top_eig={target_eig}",
    )


def calibrate_margin(target_margin: float = 0.10,
                     n_samples: int = 1024,
                     noise: float = 0.1,
                     S: int = 64, H: int = 16, W: int = 16,
                     seed: int = 12345) -> CalibResult:
    """Find (alpha, beta) such that the margins of single-modality logreg classifiers match `target_margin`."""
    def spec_margin(alpha):
        X, y = _spectral_only_features(n_samples, alpha, S, H, W, noise, seed)
        _, m = _train_logreg(X, y)
        return m

    def spat_margin(beta):
        X, y = _spatial_only_features(n_samples, beta, S, H, W, noise, seed)
        _, m = _train_logreg(X, y)
        return m

    alpha = _bisect_to_target(spec_margin, target_margin)
    beta = _bisect_to_target(spat_margin, target_margin)

    Xs, ys = _spectral_only_features(n_samples, alpha, S, H, W, noise, seed)
    Xl, yl = _spatial_only_features(n_samples, beta, S, H, W, noise, seed)
    a_s, m_s = _train_logreg(Xs, ys)
    a_l, m_l = _train_logreg(Xl, yl)

    return CalibResult(
        mode="margin",
        alpha=alpha, beta=beta,
        spectral_only_acc=a_s, spatial_only_acc=a_l,
        spectral_only_margin=m_s, spatial_only_margin=m_l,
        spectral_top_eig=_top_eigval_gram(Xs),
        spatial_top_eig=_top_eigval_gram(Xl),
        target_value=target_margin,
        note=f"target_margin={target_margin}",
    )


CALIBRATIONS = {
    "bayes": calibrate_bayes,
    "ntk": calibrate_ntk,
    "margin": calibrate_margin,
}


if __name__ == "__main__":
    print("Calibrating with all three modes (this takes ~30s on CPU)...")
    for mode in CALIBRATIONS:
        res = CALIBRATIONS[mode]()
        print(f"\n[{res.mode}] {res.note}")
        print(f"  alpha = {res.alpha:.4f}    beta = {res.beta:.4f}")
        print(f"  spectral-only logreg acc = {res.spectral_only_acc:.4f}")
        print(f"  spatial-only  logreg acc = {res.spatial_only_acc:.4f}")
        print(f"  spectral top Gram eig    = {res.spectral_top_eig:.4f}")
        print(f"  spatial  top Gram eig    = {res.spatial_top_eig:.4f}")
        print(f"  spectral margin = {res.spectral_only_margin:.4f}")
        print(f"  spatial  margin = {res.spatial_only_margin:.4f}")
