"""
synthetic/data_v2.py — contextual-broadcast generator for Experiment 2
(refocus, 2026-09-10). Kept separate from data.py: results/exp1_* depend on
the v1 generator, which is never modified.

Per-pixel binary labels y in {-1,+1}^{B x H x W}, i.i.d. Rademacher, so the
spectral cue cannot be denoised by averaging neighbours.

Fixed orthonormal directions u, v_1..v_n in R^S. For pixel p:

    x_p = alpha * y_p * u
        + beta * sum_d ( y_{p - o_d} + tau * eta_{p,d} ) * v_d
        + sigma * eps_p

with o_d the neighbour offsets (torus), eta and eps standard Gaussian.

  spectral cue : own spectrum, small amplitude alpha, isotropic noise sigma;
                 reading it requires an encoder aligned with u.
  context cue  : the neighbours of p carry y_p along "texture" directions v_d
                 with context-specific noise tau. p's own v-coordinates carry
                 the labels of p's neighbours, which are independent of y_p,
                 so the information sits ONLY in the neighbourhood. Its
                 amplitude beta is large, so a random linear encoder keeps its
                 signal-to-noise ratio (signal and noise share a direction);
                 the spectral cue's ratio is attenuated by ~sqrt(K/S).

Conditions (the training distribution is `iid` unless stated):
  iid         same distribution as training
  reversed    neighbours carry -y_p (context contradicts the label)
  ctx_random  neighbours carry an independent label field (uninformative)
  spec_only   beta = 0
  ctx_only    alpha = 0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch

CONDITIONS = ("iid", "reversed", "ctx_random", "spec_only", "ctx_only")

# The eight 3x3 offsets (dy, dx). Pixel p receives y_{p - o} along v_o, i.e.
# the pixel at p + o carries y_p along v_o: p's label is written into all
# eight neighbours, each along its own direction.
OFFSETS_RING8: Tuple[Tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


@dataclass
class ProblemSpec:
    S: int = 256
    H: int = 16
    W: int = 16
    alpha: float = 1.645
    beta: float = 5.0
    tau: float = 1.7
    sigma: float = 1.0
    offsets: Tuple[Tuple[int, int], ...] = OFFSETS_RING8

    @property
    def n_ctx(self) -> int:
        return len(self.offsets)


def make_directions(S: int, n_ctx: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Orthonormal u (S,) and V (n_ctx, S), all mutually orthogonal."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(S, n_ctx + 1, generator=g, dtype=torch.float64)
    Q, _ = torch.linalg.qr(A)  # (S, n_ctx+1), orthonormal columns
    u = Q[:, 0].contiguous()
    V = Q[:, 1:].T.contiguous()
    return u.float(), V.float()


def shift_torus(y: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """z[b, i, j] = y[b, i - dy, j - dx] on the torus."""
    return torch.roll(y, shifts=(dy, dx), dims=(1, 2))


def make_problem_v2(
    n: int,
    spec: ProblemSpec,
    u: torch.Tensor,
    V: torch.Tensor,
    seed: int,
    condition: str = "iid",
    return_fields: bool = False,
):
    """Returns X (n, H, W, S) float32, y (n, H, W) in {-1,+1} float32,
    and (optionally) the per-offset context fields C (n, n_ctx, H, W)."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    S, H, W = spec.S, spec.H, spec.W
    g = torch.Generator().manual_seed(seed)
    # Draw order is condition-independent so that the five conditions generated
    # from the SAME seed share labels y, context noise eta and spectral noise
    # eps, differing only in the assigned context field/amplitudes (paired
    # interventions; Astra refocus_01 §2.4.5). `iid` output is bitwise identical
    # to the earlier draw order (y, eta, eps).
    y = torch.randint(0, 2, (n, H, W), generator=g).float() * 2.0 - 1.0
    eta = torch.randn(n, spec.n_ctx, H, W, generator=g)
    eps = torch.randn(n, H, W, S, generator=g)
    y_alt = torch.randint(0, 2, (n, H, W), generator=g).float() * 2.0 - 1.0
    y_src = y_alt if condition == "ctx_random" else y
    sign = -1.0 if condition == "reversed" else 1.0
    alpha = 0.0 if condition == "ctx_only" else spec.alpha
    beta = 0.0 if condition == "spec_only" else spec.beta

    # Context fields: C[:, d] = sign * y_src shifted by o_d  (+ tau * noise)
    C = torch.stack([sign * shift_torus(y_src, dy, dx) for (dy, dx) in spec.offsets], dim=1)
    Cn = C + spec.tau * eta  # (n, n_ctx, H, W)

    X = alpha * y.unsqueeze(-1) * u.view(1, 1, 1, S)
    # beta * sum_d Cn[:, d] v_d  ->  (n, H, W, S)
    X = X + beta * torch.einsum("bdhw,ds->bhws", Cn, V)
    X = X + spec.sigma * eps
    X = X.float()
    if return_fields:
        return X, y, Cn
    return X, y


# ---------------------------------------------------------------- probes ---

def unfold_window(f: torch.Tensor, r: int = 2) -> torch.Tensor:
    """f: (B, C, H, W) -> (B*H*W, C*(2r+1)^2) windows on the torus."""
    B, C, H, W = f.shape
    fp = torch.cat([f[:, :, -r:], f, f[:, :, :r]], dim=2)
    fp = torch.cat([fp[:, :, :, -r:], fp, fp[:, :, :, :r]], dim=3)
    win = torch.nn.functional.unfold(fp, kernel_size=2 * r + 1)  # (B, C*k*k, H*W)
    return win.permute(0, 2, 1).reshape(B * H * W, -1)


def lda_accuracy(F_fit: torch.Tensor, y_fit: torch.Tensor,
                 F_eval: torch.Tensor, y_eval: torch.Tensor,
                 ridge: float = 1e-3) -> float:
    """Fisher LDA fitted on (F_fit, y_fit), accuracy on (F_eval, y_eval).
    y in {-1,+1}. Double precision; ridge relative to the mean variance."""
    F_fit = F_fit.double(); F_eval = F_eval.double()
    pos = y_fit > 0
    mu_p = F_fit[pos].mean(0); mu_n = F_fit[~pos].mean(0)
    Xc = torch.cat([F_fit[pos] - mu_p, F_fit[~pos] - mu_n], dim=0)
    Sigma = Xc.T @ Xc / Xc.shape[0]
    d = Sigma.shape[0]
    Sigma = Sigma + ridge * Sigma.diagonal().mean() * torch.eye(d, dtype=Sigma.dtype, device=Sigma.device)
    w = torch.linalg.solve(Sigma, mu_p - mu_n)
    thr = w @ (mu_p + mu_n) / 2.0
    pred = torch.sign(F_eval @ w - thr)
    return float((pred == y_eval.double()).double().mean())


if __name__ == "__main__":
    spec = ProblemSpec()
    u, V = make_directions(spec.S, spec.n_ctx, seed=0)
    X, y, Cn = make_problem_v2(4, spec, u, V, seed=1, return_fields=True)
    print("X", tuple(X.shape), "y", tuple(y.shape), "C", tuple(Cn.shape))
    # check: the neighbour at p + o_d carries y_p along v_d
    proj = torch.einsum("bhws,ds->bdhw", X, V)  # (B, n_ctx, H, W)
    for d, (dy, dx) in enumerate(spec.offsets):
        carried = shift_torus(y, dy, dx)  # what pixel p should carry along v_d = y_{p-o}
        corr = torch.corrcoef(torch.stack([proj[:, d].flatten(), carried.flatten()]))[0, 1]
        print(f"offset {(dy, dx)}: corr(v_d-projection, y_(p-o)) = {corr:.3f}")
    print("orthogonality max |<u,v_d>| =", float((V @ u).abs().max()), " max |V V^T - I| =",
          float((V @ V.T - torch.eye(spec.n_ctx)).abs().max()))
