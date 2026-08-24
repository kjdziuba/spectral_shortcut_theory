"""Top-k spectrum of GGN blocks via Lanczos with full reorthogonalization.

Power iteration (ggn.py) gives lambda_1 only. The Exp 1.8 finding — the
spectral block's curvature concentrates in ONE direction — is a statement
about the whole top of the spectrum, so we need lambda_1..lambda_k. Lanczos
gets top-k from ~2k matvecs; full reorthogonalization against the stored
basis suppresses the ghost-eigenvalue pathology (fine at these sizes: the
basis is m x dim floats, kept on the compute device).

`GGNBlockOperator` wraps one (model, batch, param-block) into a repeatable
matvec: the forward graph is built once and reused across all matvecs via
retain_graph, so a matvec costs one JVP + one VJP, no new forward.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .ggn import ggn_block_vector_product, math_attention, _valid_mask


class GGNBlockOperator:
    """Repeatable ``v -> G_bb v`` for one batch and one parameter block."""

    def __init__(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        params: Sequence[torch.Tensor],
        class_dim: int = 1,
        ignore_index: int = 255,
        force_math_attention: bool = True,
    ):
        self.params = [p for p in params if p.requires_grad]
        if not self.params:
            raise ValueError("empty parameter block")
        self.y = y
        self.class_dim = class_dim
        self.ignore_index = ignore_index
        ctx = math_attention() if force_math_attention else torch.enable_grad()
        with ctx:
            self.logits = model(X)
        _, self.n_valid = _valid_mask(y, self.logits, class_dim, ignore_index)
        if self.n_valid == 0:
            raise ValueError("no valid pixels in batch")
        self.dim = sum(p.numel() for p in self.params)

    def flatten(self, vs: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.cat([v.reshape(-1) for v in vs])

    def unflatten(self, flat: torch.Tensor) -> List[torch.Tensor]:
        out, i = [], 0
        for p in self.params:
            out.append(flat[i: i + p.numel()].reshape(p.shape))
            i += p.numel()
        return out

    def matvec_flat(self, flat: torch.Tensor) -> torch.Tensor:
        Gv = ggn_block_vector_product(
            self.logits, self.params, self.unflatten(flat), self.y,
            class_dim=self.class_dim, ignore_index=self.ignore_index,
            n_valid=self.n_valid,
        )
        return self.flatten(Gv)

    def rayleigh(self, flat: torch.Tensor) -> float:
        """v^T G v / v^T v for an arbitrary direction."""
        v = flat / torch.clamp(flat.norm(), min=1e-30)
        return float((self.matvec_flat(v) @ v).item())

    def hutchinson_trace(self, n_probes: int = 20, seed: int = 0) -> Tuple[float, float]:
        """(estimate, std-error) of tr(G) via Rademacher probes."""
        gen = torch.Generator(device=self.logits.device).manual_seed(seed)
        vals = []
        for _ in range(n_probes):
            z = (torch.randint(0, 2, (self.dim,), generator=gen,
                               device=self.logits.device).to(self.params[0].dtype) * 2 - 1)
            vals.append(float((self.matvec_flat(z) @ z).item()))
        t = torch.tensor(vals)
        return float(t.mean()), float(t.std() / max(len(vals), 2) ** 0.5)


def lanczos_topk(
    op: GGNBlockOperator,
    m: int = 80,
    k: int = 40,
    seed: int = 0,
    n_vectors: int = 0,
    reorth_passes: int = 2,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
    """Top-k Ritz values (desc) of the operator; optionally top Ritz vectors.

    Returns ``(ritz_vals[k], ritz_vecs[n_vectors, dim] or None, info)``.
    ``info['residuals']`` holds ||G u - lam u|| for each returned vector —
    check these before trusting a vector.
    """
    dev = op.logits.device
    dt = op.params[0].dtype
    gen = torch.Generator(device=dev).manual_seed(seed)
    q = torch.randn(op.dim, generator=gen, device=dev, dtype=dt)
    q = q / q.norm()

    Q: List[torch.Tensor] = [q]
    alphas: List[float] = []
    betas: List[float] = []

    for j in range(m):
        w = op.matvec_flat(Q[j])
        alpha = float((w @ Q[j]).item())
        alphas.append(alpha)
        w = w - alpha * Q[j] - (betas[-1] * Q[j - 1] if j > 0 else 0.0)
        # Full reorthogonalization (classical Gram-Schmidt, repeated):
        for _ in range(reorth_passes):
            for qi in Q:
                w = w - (w @ qi) * qi
        beta = float(w.norm().item())
        if beta < 1e-10 * max(abs(alpha), 1.0):
            break                                   # happy breakdown: Krylov space exhausted
        betas.append(beta)
        Q.append(w / beta)

    n = len(alphas)
    T = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        T[i, i] = alphas[i]
        if i + 1 < n:
            T[i, i + 1] = T[i + 1, i] = betas[i]
    evals, evecs = torch.linalg.eigh(T)
    order = torch.argsort(evals, descending=True)
    kk = min(k, n)
    ritz_vals = evals[order[:kk]]

    ritz_vecs = None
    residuals = []
    if n_vectors > 0:
        nv = min(n_vectors, kk)
        vecs = []
        Qm = torch.stack(Q[:n])                     # (n, dim)
        for i in range(nv):
            c = evecs[:, order[i]].to(Qm.dtype).to(dev)
            u = (c @ Qm)
            u = u / u.norm()
            vecs.append(u)
            r = op.matvec_flat(u) - float(ritz_vals[i]) * u
            residuals.append(float(r.norm().item()))
        ritz_vecs = torch.stack(vecs)

    info = dict(iters=n, residuals=residuals,
                converged_early=(n < m))
    return ritz_vals, ritz_vecs, info
