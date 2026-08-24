"""Generalized Gauss-Newton (GGN) block top-eigenvalue estimation.

The theory (Theorem 1) is stated on the GGN blocks

    G = J^T H_tau J,     J = N^{-1/2} d(logits)/d(params),
    H_tau = diag(p) - p p^T   (softmax Jacobian, per pixel),

NOT on the loss-Hessian blocks. The two differ by the functional-Hessian
term, which is nonzero and uncontrolled at initialization, so measuring
the Hessian and calling it G is measuring the wrong object (audit A24).
This module computes the real thing.

The GGN-vector product needs no second derivatives:

    G v = (1/N) J_u^T [ H_tau (J_u v) ]

where J_u is the *unnormalized* logit Jacobian and the 1/N carries the
loss averaging convention of eq:loss. Three steps:

    1. Jv   = J_u v          (JVP, via the double-backward trick)
    2. u    = H_tau (Jv)     (closed form: p*x - p*<p,x>, per pixel)
    3. G v  = J_u^T u / N    (VJP, one autograd.grad call)

Masked pixels (ignore_index) are dropped from both the average and the
Jacobian: N counts valid pixels only, matching the loss.

Run this file directly for a brute-force validation against an explicitly
materialized J^T H_tau J on a tiny model.
"""
from __future__ import annotations

import contextlib
from typing import Iterable, List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def math_attention():
    """Force SDPA onto the math backend, which supports double backward.

    PyTorch's fused/flash attention kernels implement only a first-order
    backward: ``_scaled_dot_product_efficient_attention_backward`` has no
    derivative, so the JVP inside the GGN product raises. The math backend
    is composed of primitive ops and differentiates twice. Slower and more
    memory-hungry, but this is a measurement path, not a training loop.

    No-op on PyTorch builds without ``torch.nn.attention``.
    """
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel(SDPBackend.MATH)
    except Exception:                      # pragma: no cover - old torch
        return contextlib.nullcontext()


def _global_l2_norm(vs: Sequence[torch.Tensor]) -> torch.Tensor:
    sq = torch.zeros((), device=vs[0].device, dtype=vs[0].dtype)
    for v in vs:
        sq = sq + (v * v).sum()
    return torch.sqrt(sq)


def _normalized(vs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    with torch.no_grad():
        n = torch.clamp(_global_l2_norm(vs), min=1e-12)
        return [v / n for v in vs]


def softmax_jacobian_apply(
    logits: torch.Tensor, x: torch.Tensor, class_dim: int
) -> torch.Tensor:
    """Apply H_tau = diag(p) - p p^T to ``x``, per pixel, without materializing it.

    For a single pixel with probabilities p and vector x,
    ``H_tau x = p * x - p * <p, x>``.

    Args:
        logits: raw logits; softmax is taken along ``class_dim``.
        x:      same shape as ``logits``.
        class_dim: axis holding the class scores.
    """
    p = torch.softmax(logits, dim=class_dim)
    px = p * x
    return px - p * px.sum(dim=class_dim, keepdim=True)


def _valid_mask(
    y: torch.Tensor, logits: torch.Tensor, class_dim: int, ignore_index: int
) -> Tuple[torch.Tensor, int]:
    """Boolean mask broadcastable over ``logits`` plus the valid-pixel count."""
    keep = y != ignore_index                       # (B, H, W)
    n_valid = int(keep.sum().item())
    mask = keep.unsqueeze(class_dim).to(logits.dtype)
    return mask, n_valid


def ggn_block_vector_product(
    logits: torch.Tensor,
    params: Sequence[torch.Tensor],
    v: Sequence[torch.Tensor],
    y: torch.Tensor,
    class_dim: int = 1,
    ignore_index: int = 255,
    n_valid: Optional[int] = None,
) -> List[torch.Tensor]:
    """One GGN-block matrix-vector product ``G v`` for the given parameter block.

    ``logits`` must come from a forward pass whose graph is still alive and
    which was built from ``params`` (do not call under ``no_grad``).

    Args:
        logits: (B, C, H, W) with classes on ``class_dim``.
        params: the parameter block defining the GGN block (e.g. spectral).
        v:      direction, same shapes as ``params``.
        y:      integer labels (B, H, W); ``ignore_index`` pixels are dropped.
        n_valid: pixel count override; recomputed from ``y`` when None.

    Returns:
        List of tensors shaped like ``params`` holding ``G v``.
    """
    mask, n_from_y = _valid_mask(y, logits, class_dim, ignore_index)
    N = n_from_y if n_valid is None else n_valid
    if N == 0:
        return [torch.zeros_like(p) for p in params]

    # --- step 1: JVP  Jv = d(logits)/d(params) @ v -------------------------
    # Double-backward trick: grad of (J^T dummy) w.r.t. dummy, contracted
    # with v, is J v. `dummy` is a formal variable, its value is irrelevant.
    dummy = torch.zeros_like(logits, requires_grad=True)
    jt_dummy = torch.autograd.grad(
        logits, params, grad_outputs=dummy, create_graph=True
    )
    Jv = torch.autograd.grad(
        jt_dummy, dummy, grad_outputs=list(v), retain_graph=True
    )[0]

    # --- step 2: apply the softmax Jacobian, mask, and normalize -----------
    with torch.no_grad():
        HJv = softmax_jacobian_apply(logits.detach(), Jv, class_dim)
        HJv = HJv * mask / N

    # --- step 3: VJP  G v = J^T (H_tau J v) / N ----------------------------
    Gv = torch.autograd.grad(logits, params, grad_outputs=HJv, retain_graph=True)
    return [g.detach() for g in Gv]


def top_eigenvalue_ggn_block(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    block: Union[str, nn.Module, Iterable[torch.Tensor]],
    n_iter: int = 30,
    tol: float = 1e-5,
    class_dim: int = 1,
    ignore_index: int = 255,
    seed: Optional[int] = None,
    return_history: bool = False,
    force_math_attention: bool = True,
) -> Union[float, Tuple[float, List[float]]]:
    """Top eigenvalue of the GGN block ``G_bb`` by power iteration.

    ``block`` may be an attribute name on ``model``, a submodule, or an
    explicit iterable of parameters. One batch, already on the right device.

    G is positive semidefinite, so power iteration converges to
    ``lambda_max`` without the sign ambiguity that afflicts the indefinite
    loss Hessian.
    """
    if isinstance(block, str):
        sub = model
        for part in block.split("."):
            sub = getattr(sub, part)
        params = [p for p in sub.parameters() if p.requires_grad]
    elif isinstance(block, nn.Module):
        params = [p for p in block.parameters() if p.requires_grad]
    else:
        params = [p for p in block if p.requires_grad]

    if not params:
        raise ValueError("parameter block is empty or has no trainable params")
    if y.dtype != torch.long:
        raise TypeError(f"y must be torch.long, got {y.dtype}")

    ctx = math_attention() if force_math_attention else contextlib.nullcontext()
    with ctx:
        logits = model(X)
    if logits.shape[class_dim] < 2:
        raise ValueError(
            f"expected >=2 classes on dim {class_dim}, got shape {tuple(logits.shape)}"
        )
    _, n_valid = _valid_mask(y, logits, class_dim, ignore_index)
    if n_valid == 0:
        raise ValueError("batch contains no valid (non-ignored) pixels")

    gen = None
    if seed is not None:
        gen = torch.Generator(device=params[0].device).manual_seed(seed)
    v = [
        torch.randn(p.shape, device=p.device, dtype=p.dtype, generator=gen)
        for p in params
    ]
    v = _normalized(v)

    history: List[float] = []
    eig = 0.0
    eig_old = float("inf")
    for _ in range(n_iter):
        Gv = ggn_block_vector_product(
            logits, params, v, y,
            class_dim=class_dim, ignore_index=ignore_index, n_valid=n_valid,
        )
        with torch.no_grad():
            ray = sum((gi * vi).sum() for gi, vi in zip(Gv, v))
            eig = float(ray.item())
        history.append(eig)

        nrm = float(_global_l2_norm(Gv).item())
        if nrm < 1e-20:               # G v collapsed: block has zero curvature
            eig = 0.0
            break
        v = _normalized(Gv)

        if abs(eig - eig_old) / max(abs(eig), 1e-12) < tol:
            break
        eig_old = eig

    if return_history:
        return eig, history
    return eig


def _brute_force_ggn_block(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    params: Sequence[torch.Tensor],
    class_dim: int = 1,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Materialize G_bb = (1/N) J^T H_tau J explicitly. Tiny models only."""
    logits = model(X)
    mask, N = _valid_mask(y, logits, class_dim, ignore_index)
    P = sum(p.numel() for p in params)

    # Build the full unnormalized Jacobian J_u, one logit at a time.
    flat_logits = logits.reshape(-1)
    rows = []
    for i in range(flat_logits.numel()):
        g = torch.autograd.grad(flat_logits[i], params, retain_graph=True)
        rows.append(torch.cat([gi.reshape(-1) for gi in g]))
    J = torch.stack(rows)                                   # (n_logits, P)

    # Block-diagonal H_tau over pixels, in the same flattening order.
    with torch.no_grad():
        C = logits.shape[class_dim]
        p_full = torch.softmax(logits, dim=class_dim)
        H = torch.zeros(flat_logits.numel(), flat_logits.numel(),
                        device=logits.device, dtype=logits.dtype)
        pm = p_full.movedim(class_dim, -1).reshape(-1, C)    # (n_pix, C)
        keep_flat = (y != ignore_index).reshape(-1)          # (n_pix,)
        # index of (pixel k, class c) within the original flattening
        idx = torch.arange(flat_logits.numel(), device=logits.device)
        idx = idx.reshape(logits.shape).movedim(class_dim, -1).reshape(-1, C)
        for k in range(pm.shape[0]):
            if not bool(keep_flat[k]):
                continue
            pk = pm[k]
            Hk = torch.diag(pk) - torch.outer(pk, pk)
            ii = idx[k]
            H[ii.unsqueeze(1), ii.unsqueeze(0)] = Hk
    return (J.T @ H @ J) / N


if __name__ == "__main__":
    # Validation: power iteration vs an explicitly materialized J^T H_tau J.
    torch.manual_seed(0)

    class Toy(nn.Module):
        def __init__(self, S=5, K=3, C=4):
            super().__init__()
            self.spectral = nn.Linear(S, K, bias=False)
            self.spatial = nn.Sequential(
                nn.Conv2d(K, 6, 3, padding=1), nn.Tanh(), nn.Conv2d(6, C, 1)
            )

        def forward(self, X):                    # X: (B, S, H, W)
            z = self.spectral(X.movedim(1, -1)).movedim(-1, 1)
            return self.spatial(z)               # (B, C, H, W)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Toy().to(dev).double()
    B, S, H, W = 2, 5, 3, 3
    X = torch.randn(B, S, H, W, device=dev, dtype=torch.float64)
    y = torch.randint(0, 4, (B, H, W), device=dev, dtype=torch.long)
    y[0, 0, 0] = 255                              # exercise the ignore path

    print(f"{'block':10s} {'power-iter':>14s} {'brute force':>14s} {'rel.err':>10s}")
    ok = True
    for name in ("spectral", "spatial"):
        params = [p for p in getattr(model, name).parameters() if p.requires_grad]
        eig = top_eigenvalue_ggn_block(model, X, y, name, n_iter=200, tol=1e-12, seed=0)
        G = _brute_force_ggn_block(model, X, y, params)
        exact = float(torch.linalg.eigvalsh(0.5 * (G + G.T))[-1].item())
        rel = abs(eig - exact) / max(abs(exact), 1e-30)
        ok &= rel < 1e-6
        print(f"{name:10s} {eig:14.10f} {exact:14.10f} {rel:10.2e}")

        # G must be PSD: no eigenvalue may be meaningfully negative.
        lo = float(torch.linalg.eigvalsh(0.5 * (G + G.T))[0].item())
        assert lo > -1e-10, f"{name}: GGN not PSD (min eig {lo})"

    # The loss-gradient identity: grad L = J^T r must match autograd exactly.
    logits = model(X)
    keep = (y != 255)
    loss = F.cross_entropy(
        logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1]),
        y.reshape(-1), ignore_index=255, reduction="mean",
    )
    params = [p for p in model.spectral.parameters()]
    g_auto = torch.autograd.grad(loss, params, retain_graph=True)
    with torch.no_grad():
        p_sm = torch.softmax(logits, dim=1)
        onehot = F.one_hot(y.clamp(max=3), num_classes=4).permute(0, 3, 1, 2)
        r = (p_sm - onehot) * keep.unsqueeze(1) / int(keep.sum().item())
    g_man = torch.autograd.grad(logits, params, grad_outputs=r, retain_graph=True)
    gerr = max(
        float((a - b).abs().max().item()) for a, b in zip(g_auto, g_man)
    )
    ok &= gerr < 1e-10
    print(f"\ngrad L = J^T r  max abs err: {gerr:.3e}")
    print("\nVALIDATION", "PASSED" if ok else "FAILED")
