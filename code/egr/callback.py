"""Empirical Gradient Ratio (EGR) logger.

Tracks ``||grad_theta L|| / ||grad_phi L||`` across training to diagnose gradient
starvation between the spectral (``f_theta``) and spatial (``g_phi``) blocks.

Call ``log_step`` AFTER ``loss.backward()`` and BEFORE ``optimizer.zero_grad()``.
Misordering silently yields zeros.
"""

from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn


class EGRLogger:
    """Records per-step L2 norms of spectral/spatial gradients and their ratio.

    The norm is the L2 norm of the CONCATENATED gradient vector across all params
    in the block, which equals ``sqrt(sum of per-param squared L2 norms)``.
    """

    def __init__(self, model: nn.Module, eps: float = 1e-12):
        """Snapshot ``model.spectral`` and ``model.spatial`` params.

        Args:
            model: must expose ``model.spectral`` and ``model.spatial`` submodules.
                Parameter enumeration is fixed at construction time; if the user
                reassigns these submodules, a new logger must be constructed.
            eps:   denominator floor when computing the ratio. ``1e-12`` is fine
                for fp32; bump to ``1e-6`` for fp16.
        """
        if not hasattr(model, "spectral") or not hasattr(model, "spatial"):
            raise AttributeError(
                "EGRLogger requires model.spectral and model.spatial submodules"
            )
        self.model = model
        self.eps = eps
        # Snapshot params at construction. List comprehension fixes ordering.
        # requires_grad=False params are silently skipped here by design.
        self._theta_params = [p for p in model.spectral.parameters() if p.requires_grad]
        self._phi_params = [p for p in model.spatial.parameters() if p.requires_grad]
        self._records: List[Dict[str, float]] = []
        self._warned_all_zero = False

    @torch.no_grad()
    def log_step(self, step: int) -> Dict[str, float]:
        """Compute and record gradient norms for the current step.

        MUST be called AFTER ``loss.backward()`` and BEFORE ``optimizer.zero_grad()``.
        Calling at the wrong point yields all zeros silently — the most common misuse.

        Note on accumulation: with gradient accumulation (multiple ``.backward()``
        before ``.step()``), the recorded gradients reflect the accumulated value.
        With grad clipping, call BEFORE clip unless post-clip EGR is desired.

        Returns:
            dict with keys ``step``, ``grad_theta_norm``, ``grad_phi_norm``, ``egr``.
        """
        # NOTE: per-param ``.item()`` issues one CUDA sync per param. For small
        # models this is fine. For very large models, accumulate on-device into
        # a single scalar tensor and call ``.item()`` once.
        theta_sq = 0.0
        for p in self._theta_params:
            if p.grad is None:
                continue
            theta_sq += float(p.grad.detach().pow(2).sum().item())

        phi_sq = 0.0
        for p in self._phi_params:
            if p.grad is None:
                continue
            phi_sq += float(p.grad.detach().pow(2).sum().item())

        theta_norm = theta_sq ** 0.5
        phi_norm = phi_sq ** 0.5
        egr = theta_norm / (phi_norm + self.eps)

        if not self._warned_all_zero and theta_norm == 0.0 and phi_norm == 0.0:
            import warnings
            warnings.warn(
                "EGRLogger.log_step: both grad norms are exactly 0. "
                "Did you call log_step BEFORE loss.backward() or AFTER optimizer.zero_grad()?",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_all_zero = True

        rec = {
            "step": int(step),
            "grad_theta_norm": theta_norm,
            "grad_phi_norm": phi_norm,
            "egr": egr,
        }
        self._records.append(rec)
        return rec

    def to_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame of all logged steps in call order.

        Columns: ``['step', 'grad_theta_norm', 'grad_phi_norm', 'egr']``.
        Returns an empty DataFrame with these columns if no steps were logged.
        """
        cols = ["step", "grad_theta_norm", "grad_phi_norm", "egr"]
        if not self._records:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(self._records, columns=cols)

    def reset(self) -> None:
        """Clear recorded history. Does not re-snapshot params."""
        self._records.clear()
        self._warned_all_zero = False
