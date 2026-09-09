#!/usr/bin/env python3
"""Numerical verification of the P3 interior-witness results in
``review_packet/astra/math_01.md`` (Lemma 3.1, Theorem 3.2, eq. (3.6),
Counterexamples 3.3 and 3.4).

Everything here is CPU / torch float64 / dense autograd GGN.  There is no
training: every quantity is a curvature measurement at a random draw of the
initialization, exactly as the theorems are stated.

Model (Theorem 3.2)
-------------------
    a_n   = W h_n + b            W in R^{q x M},  W_ij ~ N(0, s_w^2 / M)
                                 b in R^q,        b_i  ~ N(0, s_b^2)
    yhat_n = V ReLU(a_n)         V in R^{C x q},  V_ci ~ N(0, s_v^2 / q)

with N = 64 deterministic sites h_n in R^M satisfying ||h_n||^2 = R^2 M
(exactly, with R = 1 by construction) and lambda_max(S_h) >= a0 M, where
S_h = (1/N) sum_n h_n h_n^T.  The features are built as

    h_n = sqrt(M) * ( sqrt(rho) u0 + sqrt(1-rho) g_n ),   g_n unit, g_n _|_ u0

so ||h_n||^2 = M exactly and S_h has a top eigenvalue ~ rho M (the common
component u0), i.e. a0 ~ rho.  The features are drawn ONCE per M with a fixed
seed and are then held deterministic while W, b, V are redrawn.

Normalization convention (the note's "Normalization and a correction")
----------------------------------------------------------------------
z = yhat / sqrt(N) are the *normalized* logits, L = l(z).

  * squared loss   l(z) = 1/2 ||z - y||^2,   H = I,
                   G = J_z^T J_z          with J_z = J_yhat / sqrt(N).
  * averaged softmax CE  L = (1/N) sum_n CE(yhat_n, y_n).  In the normalized
    logit variable the output Hessian is exactly the *unaveraged* block
    diagonal H = blockdiag(H_n), H_n = diag(p_n) - p_n p_n^T, p_n =
    softmax(yhat_n).  (d^2L/dyhat^2 = (1/N) blockdiag(H_n) and dyhat/dz =
    sqrt(N) I, so the two factors of sqrt(N) cancel the 1/N.)
    G = J_z^T H J_z.
    Note the GGN does not depend on the labels at all - only on p_n - so no
    labels are generated for the CE arms.

Top-eigenvalue computation
--------------------------
G_WW is (qM) x (qM), which is 16384 x 16384 at M = 1024, q = 16.  We use the
exact transpose (Gram) identity: with B = H^{1/2} J_z of shape (NC) x (qM),

    G_WW = B^T B      and      nonzero eig(B^T B) = nonzero eig(B B^T),

so lambda_max(G_WW) = lambda_max(H^{1/2} J_z J_z^T H^{1/2}), an (NC) x (NC)
= 256 x 256 symmetric eigenproblem.  This is not an approximation.  The
script additionally forms the *dense* (qM) x (qM) GGN at the small
configurations and reports the max relative disagreement of the two top
eigenvalues (metric ``dense_vs_gram_max_rel_err``), and it validates the
autograd Jacobian against the closed form
J[n,c,i,m] = V[c,i] * 1{a_ni > 0} * h[n,m] (metric ``autograd_vs_analytic_*``).

What is checked
---------------
Part 1 (squared loss, scalar readout, eq. (3.6)):
    E lambda_max(G_WW) >= (s_v^2 / (2q)) lambda_max(S_h).
  The RHS is the *exact* expectation of the single witness Delta W = e_1 u^T
  (curvature v_1^2 Q with Q = (1/N) sum_n (h_n^T u)^2 1{a_n1 > 0}, v independent
  of Q, E v_1^2 = s_v^2/q, E Q = lambda_max(S_h)/2), so the script reports both
  the ratio for lambda_max (a slack >= 1) and the ratio for the witness itself
  (must equal 1 up to Monte-Carlo error).
  (3.6) bounds an EXPECTATION, not each draw: ``frac_draws_lam_ge_rhs`` is a
  diagnostic only, and values below 1 are not violations.

Part 2 (averaged softmax CE, Theorem 3.2 / Lemma 3.1):
    distribution of lambda_max(G_WW)/M over draws, log-log slope of the mean
    in M (must be ~ 1), and the fraction of draws with lambda_max >= c M for a
    grid of small c (must not decay with M: the positive-probability floor).
    Lemma 3.1's inequality (3.1) lambda_max(G_WW) >= lambda_max(S_{h,a}) is
    checked per draw for a = e_1.
    Counterexample 3.3: for q = 1 with identical features and zero bias the
    all-inactive probability is 1/2 and M-independent (and 2^{-q} for general
    fixed q); on that event G_WW vanishes identically.

Part 3 (Counterexample 3.4): identical features h_n = h, ||h||^2 = M, with a
    train-mode BatchNorm1d(q) inserted immediately after W h_n + b.  The
    per-site batch statistics over the N sites make the BN output constant and
    W-independent, so G_WW = 0 to numerical tolerance while lambda_max(S_h) = M
    and lambda_max(S_h^p) = p_min * M.  With eval-mode BN (frozen running
    stats) the same block is nonzero and grows linearly in M.

Output
------
Long-format CSV at results/toy_interior_witness.csv.

Usage
-----
    python code/experiments/toy_interior_witness.py
    python code/experiments/toy_interior_witness.py --draws 50 --out /tmp/x.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

ROOT = "/home/u37314kd/Projects/spectral_shortcut_theory"
DEFAULT_CSV = os.path.join(ROOT, "results", "toy_interior_witness.csv")

FIELDS = [
    "experiment", "loss", "arm", "M", "q", "C", "N",
    "s_w", "s_b", "s_v", "rho", "n_draws", "metric", "value",
]
ROWS: list[dict] = []


def rec(experiment, metric, value, *, loss="", arm="", M="", q="", C="", N="",
        s_w="", s_b="", s_v="", rho="", n_draws=""):
    """Append one long-format record."""
    ROWS.append({
        "experiment": experiment, "loss": loss, "arm": arm,
        "M": M, "q": q, "C": C, "N": N,
        "s_w": s_w, "s_b": s_b, "s_v": s_v, "rho": rho,
        "n_draws": n_draws, "metric": metric,
        "value": float(value) if value is not None else "",
    })


# --------------------------------------------------------------------------
# features and model
# --------------------------------------------------------------------------

def make_features(M: int, N: int, rho: float, seed: int) -> torch.Tensor:
    """Deterministic h_n in R^M with ||h_n||^2 = M exactly and a common
    component of relative energy rho, so lambda_max(S_h) ~ rho * M."""
    g = torch.Generator().manual_seed(seed)
    u0 = torch.randn(M, generator=g)
    u0 = u0 / u0.norm()
    G = torch.randn(N, M, generator=g)
    G = G - (G @ u0)[:, None] * u0[None, :]          # orthogonalize against u0
    G = G / G.norm(dim=1, keepdim=True)              # unit directions
    h = math.sqrt(M) * (math.sqrt(rho) * u0[None, :] + math.sqrt(1.0 - rho) * G)
    return h


def top_eig_Sh(h: torch.Tensor):
    """lambda_max(S_h) and its top unit eigenvector, via the N x N Gram."""
    N = h.shape[0]
    K = (h @ h.T) / N                                # N x N, same nonzero spectrum
    w, U = torch.linalg.eigh((K + K.T) / 2)
    lam = float(w[-1].clamp_min(0.0))
    # top eigenvector of S_h = (1/N) h^T h  is  h^T U[:, -1] / sqrt(N * lam)
    u = h.T @ U[:, -1]
    u = u / u.norm()
    return lam, u


def weighted_top_eig(h: torch.Tensor, w_site: torch.Tensor) -> torch.Tensor:
    """lambda_max of (1/N) sum_n w_n h_n h_n^T, w_n >= 0, via the N x N Gram."""
    N = h.shape[0]
    s = w_site.clamp_min(0.0).sqrt()[:, None] * h
    K = (s @ s.T) / N
    return torch.linalg.eigvalsh((K + K.T) / 2)[-1].clamp_min(0.0)


def draw_params(M, q, C, s_w, s_b, s_v, seed):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(q, M, generator=g) * (s_w / math.sqrt(M))
    b = torch.randn(q, generator=g) * s_b
    V = torch.randn(C, q, generator=g) * (s_v / math.sqrt(q))
    return W, b, V


def logits(W, b, V, h):
    """yhat_n = V ReLU(W h_n + b);  returns (N, C)."""
    return torch.relu(h @ W.T + b) @ V.T


# --------------------------------------------------------------------------
# GGN machinery
# --------------------------------------------------------------------------

def jac_W(W, b, V, h):
    """Dense autograd Jacobian d(yhat)/d(W), shape (N*C, q, M)."""
    f = lambda Wv: logits(Wv, b, V, h).reshape(-1)
    return torch.autograd.functional.jacobian(f, W, vectorize=True)


def ce_hessian_blocks(yhat):
    """H_n = diag(p_n) - p_n p_n^T for the *normalized* logits, shape (N, C, C)."""
    p = torch.softmax(yhat, dim=1)
    return torch.diag_embed(p) - p[:, :, None] * p[:, None, :], p


def blocks_sqrt(H):
    """Batched PSD square root of (N, C, C) blocks."""
    w, U = torch.linalg.eigh((H + H.transpose(-1, -2)) / 2)
    return U @ torch.diag_embed(w.clamp_min(0.0).sqrt()) @ U.transpose(-1, -2)


def top_eig_gram(Jz, Hh=None):
    """lambda_max(J_z^T H J_z) via the (NC x NC) Gram; Hh = blockdiag(H_n^{1/2})."""
    K = Jz @ Jz.T
    if Hh is not None:
        K = Hh @ K @ Hh
    return torch.linalg.eigvalsh((K + K.T) / 2)[-1].clamp_min(0.0)


def top_eig_dense(Jz, Hfull=None):
    """lambda_max of the explicit dense (qM x qM) GGN block."""
    G = Jz.T @ Jz if Hfull is None else Jz.T @ (Hfull @ Jz)
    return torch.linalg.eigvalsh((G + G.T) / 2)[-1].clamp_min(0.0)


def quad_form(Jz, dW, Hfull=None):
    """vec(dW)^T G_WW vec(dW) for a unit-Frobenius direction dW."""
    d = Jz @ dW.reshape(-1)
    return d @ d if Hfull is None else d @ (Hfull @ d)


def loglog_slope(experiment, metric, q, arm=None, min_M=None):
    """Least-squares slope of log(value) vs log(M) over the recorded rows."""
    xs, ys = [], []
    for r in ROWS:
        if r["experiment"] != experiment or r["metric"] != metric or r["q"] != q:
            continue
        if arm is not None and r["arm"] != arm:
            continue
        if min_M is not None and r["M"] < min_M:
            continue
        if r["value"] > 0:
            xs.append(math.log(r["M"]))
            ys.append(math.log(r["value"]))
    if len(xs) < 2:
        return None
    return float(np.polyfit(xs, ys, 1)[0])


# --------------------------------------------------------------------------
# Part 1 - squared loss, scalar readout, eq. (3.6)
# --------------------------------------------------------------------------

def witness_gate_mc(h, u, s_w, s_b, seed, draws, chunk=2000):
    """High-precision Monte-Carlo estimate of E[Q], Q = (1/N) sum_n (h_n.u)^2 chi_n1.

    The exact claim behind (3.6) is E[Q] = lambda_max(S_h)/2, because each site's
    first preactivation W_1.h_n + b_1 is a nondegenerate centered Gaussian, so
    E chi_n1 = 1/2 regardless of the (strong) dependence across sites.
    Q needs only row 1 of W and b_1, so it is far cheaper than a GGN draw and can
    be estimated with many more samples than lambda_max(G_WW).
    """
    N, M = h.shape
    g = torch.Generator().manual_seed(seed)
    a = (h @ u) ** 2                                    # (N,)
    out = []
    done = 0
    while done < draws:
        k = min(chunk, draws - done)
        w1 = torch.randn(k, M, generator=g) * (s_w / math.sqrt(M))
        b1 = torch.randn(k, 1, generator=g) * s_b
        chi = (w1 @ h.T + b1 > 0).to(h.dtype)           # (k, N)
        out.append((chi * a[None, :]).mean(dim=1))
        done += k
    return torch.cat(out)


def part1_squared(M_list, q_list, N, s_w, s_b, s_v, rho, draws, base_seed,
                  dense_check_budget=3, mc_draws=20000, verbose=True):
    print("\n" + "=" * 78)
    print("PART 1  squared loss, scalar readout - eq. (3.6)")
    print("  E lambda_max(G_WW)  >=  (s_v^2 / (2q)) lambda_max(S_h)")
    print("=" * 78)
    C = 1
    for M in M_list:
        h = make_features(M, N, rho, seed=base_seed + 7919 * M)
        lam_Sh, u = top_eig_Sh(h)
        rec("features", "lam_max_Sh", lam_Sh, M=M, N=N, rho=rho)
        rec("features", "a0_emp_lam_Sh_over_M", lam_Sh / M, M=M, N=N, rho=rho)
        rec("features", "max_hn_sqnorm_over_M",
            (h.pow(2).sum(1).max() / M), M=M, N=N, rho=rho)

        # ---- sharp test of the gate identity behind (3.6): E[Q] = lam_Sh / 2
        Qmc = witness_gate_mc(h, u, s_w, s_b, base_seed + 555 + M, mc_draws)
        rec("eq3_6", "mc_mean_Q", Qmc.mean(), loss="squared", M=M, N=N,
            s_w=s_w, s_b=s_b, rho=rho, n_draws=mc_draws)
        rec("eq3_6", "mc_sem_Q", Qmc.std(unbiased=True) / math.sqrt(mc_draws),
            loss="squared", M=M, N=N, s_w=s_w, s_b=s_b, rho=rho, n_draws=mc_draws)
        rec("eq3_6", "mc_ratio_EQ_over_half_lamSh", float(Qmc.mean()) / (lam_Sh / 2),
            loss="squared", M=M, N=N, s_w=s_w, s_b=s_b, rho=rho, n_draws=mc_draws)
        print(f"  M={M:5d}       | gate identity  E[Q]={float(Qmc.mean()):.5f} "
              f"vs lam_Sh/2={lam_Sh/2:.5f}  ratio="
              f"{float(Qmc.mean())/(lam_Sh/2):.5f} "
              f"(+-{float(Qmc.std())/math.sqrt(mc_draws)/(lam_Sh/2):.5f}, "
              f"{mc_draws} draws)")

        for q in q_list:
            rhs = (s_v ** 2 / (2.0 * q)) * lam_Sh          # eq. (3.6) RHS
            lams, wits, wit_err = [], [], 0.0
            dense_rel_err, n_dense = 0.0, 0
            ana_err = 0.0
            t0 = time.time()
            for d in range(draws):
                W, b, V = draw_params(M, q, C, s_w, s_b, s_v,
                                      seed=base_seed + 1000003 * d + 101 * q + M)
                J = jac_W(W, b, V, h)                       # (N*C, q, M)
                Jz = J.reshape(N * C, q * M) / math.sqrt(N)  # normalized logits

                chi = (h @ W.T + b > 0).to(h.dtype)          # (N, q) ReLU gates
                if d < 2:  # autograd vs closed form
                    Jan = torch.einsum("ci,ni,nm->ncim", V, chi, h).reshape(N * C, q, M)
                    ana_err = max(ana_err, float((Jan - J).abs().max()))

                lam = top_eig_gram(Jz)
                lams.append(float(lam))

                # single witness of the proof: Delta W = e_1 u^T (unit Frobenius)
                dW = torch.zeros(q, M)
                dW[0] = u
                wit = quad_form(Jz, dW)
                # closed form: v_1^2 * Q,  Q = (1/N) sum_n (h_n^T u)^2 chi_n1
                Q = ((h @ u) ** 2 * chi[:, 0]).mean()
                wit_cf = V[0, 0] ** 2 * Q
                wit_err = max(wit_err, float((wit - wit_cf).abs()))
                wits.append(float(wit))

                if n_dense < dense_check_budget and q * M <= 4096:
                    lam_d = top_eig_dense(Jz)
                    dense_rel_err = max(dense_rel_err,
                                        float((lam_d - lam).abs() / lam.clamp_min(1e-300)))
                    n_dense += 1

            lams = np.asarray(lams)
            wits = np.asarray(wits)
            kw = dict(loss="squared", M=M, q=q, C=C, N=N, s_w=s_w, s_b=s_b,
                      s_v=s_v, rho=rho, n_draws=draws)
            rec("eq3_6", "rhs_theory_3_6", rhs, **kw)
            rec("eq3_6", "mean_lam_max_GWW", lams.mean(), **kw)
            rec("eq3_6", "sem_lam_max_GWW", lams.std(ddof=1) / math.sqrt(draws), **kw)
            rec("eq3_6", "min_lam_max_GWW", lams.min(), **kw)
            rec("eq3_6", "ratio_mean_lam_over_rhs", lams.mean() / rhs, **kw)
            rec("eq3_6", "frac_draws_lam_ge_rhs", float((lams >= rhs).mean()), **kw)
            rec("eq3_6", "mean_witness_curvature", wits.mean(), **kw)
            rec("eq3_6", "sem_witness_curvature",
                wits.std(ddof=1) / math.sqrt(draws), **kw)
            rec("eq3_6", "ratio_mean_witness_over_rhs", wits.mean() / rhs, **kw)
            rec("eq3_6", "witness_closedform_max_abs_err", wit_err, **kw)
            rec("eq3_6", "autograd_vs_analytic_max_abs_err", ana_err, **kw)

            # high-precision version of E[witness] = E[v_1^2] E[Q] = rhs, using
            # the cheap gate samples and independent readout draws
            gv = torch.Generator().manual_seed(base_seed + 909 + 17 * q + M)
            v1sq = (torch.randn(mc_draws, generator=gv) * (s_v / math.sqrt(q))) ** 2
            wmc = v1sq * Qmc
            kwmc = dict(kw)
            kwmc["n_draws"] = mc_draws
            rec("eq3_6", "mc_mean_witness_curvature", wmc.mean(), **kwmc)
            rec("eq3_6", "mc_ratio_mean_witness_over_rhs",
                float(wmc.mean()) / rhs, **kwmc)
            rec("eq3_6", "mc_sem_ratio_witness_over_rhs",
                float(wmc.std(unbiased=True)) / math.sqrt(mc_draws) / rhs, **kwmc)
            if n_dense:
                rec("eq3_6", "dense_vs_gram_max_rel_err", dense_rel_err, **kw)
            if verbose:
                print(f"  M={M:5d} q={q:3d} | lam_Sh={float(lam_Sh):10.3f} "
                      f"rhs={float(rhs):9.4f} | E[lam_max]={lams.mean():10.4f} "
                      f"ratio={lams.mean()/float(rhs):7.3f} | "
                      f"E[witness]/rhs={wits.mean()/float(rhs):6.3f} "
                      f"({time.time()-t0:.1f}s)")

    # log-log slope of E lambda_max in M.  The full-grid slope is contaminated by
    # the smallest M: with N=64 sites and M=16 the non-common feature directions
    # (which carry (1-rho)M spread over min(N,M)-1 modes) are not yet negligible
    # against the common mode, so we also report the slope over the large-M tail.
    big = sorted(M_list)[1] if len(M_list) > 2 else None
    for q in q_list:
        for min_M, tag in [(None, "allM"), (big, f"Mge{big}")]:
            if min_M is None and tag != "allM":
                continue
            s = loglog_slope("eq3_6", "mean_lam_max_GWW", q, min_M=min_M)
            if s is None:
                continue
            rec("eq3_6", f"loglog_slope_mean_lam_vs_M_{tag}", s,
                loss="squared", q=q, N=N, rho=rho, n_draws=draws)
            print(f"  [squared] q={q}: log-log slope of E lambda_max vs M "
                  f"({tag}) = {s:.4f}")


# --------------------------------------------------------------------------
# Part 2 - averaged softmax CE, Theorem 3.2 and Lemma 3.1
# --------------------------------------------------------------------------

C_GRID = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]


def part2_ce(M_list, q_list, N, C, s_w, s_b, s_v, rho, draws, base_seed,
             dense_check_budget=3, verbose=True):
    print("\n" + "=" * 78)
    print("PART 2  averaged softmax CE - Theorem 3.2 floor + Lemma 3.1 (3.1)")
    print("=" * 78)
    for M in M_list:
        h = make_features(M, N, rho, seed=base_seed + 7919 * M)
        lam_Sh, u = top_eig_Sh(h)
        for q in q_list:
            lams, wits, lam_Sha, lam_phis = [], [], [], []
            dense_rel_err, n_dense = 0.0, 0
            wit_err = 0.0
            t0 = time.time()
            for d in range(draws):
                W, b, V = draw_params(M, q, C, s_w, s_b, s_v,
                                      seed=base_seed + 1000003 * d + 101 * q + M)
                yhat = logits(W, b, V, h)
                H, p = ce_hessian_blocks(yhat)
                Hh = torch.block_diag(*blocks_sqrt(H))       # (NC, NC)

                J = jac_W(W, b, V, h)
                Jz = J.reshape(N * C, q * M) / math.sqrt(N)
                lam = top_eig_gram(Jz, Hh)
                lams.append(float(lam))

                # full block G_phiphi over ALL parameters (W, b, V): the left
                # inequality of (3.1), lambda_max(G_phiphi) >= lambda_max(G_WW)
                Jb = torch.autograd.functional.jacobian(
                    lambda bv: logits(W, bv, V, h).reshape(-1), b, vectorize=True)
                JV = torch.autograd.functional.jacobian(
                    lambda Vv: logits(W, b, Vv, h).reshape(-1), V, vectorize=True)
                Jphi = torch.cat([Jz,
                                  Jb.reshape(N * C, -1) / math.sqrt(N),
                                  JV.reshape(N * C, -1) / math.sqrt(N)], dim=1)
                lam_phis.append(float(top_eig_gram(Jphi, Hh)))

                chi = (h @ W.T + b > 0).to(h.dtype)
                p_min = p.min(dim=1).values                  # (N,)

                # ---- Lemma 3.1 (3.1) with the unit witness direction a = e_1
                # D_n a = chi_n1 * V e_1 ;  Pi = I - 11^T/C
                Ve1 = V[:, 0]
                PiVe1 = Ve1 - Ve1.mean()
                w_site = p_min * chi[:, 0] * (PiVe1 @ PiVe1)
                lam_Sha.append(float(weighted_top_eig(h, w_site)))

                # ---- the proof's actual witness Delta W = e_1 u^T
                dW = torch.zeros(q, M)
                dW[0] = u
                Hfull = torch.block_diag(*H)
                wit = quad_form(Jz, dW, Hfull)
                # closed form: (1/N) sum_n (h_n^T u)^2 chi_n1 (V e_1)^T H_n (V e_1)
                quad_n = torch.einsum("c,ncd,d->n", Ve1, H, Ve1)
                wit_cf = (((h @ u) ** 2) * chi[:, 0] * quad_n).mean()
                wit_err = max(wit_err, float((wit - wit_cf).abs()))
                wits.append(float(wit))

                if n_dense < dense_check_budget and q * M <= 4096:
                    lam_d = top_eig_dense(Jz, Hfull)
                    dense_rel_err = max(dense_rel_err,
                                        float((lam_d - lam).abs() / lam.clamp_min(1e-300)))
                    n_dense += 1

            lams = np.asarray(lams)
            wits = np.asarray(wits)
            lam_Sha = np.asarray(lam_Sha)
            lam_phis = np.asarray(lam_phis)
            kw = dict(loss="ce", M=M, q=q, C=C, N=N, s_w=s_w, s_b=s_b, s_v=s_v,
                      rho=rho, n_draws=draws)
            rec("lemma3_1", "mean_lam_max_Gphiphi", lam_phis.mean(), **kw)
            rec("lemma3_1", "frac_draws_lam_Gphiphi_ge_lam_GWW",
                float((lam_phis >= lams - 1e-12).mean()), **kw)
            rec("lemma3_1", "mean_ratio_lam_Gphiphi_over_lam_GWW",
                float(np.mean(lam_phis / np.maximum(lams, 1e-300))), **kw)
            rec("thm3_2", "lam_max_Sh", lam_Sh, **kw)
            rec("thm3_2", "mean_lam_max_GWW", lams.mean(), **kw)
            rec("thm3_2", "sem_lam_max_GWW", lams.std(ddof=1) / math.sqrt(draws), **kw)
            rec("thm3_2", "std_lam_max_GWW", lams.std(ddof=1), **kw)
            for name, val in [("min", lams.min()), ("max", lams.max())]:
                rec("thm3_2", f"{name}_lam_max_GWW", val, **kw)
            for qq in (5, 25, 50, 75, 95):
                rec("thm3_2", f"q{qq:02d}_lam_max_GWW", np.percentile(lams, qq), **kw)
                rec("thm3_2", f"q{qq:02d}_lam_over_M", np.percentile(lams, qq) / M, **kw)
            rec("thm3_2", "mean_lam_over_M", lams.mean() / M, **kw)
            rec("thm3_2", "min_lam_over_M", lams.min() / M, **kw)
            for c in C_GRID:
                rec("thm3_2", f"frac_lam_ge_{c:g}M", float((lams >= c * M).mean()), **kw)
            # Lemma 3.1
            rec("lemma3_1", "frac_draws_lam_GWW_ge_lam_Sha",
                float((lams >= lam_Sha - 1e-12).mean()), **kw)
            rec("lemma3_1", "mean_lam_Sha", lam_Sha.mean(), **kw)
            # lambda_max(S_{h,a}) is EXACTLY zero on the draws where the witness
            # channel a = e_1 is inactive at every site - Counterexample 3.3
            # restricted to one channel.  The Lemma 3.1 bound still holds there
            # (0 <= lambda_max(G_WW)) but is vacuous, so the slack ratio is
            # reported only over the nondegenerate draws.
            nz = lam_Sha > 0
            rec("lemma3_1", "frac_draws_lam_Sha_zero", float((~nz).mean()), **kw)
            if nz.any():
                rec("lemma3_1", "mean_ratio_lam_GWW_over_lam_Sha_nonzero",
                    float(np.mean(lams[nz] / lam_Sha[nz])), **kw)
                rec("lemma3_1", "median_ratio_lam_GWW_over_lam_Sha_nonzero",
                    float(np.median(lams[nz] / lam_Sha[nz])), **kw)
            rec("lemma3_1", "frac_draws_lam_GWW_ge_witness",
                float((lams >= wits - 1e-12).mean()), **kw)
            rec("lemma3_1", "mean_witness_curvature", wits.mean(), **kw)
            rec("lemma3_1", "witness_closedform_max_abs_err", wit_err, **kw)
            if n_dense:
                rec("thm3_2", "dense_vs_gram_max_rel_err", dense_rel_err, **kw)
            if verbose:
                print(f"  M={M:5d} q={q:3d} | E[lam]={lams.mean():10.4f} "
                      f"E[lam]/M={lams.mean()/M:8.5f} "
                      f"med/M={np.percentile(lams,50)/M:8.5f} "
                      f"min/M={lams.min()/M:9.6f} | "
                      f"P(lam>=1e-2 M)={float((lams>=1e-2*M).mean()):.3f} "
                      f"({time.time()-t0:.1f}s)")

    big = sorted(M_list)[1] if len(M_list) > 2 else None
    for q in q_list:
        for src, tag in [("mean_lam_max_GWW", "mean"),
                         ("q50_lam_max_GWW", "median"),
                         ("q05_lam_max_GWW", "p05"),
                         ("min_lam_max_GWW", "min")]:
            for min_M, mtag in [(None, "allM"), (big, f"Mge{big}")]:
                if min_M is None and mtag != "allM":
                    continue
                s = loglog_slope("thm3_2", src, q, min_M=min_M)
                if s is None:
                    continue
                rec("thm3_2", f"loglog_slope_{tag}_lam_vs_M_{mtag}", s,
                    loss="ce", q=q, C=C, N=N, rho=rho, n_draws=draws)
                print(f"  [ce] q={q}: log-log slope of {tag} lambda_max vs M "
                      f"({mtag}) = {s:.4f}")
        # an empirically uniform floor constant: the smallest 5th-percentile of
        # lambda_max/M across the M grid.  P(lambda_max >= c* M) >= 0.95 at every
        # M by construction, which is the positive-probability floor of (3.5)
        # with a numerically usable constant.
        p05 = [r["value"] for r in ROWS if r["experiment"] == "thm3_2"
               and r["q"] == q and r["metric"] == "q05_lam_over_M"]
        mn = [r["value"] for r in ROWS if r["experiment"] == "thm3_2"
              and r["q"] == q and r["metric"] == "min_lam_over_M"]
        if p05:
            rec("thm3_2", "c_star_min_over_M_of_p05_lam_over_M", min(p05),
                loss="ce", q=q, C=C, N=N, rho=rho, n_draws=draws)
            rec("thm3_2", "c_star_min_over_M_of_min_lam_over_M", min(mn),
                loss="ce", q=q, C=C, N=N, rho=rho, n_draws=draws)
            print(f"  [ce] q={q}: uniform floor c* (min over M of the 5th pct of "
                  f"lambda_max/M) = {min(p05):.5f};  worst single draw = {min(mn):.5f}")


# --------------------------------------------------------------------------
# Part 2b - Counterexample 3.3, the all-inactive event
# --------------------------------------------------------------------------

def part2b_counterexample_3_3(M_list, N, s_w, s_b, rho, base_seed,
                              draws=20000, q_list=(1, 2, 4)):
    print("\n" + "=" * 78)
    print("PART 2b  Counterexample 3.3 - all-inactive probability")
    print("  identical h_n, zero bias:  P(all inactive) = 2^{-q}, M-independent")
    print("=" * 78)
    for q in q_list:
        for M in M_list:
            g = torch.Generator().manual_seed(base_seed + 31 * M + q)
            # --- arm A: the counterexample - identical features, zero bias
            hd = torch.randn(M, generator=g)
            hd = hd / hd.norm() * math.sqrt(M)               # ||h||^2 = M
            Wq = torch.randn(draws, q, M, generator=g) * (s_w / math.sqrt(M))
            pre = Wq @ hd                                     # (draws, q)
            all_off = (pre <= 0).all(dim=1).to(torch.float64).mean()
            kw = dict(arm="identical_h_zero_bias", M=M, q=q, N=N,
                      s_w=s_w, s_b=0.0, rho="", n_draws=draws)
            rec("cex3_3", "P_all_inactive", all_off, **kw)
            rec("cex3_3", "P_all_inactive_theory_2^-q", 2.0 ** (-q), **kw)

            # --- arm A2: identical features WITH bias (still centered Gaussian)
            bq = torch.randn(draws, q, generator=g) * s_b
            all_off_b = ((Wq @ hd + bq) <= 0).all(dim=1).to(torch.float64).mean()
            rec("cex3_3", "P_all_inactive", all_off_b,
                arm="identical_h_with_bias", M=M, q=q, N=N, s_w=s_w, s_b=s_b,
                n_draws=draws)

            # --- arm B: generic (rho-mixture) features, zero bias - contrast
            h = make_features(M, N, rho, seed=base_seed + 7919 * M)
            pre_g = torch.einsum("dqm,nm->dqn", Wq, h)        # (draws, q, N)
            all_off_g = (pre_g <= 0).all(dim=2).all(dim=1).to(torch.float64).mean()
            rec("cex3_3", "P_all_inactive", all_off_g,
                arm="generic_h_zero_bias", M=M, q=q, N=N, s_w=s_w, s_b=0.0,
                rho=rho, n_draws=draws)
            print(f"  q={q} M={M:5d} | identical-h P(all off)={float(all_off):.4f} "
                  f"(theory {2.0**-q:.4f})  with-bias={float(all_off_b):.4f}  "
                  f"generic-h={float(all_off_g):.5f}")

    # on the all-inactive event the whole W block must vanish exactly
    M, q, C = 256, 1, 4
    g = torch.Generator().manual_seed(base_seed + 5)
    hd = torch.randn(M, generator=g)
    hd = hd / hd.norm() * math.sqrt(M)
    h = hd[None, :].repeat(N, 1)
    found, checked, worst = 0, 0, 0.0
    for d in range(400):
        W, b, V = draw_params(M, q, C, s_w, 0.0, 1.0, seed=base_seed + 77 * d)
        if (h @ W.T + b > 0).any():
            continue
        found += 1
        if found > 5:
            break
        J = jac_W(W, b, V, h)
        worst = max(worst, float(J.abs().max()))
        checked += 1
    if checked:
        rec("cex3_3", "max_abs_JW_on_all_inactive_event", worst,
            loss="ce", arm="identical_h_zero_bias", M=M, q=q, C=C, N=N,
            s_w=s_w, s_b=0.0, n_draws=checked)
        print(f"  all-inactive draws checked={checked}: max |dyhat/dW| = {worst:.3e}")


# --------------------------------------------------------------------------
# Part 3 - Counterexample 3.4, train-mode BatchNorm erases the witness
# --------------------------------------------------------------------------

def part3_bn(M_list, q, C, N, s_w, s_b, s_v, base_seed, draws=20, verbose=True):
    print("\n" + "=" * 78)
    print("PART 3  Counterexample 3.4 - train-mode BN erases the W block")
    print("=" * 78)
    for M in M_list:
        g = torch.Generator().manual_seed(base_seed + 13 * M)
        hd = torch.randn(M, generator=g)
        hd = hd / hd.norm() * math.sqrt(M)                    # ||h||^2 = M
        h = hd[None, :].repeat(N, 1)                          # identical sites
        # S_h = (1/N) sum_n h h^T = h h^T, so lambda_max(S_h) = ||h||^2 = M
        lam_Sh, _ = top_eig_Sh(h)

        # frozen BN parameters, deliberately NOT zero so that the train-mode
        # output beta is a genuine nonzero activation pattern: the vanishing of
        # G_WW must be attributable to BN, not to a dead ReLU at exactly 0.
        gamma = torch.ones(q)
        beta = torch.linspace(-0.5, 1.0, q)
        run_mean = torch.randn(q, generator=g) * 0.3
        run_var = 0.5 + torch.rand(q, generator=g)
        eps = 1e-5

        def fwd(W, b, V, mode):
            a = h @ W.T + b                                   # (N, q)
            if mode == "no_bn":
                z = a
            elif mode == "bn_train":                          # batch stats over sites
                mu = a.mean(dim=0, keepdim=True)
                var = a.var(dim=0, unbiased=False, keepdim=True)
                z = (a - mu) / torch.sqrt(var + eps) * gamma + beta
            elif mode == "bn_eval":                           # frozen running stats
                z = (a - run_mean) / torch.sqrt(run_var + eps) * gamma + beta
            else:
                raise ValueError(mode)
            return torch.relu(z) @ V.T

        # cross-check the hand-written train-mode BN against torch's module
        Wc, bc, Vc = draw_params(M, q, C, s_w, s_b, s_v, seed=base_seed)
        bn = torch.nn.BatchNorm1d(q, eps=eps, dtype=torch.float64)
        with torch.no_grad():
            bn.weight.copy_(gamma)
            bn.bias.copy_(beta)
        bn.train()
        with torch.no_grad():
            ref = torch.relu(bn(h @ Wc.T + bc)) @ Vc.T
        rec("cex3_4", "torch_bn_module_max_abs_diff",
            float((ref - fwd(Wc, bc, Vc, "bn_train")).abs().max()),
            loss="ce", arm="bn_train", M=M, q=q, C=C, N=N)

        per_mode = {}
        for mode in ("bn_train", "bn_eval", "no_bn"):
            lams, jmax, pmins, jV = [], 0.0, [], []
            for d in range(draws):
                W, b, V = draw_params(M, q, C, s_w, s_b, s_v,
                                      seed=base_seed + 1000003 * d + M)
                yhat = fwd(W, b, V, mode)
                H, p = ce_hessian_blocks(yhat)
                Hh = torch.block_diag(*blocks_sqrt(H))
                J = torch.autograd.functional.jacobian(
                    lambda Wv: fwd(Wv, b, V, mode).reshape(-1), W, vectorize=True)
                Jz = J.reshape(N * C, q * M) / math.sqrt(N)
                lams.append(float(top_eig_gram(Jz, Hh)))
                jmax = max(jmax, float(J.abs().max()))
                pmins.append(float(p.min()))
                JV = torch.autograd.functional.jacobian(
                    lambda Vv: fwd(W, b, Vv, mode).reshape(-1), V, vectorize=True)
                jV.append(float(JV.norm()))
            lams = np.asarray(lams)
            kw = dict(loss="ce", arm=mode, M=M, q=q, C=C, N=N, s_w=s_w,
                      s_b=s_b, s_v=s_v, n_draws=draws)
            rec("cex3_4", "mean_lam_max_GWW", lams.mean(), **kw)
            rec("cex3_4", "max_lam_max_GWW", lams.max(), **kw)
            rec("cex3_4", "mean_lam_over_M", lams.mean() / M, **kw)
            rec("cex3_4", "max_abs_JW", jmax, **kw)
            rec("cex3_4", "lam_max_Sh", lam_Sh, **kw)
            rec("cex3_4", "lam_max_Sh_over_M", lam_Sh / M, **kw)
            rec("cex3_4", "mean_p_min", float(np.mean(pmins)), **kw)
            rec("cex3_4", "lam_max_Sh_p", float(np.mean(pmins)) * lam_Sh, **kw)
            rec("cex3_4", "mean_frob_JV", float(np.mean(jV)), **kw)
            rec("cex3_4", "frac_draws_lam_exactly_zero",
                float((lams == 0.0).mean()), **kw)
            per_mode[mode] = (lams.mean(), jmax)
            if verbose:
                print(f"  M={M:5d} {mode:9s} | E[lam_max(G_WW)]={lams.mean():.6e} "
                      f"max|J_W|={jmax:.3e} | lam_Sh={lam_Sh:.1f} "
                      f"p_min={np.mean(pmins):.4f} ||J_V||={np.mean(jV):.3f}")

        # relative suppression: how close to zero is "zero to numerical tolerance"
        for ref in ("bn_eval", "no_bn"):
            kwr = dict(loss="ce", arm=f"bn_train_over_{ref}", M=M, q=q, C=C, N=N,
                       n_draws=draws)
            rec("cex3_4", "rel_mean_lam_GWW", per_mode["bn_train"][0]
                / max(per_mode[ref][0], 1e-300), **kwr)
            rec("cex3_4", "rel_max_abs_JW", per_mode["bn_train"][1]
                / max(per_mode[ref][1], 1e-300), **kwr)
        print(f"  M={M:5d} suppression: E[lam]_bn_train / E[lam]_no_bn = "
              f"{per_mode['bn_train'][0]/max(per_mode['no_bn'][0],1e-300):.3e}, "
              f"max|J_W| ratio = "
              f"{per_mode['bn_train'][1]/max(per_mode['no_bn'][1],1e-300):.3e}")

    for mode in ("bn_eval", "no_bn"):
        s = loglog_slope("cex3_4", "mean_lam_max_GWW", q, arm=mode)
        if s is not None:
            rec("cex3_4", "loglog_slope_mean_lam_vs_M", s,
                loss="ce", arm=mode, q=q, C=C, N=N, n_draws=draws)
            print(f"  [{mode}] log-log slope of E lambda_max(G_WW) vs M = {s:.4f}")
    s = loglog_slope("cex3_4", "lam_max_Sh", q, arm="bn_train")
    if s is not None:
        rec("cex3_4", "loglog_slope_lam_Sh_vs_M", s,
            loss="ce", arm="bn_train", q=q, C=C, N=N)
        print(f"  [bn_train] log-log slope of lambda_max(S_h) vs M = {s:.4f} "
              f"(S_h grows linearly while G_WW is identically zero)")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--bn-draws", type=int, default=20)
    ap.add_argument("--mc-draws", type=int, default=20000,
                    help="draws for the cheap Counterexample-3.3 probability")
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--C", type=int, default=4)
    ap.add_argument("--M", type=int, nargs="+", default=[16, 64, 256, 1024])
    ap.add_argument("--q", type=int, nargs="+", default=[4, 16])
    ap.add_argument("--s-w", type=float, default=1.0)
    ap.add_argument("--s-b", type=float, default=0.5)
    ap.add_argument("--s-v", type=float, default=1.0)
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--out", type=str, default=DEFAULT_CSV)
    ap.add_argument("--skip", type=str, nargs="*", default=[],
                    help="any of: p1 p2 p2b p3")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    t0 = time.time()
    print(f"toy_interior_witness | dtype={torch.get_default_dtype()} device=cpu")
    print(f"  N={args.N} C={args.C} M={args.M} q={args.q} draws={args.draws}")
    print(f"  s_w={args.s_w} s_b={args.s_b} s_v={args.s_v} rho={args.rho} "
          f"seed={args.seed}")

    if "p1" not in args.skip:
        part1_squared(args.M, args.q, args.N, args.s_w, args.s_b, args.s_v,
                      args.rho, args.draws, args.seed, mc_draws=args.mc_draws)
    if "p2" not in args.skip:
        part2_ce(args.M, args.q, args.N, args.C, args.s_w, args.s_b, args.s_v,
                 args.rho, args.draws, args.seed)
    if "p2b" not in args.skip:
        part2b_counterexample_3_3(args.M, args.N, args.s_w, args.s_b, args.rho,
                                  args.seed, draws=args.mc_draws)
    if "p3" not in args.skip:
        part3_bn(args.M, args.q[0], args.C, args.N, args.s_w, args.s_b,
                 args.s_v, args.seed, draws=args.bn_draws)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in ROWS:
            w.writerow(r)
    print(f"\nwrote {len(ROWS)} rows -> {args.out}   ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
