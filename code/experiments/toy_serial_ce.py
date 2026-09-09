#!/usr/bin/env python3
"""
Numerical verification of Theorem 5.1 (P5, "an exactly solvable serial cross-entropy
model") from review_packet/astra/math_01.md.

Model (Theorem 5.1)
-------------------
    Y ~ Uniform{-1,+1};  two sites  x1 = (S, 0),  x2 = (0, C),  with S = C = Y.
    Shared trainable linear encoder W = (a, v):
        z1 = a*x1[0] + v*x1[1] = a*S
        z2 = a*x2[0] + v*x2[1] = v*C
    Downstream: spectral skip + M identical contextual channels with readout beta in R^M
        F = z1 + sum_j beta_j * z2 = a*S + b*v*C,     b := sum_j beta_j
    Averaged logistic loss with margin q = Y*F = a + b*v (since S = C = Y, Y^2 = 1):
        L = E[log(1 + exp(-Y F))] = log(1 + exp(-q))
    Unit-rate Euclidean gradient flow from a(0)=0, v(0)=1, beta(0)=0.

What this script checks
-----------------------
(1) Full (M+2)-parameter gradient flow integrated with scipy.solve_ivp (DOP853,
    rtol=1e-10) for M in {1,16,64,256,1024}; records a,v,b,q,r; checks the closed-form
    invariants (5.3) v = cosh(sqrt(M) a), b = sqrt(M) sinh(sqrt(M) a) and (5.4)
    q = Q_M(a); locates T_m for m = log((1-delta)/delta) = log 19 (delta = 0.05);
    verifies (5.5), (5.6), (5.7), (5.8), (5.10) and (5.11) vs the frozen comparator.
(2) Plain gradient descent (Euler) with decreasing step sizes on the full (M+2)
    parameter vector, as an independent, autograd-validated check that the discrete
    iteration converges (first order in h) to the ODE / closed-form values.
(3) Two-block GGN at initialization from explicit Jacobians and the logistic curvature
    r(1-r): lambda_theta(0) = 1/4, lambda_phi(0) = M/4, D_curv(0) = M.  Cross-checked
    against torch autograd block Hessians.
(4) The 1/sqrt(M) readout control F = z1 + M^{-1/2} sum_j gamma_j z2: its reduced
    trajectory must equal the M = 1 case for every M.
(5) The bound table (5.12): exact spectral displacement ||W(T_m) - W(0)|| vs
    (4 B_m/(M+1)) log(1/(sqrt(2) delta)) with B_m = sqrt((1+m^2)/2); ratios compared
    against the note's table (6.258, 5.045, 4.612, 4.489).

Everything is persisted to results/toy_serial_ce.csv (long format: one row per
(M, quantity) with formula / solver / GD values and relative discrepancies).

CPU only.  Usage:  python code/experiments/toy_serial_ce.py
"""

from __future__ import annotations

import csv
import math
import os
import time

import numpy as np
import torch
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq
from scipy.special import expit

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

DELTA = 0.05
M_LIST = [1, 16, 64, 256, 1024]
M_TABLE = [16, 64, 256, 1024]  # the M values appearing in the note's (5.12) table

# Note's table (5.12): M -> (exact displacement, bound, bound/displacement)
NOTE_TABLE = {
    16: (0.219021, 1.370601, 6.258),
    64: (0.071050, 0.358465, 5.045),
    256: (0.019658, 0.090662, 4.612),
    1024: (0.005064, 0.022732, 4.489),
}

RTOL = 1e-10
ATOL = 1e-13
GD_STEPS = [1_000, 10_000, 100_000, 1_000_000]  # decreasing step size h = T_m / N
GD_TORCH_STEPS = [1_000, 10_000]  # autograd cross-check of the numpy GD
DISCREPANCY_TOL = 1e-6  # relative; anything above this is reported

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_CSV = os.path.join(REPO, "results", "toy_serial_ce.csv")

M_MARGIN = math.log((1.0 - DELTA) / DELTA)  # m = log 19

torch.set_default_dtype(torch.float64)

# --------------------------------------------------------------------------------------
# Closed-form ("formula") quantities
# --------------------------------------------------------------------------------------


def psi(q):
    """Psi(q) = q + e^q - 1 (expm1 for accuracy near 0)."""
    return q + np.expm1(q)


def psi_inv(y):
    """Inverse of Psi on [0, inf).  0 <= q <= log(1+y) is a valid bracket."""
    y = float(y)
    if y <= 0.0:
        return 0.0
    hi = math.log1p(y)
    return brentq(lambda q: psi(q) - y, 0.0, hi, xtol=1e-16, rtol=8.9e-16, maxiter=300)


def Q_M(a, M):
    """(5.4)  Q_M(a) = a + (sqrt(M)/2) sinh(2 sqrt(M) a)."""
    s = math.sqrt(M)
    return a + 0.5 * s * np.sinh(2.0 * s * a)


def a_of_m(M, m):
    """Solve Q_M(a) = m for a > 0 (Q_M is strictly increasing, Q_M(0) = 0)."""
    return brentq(lambda a: Q_M(a, M) - m, 0.0, m, xtol=1e-17, rtol=8.9e-16, maxiter=300)


def v_of_a(a, M):
    """(5.3)  v = cosh(sqrt(M) a)."""
    return np.cosh(math.sqrt(M) * a)


def b_of_a(a, M):
    """(5.3)  b = sqrt(M) sinh(sqrt(M) a)."""
    s = math.sqrt(M)
    return s * np.sinh(s * a)


def T_m_quadrature(M, m):
    """
    Exact fitting time by quadrature in the trajectory coordinate a:
        dt = da / adot = da / r = (1 + e^{Q_M(a)}) da,
    so  T_m = int_0^{a_M} (1 + exp(Q_M(a))) da.
    """
    aM = a_of_m(M, m)
    val, err = quad(
        lambda a: 1.0 + math.exp(Q_M(a, M)), 0.0, aM, epsabs=0.0, epsrel=1e-13, limit=400
    )
    return val, err, aM


# --------------------------------------------------------------------------------------
# Full (M+2)-parameter gradient flow  (state = [a, v, beta_1..beta_M])
# --------------------------------------------------------------------------------------


def rhs_joint(t, y, M):
    a = y[0]
    v = y[1]
    b = y[2:].sum()
    q = a + b * v
    r = expit(-q)  # r = 1/(1+e^q), overflow safe
    dy = np.empty_like(y)
    dy[0] = r  # adot = r
    dy[1] = b * r  # vdot = b r
    dy[2:] = v * r  # betadot_j = v r  ->  bdot = M v r
    return dy


def rhs_frozen(t, y, M):
    """Frozen comparator: W = (0,1) fixed, only beta_F trained.  q_F = b_F."""
    b = y.sum()
    q = 0.0 + b * 1.0
    r = expit(-q)
    return np.full_like(y, 1.0 * r)  # betadot_j = v r with v = 1


def rhs_control(t, y, M):
    """1/sqrt(M) readout control: F = z1 + M^{-1/2} sum gamma_j z2, b = M^{-1/2} sum gamma."""
    s = 1.0 / math.sqrt(M)
    a = y[0]
    v = y[1]
    b = s * y[2:].sum()
    q = a + b * v
    r = expit(-q)
    dy = np.empty_like(y)
    dy[0] = r
    dy[1] = b * r
    dy[2:] = s * v * r  # gammadot_j = (dq/dgamma_j) r = M^{-1/2} v r
    return dy


def integrate_joint(M, m, t_end_factor=1.5, control=False):
    """Integrate to the first time q = m (terminal event).  Returns solver object + info."""
    y0 = np.zeros(M + 2)
    y0[1] = 1.0
    # T_m <= Psi(m)/(M_eff+1) by (5.5); the 1/sqrt(M) control has M_eff = 1.
    M_eff = 1 if control else M
    t_hi = t_end_factor * psi(m) / (M_eff + 1)
    f = rhs_control if control else rhs_joint
    s = 1.0 / math.sqrt(M) if control else 1.0

    def event(t, y, M=M):
        b = s * y[2:].sum()
        return (y[0] + b * y[1]) - m

    event.terminal = True
    event.direction = 1.0
    sol = solve_ivp(
        f, (0.0, t_hi), y0, args=(M,), method="DOP853", rtol=RTOL, atol=ATOL,
        dense_output=True, events=event, max_step=t_hi / 50.0,
    )
    if not sol.t_events[0].size:
        raise RuntimeError(f"margin m never reached for M={M} (control={control})")
    return sol, float(sol.t_events[0][0])


def integrate_frozen(M, t_hi):
    y0 = np.zeros(M)
    sol = solve_ivp(
        rhs_frozen, (0.0, t_hi), y0, args=(M,), method="DOP853", rtol=RTOL, atol=ATOL,
        dense_output=True, max_step=t_hi / 50.0,
    )
    return sol


def unpack(y, M, control=False):
    s = 1.0 / math.sqrt(M) if control else 1.0
    a = y[0]
    v = y[1]
    b = s * y[2:].sum(axis=0)
    q = a + b * v
    r = expit(-q)
    return a, v, b, q, r


# --------------------------------------------------------------------------------------
# Plain gradient descent (explicit Euler on the same flow), decreasing step sizes
# --------------------------------------------------------------------------------------


def grad_analytic(a, v, beta):
    """Gradient of L = log(1+exp(-q)), q = a + (sum beta) v, wrt (a, v, beta)."""
    b = beta.sum()
    q = a + b * v
    r = expit(-q)
    return -r, -b * r, np.full_like(beta, -v * r)  # dL/da, dL/dv, dL/dbeta


def gd_run(M, T, n_steps):
    """
    Plain gradient descent (explicit Euler on the same unit-rate flow), constant step
    h = T/n_steps, on the full (M+2)-dimensional parameter vector.  Decreasing h across
    calls gives the O(h) convergence check.
    """
    h = T / n_steps
    a, v = 0.0, 1.0
    beta = np.zeros(M)
    for _ in range(n_steps):
        b = float(beta.sum())
        q = a + b * v
        r = 1.0 / (1.0 + math.exp(q)) if q < 700.0 else 0.0
        # simultaneous update (plain GD, not Gauss-Seidel)
        a_new = a + h * r
        v_new = v + h * (b * r)
        beta += h * (v * r)
        a, v = a_new, v_new
    b = float(beta.sum())
    q = a + b * v
    return dict(a=a, v=v, b=b, q=q, r=float(expit(-q)), h=h,
                beta_spread=float(np.ptp(beta)))


def richardson(x_coarse, x_fine, ratio=10.0):
    """First-order (Euler) Richardson extrapolation: x* ~ (R x_fine - x_coarse)/(R-1)."""
    return (ratio * x_fine - x_coarse) / (ratio - 1.0)


def torch_loss(a, v, beta):
    """
    Literal implementation of the theorem's data/model in torch, used both for the
    autograd gradient check and for the autograd GD cross-check.

    Balanced 2-example training set Y in {-1,+1}; x1=(S,0), x2=(0,C), S=C=Y.
    Loss is the MEAN over examples of log(1+exp(-Y F)).
    """
    Y = torch.tensor([1.0, -1.0])
    x1 = torch.stack([Y, torch.zeros_like(Y)], dim=1)  # (2,2)
    x2 = torch.stack([torch.zeros_like(Y), Y], dim=1)  # (2,2)
    W = torch.stack([a, v])  # (2,)
    z1 = x1 @ W
    z2 = x2 @ W
    F = z1 + beta.sum() * z2
    return torch.nn.functional.softplus(-Y * F).mean()


def torch_grad_check(M, n_points=8, seed=0):
    """Autograd vs analytic gradients of the literal model at random parameter points."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n_points):
        a0 = float(rng.normal(0, 0.5))
        v0 = float(rng.normal(1.0, 0.5))
        beta0 = rng.normal(0, 1.0 / math.sqrt(M), size=M)
        a = torch.tensor(a0, requires_grad=True)
        v = torch.tensor(v0, requires_grad=True)
        beta = torch.tensor(beta0, requires_grad=True)
        loss = torch_loss(a, v, beta)
        loss.backward()
        ga, gv, gb = grad_analytic(a0, v0, beta0)
        ref = np.concatenate([[ga, gv], gb])
        got = np.concatenate([[a.grad.item(), v.grad.item()], beta.grad.numpy()])
        den = np.maximum(np.abs(ref), 1e-300)
        worst = max(worst, float(np.max(np.abs(got - ref) / den)))
    return worst


def gd_run_torch(M, T, n_steps):
    """Same GD but every gradient comes from torch autograd on the literal model."""
    h = T / n_steps
    a = torch.tensor(0.0, requires_grad=True)
    v = torch.tensor(1.0, requires_grad=True)
    beta = torch.zeros(M, requires_grad=True)
    for _ in range(n_steps):
        loss = torch_loss(a, v, beta)
        ga, gv, gb = torch.autograd.grad(loss, [a, v, beta])
        with torch.no_grad():
            a_new = (a - h * ga).requires_grad_(True)
            v_new = (v - h * gv).requires_grad_(True)
            beta = (beta - h * gb).requires_grad_(True)
        a, v = a_new, v_new
    b = float(beta.detach().sum())
    av, vv = float(a.detach()), float(v.detach())
    return dict(a=av, v=vv, b=b, q=av + b * vv)


# --------------------------------------------------------------------------------------
# Two-block GGN at initialization (explicit Jacobians + logistic curvature)
# --------------------------------------------------------------------------------------


def ggn_blocks_explicit(a, v, beta):
    """
    Averaged-CE GGN with the model's Euclidean parameterization, theta = (a,v), phi = beta.

    For a scalar-output model F with per-example loss l(F) = log(1+exp(-Y F)):
        l''(F) = r(1-r),  r = sigma(-Y F) = sigma(-q)
        dF/dtheta = Y (1, b),      dF/dphi = Y v 1_M
    Averaging over the balanced label set (Y^2 = 1) gives the rank-one blocks
        G_theta = r(1-r) [1,b]^T [1,b],   G_phi = r(1-r) v^2 1 1^T
    with top eigenvalues r(1-r)(1+b^2) and M r(1-r) v^2.
    """
    M = beta.shape[0]
    b = float(beta.sum())
    q = a + b * v
    r = float(expit(-q))
    curv = r * (1.0 - r)  # logistic curvature
    J_theta = np.array([1.0, b])  # dq/dtheta (Y factored out; Y^2 = 1)
    J_phi = np.full(M, v)  # dq/dphi
    G_theta = curv * np.outer(J_theta, J_theta)
    G_phi = curv * np.outer(J_phi, J_phi)
    lam_theta = float(np.linalg.eigvalsh(G_theta)[-1])  # eigensolver on the explicit block
    lam_phi = float(np.linalg.eigvalsh(G_phi)[-1])
    lam_theta_cf = curv * (1.0 + b * b)  # closed form from (c)
    lam_phi_cf = M * curv * v * v
    return dict(G_theta=G_theta, G_phi=G_phi, lam_theta=lam_theta, lam_phi=lam_phi,
                lam_theta_cf=lam_theta_cf, lam_phi_cf=lam_phi_cf, r=r, curv=curv)


def torch_block_hessians(M):
    """
    Block Hessians of the literal averaged-CE loss at init, via torch autograd.

    Because F is multilinear (no beta-beta and no theta-theta second-order terms:
    d^2F/da^2 = d^2F/da dv = d^2F/dv^2 = 0 and d^2F/dbeta_i dbeta_j = 0), the diagonal
    blocks of the Hessian coincide exactly with the diagonal GGN blocks here.  The
    cross block does not, and is not used.
    """
    def f(p):
        return torch_loss(p[0], p[1], p[2:])

    p0 = torch.zeros(M + 2)
    p0[1] = 1.0
    H = torch.autograd.functional.hessian(f, p0, vectorize=True)
    H = H.detach().numpy()
    H_theta = H[:2, :2]
    H_phi = H[2:, 2:]
    return (
        float(np.linalg.eigvalsh(H_theta)[-1]),
        float(np.linalg.eigvalsh(H_phi)[-1]),
    )


# --------------------------------------------------------------------------------------
# Result table plumbing
# --------------------------------------------------------------------------------------

ROWS = []
DISCREPANCIES = []


def rel(x, ref):
    if x is None or ref is None:
        return None
    den = abs(ref)
    if den < 1e-12:
        return abs(x - ref)  # absolute fallback for ~zero references
    return abs(x - ref) / den


def add(M, quantity, formula=None, solver=None, gd=None, bound_lo=None, bound_hi=None,
        note="", flag_solver=True, flag_gd=True):
    rs = rel(solver, formula)
    rg = rel(gd, formula)
    within = ""
    if bound_lo is not None or bound_hi is not None:
        val = formula if formula is not None else solver
        slack_ok = True
        if bound_lo is not None and val < bound_lo - 1e-11 * max(1.0, abs(bound_lo)):
            slack_ok = False
        if bound_hi is not None and val > bound_hi + 1e-11 * max(1.0, abs(bound_hi)):
            slack_ok = False
        within = "OK" if slack_ok else "VIOLATED"
        if not slack_ok:
            DISCREPANCIES.append(
                f"M={M} {quantity}: BOUND VIOLATED value={val!r} "
                f"lo={bound_lo!r} hi={bound_hi!r}"
            )
    ROWS.append(dict(
        M=M, quantity=quantity,
        formula_value=formula, solver_value=solver, gd_value=gd,
        rel_disc_solver=rs, rel_disc_gd=rg,
        bound_lower=bound_lo, bound_upper=bound_hi, within_bounds=within, note=note,
    ))
    if flag_solver and rs is not None and rs > DISCREPANCY_TOL:
        DISCREPANCIES.append(f"M={M} {quantity}: solver vs formula rel={rs:.3e} "
                             f"(formula={formula!r}, solver={solver!r})")
    if flag_gd and rg is not None and rg > DISCREPANCY_TOL:
        DISCREPANCIES.append(f"M={M} {quantity}: GD vs formula rel={rg:.3e} "
                             f"(formula={formula!r}, gd={gd!r})")


# --------------------------------------------------------------------------------------
# Main verification
# --------------------------------------------------------------------------------------


def main():
    t_start = time.time()
    m = M_MARGIN
    print("=" * 92)
    print("Theorem 5.1 numerical verification  (serial cross-entropy toy model)")
    print(f"delta = {DELTA}, m = log((1-delta)/delta) = {m!r}  (log 19 = {math.log(19)!r})")
    print(f"Psi(m) = m + e^m - 1 = {psi(m)!r}")
    print("=" * 92)

    # ---------------- global (M-independent) sanity ----------------
    add(0, "m = log((1-delta)/delta)", formula=m, solver=math.log(19.0),
        note="delta=0.05 => m = log 19")
    add(0, "r(T_m) = 1/(1+e^m) = delta", formula=DELTA, solver=float(expit(-m)),
        note="margin target expressed as residual")

    grad_err = torch_grad_check(64)
    add(0, "torch autograd vs analytic gradient (max rel, M=64, 8 pts)",
        formula=0.0, solver=grad_err,
        note="validates (5.2): adot=r, vdot=br, betadot_j=vr")
    print(f"\n[grad check] torch autograd vs analytic gradients: max rel err = {grad_err:.3e}")

    # ---------------- per-M verification ----------------
    a_M_by_M = {}
    disp_by_M = {}
    bound_by_M = {}

    B_m = math.sqrt((1.0 + m * m) / 2.0)
    log_fac = math.log(1.0 / (math.sqrt(2.0) * DELTA))

    for M in M_LIST:
        print("\n" + "-" * 92)
        print(f"M = {M}")
        print("-" * 92)

        # ---- closed-form ----
        T_form, T_err, aM = T_m_quadrature(M, m)
        vM = float(v_of_a(aM, M))
        bM = float(b_of_a(aM, M))
        qM = aM + bM * vM
        a_M_by_M[M] = aM
        disp = math.hypot(aM, vM - 1.0)  # ||W(T_m) - W(0)||, W(0) = (0,1)
        disp_by_M[M] = disp

        # ---- ODE solver ----
        t0 = time.time()
        sol, T_solver = integrate_joint(M, m)
        y_end = sol.sol(T_solver)
        a_s, v_s, b_s, q_s, r_s = unpack(y_end, M)
        beta_spread = float(np.ptp(y_end[2:]))
        print(f"  solve_ivp: {sol.nfev} rhs evals, {len(sol.t)} steps, "
              f"{time.time()-t0:.2f}s, status={sol.status}")

        # invariants along the whole trajectory (5.3), (5.4)
        ts = np.linspace(0.0, T_solver, 4001)
        Y = sol.sol(ts)
        a_t, v_t, b_t, q_t, r_t = unpack(Y, M)
        v_ref = v_of_a(a_t, M)
        b_ref = b_of_a(a_t, M)
        q_ref = Q_M(a_t, M)
        inv_v = float(np.max(np.abs(v_t - v_ref) / np.maximum(np.abs(v_ref), 1e-300)))
        inv_b = float(np.max(np.abs(b_t[1:] - b_ref[1:]) / np.maximum(np.abs(b_ref[1:]), 1e-300)))
        inv_q = float(np.max(np.abs(q_t[1:] - q_ref[1:]) / np.maximum(np.abs(q_ref[1:]), 1e-300)))

        # ---- GD (decreasing steps) ----
        gd_results = {}
        for n in GD_STEPS:
            gd_results[n] = gd_run(M, T_form, n)
        gd_fine = gd_results[GD_STEPS[-1]]
        gd_coarse = gd_results[GD_STEPS[-2]]
        gd_rich = {k: richardson(gd_coarse[k], gd_fine[k]) for k in ("a", "v", "b", "q")}
        # first-order convergence diagnostic on q
        errs = [abs(gd_results[n]["q"] - m) for n in GD_STEPS]
        ratios = [errs[i] / max(errs[i + 1], 1e-300) for i in range(len(errs) - 1)]
        print("  GD |q(T_m) - m| by steps " + str(GD_STEPS) + ": "
              + ", ".join(f"{e:.3e}" for e in errs))
        print("  GD error ratios (expect ~10 for O(h)): "
              + ", ".join(f"{x:.2f}" for x in ratios))
        for i, n in enumerate(GD_STEPS):
            add(M, f"GD raw |q(T_m) - m| at N={n} (h={gd_results[n]['h']:.3e})",
                formula=0.0, solver=errs[i],
                note="explicit-Euler discretization error; must fall ~10x per row",
                flag_solver=False)
        add(M, "GD observed convergence order (finest pair)",
            formula=1.0, solver=math.log10(max(ratios[-1], 1e-300)),
            note="log10 of the error ratio for a 10x step reduction; 1.0 = first order",
            flag_solver=False)

        gd_t = {n: gd_run_torch(M, T_form, n) for n in GD_TORCH_STEPS}
        n_x = GD_TORCH_STEPS[-1]
        tvn = max(rel(gd_t[n_x][k], gd_results[n_x][k]) for k in ("a", "v", "b", "q"))
        print(f"  torch-autograd GD vs numpy GD at N={n_x}: max rel = {tvn:.3e}")
        add(M, "GD: torch-autograd vs numpy analytic-grad (max rel over a,v,b,q)",
            formula=0.0, solver=tvn,
            note=f"N={n_x}; independent gradient path through the literal 2-site model")

        # ---- (a) trajectory + finite fitting time ----
        add(M, "a(T_m)", formula=aM, solver=float(a_s), gd=gd_fine["a"],
            bound_hi=m / (M + 1.0), note="(5.6) 0 < a_M <= m/(M+1)")
        add(M, "v(T_m)", formula=vM, solver=float(v_s), gd=gd_fine["v"],
            note="(5.3) v = cosh(sqrt(M) a)")
        add(M, "b(T_m)", formula=bM, solver=float(b_s), gd=gd_fine["b"],
            note="(5.3) b = sqrt(M) sinh(sqrt(M) a)")
        add(M, "q(T_m)", formula=m, solver=float(q_s), gd=gd_fine["q"],
            note="(5.4) q = Q_M(a); target margin m")
        add(M, "r(T_m)", formula=DELTA, solver=float(r_s), gd=gd_fine["r"],
            note="r = 1/(1+e^q) = delta at q = m")
        add(M, "T_m", formula=T_form, solver=T_solver,
            bound_lo=psi(m) / (M + 1.0 + 2.0 * m * m), bound_hi=psi(m) / (M + 1.0),
            note=f"(5.5); formula = quadrature int_0^a_M (1+e^Q_M) da, quad err={T_err:.2e}")
        add(M, "(5.5) lower bound Psi(m)/(M+1+2m^2)",
            formula=psi(m) / (M + 1.0 + 2.0 * m * m), solver=None, note="(5.5)")
        add(M, "(5.5) upper bound Psi(m)/(M+1)", formula=psi(m) / (M + 1.0),
            solver=None, note="(5.5)")
        for k, ref in (("a", aM), ("v", vM), ("b", bM), ("q", m)):
            add(M, f"GD Richardson-extrapolated {k}(T_m)", formula=ref,
                solver=None, gd=gd_rich[k],
                note="first-order Richardson from the two finest step sizes; removes the "
                     "leading O(h) Euler error and should match the closed form")
        add(M, "invariant (5.3) v = cosh(sqrt(M) a): max rel err", formula=0.0,
            solver=inv_v, bound_hi=1e-8, note="checked at 4001 points on [0,T_m]")
        add(M, "invariant (5.3) b = sqrt(M) sinh(sqrt(M) a): max rel err", formula=0.0,
            solver=inv_b, bound_hi=1e-8, note="checked at 4001 points on [0,T_m]")
        add(M, "invariant (5.4) q = Q_M(a): max rel err", formula=0.0, solver=inv_q,
            bound_hi=1e-8, note="checked at 4001 points on [0,T_m]")
        add(M, "max spread of beta_j at T_m (should be 0)", formula=0.0,
            solver=beta_spread, gd=gd_fine["beta_spread"],
            note="all M contextual readouts stay identical")

        # exact q-dot identity (5.4) at T_m
        qdot_state = float(rhs_joint(0.0, y_end, M)[0] * (1.0 + M * v_s**2 + b_s**2))
        qdot_form = (M + 1.0 + 2.0 * b_s**2) * float(r_s)
        add(M, "qdot(T_m) = (1+b^2+M v^2) r  vs  (M+1+2b^2) r",
            formula=qdot_form, solver=qdot_state, note="(5.4) using b^2 = M(v^2-1)")

        # ---- (b) suppression at matched fit ----
        bv = bM * vM
        add(M, "b(T_m) v(T_m)", formula=bv, solver=float(b_s * v_s), gd=gd_fine["b"] * gd_fine["v"],
            bound_lo=M * m / (M + 1.0), bound_hi=m,
            note="(5.6) b v = m - a_M >= M m/(M+1)")
        add(M, "m - a_M  (must equal b v)", formula=m - aM, solver=float(b_s * v_s),
            note="(5.6) identity q = a + bv = m")
        add(M, "v(T_m) - 1", formula=vM - 1.0, solver=float(v_s - 1.0), gd=gd_fine["v"] - 1.0,
            bound_lo=0.0, bound_hi=m * m / (2.0 * M), note="(5.7) 0 <= v-1 <= m^2/(2M)")
        disp_bound_57 = math.sqrt(m * m / (M + 1.0) ** 2 + m**4 / (4.0 * M * M))
        disp_s = math.hypot(float(a_s), float(v_s) - 1.0)
        disp_g = math.hypot(gd_fine["a"], gd_fine["v"] - 1.0)
        add(M, "||W(T_m) - W(0)||_2", formula=disp, solver=disp_s, gd=disp_g,
            bound_hi=disp_bound_57, note="(5.7) <= sqrt(m^2/(M+1)^2 + m^4/(4M^2))")
        add(M, "M * a_M  (-> m as M -> inf)", formula=M * aM, solver=float(M * a_s),
            bound_hi=M * m / (M + 1.0), note=f"(5.6) limit claim; m = {m:.6f}")
        add(M, "spectral-only comparator a needed for margin m", formula=m, solver=None,
            note="(5.6) removing the contextual path gives q = a, so a = m exactly")
        add(M, "suppression factor m / a_M", formula=m / aM, solver=float(m / a_s),
            bound_lo=M + 1.0, note="(5.6) at least M+1")

        # ---- (e) contextual reversal ----
        add(M, "shifted-margin Y F = 2 a_M - m", formula=2.0 * aM - m,
            solver=float(2.0 * a_s - m), bound_hi=0.0,
            note="(e) negative for every M >= 1 => error 1 under S=Y, C=-Y")

        # ---- (c) residual envelope + curvature ----
        env = 0.5 / (1.0 + (M + 1.0) * ts / 4.0)
        viol_fit = float(np.max(r_t - env))
        # extended horizon: envelope claim is "for all t >= 0"
        t_long = 20.0 * T_form
        sol_long = solve_ivp(rhs_joint, (0.0, t_long), np.concatenate([[0.0, 1.0], np.zeros(M)]),
                             args=(M,), method="DOP853", rtol=RTOL, atol=ATOL,
                             dense_output=True, max_step=t_long / 200.0)
        ts_l = np.linspace(0.0, t_long, 8001)
        _, _, _, _, r_l = unpack(sol_long.sol(ts_l), M)
        env_l = 0.5 / (1.0 + (M + 1.0) * ts_l / 4.0)
        viol_long = float(np.max(r_l - env_l))
        add(M, "(5.8) max_t [ r(t) - (1/2)/(1+(M+1)t/4) ] on [0,T_m]",
            formula=0.0, solver=viol_fit, bound_hi=1e-11,
            note="<= 0 required; positive value = envelope violation")
        add(M, "(5.8) max_t [ r(t) - envelope ] on [0, 20 T_m]",
            formula=0.0, solver=viol_long, bound_hi=1e-11,
            note="envelope is claimed for all t >= 0")

        beta0 = np.zeros(M)
        g0 = ggn_blocks_explicit(0.0, 1.0, beta0)
        lam_th0, lam_ph0 = g0["lam_theta"], g0["lam_phi"]
        th_h, ph_h = torch_block_hessians(M)
        add(M, "lambda_theta(0)", formula=0.25, solver=lam_th0, gd=th_h,
            note="(5.9) explicit-Jacobian GGN block, top eigenvalue via eigvalsh; "
                 "'gd_value' column = torch autograd Hessian block (equal here because F "
                 "is multilinear, so the diagonal Hessian blocks have no residual term)")
        add(M, "lambda_phi(0)", formula=M / 4.0, solver=lam_ph0, gd=ph_h,
            note="(5.9) explicit-Jacobian GGN block; 'gd_value' = torch autograd Hessian block")
        add(M, "D_curv(0) = lambda_phi(0)/lambda_theta(0)", formula=float(M),
            solver=lam_ph0 / lam_th0, gd=ph_h / th_h, note="(5.9)")
        add(M, "lambda_theta(0) closed form r(1-r)(1+b^2): |lam - 1/4|", formula=0.0,
            solver=abs(g0["lam_theta_cf"] - 0.25), bound_hi=0.0,
            note="EXACT in IEEE double: r(1-r)=1/4, b=0")
        add(M, "lambda_phi(0) closed form M r(1-r) v^2: |lam - M/4|", formula=0.0,
            solver=abs(g0["lam_phi_cf"] - M / 4.0), bound_hi=0.0,
            note="EXACT in IEEE double: r(1-r)=1/4, v=1")
        add(M, "lambda_theta(0) eigensolver deviation |lam - 1/4|", formula=0.0,
            solver=abs(lam_th0 - 0.25), bound_hi=1e-15, note="eigvalsh roundoff only")
        add(M, "lambda_phi(0) eigensolver deviation |lam - M/4|", formula=0.0,
            solver=abs(lam_ph0 - M / 4.0), bound_hi=1e-12 * max(1.0, M),
            note="eigvalsh roundoff only")

        # curvature along the trajectory at T_m
        gT = ggn_blocks_explicit(float(a_s), float(v_s), y_end[2:])
        add(M, "lambda_theta(T_m) = r(1-r)(1+b^2)",
            formula=DELTA * (1 - DELTA) * (1 + bM * bM), solver=gT["lam_theta"], note="(c)")
        add(M, "lambda_phi(T_m) = M r(1-r) v^2",
            formula=M * DELTA * (1 - DELTA) * vM * vM, solver=gT["lam_phi"], note="(c)")
        add(M, "D_curv(T_m) = (M v^2)/(1+b^2)",
            formula=M * vM * vM / (1.0 + bM * bM),
            solver=gT["lam_phi"] / gT["lam_theta"], note="(c) curvature disparity at fit")

        gnorm = float(r_s * math.sqrt(1.0 + b_s**2))
        add(M, "||grad_theta L(T_m)|| = r sqrt(1+b^2)",
            formula=DELTA * math.sqrt(1.0 + bM * bM), solver=gnorm,
            bound_hi=DELTA * math.sqrt(1.0 + m * m),
            note="(5.10) <= r sqrt(1+m^2)")
        gmax = float(np.max(r_t * np.sqrt(1.0 + b_t**2) / np.maximum(r_t * np.sqrt(1 + m * m), 1e-300)))
        add(M, "(5.10) max_t [ r sqrt(1+b^2) / (r sqrt(1+m^2)) ] on [0,T_m]",
            formula=None, solver=gmax, bound_hi=1.0, note="ratio must stay <= 1")
        add(M, "max_t b(t) on [0,T_m]", formula=bM, solver=float(np.max(b_t)),
            bound_hi=m, note="proof step: b <= m since bv = q-a <= m and v >= 1")

        # ---- (d) frozen comparator (5.11) ----
        sol_F = integrate_frozen(M, T_solver)
        qF_solver = sol_F.sol(ts).sum(axis=0)  # q_F = b_F (a=0, v=1)
        qF_form = np.array([psi_inv(M * t) for t in ts])  # Psi(q_F) = M t
        relF = float(np.max(np.abs(qF_solver[1:] - qF_form[1:]) / np.abs(qF_form[1:])))
        add(M, "frozen q_F(T_m)", formula=float(qF_form[-1]), solver=float(qF_solver[-1]),
            note="(d) Psi(q_F(t)) = M t")
        add(M, "frozen q_F(t): max rel err solver vs Psi^{-1}(Mt)", formula=0.0,
            solver=relF, bound_hi=1e-8, note="checked at 4001 points on [0,T_m]")
        gap = q_t - qF_form
        gap_min = float(np.min(gap))
        gap_max = float(np.max(gap))
        ptwise = (1.0 + 2.0 * m * m) * ts / 2.0
        uniform = (1.0 + 2.0 * m * m) * psi(m) / (2.0 * (M + 1.0))
        viol_pt = float(np.max(gap - ptwise))
        add(M, "(5.11) min_t [ q(t) - q_F(t) ]", formula=None, solver=gap_min,
            bound_lo=0.0, note="lower half of (5.11): q >= q_F")
        add(M, "(5.11) max_t [ q(t) - q_F(t) ]", formula=None, solver=gap_max,
            bound_hi=uniform, note="(5.11) uniform bound (1+2m^2)Psi(m)/(2(M+1))")
        add(M, "(5.11) max_t [ (q - q_F) - (1+2m^2)t/2 ]", formula=0.0, solver=viol_pt,
            bound_hi=1e-11, note="pointwise bound; must be <= 0")
        add(M, "(5.11) uniform bound value", formula=uniform, solver=None,
            note="(1+2m^2)Psi(m)/(2(M+1)); RMS logit gap is O(1/M)")
        add(M, "(5.11) gap x (M+1)  [O(1/M) diagnostic]", formula=None,
            solver=gap_max * (M + 1.0), note="should stay bounded in M")

        # ---- (4) 1/sqrt(M) readout control ----
        sol_c, T_c = integrate_joint(M, m, control=True)
        yc = sol_c.sol(T_c)
        a_c, v_c, b_c, q_c, _ = unpack(yc, M, control=True)
        T1, _, a1 = T_m_quadrature(1, m)
        add(M, "control (1/sqrt(M) readout): a(T_m)", formula=a1, solver=float(a_c),
            note="(4) must equal the M=1 value for every M")
        add(M, "control: v(T_m)", formula=float(v_of_a(a1, 1)), solver=float(v_c),
            note="(4) must equal the M=1 value")
        add(M, "control: b(T_m)", formula=float(b_of_a(a1, 1)), solver=float(b_c),
            note="(4) must equal the M=1 value")
        add(M, "control: T_m", formula=T1, solver=float(T_c),
            note="(4) must equal the M=1 fitting time")
        add(M, "control: ||W(T_m)-W(0)||", formula=math.hypot(a1, float(v_of_a(a1, 1)) - 1.0),
            solver=math.hypot(float(a_c), float(v_c) - 1.0),
            note="(4) no suppression: displacement is M-independent")

        # ---- (5) bound table (5.12) ----
        bound = 4.0 * B_m / (M + 1.0) * log_fac
        bound_by_M[M] = bound
        add(M, "B_m = sqrt((1+m^2)/2)", formula=B_m, solver=None,
            note="operator-norm cap on ||J_theta|| with symmetric logits")
        add(M, "max_t ||J_theta||_op = sqrt((1+b^2)/2)", formula=math.sqrt((1 + bM**2) / 2),
            solver=float(np.max(np.sqrt((1 + b_t**2) / 2))), bound_hi=B_m,
            note="normalized (symmetric-logit) Jacobian norm <= B_m")
        add(M, "(5.12) bound  4 B_m/(M+1) log(1/(sqrt2 delta))", formula=bound,
            solver=None, bound_lo=disp, note="must dominate the exact displacement")
        add(M, "(5.12) bound / displacement", formula=bound / disp,
            solver=bound / disp_s, note="see note-table comparison rows")

        print(f"  a_M = {aM!r}")
        print(f"  v(T_m) = {vM!r}   b(T_m) = {bM!r}")
        print(f"  T_m(formula) = {T_form!r}   T_m(solver) = {T_solver!r}  "
              f"rel = {rel(T_solver, T_form):.3e}")
        print(f"  ||W(T_m)-W(0)|| = {disp!r}   bound(5.12) = {bound!r}   "
              f"ratio = {bound/disp:.6f}")
        print(f"  invariants: v {inv_v:.2e}, b {inv_b:.2e}, q {inv_q:.2e}")
        print(f"  GGN(0): lambda_theta = {lam_th0!r}, lambda_phi = {lam_ph0!r}, "
              f"D_curv = {lam_ph0/lam_th0!r}")
        print(f"  frozen gap max = {gap_max:.6e}  (uniform bound {uniform:.6e})")

    # ---------------- monotonicity of a_M in M ----------------
    print("\n" + "-" * 92)
    print("Monotonicity / limit checks")
    print("-" * 92)
    mono = all(a_M_by_M[M_LIST[i]] > a_M_by_M[M_LIST[i + 1]] for i in range(len(M_LIST) - 1))
    add(0, "a_M strictly decreasing in M", formula=1.0, solver=1.0 if mono else 0.0,
        note="(b) claim: a_M decreases strictly with M; " +
             ", ".join(f"M={M}:{a_M_by_M[M]:.6e}" for M in M_LIST))
    print(f"  a_M strictly decreasing in M: {mono}")
    for M in M_LIST:
        print(f"    M={M:5d}  a_M={a_M_by_M[M]:.10e}  M*a_M={M*a_M_by_M[M]:.10f}  "
              f"(m={m:.10f}, m-M a_M={m - M*a_M_by_M[M]:.3e})")

    # ---------------- (5.12) note-table comparison ----------------
    print("\n" + "-" * 92)
    print("Bound table (5.12) vs the note's table")
    print("-" * 92)
    print(f"{'M':>6} {'disp(exact)':>14} {'note disp':>12} {'bound':>14} {'note bound':>12} "
          f"{'ratio':>10} {'note ratio':>10}")
    def rounding_note(computed, printed, ndec):
        ok = abs(round(computed, ndec) - printed) < 0.5 * 10.0 ** (-ndec - 3)
        return (f"'solver_value' column holds the value printed in math_01.md's table; "
                f"printed to {ndec} decimals; round(computed,{ndec}) == printed: {ok}")

    for M in M_TABLE:
        if M not in disp_by_M:
            continue
        d, bnd = disp_by_M[M], bound_by_M[M]
        nd, nb, nr = NOTE_TABLE[M]
        ratio = bnd / d
        print(f"{M:>6} {d:>14.9f} {nd:>12.6f} {bnd:>14.9f} {nb:>12.6f} "
              f"{ratio:>10.6f} {nr:>10.3f}")
        add(M, "note-table: exact displacement", formula=d, solver=nd,
            note=rounding_note(d, nd, 6))
        add(M, "note-table: bound (5.12)", formula=bnd, solver=nb,
            note=rounding_note(bnd, nb, 6))
        add(M, "note-table: bound/displacement", formula=ratio, solver=nr,
            note=rounding_note(ratio, nr, 3))

    # ---------------- normalization statement ----------------
    norm_stmt = (
        "lambda_theta(0)=1/4 and lambda_phi(0)=M/4 hold EXACTLY for the MEAN-over-examples "
        "(averaged, not summed) binary cross-entropy L = E_Y[log(1+exp(-Y F))] with a single "
        "scalar logit F whose per-example margin is q = Y F with unit scaling. Equivalently, "
        "the symmetric two-logit softmax convention (F/2,-F/2): its softmax Hessian "
        "p(1-p)[[1,-1],[-1,1]] contracted with the logit Jacobian direction (1/2,-1/2) gives "
        "the same scalar curvature r(1-r), because only the logit DIFFERENCE (=F) enters. "
        "The two normalizations that break the constants: (i) SUMMING instead of averaging "
        "over the n training examples / broadcast sites multiplies both blocks by n (e.g. "
        "broadcasting to the 2 sites and summing gives 1/2 and M/2); (ii) rescaling the logit "
        "(e.g. F/tau) multiplies both blocks by 1/tau^2. D_curv(0)=lambda_phi/lambda_theta=M "
        "is invariant under both, since they scale the two blocks identically."
    )
    add(0, "loss normalization required for lambda_theta(0)=1/4, lambda_phi(0)=M/4",
        formula=None, solver=None, note=norm_stmt)
    print("\n" + "-" * 92)
    print("Loss normalization")
    print("-" * 92)
    print(norm_stmt)

    # ---------------- persist ----------------
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fields = ["M", "quantity", "formula_value", "solver_value", "gd_value",
              "rel_disc_solver", "rel_disc_gd", "bound_lower", "bound_upper",
              "within_bounds", "note"]
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in ROWS:
            out = dict(row)
            for k in ("formula_value", "solver_value", "gd_value", "rel_disc_solver",
                      "rel_disc_gd", "bound_lower", "bound_upper"):
                out[k] = "" if out[k] is None else repr(float(out[k]))
            w.writerow(out)
    print(f"\nWrote {len(ROWS)} rows to {OUT_CSV}")

    # ---------------- discrepancy report ----------------
    print("\n" + "=" * 92)
    print(f"DISCREPANCIES > {DISCREPANCY_TOL:g} relative (or bound violations)")
    print("=" * 92)
    if not DISCREPANCIES:
        print("  none")
    for d in DISCREPANCIES:
        print("  - " + d)
    print(f"\nTotal wall time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
