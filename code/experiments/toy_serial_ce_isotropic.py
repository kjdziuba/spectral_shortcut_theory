#!/usr/bin/env python3
"""
Numerical verification of review_packet/astra/math_02.md, Section 4
("Q2: isotropic encoder initialization -- exact result and a false proposed limit"),
equations (I1)-(I6).

This is the isotropic-initialization companion to code/experiments/toy_serial_ce.py,
which checks Theorem 5.1 of math_01.md under the special initialization a(0)=0,
v(0)=1, beta(0)=0.  Here BOTH encoder coordinates start at arbitrary values and the
M replicated readouts start at UNEQUAL values with zero sum.

Model (math_02.md Section 4)
----------------------------
    q = a + b v,      b = sum_{j=1..M} beta_j,
    adot = rho,  vdot = b rho,  betadot_j = v rho  (so bdot = M v rho),
    rho = (1 + e^q)^{-1}.

    a(0) = a0 (arbitrary real, a0 < m for the non-trivial case),
    v(0) = v0 != 0 (arbitrary real),
    beta(0) = arbitrary with sum(beta(0)) = 0  ==>  b(0) = 0, q(0) = a0.

Write u = a - a0 and Delta = m - a0 > 0.

What is checked, per case
-------------------------
(I1) closed form  v = v0 cosh(sqrt(M) u),  b = sqrt(M) v0 sinh(sqrt(M) u),
     q = a0 + u + (sqrt(M) v0^2 / 2) sinh(2 sqrt(M) u),
     against the FULL (2+M)-parameter ODE (scipy solve_ivp DOP853, rtol 1e-10,
     atol 1e-12, terminal event q = m).
(I2) 0 < u_M <= Delta/(1 + M v0^2)   and   M u_M -> Delta / v0^2.
(I3) |v(T_m) - v0| <= Delta^2/(2 M |v0|^3)  and
     ||W(T_m) - W0|| <= sqrt(Delta^2/(1+M v0^2)^2 + Delta^4/(4 M^2 |v0|^6)).
(I4) update suppression (a(T_m) - a0)/(m - a0) -> 0.               <-- must vanish
     REPORTED SEPARATELY from a(T_m)/m -> a0/m and a(T_m)/a0 -> 1,  <-- must NOT vanish
     which is Astra's correction of the proposed (false) pair of limits.
(iv) Psi_{a0}(m)/(1 + M v0^2 + 2 Delta^2/v0^2) <= T_m <= Psi_{a0}(m)/(1 + M v0^2),
     with Psi_{a0}(q) = q - a0 + e^q - e^{a0}.
(I5) 0 <= q_J(t) - q_F(t) <= C0 t/(1 + e^{a0} + p0 M v0^2 t) <= C0/(p0 M v0^2)
     on [0, T_m], with p0 = e^{a0}/(1+e^{a0}), C0 = 1 + 2 Delta^2/v0^2, and the
     frozen comparator q_F defined by Psi_{a0}(q_F(t)) = M v0^2 t.
(vi) reversal-test margin 2 a(T_m) - m (limit 2 a0 - m) and the reversal error
     indicator Err = 1{2 a(T_m) < m}; also the sufficient finite-width condition
     M v0^2 > m/(m - 2 a0) for a0 < m/2.
(ctrl) 1/sqrt(M) readout normalization F = z1 + M^{-1/2} sum_j gamma_j z2 gives
     bdot = v rho, i.e. the M = 1 reduced dynamics: the suppression disappears and
     u is M-independent.

Immediate-stop convention: for a0 >= m the matched-threshold rule stops at T_m = 0,
so a(T_m) = a0, the displacement is 0 and the reversal margin is 2 a0 - m > 0.

Gaussian average (I6)
---------------------
(a0, v0) ~ N(0, sigma^2 I_2), n = 4000 paired draws, sigma in {0.5, 1.0},
m in {2.0, 4.0}, M in {64, 1024, 16384}; E[Err_reversal] is compared with
Phi(m/(2 sigma)).  The spectral-only comparator (no contextual path, so a must
reach m itself) has reversal margin m > 0, hence reversal error 0.
The closed form used for the sweep is verified against the full ODE on a subset.

Outputs
-------
    results/toy_serial_ce_isotropic.csv           (one row per case)
    results/toy_serial_ce_isotropic_gaussian.csv  (one row per (sigma, m, M))
    results/toy_serial_ce_isotropic_REPORT.md

CPU only.  Usage:
    python code/experiments/toy_serial_ce_isotropic.py
"""

from __future__ import annotations

import csv
import math
import os
import time

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq
from scipy.special import expit, ndtr

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

M_LIST = [1, 4, 16, 64, 256, 1024, 4096]
A0_LIST = [-1.5, -0.3, 0.0, 0.4, 1.7]
V0_LIST = [-1.3, -0.2, 0.05, 0.2, 1.0]
M_MARGIN_LIST = [2.0, 2.9, 4.0]

# Immediate-stop convention (a0 >= m): a smaller cross-product, since T_m = 0 exactly.
STOP_A0_M = [(2.0, 2.0), (2.5, 2.0), (2.9, 2.9), (3.5, 2.9), (4.0, 4.0), (6.0, 4.0)]
STOP_V0 = [-1.3, 0.05, 1.0]
STOP_M = [1, 64, 4096]

RTOL = 1e-10
ATOL = 1e-12

BETA_SEED = 20260910          # seed for the unequal zero-sum readout initialization
BETA_SCALE = 1.0              # beta_j(0) ~ N(0, (BETA_SCALE/sqrt(M))^2), antisymmetrized

N_GRID_LIN = 801              # closed-form gap grid: uniform part
N_GRID_GEO = 120              # closed-form gap grid: geometric refinement near t = 0
GL_ORDER = 24                 # Gauss-Legendre order for the cumulative time integral

FROZEN_ODE_MAX_M = 256        # integrate the full M-dim frozen comparator up to this M

# Gaussian sweep
GAUSS_SIGMAS = [0.5, 1.0]
GAUSS_MARGINS = [2.0, 4.0]
GAUSS_M = [64, 1024, 16384]
GAUSS_N = 4000
GAUSS_SEED = 424242
GAUSS_ODE_CHECK = {64: 40, 1024: 40, 16384: 15}   # ODE-verified draws per (sigma, m, M)

VIOL_RTOL = 1e-9              # relative slack allowed before a bound counts as violated
VIOL_ATOL = 1e-11

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_CSV = os.path.join(REPO, "results", "toy_serial_ce_isotropic.csv")
OUT_CSV_G = os.path.join(REPO, "results", "toy_serial_ce_isotropic_gaussian.csv")
OUT_MD = os.path.join(REPO, "results", "toy_serial_ce_isotropic_REPORT.md")

GL_NODES, GL_WEIGHTS = np.polynomial.legendre.leggauss(GL_ORDER)


# --------------------------------------------------------------------------------------
# Closed form (I1) and its consequences
# --------------------------------------------------------------------------------------


def Q_iso(u, a0, v0, M):
    """q as a function of u = a - a0 along the joint path:  (I1)."""
    s = math.sqrt(M)
    return a0 + u + 0.5 * s * v0 * v0 * np.sinh(2.0 * s * u)


def root_u(M, v0, Delta):
    """
    Unique u_M > 0 with Q_iso(u_M) = m, i.e.  u + (sqrt(M) v0^2/2) sinh(2 sqrt(M) u) = Delta.

    Solved in x = sqrt(M) u.  The upper bracket is the MINIMUM of
      (a) the (I2) bound sqrt(M) Delta/(1 + M v0^2)   -- valid because sinh(x) >= x, and
      (b) 0.5 asinh(Delta / A), A = sqrt(M) v0^2/2    -- where the sinh term alone hits Delta.
    (b) keeps sinh(2x) finite for tiny |v0| (where (a) alone would overflow); (a) keeps the
    bracket tight for large M.  Both are provable upper bounds, so f(x_hi) >= 0.
    """
    A = math.sqrt(M) * v0 * v0 / 2.0
    x_hi_a = math.sqrt(M) * Delta / (1.0 + M * v0 * v0)
    x_hi_b = 0.5 * math.asinh(Delta / A)
    x_hi = min(x_hi_a, x_hi_b)

    def f(x):
        return x / math.sqrt(M) + A * math.sinh(2.0 * x) - Delta

    if f(x_hi) < 0.0:                      # cannot happen mathematically; guard anyway
        raise RuntimeError(f"bad bracket M={M} v0={v0} Delta={Delta}: f(x_hi)={f(x_hi)!r}")
    x = brentq(f, 0.0, x_hi, xtol=1e-300, rtol=8.9e-16, maxiter=500)
    return x / math.sqrt(M), x


def v_of_u(u, v0, M):
    """(I1)  v = v0 cosh(sqrt(M) u)."""
    return v0 * np.cosh(math.sqrt(M) * u)


def b_of_u(u, v0, M):
    """(I1)  b = sqrt(M) v0 sinh(sqrt(M) u)."""
    s = math.sqrt(M)
    return s * v0 * np.sinh(s * u)


def psi_a0(q, a0):
    """Psi_{a0}(q) = (q - a0) + (e^q - e^{a0}) = u + e^{a0} expm1(u), u = q - a0."""
    u = q - a0
    return u + math.exp(a0) * np.expm1(u)


def psi_a0_inv(y, a0):
    """Inverse of Psi_{a0} on [0, inf).  Bracket u in [0, log1p(y e^{-a0})]."""
    y = float(y)
    if y <= 0.0:
        return a0
    hi = math.log1p(y * math.exp(-a0))     # Psi(a0+hi) >= e^{a0} expm1(hi) = y
    e0 = math.exp(a0)
    u = brentq(lambda uu: uu + e0 * math.expm1(uu) - y, 0.0, hi,
               xtol=1e-300, rtol=8.9e-16, maxiter=500)
    return a0 + u


def t_of_u_cumulative(u_grid, a0, v0, M):
    """
    Cumulative fitting time along the joint path:
        dt = da / adot = (1 + e^q) da,  q = Q_iso(u)
    so t(u) = int_0^u (1 + exp(Q_iso(s))) ds, by composite Gauss-Legendre on u_grid.
    """
    lo, hi = u_grid[:-1], u_grid[1:]
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    U = mid[:, None] + half[:, None] * GL_NODES[None, :]
    vals = (1.0 + np.exp(Q_iso(U, a0, v0, M))) @ GL_WEIGHTS * half
    return np.concatenate([[0.0], np.cumsum(vals)])


def T_m_quadrature(a0, v0, M, uM):
    """Independent adaptive-quadrature value of T_m (cross-check of the GL cumulative sum)."""
    val, err = quad(lambda u: 1.0 + math.exp(Q_iso(u, a0, v0, M)), 0.0, uM,
                    epsabs=0.0, epsrel=1e-13, limit=400)
    return val, err


# --------------------------------------------------------------------------------------
# Full (2+M)-parameter ODE
# --------------------------------------------------------------------------------------


def beta0_zero_sum(M, seed=BETA_SEED, scale=BETA_SCALE):
    """
    Unequal readout initialization with EXACTLY zero sum in IEEE double.

    beta[2k] = c_k, beta[2k+1] = -c_k.  numpy's pairwise summation combines the
    partial sums so that every +c_k is cancelled by its exact negation, giving
    sum(beta) == 0.0 bitwise (asserted by the caller).  For M = 1 the only
    zero-sum initialization is beta = 0.
    """
    if M == 1:
        return np.zeros(1)
    rng = np.random.default_rng(seed + M)
    c = rng.normal(scale=scale / math.sqrt(M), size=M // 2)
    beta = np.empty(M)
    beta[0::2] = c
    beta[1::2] = -c
    return beta


def rhs_joint(t, y, M):
    b = y[2:].sum()
    r = expit(-(y[0] + b * y[1]))
    dy = np.empty_like(y)
    dy[0] = r
    dy[1] = b * r
    dy[2:] = y[1] * r
    return dy


def rhs_control(t, y, M):
    """1/sqrt(M) readout: b = M^{-1/2} sum_j gamma_j, so bdot = v rho (M-independent)."""
    s = 1.0 / math.sqrt(M)
    b = s * y[2:].sum()
    r = expit(-(y[0] + b * y[1]))
    dy = np.empty_like(y)
    dy[0] = r
    dy[1] = b * r
    dy[2:] = s * y[1] * r
    return dy


def rhs_frozen(t, y, M, a0, v0):
    """Frozen encoder comparator: W = (a0, v0) fixed, only the readouts train."""
    b = y.sum()
    r = expit(-(a0 + b * v0))
    return np.full_like(y, v0 * r)


def integrate_case(M, a0, v0, m, t_hi, control=False):
    """Integrate to the first time q = m (terminal event).  Returns (sol, T_m)."""
    y0 = np.concatenate([[a0, v0], beta0_zero_sum(M)])
    s = 1.0 / math.sqrt(M) if control else 1.0

    def event(t, y, *_a):   # solve_ivp forwards `args` to the event functions too
        return (y[0] + s * y[2:].sum() * y[1]) - m

    event.terminal = True
    event.direction = 1.0
    sol = solve_ivp(rhs_control if control else rhs_joint, (0.0, t_hi), y0, args=(M,),
                    method="DOP853", rtol=RTOL, atol=ATOL, dense_output=True,
                    events=event, max_step=t_hi / 60.0)
    if not sol.t_events[0].size:
        raise RuntimeError(f"margin never reached: M={M} a0={a0} v0={v0} m={m} "
                           f"control={control} status={sol.status}")
    return sol, float(sol.t_events[0][0])


def unpack(y, M, control=False):
    s = 1.0 / math.sqrt(M) if control else 1.0
    a = y[0]
    v = y[1]
    b = s * y[2:].sum(axis=0)
    return a, v, b, a + b * v


# --------------------------------------------------------------------------------------
# Bookkeeping
# --------------------------------------------------------------------------------------

VIOLATIONS = []


def slack_upper(name, value, bound, tag):
    """value <= bound.  Returns the slack; records a violation if negative beyond tol."""
    sl = bound - value
    tol = VIOL_ATOL + VIOL_RTOL * max(abs(bound), abs(value), 1.0)
    if sl < -tol:
        VIOLATIONS.append(f"{tag}: {name}  value={value!r} > bound={bound!r} "
                          f"(slack={sl:.3e})")
    return sl


def slack_lower(name, value, bound, tag):
    """value >= bound."""
    sl = value - bound
    tol = VIOL_ATOL + VIOL_RTOL * max(abs(bound), abs(value), 1.0)
    if sl < -tol:
        VIOLATIONS.append(f"{tag}: {name}  value={value!r} < bound={bound!r} "
                          f"(slack={sl:.3e})")
    return sl


def relerr(x, ref):
    if ref is None or x is None:
        return None
    d = abs(ref)
    return abs(x - ref) / d if d > 1e-300 else abs(x - ref)


# --------------------------------------------------------------------------------------
# One case
# --------------------------------------------------------------------------------------


def run_case(M, a0, v0, m, do_control=True):
    t0 = time.time()
    tag = f"M={M} a0={a0} v0={v0} m={m}"
    row = dict(M=M, a0=a0, v0=v0, m=m, Delta=m - a0)

    # ---------------- immediate-stop convention ----------------
    if a0 >= m:
        row.update(
            immediate_stop=1, u_M=0.0, a_Tm_cf=a0, v_Tm_cf=v0, b_Tm_cf=0.0,
            q_Tm_cf=a0, T_m_cf=0.0, T_m_ode=0.0,
            suppression_I4=0.0, suppression_bound_I2=1.0 / (1.0 + M * v0 * v0),
            M_times_uM=0.0, Delta_over_v0sq=(m - a0) / (v0 * v0),
            a_over_m=a0 / m, a_over_a0=1.0, limit_a_over_m=a0 / m, limit_a_over_a0=1.0,
            disp=0.0, dv_abs=0.0,
            gap_sup_cf=0.0, gap_min_cf=0.0,
            reversal_margin=2.0 * a0 - m, err_reversal=0,
            reversal_margin_limit=2.0 * a0 - m, err_reversal_limit=0,
            err_reversal_spectral_only=0,
            max_rel_err_cf_vs_ode=0.0,
            note="immediate-stop convention (a0 >= m): T_m = 0, encoder unchanged",
            runtime_s=time.time() - t0,
        )
        row["slack_I2_suppression"] = row["suppression_bound_I2"]
        return row

    row["immediate_stop"] = 0
    Delta = m - a0
    s = math.sqrt(M)

    # ---------------- closed form (I1) ----------------
    uM, xM = root_u(M, v0, Delta)
    a_cf = a0 + uM
    v_cf = float(v_of_u(uM, v0, M))
    b_cf = float(b_of_u(uM, v0, M))
    q_cf = a_cf + b_cf * v_cf
    row.update(u_M=uM, a_Tm_cf=a_cf, v_Tm_cf=v_cf, b_Tm_cf=b_cf, q_Tm_cf=q_cf)
    row["abs_err_q_Tm_cf_minus_m"] = abs(q_cf - m)
    # layer-balance invariant b^2 = M (v^2 - v0^2)
    row["invariant_b2_minus_M_v2_diff"] = abs(b_cf * b_cf - M * (v_cf * v_cf - v0 * v0))

    # closed-form fitting time (two independent quadratures)
    u_lin = np.linspace(0.0, uM, N_GRID_LIN)
    u_geo = uM * np.geomspace(1e-10, 1.0, N_GRID_GEO)
    u_grid = np.unique(np.concatenate([u_lin, u_geo, [0.0, uM]]))
    t_grid = t_of_u_cumulative(u_grid, a0, v0, M)
    T_cf = float(t_grid[-1])
    T_quad, T_quad_err = T_m_quadrature(a0, v0, M, uM)
    row.update(T_m_cf=T_cf, T_m_quad=T_quad, T_m_quad_abserr=T_quad_err,
               rel_T_cf_vs_quad=relerr(T_cf, T_quad))

    # ---------------- full (2+M) ODE ----------------
    t_hi = 1.5 * T_cf
    sol, T_ode = integrate_case(M, a0, v0, m, t_hi)
    y_end = sol.y_events[0][0]
    a_o, v_o, b_o, q_o = unpack(y_end, M)
    beta0 = beta0_zero_sum(M)
    row["beta0_sum_exact"] = float(beta0.sum())
    row["beta0_ptp"] = float(np.ptp(beta0))
    # all readouts must shift by the SAME amount, so beta(T) - beta(0) is constant
    shift = y_end[2:] - beta0
    row["beta_shift_spread"] = float(np.ptp(shift))
    row.update(a_Tm_ode=float(a_o), v_Tm_ode=float(v_o), b_Tm_ode=float(b_o),
               q_Tm_ode=float(q_o), T_m_ode=T_ode)

    rel_u = relerr(float(a_o) - a0, uM)
    rel_v = relerr(float(v_o), v_cf)
    rel_b = relerr(float(b_o), b_cf)
    rel_T = relerr(T_ode, T_cf)
    row.update(rel_err_u_cf_vs_ode=rel_u, rel_err_v_cf_vs_ode=rel_v,
               rel_err_b_cf_vs_ode=rel_b, rel_err_T_cf_vs_ode=rel_T,
               max_rel_err_cf_vs_ode=max(rel_u, rel_v, rel_b, rel_T))

    # (I1) along the whole trajectory, not just at T_m
    ts = np.linspace(0.0, T_ode, 2001)
    Y = sol.sol(ts)
    a_t, v_t, b_t, q_t = unpack(Y, M)
    u_t = a_t - a0
    with np.errstate(over="ignore"):
        v_ref = v_of_u(u_t, v0, M)
        b_ref = b_of_u(u_t, v0, M)
        q_ref = Q_iso(u_t, a0, v0, M)
    # Scale-aware relative errors: v and q pass close to no zero of interest, but b(0)=0
    # and q crosses 0 whenever a0 < 0 < m, so each denominator is floored at the
    # characteristic magnitude of its own trajectory rather than at the pointwise value.
    row["I1_max_rel_err_v"] = float(
        np.max(np.abs(v_t - v_ref) / np.maximum(np.abs(v_ref), abs(v0))))
    row["I1_max_rel_err_b"] = float(
        np.max(np.abs(b_t - b_ref) / max(abs(b_cf), 1e-300)))
    row["I1_max_rel_err_q"] = float(
        np.max(np.abs(q_t - q_ref) / np.maximum(np.abs(q_ref), max(abs(m), 1.0))))

    # ---------------- (i) update suppression, eq. (I4) + (I2) ----------------
    supp = uM / Delta
    supp_bound = 1.0 / (1.0 + M * v0 * v0)
    row["suppression_I4"] = supp
    row["suppression_bound_I2"] = supp_bound
    row["slack_I2_suppression"] = slack_upper("(I2) u_M/Delta <= 1/(1+M v0^2)",
                                              supp, supp_bound, tag)
    row["slack_I2_uM"] = slack_upper("(I2) u_M <= Delta/(1+M v0^2)",
                                     uM, Delta / (1.0 + M * v0 * v0), tag)
    row["slack_I2_uM_positive"] = slack_lower("(I2) u_M > 0", uM, 0.0, tag)
    row["M_times_uM"] = M * uM
    row["Delta_over_v0sq"] = Delta / (v0 * v0)
    row["rel_M_uM_vs_limit"] = relerr(M * uM, Delta / (v0 * v0))
    row["suppression_ode"] = (float(a_o) - a0) / Delta

    # ---------------- (ii) the ratios that do NOT vanish ----------------
    row["a_over_m"] = a_cf / m
    row["limit_a_over_m"] = a0 / m
    row["rel_a_over_m_vs_limit"] = relerr(a_cf / m, a0 / m) if a0 != 0.0 else None
    row["a_over_a0"] = a_cf / a0 if a0 != 0.0 else None
    row["limit_a_over_a0"] = 1.0 if a0 != 0.0 else None
    row["a_over_a0_minus_1"] = (a_cf / a0 - 1.0) if a0 != 0.0 else None

    # ---------------- (iii) encoder displacement, eq. (I3) ----------------
    dv = abs(v_cf - v0)
    dv_bound = Delta * Delta / (2.0 * M * abs(v0) ** 3)
    disp = math.hypot(uM, v_cf - v0)
    disp_bound = math.sqrt(Delta ** 2 / (1.0 + M * v0 * v0) ** 2
                           + Delta ** 4 / (4.0 * M * M * abs(v0) ** 6))
    row.update(dv_abs=dv, dv_bound_I3=dv_bound,
               slack_I3_dv=slack_upper("(I3) |v-v0| <= Delta^2/(2M|v0|^3)",
                                       dv, dv_bound, tag),
               disp=disp, disp_bound_I3=disp_bound,
               slack_I3_disp=slack_upper("(I3) ||W(T_m)-W0|| <= sqrt(...)",
                                         disp, disp_bound, tag),
               disp_ode=math.hypot(float(a_o) - a0, float(v_o) - v0),
               disp_spectral_only=Delta,
               disp_ratio_joint_over_spectral=disp / Delta)

    # ---------------- (iv) fitting time, two-sided ----------------
    Psi_m = psi_a0(m, a0)
    T_hi_b = Psi_m / (1.0 + M * v0 * v0)
    T_lo_b = Psi_m / (1.0 + M * v0 * v0 + 2.0 * Delta * Delta / (v0 * v0))
    row.update(Psi_a0_m=Psi_m, T_bound_lower=T_lo_b, T_bound_upper=T_hi_b,
               slack_T_upper=slack_upper("T_m <= Psi/(1+Mv0^2)", T_cf, T_hi_b, tag),
               slack_T_lower=slack_lower("T_m >= Psi/(1+Mv0^2+2Delta^2/v0^2)",
                                         T_cf, T_lo_b, tag),
               slack_T_upper_ode=slack_upper("T_m(ode) <= Psi/(1+Mv0^2)",
                                             T_ode, T_hi_b, tag),
               slack_T_lower_ode=slack_lower("T_m(ode) >= Psi/(1+Mv0^2+2Delta^2/v0^2)",
                                             T_ode, T_lo_b, tag))

    # ---------------- (v) frozen-comparator gap, eq. (I5) ----------------
    p0 = math.exp(a0) / (1.0 + math.exp(a0))
    C0 = 1.0 + 2.0 * Delta * Delta / (v0 * v0)
    # closed-form joint path parametrized by u; frozen path from Psi_{a0}(q_F) = M v0^2 t
    qJ_grid = Q_iso(u_grid, a0, v0, M)
    qF_grid = np.array([psi_a0_inv(M * v0 * v0 * tt, a0) for tt in t_grid])
    gap = qJ_grid - qF_grid
    ptwise = C0 * t_grid / (1.0 + math.exp(a0) + p0 * M * v0 * v0 * t_grid)
    uni = C0 / (p0 * M * v0 * v0)
    gap_sup = float(np.max(gap))
    gap_min = float(np.min(gap))
    row.update(gap_sup_cf=gap_sup, gap_min_cf=gap_min,
               gap_bound_uniform_I5=uni,
               slack_I5_uniform=slack_upper("(I5) sup gap <= C0/(p0 M v0^2)",
                                            gap_sup, uni, tag),
               slack_I5_nonneg=slack_lower("(I5) q_J >= q_F", gap_min, 0.0, tag),
               max_excess_I5_pointwise=float(np.max(gap - ptwise)),
               slack_I5_pointwise=slack_upper(
                   "(I5) pointwise gap <= C0 t/(1+e^{a0}+p0 M v0^2 t)",
                   float(np.max(gap - ptwise)), 0.0, tag),
               C0_I5=C0, p0_I5=p0,
               gap_sup_times_M=gap_sup * M)
    # cross-check the gap with the ODE trajectory
    qF_ode = np.array([psi_a0_inv(M * v0 * v0 * tt, a0) for tt in ts])
    gap_ode = q_t - qF_ode
    row["gap_sup_ode"] = float(np.max(gap_ode))
    row["gap_min_ode"] = float(np.min(gap_ode))
    row["rel_gap_sup_cf_vs_ode"] = relerr(float(np.max(gap_ode)), gap_sup)
    # full M-dimensional frozen ODE (small M only)
    if M <= FROZEN_ODE_MAX_M:
        solF = solve_ivp(rhs_frozen, (0.0, T_ode), np.zeros(M), args=(M, a0, v0),
                         method="DOP853", rtol=RTOL, atol=ATOL, dense_output=True,
                         max_step=max(T_ode, 1e-12) / 60.0)
        qF_num = a0 + solF.sol(ts).sum(axis=0) * v0
        # q_F crosses zero whenever a0 < 0 < m, so floor the denominator at the
        # trajectory scale rather than at the pointwise value.
        row["frozen_ode_max_rel_err_vs_Psi_inv"] = float(
            np.max(np.abs(qF_num - qF_ode)
                   / np.maximum(np.abs(qF_ode), max(abs(m), 1.0))))
    else:
        row["frozen_ode_max_rel_err_vs_Psi_inv"] = None

    # ---------------- (vi) reversal test ----------------
    margin = 2.0 * a_cf - m
    row.update(reversal_margin=margin, err_reversal=int(margin < 0.0),
               reversal_margin_ode=2.0 * float(a_o) - m,
               reversal_margin_limit=2.0 * a0 - m,
               err_reversal_limit=int(2.0 * a0 - m < 0.0),
               err_reversal_spectral_only=0)
    if a0 < 0.5 * m:
        thresh = m / (m - 2.0 * a0)
        row["suff_cond_threshold"] = thresh
        row["suff_cond_M_v0sq"] = M * v0 * v0
        row["suff_cond_holds"] = int(M * v0 * v0 > thresh)
        # if the sufficient condition holds, the reversal error MUST be 1
        if M * v0 * v0 > thresh and margin >= 0.0:
            VIOLATIONS.append(f"{tag}: sufficient finite-width condition M v0^2 > "
                              f"m/(m-2a0) holds but reversal margin = {margin!r} >= 0")
        row["suff_cond_consistent"] = int(not (M * v0 * v0 > thresh and margin >= 0.0))
    else:
        row["suff_cond_threshold"] = None
        row["suff_cond_M_v0sq"] = M * v0 * v0
        row["suff_cond_holds"] = None
        row["suff_cond_consistent"] = None

    # ---------------- 1/sqrt(M) readout control ----------------
    if do_control:
        u1, _ = root_u(1, v0, Delta)               # M_eff = 1 reduced dynamics
        T1, _ = T_m_quadrature(a0, v0, 1, u1)
        solc, Tc = integrate_case(M, a0, v0, m, 1.5 * T1, control=True)
        yc = solc.y_events[0][0]
        a_c, v_c, b_c, _ = unpack(yc, M, control=True)
        row.update(ctrl_u=float(a_c) - a0, ctrl_u_M1_cf=u1,
                   ctrl_rel_err_u=relerr(float(a_c) - a0, u1),
                   ctrl_rel_err_v=relerr(float(v_c), float(v_of_u(u1, v0, 1))),
                   ctrl_rel_err_b=relerr(float(b_c), float(b_of_u(u1, v0, 1))),
                   ctrl_T_m=float(Tc), ctrl_T_m_M1_cf=T1,
                   ctrl_rel_err_T=relerr(float(Tc), T1),
                   ctrl_suppression=(float(a_c) - a0) / Delta,
                   ctrl_disp=math.hypot(float(a_c) - a0, float(v_c) - v0))
        if row["ctrl_rel_err_u"] > 1e-6:
            VIOLATIONS.append(f"{tag}: 1/sqrt(M) control u does not match the M=1 value "
                              f"(rel {row['ctrl_rel_err_u']:.3e})")

    row["note"] = ""
    row["runtime_s"] = time.time() - t0
    return row


# --------------------------------------------------------------------------------------
# Gaussian average, eq. (I6)
# --------------------------------------------------------------------------------------


def gaussian_sweep():
    rows = []
    ode_errs = []
    for sigma in GAUSS_SIGMAS:
        for m in GAUSS_MARGINS:
            rng = np.random.default_rng(GAUSS_SEED + int(1000 * sigma) + int(10 * m))
            draws = rng.normal(scale=sigma, size=(GAUSS_N, 2))   # paired across M
            a0s, v0s = draws[:, 0], draws[:, 1]
            err_limit = (2.0 * a0s - m < 0.0).astype(float)      # 1{a0 < m/2}
            phi = float(ndtr(m / (2.0 * sigma)))
            for M in GAUSS_M:
                t0 = time.time()
                margins = np.empty(GAUSS_N)
                for i in range(GAUSS_N):
                    a0, v0 = float(a0s[i]), float(v0s[i])
                    if a0 >= m:                                   # immediate stop
                        margins[i] = 2.0 * a0 - m
                        continue
                    u, _ = root_u(M, v0, m - a0)
                    margins[i] = 2.0 * (a0 + u) - m
                err = (margins < 0.0).astype(float)
                # The disagreement with the limit can only go one way: a0 >= m/2 forces
                # a(T_m) >= a0 >= m/2, hence margin >= 0 and Err = 0 = Err_limit.  And on
                # every disagreeing draw (a0 < m/2 but Err = 0) the sufficient finite-width
                # condition M v0^2 > m/(m - 2 a0) must FAIL.  Both are checked here.
                dis = (err != err_limit)
                wrong_dir = int(np.sum(dis & (err_limit == 0.0)))
                if wrong_dir:
                    VIOLATIONS.append(f"gaussian sigma={sigma} m={m} M={M}: {wrong_dir} "
                                      f"draws with Err=1 while Err_limit=0")
                if dis.any():
                    sc = M * v0s[dis] ** 2 - m / (m - 2.0 * a0s[dis])
                    n_suff_bad = int(np.sum(sc > 0.0))
                    max_Mv0sq_dis = float(np.max(M * v0s[dis] ** 2))
                else:
                    n_suff_bad, max_Mv0sq_dis = 0, 0.0
                if n_suff_bad:
                    VIOLATIONS.append(f"gaussian sigma={sigma} m={m} M={M}: {n_suff_bad} "
                                      f"draws satisfy M v0^2 > m/(m-2a0) yet Err = 0")
                p = float(err.mean())
                se = math.sqrt(max(p * (1.0 - p), 0.0) / GAUSS_N)
                p_lim = float(err_limit.mean())
                se_lim = math.sqrt(max(p_lim * (1.0 - p_lim), 0.0) / GAUSS_N)
                disagree = int(np.sum(err != err_limit))

                # ODE verification of the closed form on a subset of the same draws
                n_ode = GAUSS_ODE_CHECK[M]
                worst = 0.0
                idx = [i for i in range(GAUSS_N) if a0s[i] < m][:n_ode]
                for i in idx:
                    a0, v0 = float(a0s[i]), float(v0s[i])
                    u, _ = root_u(M, v0, m - a0)
                    T_cf, _ = T_m_quadrature(a0, v0, M, u)
                    sol, _T = integrate_case(M, a0, v0, m, 1.5 * T_cf)
                    a_o = float(sol.y_events[0][0][0])
                    worst = max(worst, relerr(a_o - a0, u))
                ode_errs.append(worst)

                rows.append(dict(
                    sigma=sigma, m=m, M=M, n_draws=GAUSS_N, seed=int(GAUSS_SEED
                        + int(1000 * sigma) + int(10 * m)),
                    E_err_reversal=p, mc_se=se,
                    E_err_reversal_limit=p_lim, mc_se_limit=se_lim,
                    Phi_m_over_2sigma=phi,
                    diff_vs_Phi=p - phi,
                    diff_vs_limit_same_draws=p - p_lim,
                    n_draws_disagreeing_with_limit=disagree,
                    frac_disagreeing=disagree / GAUSS_N,
                    n_disagreeing_wrong_direction=wrong_dir,
                    n_disagreeing_that_satisfy_suff_cond=n_suff_bad,
                    max_M_v0sq_among_disagreeing=max_Mv0sq_dis,
                    n_immediate_stop=int(np.sum(a0s >= m)),
                    E_err_reversal_spectral_only=0.0,
                    n_ode_verified=len(idx),
                    max_rel_err_cf_vs_ode=worst,
                    runtime_s=time.time() - t0,
                ))
                print(f"  sigma={sigma} m={m} M={M:6d}: E[Err]={p:.4f} (se {se:.4f})  "
                      f"limit(same draws)={p_lim:.4f}  Phi={phi:.6f}  "
                      f"disagree={disagree:5d}  ode_chk={worst:.2e}  "
                      f"[{time.time()-t0:.1f}s]")
    return rows, (max(ode_errs) if ode_errs else 0.0)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main():
    t_start = time.time()
    print("=" * 96)
    print("math_02.md Section 4 (I1)-(I6): isotropic serial-CE toy, numerical verification")
    print("=" * 96)

    cases = []
    for M in M_LIST:
        for a0 in A0_LIST:
            for v0 in V0_LIST:
                for m in M_MARGIN_LIST:
                    cases.append((M, a0, v0, m))
    n_main = len(cases)
    for M in STOP_M:
        for (a0, m) in STOP_A0_M:
            for v0 in STOP_V0:
                cases.append((M, a0, v0, m))
    print(f"{n_main} non-trivial cases + {len(cases)-n_main} immediate-stop cases "
          f"= {len(cases)} total")

    rows = []
    for k, (M, a0, v0, m) in enumerate(cases):
        rows.append(run_case(M, a0, v0, m))
        if (k + 1) % 50 == 0:
            print(f"  ... {k+1}/{len(cases)} cases  ({time.time()-t_start:.1f}s)")

    main_rows = [r for r in rows if not r["immediate_stop"]]

    # ---------------- headline diagnostics ----------------
    max_ode = max(r["max_rel_err_cf_vs_ode"] for r in main_rows)
    arg_ode = max(main_rows, key=lambda r: r["max_rel_err_cf_vs_ode"])
    max_I1 = max(max(r["I1_max_rel_err_v"], r["I1_max_rel_err_b"], r["I1_max_rel_err_q"])
                 for r in main_rows)
    max_beta_spread = max(r["beta_shift_spread"] for r in main_rows)
    max_beta_sum = max(abs(r["beta0_sum_exact"]) for r in main_rows)
    max_ctrl = max(r["ctrl_rel_err_u"] for r in main_rows)
    max_frozen = max((r["frozen_ode_max_rel_err_vs_Psi_inv"] for r in main_rows
                      if r["frozen_ode_max_rel_err_vs_Psi_inv"] is not None), default=0.0)
    max_qm = max(r["abs_err_q_Tm_cf_minus_m"] for r in main_rows)

    print("\n" + "-" * 96)
    print(f"max closed-form vs ODE relative error (u, v, b, T_m) : {max_ode:.3e}")
    print(f"    at {arg_ode['M']=} {arg_ode['a0']=} {arg_ode['v0']=} {arg_ode['m']=}")
    print(f"max (I1) rel err along whole trajectory              : {max_I1:.3e}")
    print(f"max |Q_iso(u_M) - m|                                 : {max_qm:.3e}")
    print(f"max |sum beta_j(0)| (exact zero-sum init)            : {max_beta_sum:.3e}")
    print(f"max spread of readout shifts beta(T)-beta(0)         : {max_beta_spread:.3e}")
    print(f"max frozen-ODE vs Psi^-1(M v0^2 t) rel err           : {max_frozen:.3e}")
    print(f"max 1/sqrt(M)-control vs M=1 rel err on u            : {max_ctrl:.3e}")

    # suppression / non-vanishing ratio summaries at the largest M
    print("\n" + "-" * 96)
    print("(I4) vs the false pair of limits, at M = 4096, m = 2.9")
    print(f"{'a0':>6} {'v0':>6} | {'(a-a0)/(m-a0)':>14} {'a/m':>10} {'a0/m':>10} "
          f"{'a/a0':>12} {'M u_M':>10} {'D/v0^2':>10}")
    for r in main_rows:
        if r["M"] == 4096 and r["m"] == 2.9:
            aa = "n/a" if r["a_over_a0"] is None else f"{r['a_over_a0']:.9f}"
            print(f"{r['a0']:>6} {r['v0']:>6} | {r['suppression_I4']:>14.3e} "
                  f"{r['a_over_m']:>10.6f} {r['limit_a_over_m']:>10.6f} {aa:>12} "
                  f"{r['M_times_uM']:>10.5f} {r['Delta_over_v0sq']:>10.5f}")

    # ---------------- Gaussian sweep ----------------
    print("\n" + "-" * 96)
    print(f"Gaussian average (I6): n = {GAUSS_N} paired draws per (sigma, m)")
    g_rows, g_ode_err = gaussian_sweep()

    # ---------------- persist ----------------
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)

    def fmt(x):
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return ""
        if isinstance(x, float):
            return repr(x)
        return x

    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k)) for k in fields})
    print(f"\nWrote {len(rows)} rows x {len(fields)} columns to {OUT_CSV}")

    gfields = list(g_rows[0].keys())
    with open(OUT_CSV_G, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=gfields)
        w.writeheader()
        for r in g_rows:
            w.writerow({k: fmt(r[k]) for k in gfields})
    print(f"Wrote {len(g_rows)} rows to {OUT_CSV_G}")

    # ---------------- report ----------------
    write_report(rows, main_rows, g_rows, dict(
        max_ode=max_ode, arg_ode=arg_ode, max_I1=max_I1, max_qm=max_qm,
        max_beta_sum=max_beta_sum, max_beta_spread=max_beta_spread,
        max_frozen=max_frozen, max_ctrl=max_ctrl, g_ode_err=g_ode_err,
        wall=time.time() - t_start, n_cases=len(rows), n_main=len(main_rows),
    ))

    print("\n" + "=" * 96)
    print(f"BOUND VIOLATIONS: {len(VIOLATIONS)}")
    print("=" * 96)
    for v in VIOLATIONS:
        print("  - " + v)
    if not VIOLATIONS:
        print("  none")
    print(f"\nTotal wall time: {time.time()-t_start:.1f}s")


def write_report(rows, main_rows, g_rows, S):
    L = []
    A = L.append
    A("# Isotropic serial-CE toy: numerical verification of math_02.md Section 4, (I1)-(I6)")
    A("")
    A(f"Generated by `code/experiments/toy_serial_ce_isotropic.py` "
      f"({time.strftime('%Y-%m-%d %H:%M')}, wall {S['wall']:.1f} s).")
    A("")
    A("Model: `q = a + b v`, `b = sum_j beta_j`, `adot = rho`, `vdot = b rho`, "
      "`betadot_j = v rho`, `rho = (1+e^q)^{-1}`, with `a(0)=a0`, `v(0)=v0 != 0` and "
      "**unequal** `beta_j(0)` summing to **exactly** zero (so `b(0)=0`, `q(0)=a0`).")
    A("")
    A(f"Grid: `M` in {M_LIST}, `a0` in {A0_LIST}, `v0` in {V0_LIST}, "
      f"`m` in {M_MARGIN_LIST} -> {S['n_main']} non-trivial cases, plus "
      f"{S['n_cases']-S['n_main']} immediate-stop cases (`a0 >= m`).")
    A("")
    A("ODE: `scipy.integrate.solve_ivp`, DOP853, `rtol=1e-10`, `atol=1e-12`, "
      "terminal event `q = m`, full `(2+M)`-dimensional state.")
    A("")
    A("## 1. Headline numbers")
    A("")
    A("| quantity | value |")
    A("|---|---|")
    ao = S["arg_ode"]
    A(f"| **max closed-form vs ODE relative error** (over `u=a(T_m)-a0`, `v(T_m)`, "
      f"`b(T_m)`, `T_m`; all {S['n_main']} cases) | **{S['max_ode']:.3e}** "
      f"(at M={ao['M']}, a0={ao['a0']}, v0={ao['v0']}, m={ao['m']}) |")
    A(f"| max (I1) relative error along the whole trajectory (2001 points/case) | "
      f"{S['max_I1']:.3e} |")
    A(f"| max `|Q_iso(u_M) - m|` (root accuracy) | {S['max_qm']:.3e} |")
    A(f"| max `|sum_j beta_j(0)|` (zero-sum init is bitwise exact) | "
      f"{S['max_beta_sum']:.3e} |")
    A(f"| max spread of `beta_j(T_m) - beta_j(0)` (all readouts shift equally) | "
      f"{S['max_beta_spread']:.3e} |")
    A(f"| max frozen-comparator ODE vs `Psi_a0^{{-1}}(M v0^2 t)` relative error | "
      f"{S['max_frozen']:.3e} |")
    A(f"| max `1/sqrt(M)`-readout control vs the `M=1` value of `u` | "
      f"{S['max_ctrl']:.3e} |")
    A(f"| max closed-form vs ODE relative error in the Gaussian sweep | "
      f"{S['g_ode_err']:.3e} |")
    A(f"| **bound violations** | **{len(VIOLATIONS)}** |")
    A("")
    if VIOLATIONS:
        A("### Bound violations (NOT hidden)")
        A("")
        for v in VIOLATIONS:
            A(f"- `{v}`")
    else:
        A("No bound in (I2), (I3), (I5), the two-sided `T_m` bound, or the sufficient "
          "finite-width reversal condition was violated in any case.")
    A("")

    # tightest slacks
    A("## 2. Bound slacks (worst case over the grid)")
    A("")
    A("All slacks are `bound - value` for upper bounds and `value - bound` for lower "
      "bounds; a negative number would be a violation.")
    A("")
    A("| bound | equation | min slack over all cases | at |")
    A("|---|---|---|---|")
    for key, eq, desc in [
        ("slack_I2_uM", "(I2)", "`u_M <= Delta/(1+M v0^2)`"),
        ("slack_I2_uM_positive", "(I2)", "`u_M > 0`"),
        ("slack_I3_dv", "(I3)", "`|v(T_m)-v0| <= Delta^2/(2M|v0|^3)`"),
        ("slack_I3_disp", "(I3)", "`||W(T_m)-W0|| <= sqrt(Delta^2/(1+Mv0^2)^2 + "
                                  "Delta^4/(4M^2|v0|^6))`"),
        ("slack_T_upper", "(iv)", "`T_m <= Psi_a0(m)/(1+M v0^2)`"),
        ("slack_T_lower", "(iv)", "`T_m >= Psi_a0(m)/(1+M v0^2+2Delta^2/v0^2)`"),
        ("slack_I5_nonneg", "(I5)", "`q_J(t) >= q_F(t)`"),
        ("slack_I5_pointwise", "(I5)", "`q_J-q_F <= C0 t/(1+e^{a0}+p0 M v0^2 t)`"),
        ("slack_I5_uniform", "(I5)", "`sup_t (q_J-q_F) <= C0/(p0 M v0^2)`"),
    ]:
        vals = [(r[key], r) for r in main_rows if r.get(key) is not None]
        if not vals:
            continue
        mn, r = min(vals, key=lambda p: p[0])
        A(f"| {desc} | {eq} | `{mn:.6e}` | M={r['M']}, a0={r['a0']}, v0={r['v0']}, "
          f"m={r['m']} |")
    A("")

    # (I4) vs the false limits
    A("## 3. (I4) holds; the proposed pair of limits does not")
    A("")
    A("`(a(T_m)-a0)/(m-a0)` must go to 0 (this is (I4)). `a(T_m)/m` and `a(T_m)/a0` "
      "must NOT: they converge to `a0/m` and to `1`. Below, `m = 2.9`, `v0 = 1.0`.")
    A("")
    A("| a0 | M | (a(T_m)-a0)/(m-a0) | a(T_m)/m | a0/m | a(T_m)/a0 | M u_M | Delta/v0^2 |")
    A("|---|---|---|---|---|---|---|---|")
    for a0 in A0_LIST:
        for M in M_LIST:
            r = next((x for x in main_rows if x["M"] == M and x["a0"] == a0
                      and x["v0"] == 1.0 and x["m"] == 2.9), None)
            if r is None:
                continue
            aa = "n/a (a0=0)" if r["a_over_a0"] is None else f"{r['a_over_a0']:.8f}"
            A(f"| {a0} | {M} | {r['suppression_I4']:.4e} | {r['a_over_m']:.8f} | "
              f"{r['limit_a_over_m']:.8f} | {aa} | {r['M_times_uM']:.6f} | "
              f"{r['Delta_over_v0sq']:.6f} |")
    A("")
    A("The last two columns are the (I2) limit `M u_M -> Delta/v0^2`.")
    A("")

    # displacement
    A("## 4. Encoder displacement (I3) and the spectral-only comparator")
    A("")
    A("`m = 2.9`, `a0 = -0.3`. `Delta` is the displacement of the spectral-only "
      "comparator, which must move `a` by the whole of `m - a0`.")
    A("")
    A("| v0 | M | ||W(T_m)-W0|| | (I3) bound | slack | joint/spectral-only |")
    A("|---|---|---|---|---|---|")
    for v0 in V0_LIST:
        for M in M_LIST:
            r = next((x for x in main_rows if x["M"] == M and x["a0"] == -0.3
                      and x["v0"] == v0 and x["m"] == 2.9), None)
            if r is None:
                continue
            A(f"| {v0} | {M} | {r['disp']:.6e} | {r['disp_bound_I3']:.6e} | "
              f"{r['slack_I3_disp']:.3e} | {r['disp_ratio_joint_over_spectral']:.4e} |")
    A("")

    # (I5)
    A("## 5. Frozen-comparator gap (I5)")
    A("")
    A("`sup_{[0,T_m]} (q_J - q_F)` from the closed-form parametrization, against the "
      "uniform bound `C0/(p0 M v0^2)`. `a0 = -0.3`, `m = 2.9`.")
    A("")
    A("| v0 | M | sup gap | min gap (>=0) | (I5) uniform bound | max pointwise excess |")
    A("|---|---|---|---|---|---|")
    for v0 in V0_LIST:
        for M in M_LIST:
            r = next((x for x in main_rows if x["M"] == M and x["a0"] == -0.3
                      and x["v0"] == v0 and x["m"] == 2.9), None)
            if r is None:
                continue
            A(f"| {v0} | {M} | {r['gap_sup_cf']:.6e} | {r['gap_min_cf']:.3e} | "
              f"{r['gap_bound_uniform_I5']:.6e} | {r['max_excess_I5_pointwise']:.3e} |")
    A("")
    A("The maximum pointwise excess is `max_t [ (q_J-q_F) - C0 t/(1+e^{a0}+p0 M v0^2 t) ]` "
      "and must be `<= 0`.")
    A("")

    # reversal
    A("## 6. Reversal test (vi)")
    A("")
    A("Test margin under `S=Y, C=-Y` is `2 a(T_m) - m`, with limit `2 a0 - m`. "
      "`Err = 1{margin < 0}`. `v0 = 1.0`, `m = 2.9`, so `m/2 = 1.45` and only "
      "`a0 = 1.7 > m/2` should end up correct at large M.")
    A("")
    A("| a0 | M | 2a(T_m)-m | Err | limit 2a0-m | Err(limit) | M v0^2 | m/(m-2a0) |")
    A("|---|---|---|---|---|---|---|---|")
    for a0 in A0_LIST:
        for M in M_LIST:
            r = next((x for x in main_rows if x["M"] == M and x["a0"] == a0
                      and x["v0"] == 1.0 and x["m"] == 2.9), None)
            if r is None:
                continue
            th = "n/a (a0>m/2)" if r["suff_cond_threshold"] is None \
                else f"{r['suff_cond_threshold']:.4f}"
            A(f"| {a0} | {M} | {r['reversal_margin']:+.6f} | {r['err_reversal']} | "
              f"{r['reversal_margin_limit']:+.6f} | {r['err_reversal_limit']} | "
              f"{r['suff_cond_M_v0sq']:.2f} | {th} |")
    A("")
    A("The spectral-only comparator has reversal margin `m > 0` in every case, hence "
      "reversal error `0`.")
    A("")

    # control
    A("## 7. The `1/sqrt(M)` readout normalization removes the suppression")
    A("")
    A("With `F = z1 + M^{-1/2} sum_j gamma_j z2` the readout sum obeys `bdot = v rho`, "
      "so the reduced dynamics is the `M = 1` one for every `M`. `a0 = -0.3`, "
      "`v0 = 1.0`, `m = 2.9`.")
    A("")
    A("| M | u (control, full ODE) | u (M=1 closed form) | rel err | "
      "control suppression | joint suppression |")
    A("|---|---|---|---|---|---|")
    for M in M_LIST:
        r = next((x for x in main_rows if x["M"] == M and x["a0"] == -0.3
                  and x["v0"] == 1.0 and x["m"] == 2.9), None)
        if r is None:
            continue
        A(f"| {M} | {r['ctrl_u']:.12f} | {r['ctrl_u_M1_cf']:.12f} | "
          f"{r['ctrl_rel_err_u']:.2e} | {r['ctrl_suppression']:.6f} | "
          f"{r['suppression_I4']:.4e} |")
    A("")

    # immediate stop
    n_stop = sum(1 for r in rows if r["immediate_stop"])
    A("## 8. Immediate-stop convention (`a0 >= m`)")
    A("")
    A(f"{n_stop} cases with `a0 >= m` use the matched-threshold rule and stop at "
      "`T_m = 0`: `a(T_m) = a0`, displacement `0`, suppression `0`, reversal margin "
      "`2a0 - m > 0` and reversal error `0` (agreeing with the limit, since "
      "`a0 >= m > m/2`).")
    A("")

    # (I6)
    A("## 9. Gaussian average, equation (I6)")
    A("")
    A(f"`(a0, v0) ~ N(0, sigma^2 I_2)`, `n = {GAUSS_N}` draws, the SAME draws reused "
      "across `M` (paired). `E[Err_limit]` is the sample mean of `1{a0 < m/2}` on "
      "those same draws, i.e. the Monte-Carlo estimate of `Phi(m/(2 sigma))` with the "
      "sampling noise removed from the comparison.")
    A("")
    A("| sigma | m | M | E[Err_reversal] | MC se | E[Err_limit] (same draws) | "
      "Phi(m/(2 sigma)) | E[Err] - Phi | draws disagreeing with the limit | "
      "spectral-only |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in g_rows:
        A(f"| {r['sigma']} | {r['m']} | {r['M']} | {r['E_err_reversal']:.4f} | "
          f"{r['mc_se']:.4f} | {r['E_err_reversal_limit']:.4f} | "
          f"{r['Phi_m_over_2sigma']:.6f} | {r['diff_vs_Phi']:+.4f} | "
          f"{r['n_draws_disagreeing_with_limit']} / {GAUSS_N} | "
          f"{r['E_err_reversal_spectral_only']:.1f} |")
    A("")
    A("`E[Err_reversal]` increases monotonically in `M` towards `Phi(m/(2 sigma))`. Every "
      "draw that falls short of the limit fails the sufficient finite-width condition "
      "`M v0^2 > m/(m-2 a0)` -- either because `|v0|` is small (the "
      "small-context-initialization exception that math_02.md Section 4 flags) or because "
      "`a0` sits just below `m/2`, where the threshold `m/(m-2 a0)` diverges; the "
      "`max M v0^2` column below shows both regimes occurring. Two consequences of that "
      "condition are checked on all "
      f"{GAUSS_N * len(GAUSS_M) * len(GAUSS_SIGMAS) * len(GAUSS_MARGINS)} draws:")
    A("")
    A("| sigma | m | M | disagreeing draws | of which satisfy `M v0^2 > m/(m-2a0)` "
      "(must be 0) | max `M v0^2` among them | disagreements in the wrong direction "
      "(must be 0) |")
    A("|---|---|---|---|---|---|---|")
    for r in g_rows:
        A(f"| {r['sigma']} | {r['m']} | {r['M']} | "
          f"{r['n_draws_disagreeing_with_limit']} | "
          f"{r['n_disagreeing_that_satisfy_suff_cond']} | "
          f"{r['max_M_v0sq_among_disagreeing']:.4f} | "
          f"{r['n_disagreeing_wrong_direction']} |")
    A("")
    A("(\"wrong direction\" = `Err_reversal = 1` while `Err_limit = 0`; impossible, "
      "because `a0 >= m/2` forces `a(T_m) >= a0 >= m/2`.) "
      "The spectral-only comparator has reversal error 0 at every `M`.")
    A("")
    A("## 10. Files")
    A("")
    A("- `results/toy_serial_ce_isotropic.csv` -- one row per case, every quantity and "
      "every bound slack.")
    A("- `results/toy_serial_ce_isotropic_gaussian.csv` -- one row per "
      "`(sigma, m, M)` of the (I6) sweep.")
    A("- `code/experiments/toy_serial_ce.py` -- the `a0=0, v0=1` predecessor "
      "(Theorem 5.1 of math_01.md).")
    A("")

    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(L))
    print(f"Wrote report to {OUT_MD}")


if __name__ == "__main__":
    main()
