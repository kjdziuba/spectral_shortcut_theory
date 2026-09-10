# ---------------------------------------------------------------------------
# VERIFICATION RECORD -- copied unchanged into code/audits/ as a durable audit
# artefact.  Only this header was added.
# origin  : audit of review_packet/astra/math_02.md section 4 -- (I1)-(I5) isotropic serial CE toy
#           (session scratchpad 887fb000-.../scratchpad/audit_math02_opus/i_toy_ode.py)
#           report: review_packet/astra/audit_math02_sections_2_4_6_2026-09-10.md
# author  : Claude (math_02 sections 2/4/6 auditor)
# date    : 2026-09-10
# seed    : 31337
# command : venv/bin/python3 code/audits/math02_sections_2_4_6/i_toy_ode.py
# ---------------------------------------------------------------------------
"""
AUDIT (I1)-(I5) -- isotropic serial CE toy.

   q = a + b v,  b = sum_j beta_j,
   adot = rho,  vdot = b rho,  betadot_j = v rho,  rho = 1/(1+e^q).

Independent of astra/math_02_checks.py: that script tests 12 solves at 4 widths
with UNEQUAL zero-sum readouts and checks only (I1)+the first half of (I3).
Here we integrate the FULL (2+M)-parameter system at M in {1,16,256,4096},
signed v0, negative AND positive a0, tiny |v0|, unequal zero-sum beta, and check
EVERY displayed inequality of section 4:

 (I1) closed form v = v0 cosh(sqrt(M) u), b = sqrt(M) v0 sinh(sqrt(M) u),
      q = a0 + u + sqrt(M) v0^2 sinh(2 sqrt(M) u)/2
 (I2) 0 < u_M <= Delta/(1+M v0^2)  and  M u_M -> Delta/v0^2; u_M strictly
      decreasing in M
 (I3) |v(T_m)-v0| <= Delta^2/(2 M |v0|^3)  and the W-displacement bound;
      the invariant b^2 = M(v^2-v0^2) and 0 <= b v <= Delta
 (I4) (a(T_m)-a0)/(m-a0) -> 0, while a(T_m)/a0 -> 1 and a(T_m)/m -> a0/m
      (our proposed a(T_m)/m -> 0 is FALSE)
 (I5) Psi_{a0}(m)/(1+M v0^2+2 Delta^2/v0^2) <= T_m <= Psi_{a0}(m)/(1+M v0^2);
      frozen comparator Psi_{a0}(q_F) = M v0^2 t;
      0 <= q_J-q_F <= C0 t/(1+e^{a0}+p0 M v0^2 t) <= C0/(p0 M v0^2)
 plus: beta_i - beta_j constant along the flow (closure of (a,v,b)).

Seed: 31337
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import expit

rng = np.random.default_rng(31337)
FAIL = []
TOL = 1e-7


def check(name, cond, info=""):
    if not cond:
        FAIL.append(f"{name}: {info}")
        print(f"  FAIL {name}  {info}")


def Psi(q, a0):
    return q - a0 + np.exp(q) - np.exp(a0)


def solve_full(M, a0, v0, m, beta0, tmax=1e7):
    """Integrate the full (2+M) system to first margin m; return trajectory."""
    def fun(t, y):
        a, v = y[0], y[1]
        b = y[2:].sum()
        r = expit(-(a + b * v))
        return np.r_[r, b * r, np.full(M, v * r)]

    def evt(t, y):
        return y[0] + y[1] * y[2:].sum() - m
    evt.terminal = True
    evt.direction = 1.0
    y0 = np.r_[a0, v0, beta0]
    sol = solve_ivp(fun, [0.0, tmax], y0, events=evt, rtol=3e-12, atol=1e-14,
                    method="DOP853", dense_output=True)
    assert sol.success, sol.message
    assert len(sol.t_events[0]) == 1, "margin not reached exactly once"
    return sol


def solve_frozen(M, a0, v0, m, beta0, T):
    """Frozen encoder: a, v held; only beta trains.  q_F = a0 + b v0."""
    def fun(t, y):
        b = y.sum()
        r = expit(-(a0 + b * v0))
        return np.full(M, v0 * r)
    sol = solve_ivp(fun, [0.0, T], beta0, rtol=3e-12, atol=1e-14,
                    method="DOP853", dense_output=True)
    assert sol.success
    return sol


CASES = [
    # (a0, v0, m)  -- negative/positive a0, signed v0, tiny |v0|
    (-0.7, -0.8, 2.9),
    (0.35, 1.2, 2.0),
    (1.7, 0.4, 2.5),
    (-1.5, 0.05, 1.0),      # tiny |v0|
    (0.9, -0.03, 3.0),      # tiny |v0|, negative
    (0.0, 1.0, 2.0),        # a0 = 0
]
MS = [1, 16, 256, 4096]

print("=" * 108)
print("(I1)-(I5) full (2+M)-parameter CE toy, unequal zero-sum readout init")
print("=" * 108)
hdr = (f"{'M':>6} {'a0':>6} {'v0':>7} {'m':>5} {'u_M':>11} {'bnd(I2)':>11} "
       f"{'M u_M':>10} {'D/v0^2':>10} {'|dv|':>10} {'bnd(I3)':>10} "
       f"{'T_m':>11} {'gapmax':>10} {'gapbnd':>10}")
print(hdr)

uM_by_case = {c: [] for c in CASES}
for (a0, v0, m) in CASES:
    Delta = m - a0
    assert Delta > 0
    p0 = np.exp(a0) / (1 + np.exp(a0))
    C0 = 1 + 2 * Delta**2 / v0**2
    for M in MS:
        beta0 = rng.normal(scale=0.02, size=M)
        beta0 -= beta0.mean()                      # unequal, sum zero

        # ---- (I2) root of the closed-form margin equation
        def froot(u):
            return a0 + u + np.sqrt(M) * v0**2 / 2 * np.sinh(2 * np.sqrt(M) * u) - m
        hi = Delta / (1 + M * v0**2)
        uM = brentq(froot, 0.0, hi, xtol=1e-16, rtol=8.9e-16)
        uM_by_case[(a0, v0, m)].append(uM)
        check("(I2) 0 < u_M <= Delta/(1+Mv0^2)", 0 < uM <= hi + 1e-14,
              f"M={M} uM={uM} hi={hi}")

        sol = solve_full(M, a0, v0, m, beta0)
        yT = sol.y[:, -1]
        T_m = sol.t_events[0][0]
        aT, vT = yT[0], yT[1]
        bT = yT[2:].sum()

        # ---- (I1) closed form
        x = np.sqrt(M) * uM
        v_cf = v0 * np.cosh(x)
        b_cf = np.sqrt(M) * v0 * np.sinh(x)
        check("(I1) a", abs(aT - (a0 + uM)) < TOL, f"M={M} {aT} vs {a0+uM}")
        check("(I1) v", abs(vT - v_cf) < TOL, f"M={M} {vT} vs {v_cf}")
        check("(I1) b", abs(bT - b_cf) < 1e-6 * max(1, abs(b_cf)),
              f"M={M} {bT} vs {b_cf}")
        check("(I1) q=m", abs(aT + bT * vT - m) < 1e-8, f"M={M}")

        # ---- readout differences constant (closure of (a,v,b))
        dbeta = (yT[2:] - yT[2:].mean()) - beta0
        check("beta_i-beta_j constant", np.max(np.abs(dbeta)) < 1e-10,
              f"M={M} max={np.max(np.abs(dbeta))}")

        # ---- invariants along the whole path
        ts = np.linspace(0, T_m, 400)
        Y = sol.sol(ts)
        av, vv = Y[0], Y[1]
        bv_ = Y[2:].sum(0)
        inv = bv_**2 - M * (vv**2 - v0**2)
        check("invariant b^2=M(v^2-v0^2)", np.max(np.abs(inv)) < 1e-7 * max(1, M * v0**2),
              f"M={M} max={np.max(np.abs(inv))}")
        bvprod = bv_ * vv
        check("0 <= b v <= Delta", np.all(bvprod >= -1e-10) and np.all(bvprod <= Delta + 1e-9),
              f"M={M} range=({bvprod.min()},{bvprod.max()})")

        # ---- (I3)
        dv = abs(vT - v0)
        dv_bnd = Delta**2 / (2 * M * abs(v0) ** 3)
        check("(I3) |v-v0| bound", dv <= dv_bnd + 1e-12, f"M={M} {dv} > {dv_bnd}")
        W_disp = np.hypot(aT - a0, vT - v0)
        W_bnd = np.sqrt(Delta**2 / (1 + M * v0**2) ** 2 + Delta**4 / (4 * M**2 * abs(v0) ** 6))
        check("(I3) W displacement", W_disp <= W_bnd + 1e-12, f"M={M} {W_disp} > {W_bnd}")

        # ---- (I5) fitting time
        Pm = Psi(m, a0)
        T_lo = Pm / (1 + M * v0**2 + 2 * Delta**2 / v0**2)
        T_hi = Pm / (1 + M * v0**2)
        check("(I5) T_m bounds", T_lo - 1e-12 <= T_m <= T_hi + 1e-12,
              f"M={M} {T_lo} <= {T_m} <= {T_hi}")

        # ---- (I5) frozen comparator + gap
        solF = solve_frozen(M, a0, v0, m, beta0, T_m)
        tsg = np.linspace(0, T_m, 600)
        bF = solF.sol(tsg).sum(0)
        qF = a0 + bF * v0
        qJ = Y[0][:0]  # placeholder
        YJ = sol.sol(tsg)
        qJ = YJ[0] + YJ[1] * YJ[2:].sum(0)
        # frozen closed form Psi(q_F) = M v0^2 t
        check("(I5) Psi(q_F)=M v0^2 t",
              np.max(np.abs(Psi(qF, a0) - M * v0**2 * tsg)) < 1e-7 * max(1, M * v0**2 * T_m),
              f"M={M} max={np.max(np.abs(Psi(qF,a0)-M*v0**2*tsg))}")
        gap = qJ - qF
        gbnd = C0 * tsg / (1 + np.exp(a0) + p0 * M * v0**2 * tsg)
        check("(I5) 0 <= gap", gap.min() >= -1e-9, f"M={M} min={gap.min()}")
        check("(I5) gap <= C0 t/(...)", np.all(gap <= gbnd + 1e-9),
              f"M={M} max excess={np.max(gap-gbnd)}")
        check("(I5) gap <= C0/(p0 M v0^2)", gap.max() <= C0 / (p0 * M * v0**2) + 1e-9,
              f"M={M}")

        print(f"{M:6d} {a0:6.2f} {v0:7.3f} {m:5.2f} {uM:11.4e} {hi:11.4e} "
              f"{M*uM:10.5f} {Delta/v0**2:10.5f} {dv:10.3e} {dv_bnd:10.3e} "
              f"{T_m:11.4e} {gap.max():10.3e} {C0/(p0*M*v0**2):10.3e}")

# ---- (I2) monotone decrease in M and the limit M u_M -> Delta/v0^2
print()
print("(I2) monotonicity in M and the limit  M u_M -> Delta/v0^2")
for (a0, v0, m), us in uM_by_case.items():
    Delta = m - a0
    check("(I2) u_M strictly decreasing in M",
          all(us[i] > us[i + 1] for i in range(len(us) - 1)), f"{us}")
    # very large M for the limit
    big, rels = [], []
    for M in [10**6, 10**8, 10**10]:
        hi = Delta / (1 + M * v0**2)
        u = brentq(lambda u: a0 + u + np.sqrt(M) * v0**2 / 2 * np.sinh(2 * np.sqrt(M) * u) - m,
                   0.0, hi, xtol=1e-300, rtol=8.9e-16)
        big.append(M * u)
        rels.append(abs(M * u - Delta / v0**2) / (Delta / v0**2))
    rel = rels[-1]
    # Expansion of the root equation Delta = u(1+M v0^2) + (2/3) M^2 v0^2 u^3 + ...
    # with M u = (Delta/v0^2)(1+eps) gives, to leading order,
    #      eps = -[ 1/(M v0^2)  +  (2/3) Delta^2/(M v0^4) ].
    # The SECOND term dominates for small |v0|: the limit is controlled by
    # M v0^4, i.e. a FOURTH power of |v0|, not M v0^2.
    Mb = 10**10
    pred = 1.0 / (Mb * v0**2) + (2.0 / 3.0) * Delta**2 / (Mb * v0**4)
    print(f"  a0={a0:6.2f} v0={v0:7.3f} m={m:4.1f}: M u_M at M=1e6,1e8,1e10 = "
          f"{big[0]:.6f}, {big[1]:.6f}, {big[2]:.6f}   target {Delta/v0**2:.6f}  "
          f"rel err {rel:.2e}  predicted eps = {pred:.2e}")
    check("(I2) limit M u_M -> Delta/v0^2 (rate-aware tol)", rel < 3 * max(pred, 1e-12),
          f"rel={rel} pred={pred}")
    check("(I2) monotone convergence", rels[0] > rels[1] > rels[2], f"{rels}")
    check("(I2) rate matches 1/(Mv0^2)+(2/3)D^2/(Mv0^4)", abs(rel / pred - 1) < 0.02,
          f"rel/pred={rel/pred}")

# ---- (I4): our proposed Q2 limits vs Astra's correction
print()
print("(I4) our Q2 proposal   a(T_m)/a0 -> 1   AND   a(T_m)/m -> 0")
print(f"{'a0':>7} {'v0':>7} {'m':>5} {'M':>8} {'a(T)/a0':>10} {'a(T)/m':>10} "
      f"{'a0/m':>10} {'(a-a0)/Delta':>13}")
for (a0, v0, m) in [(-0.7, -0.8, 2.9), (0.35, 1.2, 2.0), (1.7, 0.4, 2.5)]:
    Delta = m - a0
    for M in [1, 256, 10**6, 10**10]:
        hi = Delta / (1 + M * v0**2)
        u = brentq(lambda u: a0 + u + np.sqrt(M) * v0**2 / 2 * np.sinh(2 * np.sqrt(M) * u) - m,
                   0.0, hi, xtol=1e-300, rtol=8.9e-16)
        aT = a0 + u
        print(f"{a0:7.2f} {v0:7.3f} {m:5.2f} {M:8.0e} {aT/a0:10.6f} {aT/m:10.6f} "
              f"{a0/m:10.6f} {u/Delta:13.3e}")
    check("(I4) a(T)/a0 -> 1", abs(aT / a0 - 1) < 1e-8, f"{aT/a0}")
    check("(I4) a(T)/m -> a0/m NOT 0", abs(aT / m - a0 / m) < 1e-8 and abs(a0 / m) > 1e-3,
          f"a(T)/m={aT/m} a0/m={a0/m}")
    check("(I4) (a-a0)/Delta -> 0", u / Delta < 1e-8, f"{u/Delta}")

print()
print("FAILURES:", FAIL if FAIL else "none")
