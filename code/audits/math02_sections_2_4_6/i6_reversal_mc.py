# ---------------------------------------------------------------------------
# VERIFICATION RECORD -- copied unchanged into code/audits/ as a durable audit
# artefact.  Only this header was added.
# origin  : audit of review_packet/astra/math_02.md section 4 -- (I6) reversal-error Gaussian average
#           (session scratchpad 887fb000-.../scratchpad/audit_math02_opus/i6_reversal_mc.py)
#           report: review_packet/astra/audit_math02_sections_2_4_6_2026-09-10.md
# author  : Claude (math_02 sections 2/4/6 auditor)
# date    : 2026-09-10
# seed    : 77007
# command : venv/bin/python3 code/audits/math02_sections_2_4_6/i6_reversal_mc.py
# ---------------------------------------------------------------------------
"""
AUDIT (I6) -- reversal error under isotropic Gaussian encoder initialization.

Claim: with (a0,v0) ~ N(0, sigma^2 I_2), zero-sum readout, m > 0, and the
immediate-stop convention for a0 >= m,
    lim_{M->inf} E_{W0}[Err_reversal] = Phi(m/(2 sigma)).
Supporting claims:
  * test margin at a fitted run = 2 a(T_m) - m  -> 2 a0 - m
  * for a0 < m/2: error is 1 once  M v0^2 > m/(m - 2 a0)   (finite-width suff. cond.)
  * for a0 > m/2: correct at EVERY M (since u_M > 0)
  * a0 >= m: immediate stop, test margin = a0 > 0, correct
  * Gaussian tail bounds  |a0| <= sigma sqrt(2 log(4/rho)),  |v0| >= rho sigma sqrt(pi/2)/2
    each fail with prob <= rho/2.

Monte-Carlo over (a0, v0) at several M, plus a check of the exact finite-width
sufficient condition on a grid.

Seed: 77007
"""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

rng = np.random.default_rng(77007)
FAIL = []


def check(name, cond, info=""):
    if not cond:
        FAIL.append(f"{name}: {info}")
        print(f"  FAIL {name}  {info}")


def u_root(M, a0, v0, m):
    """First-margin displacement u_M solving  a0 + u + sqrt(M) v0^2 sinh(2 sqrt(M) u)/2 = m."""
    D = m - a0
    hi = D / (1 + M * v0**2)
    if hi <= 0:
        return 0.0
    f = lambda u: a0 + u + np.sqrt(M) * v0**2 / 2 * np.sinh(2 * np.sqrt(M) * u) - m
    return brentq(f, 0.0, hi, xtol=1e-300, rtol=8.9e-16)


def reversal_err(M, a0, v0, m):
    """1 if the reversal test margin 2 a(T_m) - m is <= 0, else 0.
       Immediate-stop convention for a0 >= m (then a = a0, b = 0, test margin a0).

       EXACT sign test, no root finding: q(u) = a0 + u + sqrt(M) v0^2 sinh(2 sqrt(M) u)/2
       is strictly increasing, so with u* = m/2 - a0,
           u_M <= u*   <=>   q(u*) >= m   <=>   sqrt(M) v0^2 sinh(2 sqrt(M) u*) >= m.
       Evaluated in log space when the sinh argument is large (avoids overflow)."""
    if a0 >= m:
        return 0.0                              # a0 >= m > 0: test margin a0 > 0
    ustar = m / 2.0 - a0
    if ustar <= 0.0:                            # a0 >= m/2  ->  u_M > 0 > u*, correct
        return 0.0
    arg = 2.0 * np.sqrt(M) * ustar
    lhs_log = 0.5 * np.log(M) + 2 * np.log(abs(v0))
    if arg > 20.0:                              # sinh(arg) = e^arg/2 (rel err < 1e-17)
        lhs_log += arg - np.log(2.0)
    else:
        lhs_log += np.log(np.sinh(arg))
    return 1.0 if lhs_log >= np.log(m) else 0.0


sigma = 1.0
print("=" * 84)
print("(I6) Monte-Carlo reversal error vs Phi(m/(2 sigma))")
print("=" * 84)
print(f"{'m':>5} {'sigma':>6} {'M':>10} {'draws':>8} {'MC err':>10} "
      f"{'Phi(m/2s)':>10} {'diff':>10} {'2 s.e.':>9}")
NDRAW = 40000
for m in [0.5, 2.0, 5.0]:
    tgt = norm.cdf(m / (2 * sigma))
    A0 = rng.normal(scale=sigma, size=NDRAW)
    V0 = rng.normal(scale=sigma, size=NDRAW)
    for M in [1, 10**2, 10**4, 10**8, 10**14]:
        e = np.array([reversal_err(M, a, v, m) for a, v in zip(A0, V0)])
        mc = e.mean()
        se = np.sqrt(mc * (1 - mc) / NDRAW)
        print(f"{m:5.1f} {sigma:6.1f} {M:10.0e} {NDRAW:8d} {mc:10.5f} {tgt:10.5f} "
              f"{mc-tgt:+10.5f} {2*se:9.5f}")
    check("(I6) limit = Phi(m/(2 sigma))", abs(mc - tgt) <= 3 * se + 2e-4,
          f"m={m} mc={mc} tgt={tgt} se={se}")

# ---- monotone approach: the MC error should increase to Phi(m/(2 sigma)) in M
print()
print("(I6) the limit indicator is 1{a0 < m/2}: check per-draw agreement at huge M")
m, M = 2.0, 10**16
A0 = rng.normal(scale=sigma, size=20000)
V0 = rng.normal(scale=sigma, size=20000)
e = np.array([reversal_err(M, a, v, m) for a, v in zip(A0, V0)])
ind = (A0 < m / 2).astype(float)
dis = float(np.mean(e != ind))
print(f"  per-draw disagreement with 1{{a0<m/2}} at M=1e16: {dis:.6f} "
      f"({int(dis*20000)} of 20000)")
check("(I6) per-draw -> 1{a0<m/2}", dis < 5e-3, f"dis={dis}")

# ---- exact finite-width sufficient condition  M v0^2 > m/(m-2 a0)
print()
print("(I6) finite-width sufficient condition  M v0^2 > m/(m - 2 a0)  =>  error 1")
bad = 0
tested = 0
for m in [0.7, 2.0, 4.0]:
    for a0 in np.linspace(-3.0, m / 2 - 1e-3, 60):
        for v0 in [0.05, 0.2, 0.7, 1.5]:
            need = m / (m - 2 * a0)
            Mstar = int(np.ceil(need / v0**2)) + 1
            for M in [Mstar, 2 * Mstar, 10 * Mstar]:
                tested += 1
                if reversal_err(M, a0, v0, m) != 1.0:
                    bad += 1
print(f"  {tested} (m,a0,v0,M) combinations with M v0^2 > m/(m-2a0): "
      f"{bad} failed to give error 1")
check("(I6) sufficient condition", bad == 0, f"{bad}/{tested}")

# a0 > m/2  ->  correct at EVERY M
bad2, tested2 = 0, 0
for m in [0.7, 2.0, 4.0]:
    for a0 in np.linspace(m / 2 + 1e-3, m - 1e-3, 40):
        for v0 in [0.02, 0.3, 2.0]:
            for M in [1, 3, 17, 1000, 10**7]:
                tested2 += 1
                if reversal_err(M, a0, v0, m) != 0.0:
                    bad2 += 1
print(f"  a0 in (m/2, m): {tested2} combinations, {bad2} misclassified "
      f"(expect 0 at EVERY M, since u_M > 0)")
check("(I6) a0>m/2 correct at every M", bad2 == 0, f"{bad2}/{tested2}")

# ---- Gaussian tail bounds
print()
print("Gaussian tail bounds for (a0, v0) (each should fail with prob <= rho/2)")
NT = 4_000_000
A = rng.normal(scale=sigma, size=NT)
V = rng.normal(scale=sigma, size=NT)
print(f"{'rho':>7} {'|a0| bnd':>10} {'MC P(>)':>10} {'EXACT P(>)':>12} {'rho/2':>8} | "
      f"{'|v0| bnd':>10} {'MC P(<)':>10} {'EXACT P(<)':>12} {'rho/2':>8}")
for rho in [0.5, 0.2, 0.05, 0.01]:
    ta = sigma * np.sqrt(2 * np.log(4 / rho))
    tv = rho * sigma * np.sqrt(np.pi / 2) / 2
    pa = float(np.mean(np.abs(A) > ta))
    pv = float(np.mean(np.abs(V) < tv))
    pa_ex = 2 * norm.sf(ta / sigma)
    pv_ex = norm.cdf(tv / sigma) - norm.cdf(-tv / sigma)
    print(f"{rho:7.3f} {ta:10.5f} {pa:10.6f} {pa_ex:12.8f} {rho/2:8.4f} | "
          f"{tv:10.5f} {pv:10.6f} {pv_ex:12.8f} {rho/2:8.4f}")
    # EXACT checks are the mathematical claim; MC checks allow 4 s.e. of noise.
    check("|a0| tail EXACT", pa_ex <= rho / 2 + 1e-15, f"rho={rho} {pa_ex}")
    check("|v0| anti-concentration EXACT", pv_ex <= rho / 2 + 1e-15, f"rho={rho} {pv_ex}")
    se_a = np.sqrt(max(pa, 1e-9) * (1 - pa) / len(A))
    se_v = np.sqrt(max(pv, 1e-9) * (1 - pv) / len(V))
    check("|a0| tail MC", pa <= rho / 2 + 4 * se_a, f"rho={rho} pa={pa}")
    check("|v0| anti-concentration MC", pv <= rho / 2 + 4 * se_v,
          f"rho={rho} pv={pv} (excess {(pv-rho/2)/se_v:.2f} s.e.)")
print("  (the |v0| bound is EXACTLY tight to first order: 2 tv phi(0) = rho/2, so the")
print("   true probability sits just below rho/2 and MC noise can straddle it.)")

# ---- rho-uniformity of the (I2) limit needs M v0^4, i.e. rho^{-4}
print()
print("rho-uniform substitution: |v0| >= rho sigma sqrt(pi/2)/2 gives")
print("   1/(M v0^2) ~ rho^-2/M   BUT   (2/3)Delta^2/(M v0^4) ~ rho^-4/M")
for rho in [0.5, 0.1, 0.05]:
    tv = rho * sigma * np.sqrt(np.pi / 2) / 2
    D = 3.0
    print(f"   rho={rho:5.2f}: |v0|>={tv:.5f}; M needed for eps<=0.01 is "
          f"M >= {(1/tv**2 + (2/3)*D**2/tv**4)/0.01:.4g}")

print()
print("FAILURES:", FAIL if FAIL else "none")
