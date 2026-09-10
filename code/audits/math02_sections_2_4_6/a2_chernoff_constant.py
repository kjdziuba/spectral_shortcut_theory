# ---------------------------------------------------------------------------
# VERIFICATION RECORD -- copied unchanged into code/audits/ as a durable audit
# artefact.  Only this header was added.
# origin  : audit of review_packet/astra/math_02.md section 2.2 -- (A2) constant stress on the extremal family
#           (session scratchpad 887fb000-.../scratchpad/audit_math02_opus/a2_chernoff_constant.py)
#           report: review_packet/astra/audit_math02_sections_2_4_6_2026-09-10.md
# author  : Claude (math_02 sections 2/4/6 auditor)
# date    : 2026-09-10
# seed    : 5150
# command : venv/bin/python3 code/audits/math02_sections_2_4_6/a2_chernoff_constant.py
# ---------------------------------------------------------------------------
"""
AUDIT (A2), part 3 -- stress the CHERNOFF CONSTANT itself.

In the ReLU instance the bound is numerically vacuous (kappa_* is tiny), so the
inequality is never actually tested there.  The load-bearing step in A2 is the
bounded-PSD matrix Chernoff lower tail with delta = 1/2.  Here we test it on the
EXTREMAL family for that inequality -- independent diagonal spikes
   Y_i = R e_{J_i} e_{J_i}^T,  J_i ~ Unif{1..N}
so lambda_min(sum Y_i) = R * min_j (bin count) and mu_min = M R / N.  The
inequality is then a balls-in-bins lower tail, where matrix Chernoff is tight up
to the dimension factor.

We check, for delta = 1/2:
   P{lambda_min(sum Y_i) <= mu_min/2} <= N (e^{-d}/(1-d)^{1-d})^{mu_min/R}   [Tropp, exact]
                                      <= N exp(-mu_min/(8R))                 [simplified]
and that Astra's chain (mu_min >= M kappa/2  =>  N exp(-M kappa/(16 R))) is the
simplified form, hence valid but conservative by 0.153426/2 / (1/16) = 1.227x.

Seed: 5150
"""
import numpy as np

rng = np.random.default_rng(5150)
FAIL = []


def check(name, cond, info=""):
    if not cond:
        FAIL.append(f"{name}: {info}")
        print(f"  FAIL {name}  {info}")


c_exact = -np.log(np.exp(-0.5) / np.sqrt(0.5))     # 0.1534264...
print(f"Tropp delta=1/2 exponent constant  -log(e^-.5 / .5^.5) = {c_exact:.7f}")
print(f"simplified delta^2/2 = 0.125  <=  {c_exact:.7f} : "
      f"{'simplified is valid (smaller exponent)' if 0.125 <= c_exact else 'INVALID'}")
check("simplified <= exact", 0.125 <= c_exact)
print()
print(f"{'N':>4} {'M':>6} {'mu/R':>7} {'trials':>8} {'emp P':>10} "
      f"{'Tropp exact':>12} {'simplified':>12} {'tight?':>8}")
for N, M in [(4, 20), (4, 40), (4, 80), (8, 40), (8, 80), (8, 160), (16, 160), (16, 320)]:
    T = 200000
    J = rng.integers(0, N, size=(T, M))
    counts = np.zeros((T, N), dtype=np.int32)
    for j in range(N):
        counts[:, j] = (J == j).sum(1)
    lam = counts.min(1)                       # lambda_min(sum Y_i)/R
    mu_over_R = M / N                          # mu_min/R
    emp = float(np.mean(lam <= mu_over_R / 2))
    tropp = N * np.exp(-c_exact * mu_over_R)
    simp = N * np.exp(-0.125 * mu_over_R)
    print(f"{N:4d} {M:6d} {mu_over_R:7.1f} {T:8d} {emp:10.5f} {tropp:12.5f} "
          f"{simp:12.5f} {emp/max(tropp,1e-300):8.3f}")
    check("Tropp exact holds", emp <= tropp + 4 * np.sqrt(max(emp, 1e-7) / T),
          f"N={N} M={M} emp={emp} > {tropp}")
    check("simplified holds", emp <= simp + 4 * np.sqrt(max(emp, 1e-7) / T),
          f"N={N} M={M} emp={emp} > {simp}")
    check("simplified >= exact", simp >= tropp)

print()
print("Interpretation: the 'tight?' column is empirical/Tropp-exact.  Values in the")
print("0.2-0.8 range on the extremal family show the constant is the right order,")
print("i.e. Astra's 1/16 is not an artefact but the standard simplification.")
print()
print("FAILURES:", FAIL if FAIL else "none")
