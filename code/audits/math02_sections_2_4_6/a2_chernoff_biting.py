# ---------------------------------------------------------------------------
# VERIFICATION RECORD -- copied unchanged into code/audits/ as a durable audit
# artefact.  Only this header was added.
# origin  : audit of review_packet/astra/math_02.md section 2.2 -- (A2) truncated matrix Chernoff, second parameterisation
#           (session scratchpad 887fb000-.../scratchpad/audit_math02_opus/a2_chernoff_biting.py)
#           report: review_packet/astra/audit_math02_sections_2_4_6_2026-09-10.md
# author  : Claude (math_02 sections 2/4/6 auditor)
# date    : 2026-09-10
# seed    : 909090
# command : venv/bin/python3 code/audits/math02_sections_2_4_6/a2_chernoff_biting.py
# ---------------------------------------------------------------------------
"""
AUDIT (A2), part 2 -- a regime where the Chernoff bound is NON-VACUOUS, so the
inequality is actually tested rather than trivially satisfied.

Small N and a large bias variance push kappa_* close to its cap s_max^2/(2N),
which makes R/kappa_* small enough that N exp[-M kappa_*/(16R)] < 1 at
simulable M.  We then compare the empirical failure frequency of
   lambda_min(sum_i X_i) < M kappa_*/4
to Astra's bound and to Tropp's exact delta=1/2 lower tail.

Seed: 909090
"""
import numpy as np

rng = np.random.default_rng(909090)
FAIL = []


def check(name, cond, info=""):
    if not cond:
        FAIL.append(f"{name}: {info}")
        print(f"  FAIL {name}  {info}")


N, K, s_w, s_b = 2, 1, 0.05, 1.0     # nearly pure-bias ReLU features
Z = np.array([[1.0], [-1.0]])
R_z = 1.0
s_max2 = s_w**2 * R_z**2 / K + s_b**2
s_max = np.sqrt(s_max2)


def draw(n):
    W = rng.normal(size=(n, K)) * (s_w / np.sqrt(K))
    b = rng.normal(size=(n, 1)) * s_b
    G = W @ Z.T + b
    return np.maximum(G, 0.0)


NMC = 8_000_000
Psi = draw(NMC)
Kstar = (Psi.T @ Psi) / (NMC * N)
kap = float(np.linalg.eigvalsh(Kstar)[0])
nx = (Psi**2).sum(1) / N
print(f"N={N} K={K} s_w={s_w} s_b={s_b}")
print(f"s_max^2 = {s_max2:.6f};  kappa_* = {kap:.6e};  cap s_max^2/(2N) = {s_max2/(2*N):.6e}"
      f"  (ratio to cap {kap/(s_max2/(2*N)):.3f})")

R = s_max2 * 0.5
while 2 * N * (R + 2 * s_max2) * np.exp(-R / (2 * s_max2)) > kap / 2:
    R *= 1.001
print(f"chosen R = {R:.6f} = {R/s_max2:.3f} s_max^2;  "
      f"tail bound = {2*N*(R+2*s_max2)*np.exp(-R/(2*s_max2)):.4e} <= kappa_*/2 = {kap/2:.4e}")

emp_tail = float(np.mean(nx * (nx > R)))
bnd_tail = 2 * N * (R + 2 * s_max2) * np.exp(-R / (2 * s_max2))
print(f"MC E[||X|| 1_{{>R}}] = {emp_tail:.4e}  <= bound {bnd_tail:.4e}  "
      f"(slack {bnd_tail/max(emp_tail,1e-300):.1f}x)")
check("tail bound", emp_tail <= bnd_tail)

mask = nx <= R
Ktr = (Psi[mask].T @ Psi[mask]) / (NMC * N)
print(f"lambda_min(E[X 1_R]) = {np.linalg.eigvalsh(Ktr)[0]:.6e} >= kappa_*/2 = {kap/2:.6e}")
check("truncated mean", np.linalg.eigvalsh(Ktr)[0] >= kap / 2)

c_tropp = -np.log(np.exp(-0.5) / np.sqrt(0.5))
print()
print(f"{'M':>8} {'trials':>8} {'emp fail':>12} {'Astra bnd':>12} {'Tropp exact':>12}"
      f" {'emp<=Astra':>11}")
for M in [40, 80, 160, 320, 640, 1280]:
    T = 20000
    lam = np.empty(T)
    for j in range(T):
        P2 = draw(M)
        lam[j] = np.linalg.eigvalsh(P2.T @ P2 / N)[0]
    emp = float(np.mean(lam < M * kap / 4))
    astra = N * np.exp(-M * kap / (16 * R))
    tropp = N * np.exp(-c_tropp * (M * kap / 2) / R)
    ok = emp <= astra
    print(f"{M:8d} {T:8d} {emp:12.5f} {astra:12.5f} {tropp:12.5f} {str(ok):>11}")
    check("Chernoff holds", emp <= astra + 4 * np.sqrt(max(emp, 1e-6) / T),
          f"M={M} emp={emp} > {astra}")
    check("Astra conservative", astra >= tropp)

print()
print("FAILURES:", FAIL if FAIL else "none")
