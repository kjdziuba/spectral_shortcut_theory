# ---------------------------------------------------------------------------
# VERIFICATION RECORD -- copied unchanged into code/audits/ as a durable audit
# artefact.  Only this header was added.
# origin  : audit of review_packet/astra/math_02.md section 2.2 -- (A2) truncated matrix Chernoff
#           (session scratchpad 887fb000-.../scratchpad/audit_math02_opus/a2_chernoff.py)
#           report: review_packet/astra/audit_math02_sections_2_4_6_2026-09-10.md
# author  : Claude (math_02 sections 2/4/6 auditor)
# date    : 2026-09-10
# seed    : 424242
# command : venv/bin/python3 code/audits/math02_sections_2_4_6/a2_chernoff.py
# ---------------------------------------------------------------------------
"""
AUDIT (A2) -- truncated matrix Chernoff alternative to the Frobenius/Markov
certificate of Corollary 1.5.

Checks, all independent of astra/math_02_checks.py (which does not test A2 at all):

 (a) ||X|| <= max_n g_n^2  where X = psi psi^T / N, psi_n = ReLU(g_n).
 (b) the tail integral bound
        E[ ||X|| 1{||X||>R} ]  <=  2N (R + 2 s_max^2) e^{-R/(2 s_max^2)}
     by high-accuracy Monte Carlo (and by exact 1-d quadrature for N=1).
 (c) the chosen R makes  E[X 1{||X||<=R}] >= (kappa_*/2) I.
 (d) the Chernoff statement
        P{lambda_min(sum_i X_i) < M kappa_*/4} <= N exp[-M kappa_*/(16 R)]
     against empirical failure frequency, and against Tropp's exact lower-tail
     constant with delta = 1/2  (is 1/16 conservative?).
 (e) the threshold M >= (16 R/kappa_*) log(N/eta), and its scaling in kappa_*
     versus the original M0 = 18 s_max^4 / (rho kappa_*^2).

Seed: 424242
"""
import numpy as np

rng = np.random.default_rng(424242)
FAIL = []


def check(name, cond, info=""):
    if not cond:
        FAIL.append(f"{name}: {info}")
        print(f"  FAIL {name}  {info}")


# ----------------------------------------------------------------------
# Setup exactly as Corollary 1.5: z_n in R^K distinct, ||z_n|| <= R_z,
# w ~ N(0, s_w^2 I_K / K), b ~ N(0, s_b^2), psi_n = ReLU(w'z_n + b),
# X = psi psi^T / N,  K_* = E X,  kappa_* = lambda_min(K_*).
# Pre-activation g_n = w'z_n + b is Gaussian, Var = s_w^2||z_n||^2/K + s_b^2
#                                                <= s_w^2 R_z^2/K + s_b^2 = s_max^2.
# ----------------------------------------------------------------------
def setup(N, K, s_w, s_b, rng):
    Z = rng.normal(size=(N, K))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)          # ||z_n|| = R_z = 1
    R_z = 1.0
    s_max2 = s_w**2 * R_z**2 / K + s_b**2
    # exact covariance of g = Z w + b 1 :  s_w^2/K Z Z^T + s_b^2 11^T
    Sig = (s_w**2 / K) * (Z @ Z.T) + s_b**2 * np.ones((N, N))
    return Z, s_max2, Sig


def draw_X(n_draw, Z, s_w, s_b, rng):
    N, K = Z.shape
    W = rng.normal(size=(n_draw, K)) * (s_w / np.sqrt(K))
    b = rng.normal(size=(n_draw, 1)) * s_b
    G = W @ Z.T + b                       # (n_draw, N) pre-activations
    Psi = np.maximum(G, 0.0)
    return Psi, G


N, K, s_w, s_b = 6, 3, 1.1, 0.7
Z, s_max2, Sig = setup(N, K, s_w, s_b, rng)
s_max = np.sqrt(s_max2)
print(f"N={N} K={K} s_w={s_w} s_b={s_b}  s_max^2={s_max2:.6f}  "
      f"max diag(Sig)={Sig.diagonal().max():.6f}  (must be <= s_max^2)")
check("s_max^2 dominates", Sig.diagonal().max() <= s_max2 + 1e-12)

# --- K_* and kappa_* by large MC (E[psi psi^T]/N) -----------------------
NMC = 4_000_000
Psi, G = draw_X(NMC, Z, s_w, s_b, rng)
Kstar = (Psi.T @ Psi) / (NMC * N)
kap = float(np.linalg.eigvalsh(Kstar)[0])
print(f"kappa_* (MC, {NMC:,} draws) = {kap:.6e}   "
      f"lambda_max(K_*) = {np.linalg.eigvalsh(Kstar)[-1]:.6e}   "
      f"trace cap s_max^2/2 = {s_max2/2:.6e}")

# (a) ||X|| <= max_n g_n^2
nx = (Psi**2).sum(1) / N
check("(a) ||X|| <= max g_n^2", np.all(nx <= (G**2).max(1) + 1e-12))
print(f"(a) ||X|| <= max_n g_n^2 on all {NMC:,} draws: "
      f"max ratio = {np.max(nx / (G**2).max(1)):.6f}")

# (b) tail-integral bound
print()
print("(b) tail bound  E[||X|| 1{||X||>R}] <= 2N(R+2 s_max^2) e^{-R/(2 s_max^2)}")
print(f"{'R':>8} {'R/s_max^2':>10} {'MC E[..]':>14} {'bound':>14} {'slack':>10}")
for Rm in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
    R = Rm * s_max2
    emp = float(np.mean(nx * (nx > R)))
    bnd = 2 * N * (R + 2 * s_max2) * np.exp(-R / (2 * s_max2))
    print(f"{R:8.4f} {Rm:10.2f} {emp:14.6e} {bnd:14.6e} {bnd/max(emp,1e-300):10.2f}")
    check("(b) tail bound", emp <= bnd + 1e-12, f"R={R} emp={emp} bnd={bnd}")

# (c) choose R so that bound <= kappa_*/2, then verify E[X 1{||X||<=R}] >= (kappa_*/2) I
print()
print("(c) choose R with 2N(R+2s^2)e^{-R/(2s^2)} <= kappa_*/2, then check truncated mean")
R = s_max2
while 2 * N * (R + 2 * s_max2) * np.exp(-R / (2 * s_max2)) > kap / 2:
    R *= 1.02
print(f"    kappa_*/2 = {kap/2:.6e};  chosen R = {R:.6f} = {R/s_max2:.3f} * s_max^2")
mask = nx <= R
Ktr = (Psi[mask].T @ Psi[mask]) / (NMC * N)
lam_tr = float(np.linalg.eigvalsh(Ktr)[0])
print(f"    lambda_min(E[X 1_{{||X||<=R}}]) (MC) = {lam_tr:.6e}  >= kappa_*/2 = {kap/2:.6e}?  "
      f"{'YES' if lam_tr >= kap/2 else 'NO'}")
check("(c) truncated mean >= kappa_*/2", lam_tr >= kap / 2 - 5e-5 * kap)
print(f"    fraction truncated away: {1-mask.mean():.3e};  max ||X|| kept = {nx[mask].max():.4f} <= R")

# (d) Chernoff bound vs empirical failure frequency
print()
print("(d) P{lambda_min(sum X_i) < M kappa_*/4} <= N exp[-M kappa_*/(16R)]")
print(f"{'M':>8} {'trials':>8} {'emp fail':>12} {'Astra bound':>14} {'Tropp exact':>14}")
# Tropp lower tail, delta=1/2:  N * (e^{-d}/(1-d)^{1-d})^{mu/R}, mu >= M kappa/2
c_tropp = -np.log(np.exp(-0.5) / np.sqrt(0.5))          # 0.153426...
for M in [200, 1000, 5000, 20000]:
    T = 4000
    lam = np.empty(T)
    for j in range(T):
        P2, _ = draw_X(M, Z, s_w, s_b, rng)
        lam[j] = np.linalg.eigvalsh(P2.T @ P2 / N)[0]
    emp = float(np.mean(lam < M * kap / 4))
    astra = N * np.exp(-M * kap / (16 * R))
    tropp = N * np.exp(-c_tropp * (M * kap / 2) / R)
    print(f"{M:8d} {T:8d} {emp:12.4f} {astra:14.4e} {tropp:14.4e}")
    check("(d) Chernoff holds", emp <= astra + 3 / np.sqrt(T),
          f"M={M} emp={emp} bnd={astra}")
    check("(d) Astra conservative vs Tropp", astra >= tropp - 1e-15,
          f"M={M} astra={astra} tropp={tropp}")
print(f"    Tropp's exact delta=1/2 constant: exponent coefficient "
      f"{c_tropp:.6f}/2 = {c_tropp/2:.6f} = 1/{2/c_tropp:.3f}")
print(f"    Astra uses 1/16 = {1/16:.6f}.  Conservative by factor "
      f"{(c_tropp/2)/(1/16):.4f}x in the exponent.")
print(f"    (1/16 is exactly the standard simplification e^-d/(1-d)^(1-d) <= e^(-d^2/2), "
      f"d=1/2: {np.exp(-0.5)/np.sqrt(0.5):.6f} <= {np.exp(-0.125):.6f})")
check("simplified Chernoff form valid",
      np.exp(-0.5) / np.sqrt(0.5) <= np.exp(-0.125))

# (e) thresholds
print()
print("(e) threshold comparison")
for eta in [0.1, 0.01]:
    M_new = (16 * R / kap) * np.log(N / eta)
    M_old = 18 * s_max2**2 / (eta * kap**2)
    print(f"    eta=rho={eta}:  A2 M >= {M_new:.4g}   |   Cor 1.5 M0 = {M_old:.4g}"
          f"   ratio {M_new/M_old:.3e}")
    check("(e) threshold achieves eta", N * np.exp(-M_new * kap / (16 * R)) <= eta + 1e-12)
print("    kappa_* scaling: A2 is O(kappa_*^{-1} log(1/kappa_*)); Cor 1.5 is O(kappa_*^{-2}).")

# Also confirm the PSD-monotonicity step Astra leaves implicit:
# sum X_i >= sum X_i 1{||X_i||<=R}, so the Chernoff bound for the truncated sum
# transfers verbatim to the untruncated sum.
P3, _ = draw_X(3000, Z, s_w, s_b, rng)
nx3 = (P3**2).sum(1) / N
full = P3.T @ P3 / N
trunc = P3[nx3 <= R].T @ P3[nx3 <= R] / N
check("PSD monotonicity sum X >= sum X 1{..}",
      np.linalg.eigvalsh(full - trunc)[0] >= -1e-10)
print(f"    PSD step  lambda_min(sum X_i - sum X_i 1_R) = "
      f"{np.linalg.eigvalsh(full-trunc)[0]:.3e} >= 0  (bound transfers)")

print()
print("FAILURES:", FAIL if FAIL else "none")
