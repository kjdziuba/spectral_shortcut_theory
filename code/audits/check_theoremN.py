# ---------------------------------------------------------------------------
# VERIFICATION RECORD -- copied unchanged into code/audits/ as a durable audit
# artefact.  Only this header was added.
# origin  : Theorem N audit of review_packet/astra/math_02.md section 3
#           (session scratchpad 887fb000-.../scratchpad/check_theoremN.py)
# author  : Claude (Theorem N auditor)
# date    : 2026-09-10
# seed    : numpy default_rng(0), plus per-block default_rng(seed) draws
# command : venv/bin/python3 code/audits/check_theoremN.py
# ---------------------------------------------------------------------------
"""Numerical cross-checks for the audit of Theorem N (math_02.md section 3).

Checks (all deterministic-algebra or Monte-Carlo, none rely on the huge M thresholds):
  1. Z inequality: sqrt(((||Th0||+1)X)^2+1) <= Z0 + X.
  2. Three-term decomposition identity V g W - V0 g0 W0 = (V-V0) g W0 + V g (W-W0) + V0 (g-g0) W0.
  3. Encoder-Jacobian perturbation bound with the *actual* per-unit preactivation-change radius tau_i
     (gate-flip units are inside the boundary set; bound X[|dV||W0| + |V||dW| + B_M] holds).
  4. Feature perturbation: ||H-H0||_F <= X||W0||_F||dTheta||_F + Z||d omega||.
  5. Preactivation change <= X||w_i0|| r_theta + Z ||d omega_i||.
  6. Moment bounds (N5): E||H0||_F^2/M <= s_max^2/2, E||W0||_F^2/M = s_w^2, E||V0||_F^2 = C s_v^2,
     E||J_Theta,0||_F^2 <= C s_v^2 s_w^2 X^2, E||e0||^2 <= C s_v^2 s_max^2 + 2E||beta0||^2 + 2||y||^2.
  7. Boundary mass (N7): E B_M <= C_bd for a *given* tau_i(w_i0) = (X||w_i0|| + Z R_phi)/sqrt(M).
  8. E[ReLU(g)^4] = 3 s^4 / 2, and the Frobenius/Chebyshev threshold constant 42.
  9. (A1) on random noncommuting PSD kernels along the *nonlinear* flow of the actual model (small instance).
"""
import numpy as np

rng = np.random.default_rng(0)

def relu(u):
    return np.maximum(u, 0.0)

# ---------------------------------------------------------------- 1. Z inequality
worst = -np.inf
for _ in range(20000):
    a = rng.exponential(1.0) * rng.choice([0.01, 1, 10])   # a = ||Theta0||_op X >= 0
    X = rng.exponential(1.0) * rng.choice([0.01, 1, 10])
    lhs = np.sqrt((a + X) ** 2 + 1)
    rhs = np.sqrt(1 + a ** 2) + X
    worst = max(worst, lhs - rhs)
print(f"[1] Z inequality: max(lhs-rhs) = {worst:.3e}  (must be <= 0)")

# ---------------------------------------------------------------- model helpers
def make_instance(N=7, D=5, K=4, C=3, M=64, s_w=1.3, s_b=0.7, s_v=0.9, seed=None):
    r = np.random.default_rng(seed)
    x = r.normal(size=(N, D))
    y = r.normal(size=(N, C))
    Th0 = r.normal(size=(K, D)) / np.sqrt(D)
    W0 = r.normal(size=(M, K)) * s_w / np.sqrt(K)
    b0 = r.normal(size=M) * s_b
    V0 = r.normal(size=(C, M)) * s_v / np.sqrt(M)
    beta0 = r.normal(size=C)
    return dict(x=x, y=y, Th0=Th0, W0=W0, b0=b0, V0=V0, beta0=beta0, N=N, D=D, K=K, C=C, M=M,
                s_w=s_w, s_b=s_b, s_v=s_v)

def preact(Th, W, b, x):
    # (N, M)
    return x @ Th.T @ W.T + b[None, :]

def jac_theta(Th, W, b, V, x, gates=None):
    """Normalized encoder Jacobian as an (N*C) x (K*D) matrix. gates: (N,M) in [0,1] or None (=ReLU')."""
    N = x.shape[0]
    G = preact(Th, W, b, x)
    if gates is None:
        gates = (G > 0).astype(float)
    C, M = V.shape
    K, D = Th.shape
    J = np.zeros((N, C, K, D))
    for n in range(N):
        # V diag(gates_n) W : (C,K); then outer with x_n
        A = (V * gates[n][None, :]) @ W          # (C,K)
        J[n] = A[:, :, None] * x[n][None, None, :]
    return J.reshape(N * C, K * D) / np.sqrt(N)

def features(Th, W, b, x):
    N = x.shape[0]
    return relu(preact(Th, W, b, x)) / np.sqrt(N)

def jac_omega(Th, W, b, V, x, gates=None):
    """Normalized (W,b) Jacobian: (N*C) x (M*(K+1))."""
    N = x.shape[0]
    G = preact(Th, W, b, x)
    if gates is None:
        gates = (G > 0).astype(float)
    C, M = V.shape
    K = W.shape[1]
    xi = np.concatenate([x @ Th.T, np.ones((N, 1))], axis=1)   # (N, K+1)
    J = np.zeros((N, C, M, K + 1))
    for n in range(N):
        J[n] = (V * gates[n][None, :])[:, :, None] * xi[n][None, None, :]
    return J.reshape(N * C, M * (K + 1)) / np.sqrt(N)

# ---------------------------------------------------------------- 2. decomposition identity
inst = make_instance(seed=1)
x, Th0, W0, b0, V0 = inst["x"], inst["Th0"], inst["W0"], inst["b0"], inst["V0"]
g0 = (preact(Th0, W0, b0, x) > 0).astype(float)
worst = 0.0
for _ in range(200):
    V = V0 + 0.3 * rng.normal(size=V0.shape)
    W = W0 + 0.3 * rng.normal(size=W0.shape)
    g = rng.uniform(size=g0.shape)  # arbitrary gates in [0,1]
    for n in range(x.shape[0]):
        lhs = (V * g[n]) @ W - (V0 * g0[n]) @ W0
        rhs = ((V - V0) * g[n]) @ W0 + (V * g[n]) @ (W - W0) + (V0 * (g[n] - g0[n])) @ W0
        worst = max(worst, np.abs(lhs - rhs).max())
print(f"[2] decomposition identity: max abs error = {worst:.3e}")

# ---------------------------------------------------------------- 3,4,5. perturbation bounds with actual radii
def perturbation_checks(inst, r_theta, r_phi, trials=300, allow_zero_gate_selection=True):
    x, y, Th0, W0, b0, V0 = (inst[k] for k in ("x", "y", "Th0", "W0", "b0", "V0"))
    N, C, M, K = inst["N"], inst["C"], inst["M"], inst["K"]
    X = np.linalg.norm(x, axis=1).max()
    Th0_op = np.linalg.norm(Th0, 2)
    Z0 = np.sqrt(1 + (Th0_op * X) ** 2)
    Z = Z0 + X                                  # valid if r_theta <= 1
    assert r_theta <= 1.0
    G0 = preact(Th0, W0, b0, x)
    gates0 = (G0 > 0).astype(float)
    J0 = jac_theta(Th0, W0, b0, V0, x)
    H0 = features(Th0, W0, b0, x)
    wnorm0 = np.linalg.norm(W0, axis=1)
    vnorm0 = np.linalg.norm(V0, axis=0)
    # actual per-unit preactivation-change radius used in the proof (with r_theta, r_phi in place of R/M, R/sqrt M)
    tau = X * wnorm0 * r_theta + Z * r_phi
    boundary = np.abs(G0).min(axis=0) <= tau
    B_M = np.sum(vnorm0 * wnorm0 * boundary)
    viol = dict(gateflip_outside_boundary=0, jac=0.0, feat=0.0, preact=0.0)
    for _ in range(trials):
        dTh = rng.normal(size=Th0.shape); dTh *= r_theta * rng.uniform() / np.linalg.norm(dTh)
        dphi = rng.normal(size=M * (K + 1) + C * M + C); dphi *= r_phi * rng.uniform() / np.linalg.norm(dphi)
        dW = dphi[:M * K].reshape(M, K); db = dphi[M * K:M * (K + 1)]
        dV = dphi[M * (K + 1):M * (K + 1) + C * M].reshape(C, M)
        Th, W, b, V = Th0 + dTh, W0 + dW, b0 + db, V0 + dV
        G = preact(Th, W, b, x)
        # 5. preactivation change bound
        domega = np.sqrt(np.sum(dW ** 2, axis=1) + db ** 2)           # per unit
        bound5 = X * wnorm0 * r_theta + Z * domega                     # (M,)
        viol["preact"] = max(viol["preact"], (np.abs(G - G0).max(axis=0) - bound5).max())
        # gates: ReLU' plus optionally a random selection where |G| is tiny (simulate zero-preactivation selection)
        gates = (G > 0).astype(float)
        if allow_zero_gate_selection:
            tiny = np.abs(G) < 1e-3
            gates[tiny] = rng.uniform(size=tiny.sum())
        changed_units = np.any(gates != gates0, axis=0)
        viol["gateflip_outside_boundary"] += int(np.any(changed_units & ~boundary))
        # 3. Jacobian perturbation bound
        J = jac_theta(Th, W, b, V, x, gates=gates)
        lhs = np.linalg.norm(J - J0, 2)
        rhs = X * (np.linalg.norm(dV) * np.linalg.norm(W0) + np.linalg.norm(V) * np.linalg.norm(dW) + B_M)
        viol["jac"] = max(viol["jac"], lhs - rhs)
        # 4. feature perturbation bound
        H = features(Th, W, b, x)
        lhs = np.linalg.norm(H - H0)
        rhs = X * np.linalg.norm(W0) * np.linalg.norm(dTh) + Z * np.linalg.norm(dphi[:M * (K + 1)])
        viol["feat"] = max(viol["feat"], lhs - rhs)
    return viol, B_M, boundary.sum()

for seed, (rt, rp) in zip([2, 3, 4], [(0.05, 0.2), (0.5, 0.5), (1.0, 1.0)]):
    inst = make_instance(seed=seed, M=64)
    viol, B_M, nb = perturbation_checks(inst, rt, rp)
    print(f"[3-5] seed={seed} r_theta={rt} r_phi={rp}: gate-flip-outside-boundary count={viol['gateflip_outside_boundary']}, "
          f"max(jac lhs-rhs)={viol['jac']:.3e}, max(feat lhs-rhs)={viol['feat']:.3e}, "
          f"max(preact lhs-rhs)={viol['preact']:.3e}; boundary units={nb}/{inst['M']}, B_M={B_M:.3f}")

# ---------------------------------------------------------------- 6. moment bounds (N5), Monte Carlo
def moments(N=6, D=4, K=3, C=2, M=128, s_w=1.1, s_b=0.6, s_v=0.8, reps=4000, seed=5):
    r = np.random.default_rng(seed)
    x = r.normal(size=(N, D)); y = r.normal(size=(N, C)); Th0 = r.normal(size=(K, D)) / np.sqrt(D)
    X = np.linalg.norm(x, axis=1).max()
    smax2 = s_w ** 2 * np.max(np.sum((x @ Th0.T) ** 2, axis=1)) / K + s_b ** 2
    ynorm2 = np.sum(y ** 2) / N
    acc = dict(H=0.0, W=0.0, V=0.0, J=0.0, e=0.0, beta=0.0)
    for _ in range(reps):
        W0 = r.normal(size=(M, K)) * s_w / np.sqrt(K); b0 = r.normal(size=M) * s_b
        V0 = r.normal(size=(C, M)) * s_v / np.sqrt(M); beta0 = r.normal(size=C) * 0.5
        H0 = features(Th0, W0, b0, x)
        acc["H"] += np.sum(H0 ** 2); acc["W"] += np.sum(W0 ** 2); acc["V"] += np.sum(V0 ** 2)
        acc["J"] += np.sum(jac_theta(Th0, W0, b0, V0, x) ** 2)
        f = relu(preact(Th0, W0, b0, x)) @ V0.T + beta0[None, :]
        acc["e"] += np.sum((f - y) ** 2) / N
        acc["beta"] += np.sum(beta0 ** 2)
    for k in acc: acc[k] /= reps
    print(f"[6] E||H0||_F^2/M = {acc['H']/M:.4f} <= s_max^2/2 = {smax2/2:.4f}")
    print(f"[6] E||W0||_F^2/M = {acc['W']/M:.4f}  vs s_w^2 = {s_w**2:.4f}")
    print(f"[6] E||V0||_F^2   = {acc['V']:.4f}  vs C s_v^2 = {C*s_v**2:.4f}")
    print(f"[6] E||J_Theta,0||_F^2 = {acc['J']:.4f} <= C s_v^2 s_w^2 X^2 = {C*s_v**2*s_w**2*X**2:.4f} "
          f"(tight form C s_v^2 s_w^2 tr Sigma_X = {C*s_v**2*s_w**2*np.mean(np.sum(x**2,axis=1)):.4f})")
    print(f"[6] E||e0||^2 = {acc['e']:.4f} <= C s_v^2 s_max^2 + 2E||beta||^2 + 2||y||^2 = "
          f"{C*s_v**2*smax2 + 2*acc['beta'] + 2*ynorm2:.4f}")
moments()

# ---------------------------------------------------------------- 7. boundary mass (N7)
def boundary_mass(N=6, D=4, K=3, C=2, M=256, s_w=1.1, s_b=0.6, s_v=0.8, R_phi=3.0, reps=3000, seed=6):
    r = np.random.default_rng(seed)
    x = r.normal(size=(N, D)); Th0 = r.normal(size=(K, D)) / np.sqrt(D)
    X = np.linalg.norm(x, axis=1).max()
    Z = np.sqrt(1 + (np.linalg.norm(Th0, 2) * X) ** 2) + X
    acc = 0.0
    for _ in range(reps):
        W0 = r.normal(size=(M, K)) * s_w / np.sqrt(K); b0 = r.normal(size=M) * s_b
        V0 = r.normal(size=(C, M)) * s_v / np.sqrt(M)
        wn = np.linalg.norm(W0, axis=1); vn = np.linalg.norm(V0, axis=0)
        tau = (X * wn + Z * R_phi) / np.sqrt(M)
        G0 = preact(Th0, W0, b0, x)
        bd = np.abs(G0).min(axis=0) <= tau
        acc += np.sum(vn * wn * bd)
    EB = acc / reps
    C_bd = 2 * N * s_v * np.sqrt(C) / (s_b * np.sqrt(2 * np.pi)) * (X * s_w ** 2 + Z * R_phi * s_w)
    print(f"[7] M={M}: E B_M (MC) = {EB:.4f} <= C_bd = {C_bd:.4f}")
boundary_mass(M=256); boundary_mass(M=1024)

# ---------------------------------------------------------------- 8. fourth moment & threshold constant
g = rng.normal(size=4_000_000) * 1.7
print(f"[8] E[ReLU(g)^4]/s^4 = {np.mean(relu(g)**4)/1.7**4:.4f} (theory 1.5); E[ReLU(g)^2]/s^2 = {np.mean(relu(g)**2)/1.7**2:.4f} (theory 0.5)")
print(f"[8] Frobenius: E||.||_F^2 <= 3 s^4/(2M); Markov at kappa*/2 -> 6 s^4/(M kappa*^2); = rho/7 -> M >= 42 s^4/(rho kappa*^2). 6*7 = {6*7}")

# ---------------------------------------------------------------- 9. (A1) along the actual nonlinear flow (tiny instance, explicit Euler)
def a1_check(seed=7, N=5, D=3, K=2, C=2, M=40, steps=20000, dt=2e-4):
    inst = make_instance(N=N, D=D, K=K, C=C, M=M, seed=seed)
    x, y = inst["x"], inst["y"]
    Th, W, b, V, beta = (inst[k].copy() for k in ("Th0", "W0", "b0", "V0", "beta0"))
    def resid(Th, W, b, V, beta):
        return (relu(preact(Th, W, b, x)) @ V.T + beta[None, :] - y) / np.sqrt(N)   # (N,C)
    L0 = 0.5 * np.sum(resid(Th, W, b, V, beta) ** 2)
    ratio_max = 0.0; int_theta = 0.0; t = 0.0
    a_max = 0.0; kappa_min = np.inf
    for s in range(steps):
        e = resid(Th, W, b, V, beta).reshape(-1)
        JT = jac_theta(Th, W, b, V, x); Jw = jac_omega(Th, W, b, V, x)
        H = features(Th, W, b, x)
        gT = JT.T @ e
        gw = Jw.T @ e
        gV = (H.T @ e.reshape(N, C)).T          # dL/dV = H^T E  -> (C,M)
        gbeta = e.reshape(N, C).sum(axis=0) / np.sqrt(N)
        # pointwise kernel quadratic forms
        qT = float(e @ (JT @ (JT.T @ e))); qphi = float(gw @ gw + np.sum(gV ** 2) + np.sum(gbeta ** 2))
        ne2 = float(e @ e)
        if ne2 > 1e-14:
            a_max = max(a_max, qT / ne2); kappa_min = min(kappa_min, qphi / ne2)
        int_theta += dt * qT
        Th -= dt * gT.reshape(Th.shape); W -= dt * gw[:M * K].reshape(M, K); b -= dt * gw[M * K:]
        V -= dt * gV; beta -= dt * gbeta
        t += dt
        if s % 2000 == 0 and s > 0:
            L = 0.5 * np.sum(resid(Th, W, b, V, beta) ** 2)
            share = int_theta / (L0 - L)
            ratio_max = max(ratio_max, share)
    L = 0.5 * np.sum(resid(Th, W, b, V, beta) ** 2)
    share = int_theta / (L0 - L)
    print(f"[9] (A1) nonlinear flow: encoder share = {share:.4f}, max over horizons = {ratio_max:.4f}, "
          f"a/(a+kappa) with a=max q_T/|e|^2={a_max:.3f}, kappa=min q_phi/|e|^2={kappa_min:.3f} -> {a_max/(a_max+kappa_min):.4f}; "
          f"L0={L0:.3f}, L(T)={L:.2e}")
a1_check()
