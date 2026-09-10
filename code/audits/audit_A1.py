# ---------------------------------------------------------------------------
# VERIFICATION RECORD -- copied unchanged into code/audits/ as a durable audit
# artefact.  Only this header was added.
# origin  : session scratchpad 887fb000-.../scratchpad/audit_A1.py
# author  : Claude (session audit of math_01.md)
# date    : 2026-09-09
# seed    : numpy default_rng(0)
# command : venv/bin/python3 code/audits/audit_A1.py
# ---------------------------------------------------------------------------
"""Numerical audit of math_01.md, assignment A1 (P4 + P1 items).
No repo files touched. Every check prints PASS/FAIL with the worst ratio."""
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
rng = np.random.default_rng(0)

def report(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")

# ---------------- (4.2): sum_{k<K} 1/(1+mu k) <= 1 + log(1+mu(K-1))/mu ----------------
ok = True; worst = 0
for mu in [0.01, 0.1, 1.0, 10.0]:
    for K in [1, 2, 3, 10, 100, 1000]:
        s = sum(1.0/(1+mu*k) for k in range(K))
        b = 1 + np.log(1+mu*(K-1))/mu
        ok &= s <= b + 1e-12; worst = max(worst, s/b)
report("(4.2) discrete sum bound", ok, f"max sum/bound={worst:.4f}")

# ---------------- (4.3): K_delta <= K_* and log(1+mu(K_delta-1)) < log(C/delta) ----------------
ok = True
for mu in [0.05, 0.3, 2.0]:
    for C in [1.0, 3.0]:
        for delta in [0.9, 0.5, 0.1, 0.01]:
            if delta >= C: continue
            x = (C/delta - 1)/mu
            Kstar = int(np.ceil(x))
            R = lambda k: C/(1+mu*k)        # take the envelope itself as the residual (worst case)
            Kd = next(k for k in range(10**7) if R(k) <= delta)
            ok &= (Kd <= Kstar)
            if Kd >= 1:
                ok &= (1 + mu*(Kd-1) < C/delta)
report("(4.3) K_delta <= K_* and log argument < C/delta", ok)

# ---------------- (4.4): variable steps, envelope in flow time ----------------
ok = True; worst = 0
for trial in range(200):
    mu = 10**rng.uniform(-2, 1); K = rng.integers(1, 200)
    eta = rng.uniform(0, 0.5, size=K) * (rng.random(K) > 0.2)   # include zero steps
    t = np.concatenate([[0], np.cumsum(eta)])
    f = lambda s: 1/(1+mu*s)
    lhs = sum(eta[k]*f(t[k]) for k in range(K))                   # sum eta_k R_k / C with B=C=1
    rhs = np.log(1+mu*t[K])/mu + eta.max()*(1 - 1/(1+mu*t[K]))
    ok &= lhs <= rhs + 1e-12; worst = max(worst, lhs/max(rhs,1e-300))
report("(4.4) left-Riemann bound with eta_max", ok, f"max lhs/rhs={worst:.4f}")

# ---------------- (4.5)/(4.6): heavy-ball exact weights ----------------
ok = True
for trial in range(50):
    K = rng.integers(1, 40); d = 5; beta = rng.uniform(0, 0.99); eta = rng.uniform(0.01, 1)
    g = rng.normal(size=(K, d)); a = rng.uniform(0, 1, size=K); v0 = rng.normal(size=d)
    th = np.zeros(d); v = v0.copy()
    for k in range(K):
        v = beta*v + a[k]*g[k]; th = th - eta*v
    # (4.5)
    pred = -v0*sum(eta*beta**(k+1) for k in range(K)) - sum(a[i]*g[i]*sum(eta*beta**(k-i) for k in range(i, K)) for i in range(K))
    ok &= np.allclose(th, pred)
    # (4.6)
    bnd = eta*beta*(1-beta**K)/(1-beta)*np.linalg.norm(v0) + eta*sum((1-beta**(K-i))/(1-beta)*a[i]*np.linalg.norm(g[i]) for i in range(K))
    ok &= np.linalg.norm(th) <= bnd + 1e-9
report("(4.5)/(4.6) momentum identity and bound", ok)

# ---------------- (4.7): Adam with stabilizer ----------------
ok = True; worst = 0
for trial in range(50):
    K = rng.integers(1, 60); d = 4; b1 = rng.uniform(0, 0.99); b2 = rng.uniform(0.5, 0.999)
    zeta = 10**rng.uniform(-8, -1); eta = rng.uniform(0.001, 0.1)
    g = rng.normal(size=(K, d)) * 10**rng.uniform(-6, 0, size=(K, 1))
    th = np.zeros(d); m = np.zeros(d); v = np.zeros(d)
    for k in range(K):
        m = b1*m + (1-b1)*g[k]; v = b2*v + (1-b2)*g[k]**2
        mh = m/(1-b1**(k+1)); vh = v/(1-b2**(k+1))
        th = th - eta*mh/(np.sqrt(vh)+zeta)
    bnd = eta/(zeta*(1-b1))*sum(np.linalg.norm(g[i]) for i in range(K))
    ok &= np.linalg.norm(th) <= bnd + 1e-12; worst = max(worst, np.linalg.norm(th)/bnd)
report("(4.7) Adam bound", ok, f"max ratio={worst:.3e} (loose by design)")
# zeta=0 remark: first step magnitude eta regardless of |g0|
g0 = 1e-9; m = (1-0.9)*g0/(1-0.9); v = (1-0.999)*g0**2/(1-0.999)
report("zeta=0 first Adam step has magnitude eta", np.isclose(abs(m/np.sqrt(v)), 1.0))

# ---------------- Theorem 1.1 (1.1): scalar Gronwall check on a linear strongly monotone field ----------------
# phi_F' = -alpha phi_F ; phi_J' = -alpha phi_J + forcing(t), |forcing| <= K d(t), same start
ok = True
for trial in range(20):
    alpha = rng.uniform(0, 3); Kf = rng.uniform(0.1, 2); mu = rng.uniform(0.1, 5)
    d = lambda t: np.log(1+mu*t)/mu
    sol = solve_ivp(lambda t, y: [-alpha*y[0] + Kf*d(t)*np.sin(3*t), -alpha*y[1]], [0, 5], [1.0, 1.0], rtol=1e-10, atol=1e-12, dense_output=True)
    for t in np.linspace(0, 5, 50):
        Delta = abs(sol.sol(t)[0]-sol.sol(t)[1])
        from scipy.integrate import quad
        rhs = Kf*quad(lambda s: np.exp(-alpha*(t-s))*d(s), 0, t)[0]
        psi = (1-np.exp(-alpha*t))/alpha if alpha > 0 else t
        ok &= Delta <= rhs + 1e-8 and rhs <= Kf*psi*d(t) + 1e-8
report("(1.1)/(1.3) Gronwall comparison and psi_alpha bound", ok)

# ---------------- Counterexample 1.2 ----------------
a, b = 3.0, 1.0
f = lambda x, y: 0.5*(x+a*y**2)**2
grad = lambda x, y: np.array([(x+a*y**2), (x+a*y**2)*2*a*y])
# PL: |grad|^2 >= 2 f
pts = rng.normal(size=(1000, 2))*3
ok = all(np.dot(grad(x, y), grad(x, y)) >= 2*f(x, y) - 1e-12 for x, y in pts)
report("CE1.2 PL inequality |grad f|^2 >= 2f", ok)
# Hessian at (-b,0)
H = np.array([[1, 0], [0, 2*a*(-b)]])
report("CE1.2 Hessian eigenvalues at start (1, -2ab)", np.allclose(np.linalg.eigvalsh(H), sorted([1, -2*a*b])))
# variational eq: xi' = 2ab e^{-t} xi ; compare to actual perturbed trajectory expansion (nonlinear ODE)
eps = 1e-6
solA = solve_ivp(lambda t, z: -grad(*z), [0, 1], [-b, 0.0], rtol=1e-12, atol=1e-14)
solB = solve_ivp(lambda t, z: -grad(*z), [0, 1], [-b, eps], rtol=1e-12, atol=1e-14)
num_factor = (solB.y[1, -1]-solA.y[1, -1])/eps
pred_factor = np.exp(2*a*b*(1-np.exp(-1)))
report("CE1.2 linearized expansion factor at t=1", np.isclose(num_factor, pred_factor, rtol=1e-3), f"numeric={num_factor:.5f} formula={pred_factor:.5f}")

# ---------------- Theorem 1.3 with NON-commuting kernels ----------------
ok = True; worst = 0; worst6 = 0; worst7 = 0; share_worst = 0
for trial in range(200):
    n = 6
    Q1 = rng.normal(size=(n, n)); Kphi = Q1@Q1.T + rng.uniform(0.2, 2)*np.eye(n)      # >= kappa I
    A = rng.normal(size=(n, rng.integers(1, 8))) * rng.uniform(0.1, 2); Kth = A@A.T
    kappa = np.linalg.eigvalsh(Kphi).min(); aa = np.linalg.eigvalsh(Kth).max()
    assert np.linalg.norm(Kphi@Kth - Kth@Kphi) > 1e-6   # genuinely non-commuting
    e0 = rng.normal(size=n)
    ts = np.linspace(0, 6/kappa, 300)
    theta_disp = 0.0; energy = 0.0
    for i, t in enumerate(ts):
        eJ = expm(-(Kphi+Kth)*t)@e0; eF = expm(-Kphi*t)@e0
        diff = np.linalg.norm(eJ-eF); bnd = aa*t*np.exp(-kappa*t)*np.linalg.norm(e0)
        ok &= diff <= bnd + 1e-9; worst = max(worst, diff/max(bnd, 1e-300))
        ok &= diff <= aa/(np.e*kappa)*np.linalg.norm(e0) + 1e-9
    # (1.6) and (1.7) by quadrature on a fine grid
    tt = np.linspace(0, 40/kappa, 4000); dt = tt[1]-tt[0]
    gth = np.array([A.T@(expm(-(Kphi+Kth)*t)@e0) for t in tt])
    speeds = np.linalg.norm(gth, axis=1)
    disp = np.concatenate([[0.0], np.cumsum(0.5*(speeds[1:]+speeds[:-1]))*dt])   # trapezoid: disp[i] ~ int_0^{t_i}
    bnd6 = np.sqrt(aa)/kappa*(1-np.exp(-kappa*tt))*np.linalg.norm(e0)
    ok &= np.all(disp <= bnd6 + 1e-6*np.linalg.norm(e0)); worst6 = max(worst6, (disp/np.maximum(bnd6, 1e-300))[1:].max())
    E = np.sum(np.linalg.norm(gth, axis=1)**2)*dt; bnd7 = aa/(2*kappa)*np.linalg.norm(e0)**2
    ok &= E <= bnd7 + 1e-6; worst7 = max(worst7, E/bnd7)
    share = E/(0.5*np.linalg.norm(e0)**2); ok &= share <= min(1, aa/kappa) + 1e-6
report("Thm 1.3 (1.5)-(1.7) non-commuting Duhamel bounds + share<=a/kappa", ok, f"max ratios: (1.5) {worst:.3f}, (1.6) {worst6:.3f}, (1.7) {worst7:.3f}")

# ---------------- Counterexample 1.4 ----------------
for M in [1, 10, 1000]:
    Kphi = np.diag([M, 0.]); Kth = np.diag([0., 1.]); e0 = np.array([0., 1.])
    t = 0.7
    diff = np.linalg.norm(expm(-(Kphi+Kth)*t)@e0 - expm(-Kphi*t)@e0)
    report(f"CE1.4 M={M}: |zJ-zF|=1-e^-t", np.isclose(diff, 1-np.exp(-t)))

# ---------------- Margin -> disagreement bound ----------------
ok = True; worst_note = 0; worst_sharp = 0
for trial in range(300):
    N, C = 400, 4; gamma = rng.uniform(0.1, 2); h_scale = rng.uniform(0.01, 1.5)
    yF = rng.normal(size=(N, C))*2; yJ = yF + rng.normal(size=(N, C))*h_scale*rng.random((N, 1))**3
    h = np.sqrt(np.mean(np.sum((yJ-yF)**2, axis=1)))       # normalized-logit L2 distance
    srt = np.sort(yF, axis=1); margin = srt[:, -1]-srt[:, -2]
    b_gamma = np.mean(margin <= gamma)
    dis = np.mean(np.argmax(yF, 1) != np.argmax(yJ, 1))
    ok &= dis <= b_gamma + 4*h**2/gamma**2 + 1e-12
    ok &= dis <= b_gamma + 2*h**2/gamma**2 + 1e-12      # sharper version (l2 argument)
    worst_note = max(worst_note, dis/(b_gamma + 4*h**2/gamma**2)); worst_sharp = max(worst_sharp, dis/(b_gamma+2*h**2/gamma**2))
report("disagreement <= b_gamma + 4h^2/gamma^2 (note) and <= b_gamma + 2h^2/gamma^2 (sharper)", ok, f"max ratios {worst_note:.3f} / {worst_sharp:.3f}")

# ---------------- Corollary 1.5: fourth moment, kernel structure, concentration constant ----------------
# E ReLU(g)^4 = 3 s^4 / 2
g = rng.normal(size=4_000_000)*1.3
report("E ReLU(N(0,s^2))^4 = 3 s^4/2", np.isclose(np.mean(np.maximum(g, 0)**4), 1.5*1.3**4, rtol=0.01), f"{np.mean(np.maximum(g,0)**4):.4f} vs {1.5*1.3**4:.4f}")
report("E ReLU(N(0,s^2))^2 = s^2/2", np.isclose(np.mean(np.maximum(g, 0)**2), 0.5*1.3**2, rtol=0.01))

# head-weight kernel = (HH^T/N) kron I_C ; K_theta trace bound ; e0 second moment
N, K, M, C, S = 7, 3, 50, 3, 5
sw, sb = 1.1, 0.7
X = rng.normal(size=(N, S)); W0 = rng.normal(size=(K, S)); Z = X@W0.T
def sample():
    W1 = rng.normal(size=(M, K))*sw/np.sqrt(K); b1 = rng.normal(size=M)*sb
    V = rng.normal(size=(C, M))*sw/np.sqrt(M); b2 = rng.normal(size=C)*sb
    pre = Z@W1.T + b1; H = np.maximum(pre, 0); D = (pre > 0).astype(float)
    return W1, b1, V, b2, H, D
W1, b1, V, b2, H, D = sample()
# Jacobian of normalized logits wrt V: rows (n,c), cols (c',i)
BV = np.zeros((N*C, C*M))
for n in range(N):
    for c in range(C):
        BV[n*C+c, c*M:(c+1)*M] = H[n]/np.sqrt(N)
KV = BV@BV.T
report("K_phi^V == (HH^T/N) kron I_C", np.allclose(KV, np.kron(H@H.T/N, np.eye(C))))
# E||J_theta||_F^2 <= C sw^4 tr(Sigma_X): Monte Carlo
SigX = X.T@X/N; CG = C*sw**4*np.trace(SigX)
vals = []
for _ in range(3000):
    W1, b1, V, b2, H, D = sample()
    J = np.zeros((N*C, K*S))
    for n in range(N):
        Jn = (V*D[n])@W1           # C x K
        J[n*C:(n+1)*C, :] = np.kron(Jn, X[n][None, :])/np.sqrt(N)   # d yhat_n / d vec(W0) with vec row-major (K x S)
    vals.append(np.sum(J**2))
report("E||J_theta||_F^2 <= C s_w^4 tr(Sigma_X)", np.mean(vals) <= CG*1.02, f"MC={np.mean(vals):.4f} bound={CG:.4f}")
# E||e0||^2 <= 2C(sw^2 smax^2/2 + sb^2) + 2||y||^2 with one-hot normalized targets
R = np.linalg.norm(Z, axis=1).max(); smax2 = sw**2*R**2/K + sb**2
y = np.zeros((N, C)); y[np.arange(N), rng.integers(0, C, N)] = 1; ynorm = y/np.sqrt(N)
e2 = []
for _ in range(3000):
    W1, b1, V, b2, H, D = sample()
    z0 = (H@V.T + b2)/np.sqrt(N); e2.append(np.sum((z0-ynorm)**2))
Cr = 2*C*(sw**2*smax2/2 + sb**2) + 2*np.sum(ynorm**2)
report("E||e0||^2 <= C_r", np.mean(e2) <= Cr, f"MC={np.mean(e2):.4f} C_r={Cr:.4f}")
# concentration constant: E||HH^T/(NM) - K_*||_F^2 <= 3 s_max^4/(2M)
Kstar = np.zeros((N, N)); psi_all = []
for _ in range(20000):
    w = rng.normal(size=K)*sw/np.sqrt(K); bb = rng.normal()*sb
    psi = np.maximum(Z@w + bb, 0); psi_all.append(psi)
psi_all = np.array(psi_all); Kstar = psi_all.T@psi_all/len(psi_all)/N
kappa_star = np.linalg.eigvalsh(Kstar).min()
errs = []
for _ in range(2000):
    Hm = psi_all[rng.integers(0, len(psi_all), M)].T     # N x M
    errs.append(np.sum((Hm@Hm.T/(N*M) - Kstar)**2))
report("E||HH^T/(NM)-K_*||_F^2 <= 3 s_max^4/(2M)", np.mean(errs) <= 1.5*smax2**2/M, f"MC={np.mean(errs):.4e} bound={1.5*smax2**2/M:.4e}")
print(f"kappa_* = {kappa_star:.4e};  trivial cap s_max^2/(2N) = {smax2/(2*N):.4e};  implied M_0 >= 72 N^2/rho = {72*N**2:.0f}/rho")
print(f"note's M_0 with rho=0.1: {int(np.ceil(18*smax2**2/(0.1*kappa_star**2)))}")
