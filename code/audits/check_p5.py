# ---------------------------------------------------------------------------
# VERIFICATION RECORD -- copied unchanged into code/audits/ as a durable audit
# artefact.  Only this header was added.
# origin  : session scratchpad 887fb000-.../scratchpad/check_p5.py
# author  : Claude (session audit of math_01.md)
# date    : 2026-09-09
# seed    : none (deterministic)
# command : venv/bin/python3 code/audits/check_p5.py
# ---------------------------------------------------------------------------
"""Independent audit of Theorem 5.1 (math_01.md, P5) numbers.

1. Solve Q_M(a) = m by bisection, evaluate exact displacement from (5.3),
   compare with the note's table and bound (5.12).
2. Integrate the FULL (M+2)-parameter gradient flow (a, v, beta_1..beta_M)
   with an adaptive ODE solver, find first time q >= m, compare with the
   invariants (5.3), (5.4), (5.5), (5.8), (5.11).
3. Build the two-logit softmax-CE GGN numerically (packet convention:
   logits (F/2,-F/2), residual N^{-1/2}-stacked, G = J^T H J) and check
   lambda_theta = r(1-r)(1+b^2), lambda_phi = M r(1-r) v^2, R = sqrt(2) r.
4. Check the 1/sqrt(M) readout control gives the M=1 reduced dynamics.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

delta = 0.05
m = np.log((1 - delta) / delta)          # = log 19
Bm = np.sqrt((1 + m**2) / 2)
Psi = lambda q: q + np.exp(q) - 1
print(f"m = {m:.9f}, B_m = {Bm:.9f}, Psi(m) = {Psi(m):.6f}")
print(f"log(1/(sqrt2*delta)) = {np.log(1/(np.sqrt(2)*delta)):.6f}")
print()

def Q(a, M):
    return a + 0.5 * np.sqrt(M) * np.sinh(2 * np.sqrt(M) * a)

print("=== Table check (solve Q_M(a_M)=m; displacement from (5.3)) ===")
print(f"{'M':>5} {'a_M':>12} {'v-1':>12} {'disp':>12} {'note':>10} {'bound':>12} {'note':>10} {'ratio':>8} {'sharp_alt':>10} {'M*a_M':>8}")
table_note = {16: (0.219021, 1.370601), 64: (0.071050, 0.358465),
              256: (0.019658, 0.090662), 1024: (0.005064, 0.022732)}
for M in [1, 4, 16, 64, 256, 1024, 4096]:
    aM = brentq(lambda a: Q(a, M) - m, 0.0, m)
    v = np.cosh(np.sqrt(M) * aM)
    b = np.sqrt(M) * np.sinh(np.sqrt(M) * aM)
    disp = np.hypot(aM, v - 1)
    bound = 4 * Bm / (M + 1) * np.log(1 / (np.sqrt(2) * delta))
    # sharper bound available from the same integration: ||dW/dt|| = r sqrt(1+b^2), int r dt = a_M
    sharp = np.sqrt(1 + m**2) * aM
    # checks of (5.6),(5.7)
    assert aM <= m / (M + 1) + 1e-12, (M, aM, m/(M+1))
    assert v - 1 <= m**2 / (2 * M) + 1e-12
    assert abs(b * v + aM - m) < 1e-9
    assert b <= m
    note_d, note_b = table_note.get(M, (np.nan, np.nan))
    print(f"{M:>5} {aM:12.6f} {v-1:12.6f} {disp:12.6f} {note_d:10.6f} {bound:12.6f} {note_b:10.6f} {bound/disp:8.3f} {sharp:10.6f} {M*aM:8.4f}")
print(f"asymptotic bound/disp ratio -> {4*Bm*np.log(1/(np.sqrt(2)*delta)) / (m*np.sqrt(1+m**2/4)):.4f}")
print()

print("=== Full (M+2)-parameter flow, adaptive solver ===")
def rhs(t, y, M, alpha=1.0):
    a, v = y[0], y[1]
    beta = y[2:]
    b = alpha * beta.sum()
    q = a + b * v
    r = 1.0 / (1.0 + np.exp(q))
    da = r
    dv = b * r
    dbeta = np.full(M, alpha * v * r)
    return np.concatenate(([da, dv], dbeta))

for M in [16, 64, 256]:
    y0 = np.concatenate(([0.0, 1.0], np.zeros(M)))
    def hit(t, y, M=M):
        a, v = y[0], y[1]; b = y[2:].sum(); return a + b*v - m
    hit.terminal = True; hit.direction = 1
    sol = solve_ivp(rhs, [0, 50.0], y0, args=(M,), events=hit, rtol=1e-11, atol=1e-13, dense_output=True)
    Tm = sol.t_events[0][0]
    yT = sol.y_events[0][0]
    a, v = yT[0], yT[1]; b = yT[2:].sum()
    q = a + b * v
    # invariants
    print(f"M={M}: T_m={Tm:.6e}  [Psi(m)/(M+1+2m^2)={Psi(m)/(M+1+2*m**2):.6e}, Psi(m)/(M+1)={Psi(m)/(M+1):.6e}]")
    print(f"   a={a:.6f}  v-cosh(sqrtM a)={v-np.cosh(np.sqrt(M)*a):.2e}  b-sqrtM sinh={b-np.sqrt(M)*np.sinh(np.sqrt(M)*a):.2e}  q-m={q-m:.1e}")
    print(f"   disp ||W(T_m)-W(0)|| = {np.hypot(a, v-1):.6f}")
    # residual envelope (5.8) along trajectory
    ts = np.linspace(0, Tm, 2000)
    Y = sol.sol(ts)
    qs = Y[0] + Y[2:].sum(0) * Y[1]
    rs = 1 / (1 + np.exp(qs))
    env = 0.5 / (1 + (M + 1) * ts / 4)
    print(f"   max r/envelope on [0,T_m] = {np.max(rs/env):.6f} (must be <=1)")
    # frozen comparator: Psi(q_F)=M t  -> q_F = Psi^{-1}(M t)
    qF = np.array([brentq(lambda q: Psi(q) - M * t, 0, 50) for t in ts])
    diff = qs - qF
    print(f"   max (q-q_F) = {diff.max():.6f}; bound (1+2m^2)t/2 at T_m = {(1+2*m**2)*Tm/2:.6f}; (1+2m^2)Psi(m)/(2(M+1)) = {(1+2*m**2)*Psi(m)/(2*(M+1)):.6f}; min diff={diff.min():.2e}")
    # b<=m check
    bs = Y[2:].sum(0)
    print(f"   max b on [0,T_m] = {bs.max():.4f} <= m={m:.4f}: {bs.max()<=m}")
print()

print("=== 1/sqrt(M) readout control: reduced dynamics == M=1 ===")
M = 64
y0 = np.concatenate(([0.0, 1.0], np.zeros(M)))
solc = solve_ivp(rhs, [0, 5.0], y0, args=(M, 1/np.sqrt(M)), rtol=1e-11, atol=1e-13)
y1 = np.array([0.0, 1.0, 0.0])
sol1 = solve_ivp(rhs, [0, 5.0], y1, args=(1,), rtol=1e-11, atol=1e-13, t_eval=solc.t)
bc = solc.y[2:].sum(0) / np.sqrt(M)
print(f"max |a_ctrl - a_M1| = {np.max(np.abs(solc.y[0]-sol1.y[0])):.2e}, max |b_ctrl - b_M1| = {np.max(np.abs(bc - sol1.y[2])):.2e}")
print()

print("=== Two-logit softmax-CE GGN in packet convention (numerical) ===")
def ggn_and_residual(a, v, beta):
    """Data: Y in {+1,-1} (N=2 samples). logits yhat=(F/2,-F/2), F=aS+b v C, S=C=Y.
    Class index: Y=+1 -> class 0, Y=-1 -> class 1. Returns lam_theta, lam_phi, RMS residual."""
    M = len(beta); b = beta.sum()
    params = np.concatenate(([a, v], beta))
    N = 2
    Js, Hs, rs = [], [], []
    for Y in [+1.0, -1.0]:
        F = a * Y + b * v * Y
        yhat = np.array([F / 2, -F / 2])
        p = np.exp(yhat - yhat.max()); p /= p.sum()
        onehot = np.array([1.0, 0.0]) if Y > 0 else np.array([0.0, 1.0])
        # dF/dparams
        dF = np.concatenate(([Y, b * Y], np.full(M, v * Y)))
        J = np.outer(np.array([0.5, -0.5]), dF)      # 2 x (M+2), unnormalized per-sample logit Jacobian
        H = np.diag(p) - np.outer(p, p)
        Js.append(J); Hs.append(H); rs.append(p - onehot)
    G = sum(J.T @ H @ J for J, H in zip(Js, Hs)) / N
    Gtt = G[:2, :2]; Gpp = G[2:, 2:]
    R = np.sqrt(sum(np.sum(r**2) for r in rs) / N)
    grad = sum(J.T @ r for J, r in zip(Js, rs)) / N
    return np.linalg.eigvalsh(Gtt).max(), np.linalg.eigvalsh(Gpp).max(), R, grad

for (M, a, v, bb) in [(16, 0.0, 1.0, 0.0), (16, 0.1, 1.1, 2.0), (64, 0.05, 1.03, 1.5)]:
    beta = np.full(M, bb / M)
    lt, lp, R, grad = ggn_and_residual(a, v, beta)
    q = a + bb * v; r = 1 / (1 + np.exp(q))
    print(f"M={M}, a={a}, v={v}, b={bb}: lam_theta={lt:.6f} (formula {r*(1-r)*(1+bb**2):.6f}); "
          f"lam_phi={lp:.6f} (formula {M*r*(1-r)*v**2:.6f}); D={lp/lt:.3f}; R={R:.6f} (sqrt2 r={np.sqrt(2)*r:.6f}); "
          f"||grad_theta||={np.linalg.norm(grad[:2]):.6f} (r sqrt(1+b^2)={r*np.sqrt(1+bb**2):.6f})")
# operator norm of normalized J_theta
M=16; a, v, bb = 0.1, 1.1, 2.0
Y=1.0; dF=np.array([Y, bb*Y]); J=np.outer(np.array([0.5,-0.5]), dF)
JtJ = J.T@J  # same for both samples -> average is the same
print(f"||J_theta||_op = {np.sqrt(np.linalg.eigvalsh(JtJ).max()):.6f} vs sqrt((1+b^2)/2)={np.sqrt((1+bb**2)/2):.6f}")

print()
print("=== (5.12) bound with C=1 vs the sharper C=1/sqrt2 choice ===")
for M in [16, 64, 256, 1024]:
    mu = (M+1)/4
    b1 = Bm/mu*np.log(1/(np.sqrt(2)*delta))
    b2 = Bm*(1/np.sqrt(2))/mu*np.log((1/np.sqrt(2))/(np.sqrt(2)*delta))
    print(f"M={M}: C=1 -> {b1:.6f}; C=1/sqrt2 -> {b2:.6f}")
