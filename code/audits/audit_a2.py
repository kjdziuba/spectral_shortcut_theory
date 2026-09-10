# ---------------------------------------------------------------------------
# VERIFICATION RECORD -- copied unchanged into code/audits/ as a durable audit
# artefact.  Only this header was added.
# origin  : session scratchpad 887fb000-.../scratchpad/audit_a2.py
# author  : Claude (session audit of math_01.md)
# date    : 2026-09-09
# seed    : numpy default_rng(0)
# command : venv/bin/python3 code/audits/audit_a2.py
# ---------------------------------------------------------------------------
"""Numerical spot-checks for math_01.md, assignment A2 (P3 and P2).
All checks use the paper conventions: logits stacked and divided by sqrt(N),
averaged softmax CE, r = grad of loss in logits, GGN = J^T H J.
"""
import numpy as np, torch, math
torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(0)

def softmax_H(y):
    p = torch.softmax(y, -1)
    return torch.diag_embed(p) - p[..., :, None] * p[..., None, :]

# ---------------------------------------------------------------- P2 first correction
print("== P2 CE counterexample: one example, logits (theta,0), theta=0, label 1 ==")
theta = torch.zeros(1)
y = torch.stack([theta[0], torch.zeros(())])
p = torch.softmax(y, 0)
r = p - torch.tensor([1.0, 0.0])
J = torch.tensor([1.0, 0.0])
grad = J @ r
H = torch.diag(p) - torch.outer(p, p)
G = J @ H @ J
print(f"r={r.numpy()}, |grad|^2={grad.item()**2:.4f}, G={G.item():.4f}, |r|^2={(r@r).item():.4f}, "
      f"G|r|^2={(G*(r@r)).item():.4f}  -> |grad|^2 > G|r|^2 : {grad.item()**2 > (G*(r@r)).item()}")
Hp = torch.linalg.pinv(H)
print(f"(2.2) check: G * r^T H^+ r = {(G * (r@Hp@r)).item():.4f}  (equals |grad|^2 = {grad.item()**2:.4f})")
# general p: ratio |grad|^2 / (G |r|^2) = 1/(2p(1-p))
for pp in [0.5, 0.1, 0.01]:
    print(f"  p={pp}: ratio |grad|^2/(G|r|^2) = {1/(2*pp*(1-pp)):.2f}")

# ---------------------------------------------------------------- Lemma 3.1 / Theorem 3.2 setup
def make_features(N, M, R=1.0, rho=0.6, rng=rng):
    """features with a common direction of Theta(M) energy, ||h_n||^2 <= R^2 M."""
    u = rng.standard_normal(M); u /= np.linalg.norm(u)
    H = rng.standard_normal((N, M)) / np.sqrt(M) * np.sqrt(1 - rho**2)
    H += rho * np.outer(rng.choice([1.0, 1.0, 1.0, -0.2], N), u)
    # rescale each row to norm <= R sqrt(M) (rows have norm ~ 1 here; scale up to R sqrt(M)*0.9)
    H = H / np.linalg.norm(H, axis=1, keepdims=True) * (0.9 * R * np.sqrt(M)) * rng.uniform(0.5, 1.0, (N, 1))
    return torch.tensor(H)

def ggn_quadform_W(h, W, b, V, dW):
    """vec(dW)^T G_WW vec(dW) for logits V ReLU(W h_n + b), averaged softmax CE, via autograd JVP."""
    N = h.shape[0]
    def logits(Wm):
        return torch.relu(h @ Wm.T + b) @ V.T   # (N, C)
    y, dy = torch.func.jvp(logits, (W,), (dW,))
    Hn = softmax_H(y)                            # (N, C, C)
    return (torch.einsum('nc,ncd,nd->n', dy, Hn, dy).sum() / N).item(), y, dy

def dense_ggn_W(h, W, b, V):
    """Dense G_WW = (1/N) sum_n J_n^T H_n J_n with J_n = d logits_n / d vec(W)."""
    N, M = h.shape; q = W.shape[0]
    def logits_vec(wvec):
        return torch.relu(h @ wvec.view(q, M).T + b) @ V.T
    Jfull = torch.func.jacrev(logits_vec)(W.reshape(-1))  # (N, C, qM)
    y = logits_vec(W.reshape(-1))
    Hn = softmax_H(y)
    return torch.einsum('nci,ncd,ndj->ij', Jfull, Hn, Jfull) / N

print("\n== Lemma 3.1 witness identity + dense GGN check (small sizes) ==")
N, M, q, C = 40, 12, 3, 4
s_w, s_b, s_v = 1.0, 0.5, 1.0
h = make_features(N, M)
S_h = h.T @ h / N
evals, evecs = torch.linalg.eigh(S_h); u = evecs[:, -1]; A = evals[-1].item()
W = torch.randn(q, M) * s_w / math.sqrt(M); b = torch.randn(q) * s_b; V = torch.randn(C, q) * s_v / math.sqrt(q)
dW = torch.zeros(q, M); dW[0] = u
qf, y, dy = ggn_quadform_W(h, W, b, V, dW)
# closed form (h_n^T u) chi_n V e1
t = h @ W.T + b; chi = (t[:, 0] > 0).double()
dy_formula = (h @ u)[:, None] * chi[:, None] * V[:, 0][None, :]
print(f"logit perturbation formula max abs err: {(dy - dy_formula).abs().max().item():.2e}")
Gd = dense_ggn_W(h, W, b, V)
lam_max = torch.linalg.eigvalsh(Gd)[-1].item()
print(f"quadform along witness = {qf:.6f}; dense lambda_max(G_WW) = {lam_max:.6f}; witness <= lambda_max: {qf <= lam_max + 1e-12}")
# S_{h,a} bound of (3.1) with a = e1: D_n a = V diag(chi_n) e1 = chi_n V e1
Pi = torch.eye(C) - torch.ones(C, C) / C
pmin = torch.softmax(y, -1).min(-1).values
S_ha = (pmin * (Pi @ V[:, 0]).norm()**2 * chi)[:, None, None] * (h[:, :, None] * h[:, None, :])
S_ha = S_ha.sum(0) / N
print(f"(3.1): lambda_max(G_WW)={lam_max:.6f} >= lambda_max(S_h,a)={torch.linalg.eigvalsh(S_ha)[-1].item():.6f}")
print(f"dense sanity: max|G - G^T| = {(Gd - Gd.T).abs().max().item():.1e}")

# ---------------------------------------------------------------- Theorem 3.2 constants, Monte Carlo
print("\n== Theorem 3.2: Pr(Q>=A/4), Pr(U>A/8), joint event, curvature floor on the event ==")
def thm32_trial(h, u, A, q, C, s_w, s_b, s_v):
    N, M = h.shape
    W = torch.randn(q, M) * s_w / math.sqrt(M); b = torch.randn(q) * s_b; V = torch.randn(C, q) * s_v / math.sqrt(q)
    a_n = (h @ u) ** 2
    t = h @ W.T + b; chi = (t[:, 0] > 0).double()
    Q = (a_n * chi).mean().item()
    R = 1.0
    s_max2 = s_w**2 * R**2 + s_b**2
    L0 = math.sqrt(96 * q * s_max2)
    relu = torch.relu(t); untame = (relu.norm(dim=1) > L0).double()
    U = (a_n * untame).mean().item()
    EV = (V.norm() <= 2) and ((torch.eye(C) - torch.ones(C, C) / C) @ V[:, 0]).norm() >= 0.5
    dW = torch.zeros(q, M); dW[0] = u
    qf, y, dy = ggn_quadform_W(h, W, b, V, dW)
    c_p = math.exp(-4 * L0) / C
    return Q, U, bool(EV), qf, A, c_p, L0

for M in [16, 64, 256]:
    h = make_features(200, M)
    S_h = h.T @ h / 200
    evals, evecs = torch.linalg.eigh(S_h); u = evecs[:, -1]; A = evals[-1].item()
    res = [thm32_trial(h, u, A, q=3, C=4, s_w=1.0, s_b=0.5, s_v=1.0) for _ in range(400)]
    Q = np.array([r[0] for r in res]); U = np.array([r[1] for r in res]); EV = np.array([r[2] for r in res]); qf = np.array([r[3] for r in res])
    c_p, L0 = res[0][5], res[0][6]
    ev = (Q >= A / 4) & (U <= A / 8)
    on = ev & EV
    print(f"M={M:4d} A={A:8.3f} (A/M={A/M:.3f}) EQ/A={Q.mean()/A:.3f}  Pr(Q>=A/4)={np.mean(Q>=A/4):.3f} (>=1/3)  "
          f"Pr(U>A/8)={np.mean(U>A/8):.3f} (<=1/12)  Pr(joint)={ev.mean():.3f} (>=1/4)  Pr(E_V)={EV.mean():.3f}")
    print(f"        L0={L0:.2f}, c_p=e^-4L0/C={c_p:.2e}; on event: min quadform/(c_p A/32) = "
          f"{(qf[on].min()/(c_p*A/32)) if on.any() else float('nan'):.3e}  (>=1 required); E[quadform]/M = {qf.mean()/M:.4f}")

# ---------------------------------------------------------------- (3.6) squared loss expectation
print("\n== (3.6) squared loss: E[lambda_max(G_WW)] >= s_v^2/(2q) lambda_max(S_h); check E[v1^2 Q] identity ==")
for M in [16, 64]:
    h = make_features(200, M); S_h = h.T @ h / 200
    evals, evecs = torch.linalg.eigh(S_h); u = evecs[:, -1]; A = evals[-1].item()
    q = 3; s_v = 1.3; vals = []
    for _ in range(3000):
        W = torch.randn(q, M) / math.sqrt(M); b = torch.randn(q) * 0.5; v = torch.randn(q) * s_v / math.sqrt(q)
        t = h @ W.T + b; chi = (t[:, 0] > 0).double()
        vals.append((v[0]**2 * ((h @ u)**2 * chi).mean()).item())
    print(f"M={M}: MC E[v1^2 Q]={np.mean(vals):.4f} +- {np.std(vals)/np.sqrt(len(vals)):.4f};  s_v^2/(2q) lambda_max(S_h) = {s_v**2/(2*q)*A:.4f}")

# ---------------------------------------------------------------- Counterexample 3.3
print("\n== Counterexample 3.3: q=1, common h, zero bias: Pr(W block == 0) ==")
M = 64; h = torch.ones(50, M); zeros = 0
for _ in range(2000):
    W = torch.randn(1, M) / math.sqrt(M)
    zeros += int((h @ W.T <= 0).all())
print(f"Pr(all preacts <= 0) = {zeros/2000:.3f} (theory 1/2); on that event ReLU' = 0 at every site -> G_WW = 0 exactly")

# ---------------------------------------------------------------- Counterexample 3.4 (train-mode BN)
print("\n== Counterexample 3.4: train-mode BN after W h with constant features ==")
M, q, C, N = 32, 4, 3, 20
h = torch.ones(N, M) * (1.0)          # ||h||^2 = M
W = torch.randn(q, M, requires_grad=True) / math.sqrt(M)
gamma = torch.randn(q); beta = torch.randn(q) * 0.7
V = torch.randn(C, q) / math.sqrt(q)
for eps in [1e-5, 1e-1]:
    def logits_of_W(Wm):
        t = h @ Wm.T                                   # (N, q), constant across n
        mu = t.mean(0, keepdim=True); var = t.var(0, unbiased=False, keepdim=True)
        o = gamma * (t - mu) / torch.sqrt(var + eps) + beta   # train-mode BN
        return torch.relu(o) @ V.T
    Jw = torch.func.jacrev(logits_of_W)(W.detach())   # (N, C, q, M)
    out = logits_of_W(W.detach())
    print(f"eps={eps}: max|d logits/dW| = {Jw.abs().max().item():.2e}; BN output == beta at every site: "
          f"{torch.allclose((h@W.detach().T - (h@W.detach().T).mean(0))*0 + beta, beta)}; logits constant across sites: {(out - out[0]).abs().max().item():.1e}")
    # also check the torch.nn.BatchNorm1d module in train mode
    bn = torch.nn.BatchNorm1d(q, eps=eps); bn.train()
    with torch.no_grad(): bn.weight.copy_(gamma); bn.bias.copy_(beta)
    def logits_bn(Wm):
        return torch.relu(bn(h @ Wm.T)) @ V.T
    Jw2 = torch.func.jacrev(logits_bn)(W.detach())
    print(f"          nn.BatchNorm1d(train): max|d logits/dW| = {Jw2.abs().max().item():.2e}")
p = torch.softmax(out, -1); pmin = p.min(-1).values
S_hp = (pmin[:, None, None] * (h[:, :, None] * h[:, None, :])).sum(0) / N
print(f"lambda_max(S_h^p) = {torch.linalg.eigvalsh(S_hp)[-1].item():.3f} = pmin*M = {pmin[0].item()*M:.3f}  (proportional to M)")
# BN Jacobian wrt input at t constant: (gamma/sqrt(eps)) (I - 11^T/N) annihilates constant vectors
eps = 1e-5
t0 = (h @ W.detach().T)
def bn_fn(t):
    mu = t.mean(0, keepdim=True); var = t.var(0, unbiased=False, keepdim=True)
    return gamma * (t - mu) / torch.sqrt(var + eps) + beta
Jbn = torch.func.jacrev(bn_fn)(t0)  # (N,q,N,q)
const_pert = torch.ones(N, 1) * torch.randn(1, q)
print(f"D_BN applied to site-constant perturbation: max = {torch.einsum('ncmd,md->nc', Jbn, const_pert).abs().max().item():.2e}; "
      f"D_BN finite: {torch.isfinite(Jbn).all().item()}, ||D_BN||_max = {Jbn.abs().max().item():.2e} (~gamma/sqrt(eps)={ (gamma.abs().max()/math.sqrt(eps)).item():.2e})")

# ---------------------------------------------------------------- (2.5) exact ratio
print("\n== (2.5) exact energy ratio, numerical integration ==")
lam_v, lam_u, gam_v, gam_u, a_v, a_u, T = 50.0, 2.0, 60.0, 3.0, 0.8, 1.5, 0.7
tt = np.linspace(0, T, 200001)
Ev = np.trapz(lam_v * (a_v * np.exp(-gam_v * tt))**2, tt); Eu = np.trapz(lam_u * (a_u * np.exp(-gam_u * tt))**2, tt)
formula = (lam_v/lam_u) * (a_v**2/a_u**2) * (gam_u/gam_v) * (1 - np.exp(-2*gam_v*T)) / (1 - np.exp(-2*gam_u*T))
print(f"numeric ratio = {Ev/Eu:.6f}, formula (2.5) = {formula:.6f}, rel err = {abs(Ev/Eu-formula)/formula:.1e}")
# Counterexample 2.3 limit
L, l = 100.0, 1.0
for T in [0.01, 0.1, 1, 10]:
    print(f"  CE2.3: T={T}: E_v/E_u = {(1-np.exp(-2*L*T))/(1-np.exp(-2*l*T)):.3f}  (curvature ratio L/l = {L/l})")
