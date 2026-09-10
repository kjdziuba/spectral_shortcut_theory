# ---------------------------------------------------------------------------
# VERIFICATION RECORD -- copied unchanged into code/audits/ as a durable audit
# artefact.  Only this header was added.
# origin  : session scratchpad 887fb000-.../scratchpad/audit_a2_bn.py
# author  : Claude (session audit of math_01.md)
# date    : 2026-09-09
# seed    : torch.manual_seed(1)
# command : venv/bin/python3 code/audits/audit_a2_bn.py
# ---------------------------------------------------------------------------
"""Counterexample 3.4 with the real nn.BatchNorm1d module (train mode), plus (2.5)."""
import numpy as np, torch, math
torch.set_default_dtype(torch.float64)
torch.manual_seed(1)
M, q, C, N = 32, 4, 3, 20
h = torch.ones(N, M)                     # ||h||^2 = M, constant across sites
W0 = torch.randn(q, M) / math.sqrt(M)
gamma = torch.randn(q); beta = torch.randn(q) * 0.7
V = torch.randn(C, q) / math.sqrt(q)
for eps in [1e-5, 1e-1]:
    bn = torch.nn.BatchNorm1d(q, eps=eps, track_running_stats=False); bn.train()
    with torch.no_grad(): bn.weight.copy_(gamma); bn.bias.copy_(beta)
    def logits_bn(Wm):
        return torch.relu(bn(h @ Wm.T)) @ V.T
    Jw = torch.autograd.functional.jacobian(logits_bn, W0)      # (N, C, q, M)
    out = logits_bn(W0)
    print(f"nn.BatchNorm1d(train, eps={eps}): max|d logits/dW| = {Jw.abs().max().item():.2e}; "
          f"BN out == beta: {torch.allclose(bn(h @ W0.T), beta.expand(N, q))}; logits site-constant: {(out-out[0]).abs().max().item():.1e}")
    # gradient wrt gamma vanishes too, wrt beta does not
    g = torch.autograd.functional.jacobian(lambda gm: torch.relu(gm * (h@W0.T - (h@W0.T).mean(0)) / torch.sqrt((h@W0.T).var(0, unbiased=False) + eps) + beta) @ V.T, gamma)
    print(f"   d logits/d gamma max = {g.abs().max().item():.2e} (gamma is also dead)")
p = torch.softmax(out, -1); pmin = p.min(-1).values
S_hp = (pmin[:, None, None] * (h[:, :, None] * h[:, None, :])).sum(0) / N
print(f"lambda_max(S_h^p) = {torch.linalg.eigvalsh(S_hp)[-1].item():.3f} = pmin*M = {pmin[0].item()*M:.3f}  (proportional to M, while G_WW = 0)")

# Non-constant features: BN kills exactly the site-constant part of each channel perturbation.
print("\n-- non-constant features: what train-mode BN does to the mean-direction witness --")
eps = 1e-5
hn = torch.ones(N, M) + 0.3 * torch.randn(N, M)          # mean direction carries Theta(M) energy
u = hn.mean(0); u = u / u.norm()
dW = torch.zeros(q, M); dW[0] = u                          # supplement's witness direction (mean feature)
pert = hn @ dW.T                                           # (N, q): perturbation of BN input, channel 0 only
t0 = hn @ W0.T
def bn_fn(t):
    mu = t.mean(0, keepdim=True); var = t.var(0, unbiased=False, keepdim=True)
    return gamma * (t - mu) / torch.sqrt(var + eps) + beta
_, dout = torch.func.jvp(bn_fn, (t0,), (pert,))
e_in = (pert**2).sum(0) / N; e_out = (dout**2).sum(0) / N
print(f"channel-0 witness energy: incoming (1/N)sum (h_n^T u)^2 = {e_in[0].item():.2f} (~M={M}); "
      f"post-BN = {e_out[0].item():.3f}; centered incoming energy Var_n(h_n^T u) = {pert[:,0].var(unbiased=False).item():.3f}")
print("   -> BN passes only the across-site variance of (h_n^T u) (times gamma/sigma); the Theta(M) mean-energy is annihilated.")

# ---------------------------------------------------------------- (2.5) exact ratio
print("\n== (2.5) exact energy ratio, numerical integration ==")
lam_v, lam_u, gam_v, gam_u, a_v, a_u, T = 50.0, 2.0, 60.0, 3.0, 0.8, 1.5, 0.7
tt = np.linspace(0, T, 200001)
Ev = np.trapezoid(lam_v * (a_v * np.exp(-gam_v * tt))**2, tt); Eu = np.trapezoid(lam_u * (a_u * np.exp(-gam_u * tt))**2, tt)
formula = (lam_v/lam_u) * (a_v**2/a_u**2) * (gam_u/gam_v) * (1 - np.exp(-2*gam_v*T)) / (1 - np.exp(-2*gam_u*T))
print(f"numeric ratio = {Ev/Eu:.6f}, formula (2.5) = {formula:.6f}, rel err = {abs(Ev/Eu-formula)/formula:.1e}")
L, l = 100.0, 1.0
for T in [0.01, 0.1, 1, 10]:
    print(f"  CE2.3: T={T}: E_v/E_u = {(1-np.exp(-2*L*T))/(1-np.exp(-2*l*T)):.3f}  (curvature ratio L/l = {L/l})")
# CE 2.2 realization check: z=(sqrt(L) th1, sqrt(l) th2), r0=e2, squared loss, flow
A = np.diag([np.sqrt(L), np.sqrt(l)]); r = np.array([0.0, 1.0]); dt = 1e-4; Ev = Eu = 0.0
for _ in range(20000):
    Ev += (A[:, 0] @ r)**2 * dt; Eu += (A[:, 1] @ r)**2 * dt; r = r - dt * (A @ A.T @ r)
print(f"  CE2.2 (Euler, T=2): E_v={Ev:.2e}, E_u={Eu:.4f} (theory 0 and (1-e^-4)/2={ (1-np.exp(-4))/2:.4f})")
