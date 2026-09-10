# ---------------------------------------------------------------------------
# VERIFICATION RECORD -- copied unchanged into code/audits/ as a durable audit
# artefact.  Only this header was added.
# origin  : audit of review_packet/astra/math_02.md section 6 -- (B1)-(B4) BatchNorm statements
#           (session scratchpad 887fb000-.../scratchpad/audit_math02_opus/b_bn_torch.py)
#           report: review_packet/astra/audit_math02_sections_2_4_6_2026-09-10.md
# author  : Claude (math_02 sections 2/4/6 auditor)
# date    : 2026-09-10
# seed    : 13579
# command : venv/bin/python3 code/audits/math02_sections_2_4_6/b_bn_torch.py
# ---------------------------------------------------------------------------
"""
AUDIT (B1)-(B4) -- train-mode BatchNorm derivative, contraction, the necessity
counterexample, and the function-preserving scale identity.

Independent of astra/math_02_checks.py, which uses a hand-written numpy BN.
Here everything is checked against the REAL torch.nn.BatchNorm2d in train mode
via torch.func.jvp, so the normalization group (N*H*W per channel) is torch's.

 (B1) D_BN = (gamma/sigma~)(P - t t^T/(L(s^2+eps))) as an operator, and its
      spectrum: 0 on constants, 1 on centred _|_ t, eps/(s^2+eps) on t.
 (B2) ||D_BN d||^2 <= gamma^2/(s^2+eps) ||P d||^2, with the extremal cases.
 (B3) function-preserving scaling J_W(cW) = c^{-1} J_W(W),
      G_WW(cW) = c^{-2} G_WW(W); the exact eps co-scaling BN_{c^2 eps}(c a)=BN_eps(a);
      the invariant ||W||_F^2 lambda_max(G_WW); the size of the DEPARTURE at
      fixed eps=1e-5 when the pre-BN variance is comparable to eps; and the
      failure of the scalar invariant under CHANNELWISE rescaling.
 (B2c) the necessity counterexample: rows (1,0),(0,1),(-1,-1) zero-padded to
      dimension M, single BN channel, w_M = e1/sqrt(M), gamma=1, eps=0, logits
      (BN(w_M^T h), 0): weight-GGN top eigenvalue exactly M x const while the
      centred input Gram is constant.
 (B4) the algebra gamma^2(M+K)/(c0 + eps(M+K)).

Seed: 13579
"""
import numpy as np
import torch
from torch.func import jvp

torch.manual_seed(13579)
torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(13579)
FAIL = []


def check(name, cond, info=""):
    if not cond:
        FAIL.append(f"{name}: {info}")
        print(f"  FAIL {name}  {info}")


# =====================================================================
# (B1) D_BN against torch.nn.BatchNorm2d(train) via jvp
# =====================================================================
print("=" * 78)
print("(B1) train-mode BatchNorm2d derivative vs the closed form")
print("=" * 78)
for (Nb, C, H, W, eps, gam) in [(4, 3, 5, 5, 1e-5, 1.3), (3, 2, 4, 6, 0.0, 1.0),
                                (6, 1, 3, 3, 1e-2, -0.7)]:
    bn = torch.nn.BatchNorm2d(C, eps=eps, affine=True, track_running_stats=False)
    with torch.no_grad():
        bn.weight.fill_(gam)
        bn.bias.normal_()
    bn.train()
    x = torch.randn(Nb, C, H, W) * 0.8 + 0.3

    L = Nb * H * W                                     # torch's normalization group
    err_op, err_eig = 0.0, 0.0
    for c in range(C):
        a = x[:, c].reshape(-1).clone()                 # the group, flattened
        P = torch.eye(L) - torch.ones(L, L) / L
        t = P @ a
        s2 = float(t @ t) / L
        sig = float(np.sqrt(s2 + eps))
        D = (gam / sig) * (P - torch.outer(t, t) / (L * (s2 + eps)))

        # jvp through the real torch BN, restricted to this channel
        for _ in range(6):
            d = torch.randn(L)
            dx = torch.zeros_like(x)
            dx[:, c] = d.reshape(Nb, H, W)
            _, out_t = jvp(lambda z: bn(z), (x,), (dx,))
            got = out_t[:, c].reshape(-1)
            err_op = max(err_op, float((got - D @ d).abs().max()))

        # spectrum of the middle factor
        Q = P - torch.outer(t, t) / (L * (s2 + eps))
        one = torch.ones(L) / np.sqrt(L)
        err_eig = max(err_eig, float((Q @ one).abs().max()))                      # 0
        tn = t / t.norm()
        err_eig = max(err_eig, float((Q @ tn - (eps / (s2 + eps)) * tn).abs().max()))
        # a centred direction orthogonal to t
        g = torch.randn(L)
        g = P @ g
        g = g - (g @ tn) * tn
        g = g / g.norm()
        err_eig = max(err_eig, float((Q @ g - g).abs().max()))
        ev = torch.linalg.eigvalsh(Q)
        check("(B1) spectrum set", abs(float(ev[-1]) - 1.0) < 1e-10
              and abs(float(ev[0]) - min(0.0, eps / (s2 + eps))) < 1e-10)
    print(f"  N={Nb} C={C} HxW={H}x{W} eps={eps:<7g} gamma={gam:<5g} L={L:<4d}"
          f" max|jvp - D_BN d| = {err_op:.3e}   max eig residual = {err_eig:.3e}")
    check("(B1) jvp matches closed form", err_op < 1e-11, f"{err_op}")
    check("(B1) eigenvalues", err_eig < 1e-11, f"{err_eig}")

# =====================================================================
# (B2) contraction bound, incl. the extremal directions
# =====================================================================
print()
print("(B2) ||D_BN d||^2 <= gamma^2/(s^2+eps) ||P d||^2")
print(f"{'eps':>9} {'s^2':>10} {'ratio(rand)':>13} {'ratio(d=1)':>12} "
      f"{'ratio(d=t)':>12} {'ratio(d _|_ t)':>15}")
for eps in [0.0, 1e-8, 1e-5, 1e-2, 1.0]:
    L = 40
    a = torch.randn(L) * 0.5
    P = torch.eye(L) - torch.ones(L, L) / L
    t = P @ a
    s2 = float(t @ t) / L
    gam = 1.0
    D = (gam / np.sqrt(s2 + eps)) * (P - torch.outer(t, t) / (L * (s2 + eps)))
    cap = gam**2 / (s2 + eps)

    def ratio(d):
        pd = float((P @ d) @ (P @ d))
        if pd < 1e-30:
            return 0.0
        return float((D @ d) @ (D @ d)) / (cap * pd)

    rr = max(ratio(torch.randn(L)) for _ in range(500))
    r1 = ratio(torch.ones(L))
    rt = ratio(t)
    gperp = P @ torch.randn(L)
    gperp = gperp - (gperp @ t) / (t @ t) * t
    rp = ratio(gperp)
    print(f"{eps:9g} {s2:10.5f} {rr:13.6f} {r1:12.6f} {rt:12.3e} {rp:15.6f}")
    check("(B2) bound", rr <= 1 + 1e-12 and rt <= 1 + 1e-12 and rp <= 1 + 1e-12)
    check("(B2) attained on centred _|_ t", abs(rp - 1.0) < 1e-12, f"{rp}")
    check("(B2) t-direction attenuated to (eps/(s2+eps))^2",
          abs(rt - (eps / (s2 + eps)) ** 2) < 1e-10, f"{rt}")

# =====================================================================
# (B2c) the necessity counterexample
# =====================================================================
print()
print("(B2c) necessity counterexample: rows (1,0),(0,1),(-1,-1) zero-padded")
print(f"{'M':>7} {'lam_max(centred Gram)':>22} {'lam_max(G_WW)':>16} "
      f"{'/M':>16} {'BN out (M-indep)':>20}")
base = None
for M in [2, 4, 16, 128, 1024, 8192]:
    Hm = torch.zeros(3, M)
    Hm[0, 0] = 1.0
    Hm[1, 1] = 1.0
    Hm[2, 0] = -1.0
    Hm[2, 1] = -1.0
    # centred feature covariance (1/L) sum (h - hbar)(h - hbar)^T ; mean is 0 here
    Hc = Hm - Hm.mean(0, keepdim=True)
    S = (Hc.T @ Hc) / 3.0
    lam_S = float(torch.linalg.eigvalsh(S)[-1])

    w = torch.zeros(M)
    w[0] = 1.0 / np.sqrt(M)
    L3 = 3
    P3 = torch.eye(L3) - torch.ones(L3, L3) / L3

    def bn_logit(wv, eps=0.0):
        a = Hm @ wv
        t = P3 @ a
        s2 = (t @ t) / L3
        return t / torch.sqrt(s2 + eps)

    z = bn_logit(w)
    # logits (z_n, 0) -> binary softmax
    p = torch.sigmoid(z)
    # J = d z / d w  (3 x M)
    J = torch.autograd.functional.jacobian(bn_logit, w)
    # GGN in w: (1/3) sum_n p(1-p) J_n J_n^T   [Hessian of binary CE wrt z_n]
    Gww = (J.T @ ((p * (1 - p)).unsqueeze(1) * J)) / L3
    lam_G = float(torch.linalg.eigvalsh(Gww)[-1])
    if base is None:
        base = lam_G / M
    print(f"{M:7d} {lam_S:22.12f} {lam_G:16.8f} {lam_G/M:16.12f} "
          f"{str([round(float(v),8) for v in z]):>20}")
    check("(B2c) centred Gram constant = 1", abs(lam_S - 1.0) < 1e-12, f"M={M} {lam_S}")
    check("(B2c) lam_max(G_WW) = M x const", abs(lam_G / M - base) < 1e-12,
          f"M={M} {lam_G/M} vs {base}")
print(f"  => centred Gram flat at 1 for every M, while lam_max(G_WW) = M x "
      f"{base:.12f}.  Necessity of a Theta(M) centred Gram is REFUTED.")

# with fixed eps > 0 the growth saturates
print()
print("  same counterexample at FIXED eps > 0 (stabilisation cutoff):")
for eps in [0.0, 1e-5, 1e-2]:
    row = []
    for M in [2, 128, 8192, 10**6]:
        Hm = torch.zeros(3, M)
        Hm[0, 0] = 1.0; Hm[1, 1] = 1.0; Hm[2, 0] = -1.0; Hm[2, 1] = -1.0
        w = torch.zeros(M); w[0] = 1.0 / np.sqrt(M)
        L3 = 3; P3 = torch.eye(L3) - torch.ones(L3, L3) / L3
        # closed form: only coords 0,1 of w matter, so restrict to a 3x2 problem
        H2 = Hm[:, :2]; w2 = w[:2]
        def f2(wv):
            a = H2 @ wv
            t = P3 @ a
            s2 = (t @ t) / L3
            return t / torch.sqrt(s2 + eps)
        z = f2(w2); p = torch.sigmoid(z)
        J2 = torch.autograd.functional.jacobian(f2, w2)
        G2 = (J2.T @ ((p * (1 - p)).unsqueeze(1) * J2)) / L3
        row.append(float(torch.linalg.eigvalsh(G2)[-1]))
    print(f"    eps={eps:<8g} lam_max(G_WW) at M=2,128,8192,1e6: "
          + ", ".join(f"{v:.4g}" for v in row))

# =====================================================================
# (B3) function-preserving scale identity, eps co-scaling, invariant
# =====================================================================
print()
print("(B3) J_W(cW) = c^{-1} J_W(W), G_WW(cW) = c^{-2} G_WW(W); BN_{c^2 eps}(c a) = BN_eps(a)")
Nb, Cin, Cout, H, W_, eps0 = 5, 3, 4, 6, 6, 0.0
h = torch.randn(Nb, Cin, H, W_)
Wt = torch.randn(Cout, Cin, 3, 3) * 0.5
head = torch.randn(2, Cout, 1, 1)          # 1x1 conv to 2 logits, downstream of BN


def net(Wv, c_eps):
    bn = torch.nn.BatchNorm2d(Cout, eps=c_eps, affine=True, track_running_stats=False)
    with torch.no_grad():
        bn.weight.fill_(1.0)
        bn.bias.zero_()
    bn.train()
    pre = torch.nn.functional.conv2d(h, Wv, padding=1)
    z = bn(pre)
    return torch.nn.functional.conv2d(z, head)     # (Nb,2,H,W)


print(f"{'c':>8} {'max|z(cW)-z(W)|':>18} {'||J(cW)||/||J(W)||':>20} {'x c':>10} "
      f"{'lam(cW)/lam(W)':>16} {'x c^2':>10} {'||W||^2 lam':>14}")
z0 = net(Wt, eps0)
J0 = torch.autograd.functional.jacobian(lambda Wv: net(Wv, eps0), Wt).reshape(-1, Wt.numel())
G0 = J0.T @ J0
lam0 = float(torch.linalg.eigvalsh(G0)[-1])
inv0 = float((Wt**2).sum()) * lam0
for c in [0.1, 0.5, 2.0, 7.0]:
    zc = net(c * Wt, c * c * eps0)
    Jc = torch.autograd.functional.jacobian(lambda Wv: net(Wv, c * c * eps0),
                                            c * Wt).reshape(-1, Wt.numel())
    Gc = Jc.T @ Jc
    lamc = float(torch.linalg.eigvalsh(Gc)[-1])
    invc = float(((c * Wt) ** 2).sum()) * lamc
    dz = float((zc - z0).abs().max())
    rJ = float(Jc.norm() / J0.norm())
    print(f"{c:8g} {dz:18.3e} {rJ:20.12f} {rJ*c:10.8f} {lamc/lam0:16.10f} "
          f"{lamc/lam0*c*c:10.8f} {invc/inv0:14.10f}")
    check("(B3) function preserved", dz < 1e-11, f"c={c} {dz}")
    check("(B3) J scaling", abs(rJ * c - 1) < 1e-9, f"c={c} {rJ*c}")
    check("(B3) G scaling", abs(lamc / lam0 * c * c - 1) < 1e-8, f"c={c}")
    check("(B3) ||W||^2 lam invariant", abs(invc / inv0 - 1) < 1e-8, f"c={c}")

# exact eps co-scaling identity on the BN map itself
print()
print("  eps co-scaling identity BN_{c^2 eps}(c a) = BN_eps(a):")
for eps in [1e-5, 1e-2, 1.0]:
    a = torch.randn(50) * 0.3
    P = torch.eye(50) - torch.ones(50, 50) / 50

    def BN(v, e):
        t = P @ v
        return t / torch.sqrt((t @ t) / 50 + e)
    worst = max(float((BN(c * a, c * c * eps) - BN(a, eps)).abs().max())
                for c in [0.03, 0.4, 3.0, 25.0])
    print(f"    eps={eps:<8g} max deviation over c in {{0.03,0.4,3,25}} = {worst:.3e}")
    check("(B3) eps co-scaling", worst < 1e-12, f"eps={eps} {worst}")

# departure at FIXED eps when pre-BN variance ~ eps
print()
print("  DEPARTURE at FIXED eps=1e-5: G_WW(cW)/G_WW(W) vs the exact c^{-2} law,")
print("  as the pre-BN variance s^2 approaches eps")
eps_fix = 1e-5
print(f"{'s^2(W)':>12} {'s^2/eps':>10} {'c':>7} {'exact c^-2':>12} "
      f"{'observed':>12} {'rel dev':>11}")
for scale in [1.0, 1e-1, 1e-2, 3e-3, 1e-3]:
    Wl = Wt * scale
    pre = torch.nn.functional.conv2d(h, Wl, padding=1)
    s2 = float(pre[:, 0].var(unbiased=False))
    Jl = torch.autograd.functional.jacobian(lambda Wv: net(Wv, eps_fix),
                                            Wl).reshape(-1, Wl.numel())
    laml = float(torch.linalg.eigvalsh(Jl.T @ Jl)[-1])
    for c in [0.5, 2.0]:
        Jc = torch.autograd.functional.jacobian(lambda Wv: net(Wv, eps_fix),
                                                c * Wl).reshape(-1, Wl.numel())
        lamc = float(torch.linalg.eigvalsh(Jc.T @ Jc)[-1])
        obs = lamc / laml
        print(f"{s2:12.3e} {s2/eps_fix:10.2f} {c:7g} {1/c**2:12.6f} {obs:12.6f} "
              f"{abs(obs*c*c-1):11.3e}")

# failure of the scalar invariant under CHANNELWISE rescaling
print()
print("  the scalar invariant ||W||^2 lam_max(G_WW) is NOT preserved by")
print("  CHANNELWISE (per-output-channel) rescaling, which is equally function-preserving:")
cs = torch.tensor([0.2, 1.0, 3.0, 8.0]).reshape(Cout, 1, 1, 1)
Wch = Wt * cs
zch = net(Wch, eps0)
Jch = torch.autograd.functional.jacobian(lambda Wv: net(Wv, eps0), Wch).reshape(-1, Wt.numel())
lamch = float(torch.linalg.eigvalsh(Jch.T @ Jch)[-1])
invch = float((Wch**2).sum()) * lamch
print(f"    max|z(diag(c)W)-z(W)| = {float((zch-z0).abs().max()):.3e}  (function preserved)")
print(f"    ||W||^2 lam:  {inv0:.6f}  ->  {invch:.6f}   (ratio {invch/inv0:.4f})")
check("(B3) channelwise is function-preserving", float((zch - z0).abs().max()) < 1e-11)
check("(B3) invariant BREAKS channelwise", abs(invch / inv0 - 1) > 1e-3,
      f"ratio={invch/inv0}")

# =====================================================================
# (B4) algebra
# =====================================================================
print()
print("(B4) squared BN gain gamma^2/(s^2+eps) with s^2 = c0/(M+K)")
c0, K, gam = 2.0, 64, 1.1
for eps in [0.0, 1e-5, 1e-2]:
    vals = [(M, gam**2 / (c0 / (M + K) + eps), gam**2 * (M + K) / (c0 + eps * (M + K)))
            for M in [48, 192, 384, 10**5]]
    for M, lhs, rhs in vals:
        check("(B4) algebra", abs(lhs - rhs) < 1e-9 * max(1, abs(lhs)), f"{lhs} {rhs}")
    print(f"  eps={eps:<8g} gain^2 at M=48,192,384,1e5: "
          + ", ".join(f"{v[1]:.4g}" for v in vals)
          + ("   (plateau at gamma^2/eps = %.4g)" % (gam**2 / eps) if eps > 0 else "   (no plateau)"))

print()
print("FAILURES:", FAIL if FAIL else "none")
