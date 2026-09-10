# ---------------------------------------------------------------------------
# VERIFICATION RECORD -- copied unchanged into code/audits/ as a durable audit
# artefact.  Only this header was added.
# origin  : audit of review_packet/astra/math_02.md section 2.3 -- the six bullets
#           (session scratchpad 887fb000-.../scratchpad/audit_math02_opus/c23_bullets.py)
#           report: review_packet/astra/audit_math02_sections_2_4_6_2026-09-10.md
# author  : Claude (math_02 sections 2/4/6 auditor)
# date    : 2026-09-10
# seed    : 246810
# command : venv/bin/python3 code/audits/math02_sections_2_4_6/c23_bullets.py
# ---------------------------------------------------------------------------
"""
AUDIT section 2.3 bullets.

 (b3) Unequal replicated readouts keep (a,v,b) closed:  derivatives identical
      => beta_i - beta_j constant; the loss depends only on b = sum beta_j.
      (Checked as a REDUCTION: the full (2+M) flow projects exactly onto the
       3-variable flow for ANY beta0, and only the SUM enters.)
 (b5) softmax at finite logits: r = p - y in range(H), H = diag(p) - p p^T;
      r^T H^dagger r <= ||r||^2 / lambda_min^+(H); and on confidently CORRECT
      one-hot examples the quadratic form SHRINKS (it diverges only when
      confidently WRONG) -- so "it diverges as predictions saturate" is false.
 (b6) clipping BEFORE momentum vs clipping the accumulated velocity:
      v_{k+1} = beta v_k + a_k g_k     -> (4.6) holds WITH the a_i factors
      v_{k+1} = a_k (beta v_k + g_k)   -> (4.6) needs a_i replaced by 1
      Also confirm which one PyTorch's clip_grad_norm_ + SGD(momentum) is.

Seed: 246810
"""
import numpy as np
import torch

rng = np.random.default_rng(246810)
FAIL = []


def check(name, cond, info=""):
    if not cond:
        FAIL.append(f"{name}: {info}")
        print(f"  FAIL {name}  {info}")


# =====================================================================
# (b3) closure of (a, v, b) with unequal replicated readouts
# =====================================================================
print("=" * 78)
print("(b3) unequal replicated readouts keep the (a,v,b) dynamics closed")
print("=" * 78)
from scipy.integrate import solve_ivp
from scipy.special import expit

print(f"{'M':>6} {'beta0 spread':>13} {'sum beta0':>11} {'max|full-reduced|':>18} "
      f"{'max|d(beta_i-beta_j)|':>22}")
for M in [1, 5, 64, 500]:
    for spread in [0.0, 0.3, 3.0]:
        beta0 = rng.normal(scale=spread, size=M) if spread > 0 else np.zeros(M)
        b0 = beta0.sum()                       # deliberately NOT zero when spread>0
        a0, v0 = 0.4, -0.9
        T = 3.0

        def full(t, y):
            a, v = y[0], y[1]
            b = y[2:].sum()
            r = expit(-(a + b * v))
            return np.r_[r, b * r, np.full(M, v * r)]

        def red(t, y):
            a, v, b = y
            r = expit(-(a + b * v))
            return [r, b * r, M * v * r]

        sF = solve_ivp(full, [0, T], np.r_[a0, v0, beta0], rtol=1e-12, atol=1e-14,
                       method="DOP853", dense_output=True)
        sR = solve_ivp(red, [0, T], [a0, v0, b0], rtol=1e-12, atol=1e-14,
                       method="DOP853", dense_output=True)
        ts = np.linspace(0, T, 200)
        YF, YR = sF.sol(ts), sR.sol(ts)
        avbF = np.vstack([YF[0], YF[1], YF[2:].sum(0)])
        err = float(np.max(np.abs(avbF - YR)))
        # beta_i - beta_j constant
        d = YF[2:] - YF[2:][0:1]
        d0 = (beta0 - beta0[0])[:, None]
        derr = float(np.max(np.abs(d - d0)))
        print(f"{M:6d} {spread:13.2f} {b0:11.4f} {err:18.3e} {derr:22.3e}")
        check("(b3) full == reduced", err < 1e-8, f"M={M} {err}")
        check("(b3) beta_i-beta_j constant", derr < 1e-9, f"M={M} {derr}")
print("  => closure holds for ANY beta0 (equal init is needed only for beta_j = b/M);")
print("     with sum beta0 = 0 the old aggregate formulas apply verbatim.")

# =====================================================================
# (b5) softmax pseudoinverse
# =====================================================================
print()
print("=" * 78)
print("(b5) softmax at finite logits: r in range(H), and the saturation behaviour")
print("=" * 78)
# range(H) = ker(H)^perp = 1^perp exactly when every p_c > 0 (H symmetric, rank C-1).
# So "r in range(H)" is EXACTLY "sum_c r_c = 0", which holds since r = p - y with
# sum p = sum y = 1.  We verify (i) H 1 = 0 exactly, (ii) rank(H) = C-1,
# (iii) sum r = 0, and (iv) the pseudoinverse quadratic form computed STABLY on an
# orthonormal basis of 1^perp (np.linalg.pinv with a fixed rcond silently truncates
# the true smallest positive eigenvalue at saturated logits and is not usable here).
worst_res = 0.0
worst_ratio = 0.0
worst_sum = 0.0
for C in [2, 3, 5, 12]:
    B = np.linalg.qr(np.eye(C) - np.ones((C, C)) / C)[0][:, : C - 1]   # basis of 1^perp
    for _ in range(400):
        z = rng.normal(scale=rng.uniform(0.1, 8.0), size=C)
        p = np.exp(z - z.max())
        p /= p.sum()
        y = np.zeros(C)
        y[rng.integers(C)] = 1.0
        r = p - y
        Hm = np.diag(p) - np.outer(p, p)
        worst_sum = max(worst_sum, abs(r.sum()))
        worst_res = max(worst_res, float(np.abs(Hm @ np.ones(C)).max()))
        check("(b5) H 1 = 0", float(np.abs(Hm @ np.ones(C)).max()) < 1e-15)
        check("(b5) r in range(H) <=> sum r = 0", abs(r.sum()) < 1e-14)
        Hs = B.T @ Hm @ B                      # H restricted to 1^perp, invertible
        rs = B.T @ r
        q = float(rs @ np.linalg.solve(Hs, rs))
        lam_pos = float(np.linalg.eigvalsh(Hs)[0])
        check("(b5) r'H^+ r <= ||r||^2/lam_min^+",
              q <= (r @ r) / lam_pos * (1 + 1e-9) + 1e-12,
              f"{q} > {(r@r)/lam_pos}")
        worst_ratio = max(worst_ratio, q / ((r @ r) / lam_pos))
print(f"  max |H 1|_inf over 1600 draws        = {worst_res:.3e}  (1 in ker H exactly)")
print(f"  max |sum_c r_c| over 1600 draws      = {worst_sum:.3e}  (=> r in range(H) = 1^perp)")
print(f"  max (r'H^+ r)/(||r||^2/lam_min^+)    = {worst_ratio:.6f}  (<= 1: uniform lower")
print(f"     softmax curvature suffices to replace r'H^+ r by const * ||r||^2)")

print()
print("  saturation behaviour, binary case p = (1-e, e), y = e_1:")
print(f"{'eps':>10} {'CORRECT r H^+ r':>18} {'WRONG r H^+ r':>16}")
for e in [1e-1, 1e-2, 1e-4, 1e-8]:
    # confidently CORRECT: p_true = 1-e
    p = np.array([1 - e, e]); y = np.array([1.0, 0.0]); r = p - y
    Hm = np.diag(p) - np.outer(p, p)
    qc = float(r @ np.linalg.pinv(Hm, rcond=1e-16) @ r)
    # confidently WRONG: p_true = e
    p2 = np.array([e, 1 - e]); r2 = p2 - y
    H2 = np.diag(p2) - np.outer(p2, p2)
    qw = float(r2 @ np.linalg.pinv(H2, rcond=1e-16) @ r2)
    print(f"{e:10.0e} {qc:18.6e} {qw:16.6e}")
    check("(b5) correct shrinks", qc < 2 * e / (1 - e) + 1e-9, f"{qc}")
    check("(b5) wrong diverges", qw > (1 - e) / e * 0.5, f"{qw}")
print("  => the quadratic form -> 0 on confidently CORRECT examples (e/(1-e)) and")
print("     -> infinity only on confidently WRONG ones ((1-e)/e).  Astra is right:")
print("     'it diverges as predictions saturate' is NOT universal.")

# =====================================================================
# (b6) clipping order
# =====================================================================
print()
print("=" * 78)
print("(b6) clipping BEFORE momentum vs clipping the accumulated velocity")
print("=" * 78)
beta, K, eta, cmax = 0.9, 60, 0.05, 1.0
g = rng.normal(size=(K, 7)) * rng.uniform(0.2, 6.0, size=(K, 1))
a = np.minimum(1.0, cmax / np.linalg.norm(g, axis=1))     # per-step clip factors

# form A: clip gradients, then accumulate  (PyTorch clip_grad_norm_ + SGD momentum)
vA = np.zeros(7); thA = np.zeros(7)
for k in range(K):
    vA = beta * vA + a[k] * g[k]
    thA = thA - eta * vA
# form B: accumulate, then clip the velocity
vB = np.zeros(7); thB = np.zeros(7)
for k in range(K):
    vB = beta * vB + g[k]
    s = min(1.0, cmax / np.linalg.norm(vB))
    vB = s * vB
    thB = thB - eta * vB

bound_with_a = eta * sum((1 - beta ** (K - i)) / (1 - beta) * a[i] * np.linalg.norm(g[i])
                         for i in range(K))
bound_no_a = eta * sum((1 - beta ** (K - i)) / (1 - beta) * np.linalg.norm(g[i])
                       for i in range(K))
print(f"  ||theta_K - theta_0||   form A (clip-then-accumulate) = {np.linalg.norm(thA):.6f}")
print(f"  ||theta_K - theta_0||   form B (clip the velocity)    = {np.linalg.norm(thB):.6f}")
print(f"  (4.6) bound WITH a_i   = {bound_with_a:.6f}")
print(f"  (4.6) bound with a_i=1 = {bound_no_a:.6f}")
check("(b6) form A obeys (4.6) with a_i", np.linalg.norm(thA) <= bound_with_a + 1e-12)
check("(b6) form B obeys the a_i=1 bound", np.linalg.norm(thB) <= bound_no_a + 1e-12)
viol = np.linalg.norm(thB) > bound_with_a
print(f"  does form B violate the WITH-a_i bound here?  {viol}  "
      f"(ratio {np.linalg.norm(thB)/bound_with_a:.4f})")

# Random search for a form-B violation of the WITH-a_i bound, plus the one-line
# induction showing there is none:
#     ||v_{k+1}^B|| <= min(c, beta||v_k^B|| + ||g_k||) <= sum_i beta^{k-i} a_i ||g_i||
# (if ||g_k|| <= c use the second argument; if ||g_k|| > c then min = c = a_k||g_k||).
print()
print("  random search for a form-B violation of the WITH-a_i bound (10000 trials):")
worst = 0.0
worst_v = 0.0
for _ in range(10000):
    Kx = int(rng.integers(2, 60))
    bx = float(rng.uniform(0.0, 0.995))
    cx = float(10 ** rng.uniform(-2, 2))
    dm = int(rng.integers(1, 5))
    gx = rng.normal(size=(Kx, dm)) * (10 ** rng.uniform(-2, 2, size=(Kx, 1)))
    ax = np.minimum(1.0, cx / np.linalg.norm(gx, axis=1))
    ex = float(10 ** rng.uniform(-2, 0))
    vB = np.zeros(dm); thB = np.zeros(dm)
    vnorm_bnd = 0.0
    ok_v = True
    for k in range(Kx):
        raw = bx * vB + gx[k]
        vB = min(1.0, cx / np.linalg.norm(raw)) * raw
        thB = thB - ex * vB
        vnorm_bnd = bx * vnorm_bnd + ax[k] * np.linalg.norm(gx[k])
        if np.linalg.norm(vB) > vnorm_bnd + 1e-12:
            ok_v = False
    bwa = ex * sum((1 - bx ** (Kx - i)) / (1 - bx) * ax[i] * np.linalg.norm(gx[i])
                   for i in range(Kx)) if bx < 1 else np.inf
    worst = max(worst, np.linalg.norm(thB) / max(bwa, 1e-300))
    if not ok_v:
        worst_v = 1.0
print(f"    max ||theta_K^B|| / [(4.6) WITH a_i] = {worst:.6f}   "
      f"(a violation would exceed 1)")
print(f"    per-step ||v_k^B|| <= sum beta^(k-i) a_i ||g_i||  violated in any trial? "
      f"{bool(worst_v)}")
check("(b6) form B also obeys the with-a_i bound", worst <= 1.0 + 1e-9, f"{worst}")
check("(b6) form B velocity induction", worst_v == 0.0)
print("    => (4.6) WITH the a_i factors is in fact TRUE for form B as well, by that")
print("       induction.  Our A1-E4 correction was about what the COARSE recurrence")
print("       ||v_{k+1}|| <= beta||v_k|| + ||g_k|| proves, not about the truth of (4.6).")

# Confirm PyTorch's actual arm is form A
print()
print("  PyTorch check: clip_grad_norm_ then SGD(momentum) == form A?")
pp = torch.zeros(7, requires_grad=True, dtype=torch.float64)
opt = torch.optim.SGD([pp], lr=eta, momentum=beta, dampening=0.0, nesterov=False)
for k in range(K):
    opt.zero_grad()
    pp.grad = torch.tensor(g[k], dtype=torch.float64)
    torch.nn.utils.clip_grad_norm_([pp], cmax)
    opt.step()
# torch's clip_grad_norm_ uses clamp(max_norm/(total_norm + 1e-6), max=1.0):
# mirror the 1e-6 exactly (this is the repo's own documented convention).
a_t = np.minimum(1.0, cmax / (np.linalg.norm(g, axis=1) + 1e-6))
vT = np.zeros(7); thT = np.zeros(7)
for k in range(K):
    vT = beta * vT + a_t[k] * g[k]
    thT = thT - eta * vT
d = float(np.abs(pp.detach().numpy() - thT).max())
print(f"    max|torch - form A (with torch's 1e-6)| = {d:.3e}  -> PyTorch implements v_{{k+1}} = beta v_k + a_k g_k")
check("(b6) PyTorch == form A", d < 1e-12, f"{d}")

print()
print("FAILURES:", FAIL if FAIL else "none")
