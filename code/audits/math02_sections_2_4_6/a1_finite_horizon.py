# ---------------------------------------------------------------------------
# VERIFICATION RECORD -- copied unchanged into code/audits/ as a durable audit
# artefact.  Only this header was added.
# origin  : audit of review_packet/astra/math_02.md section 2.1 -- (A1) finite-horizon loss attribution
#           (session scratchpad 887fb000-.../scratchpad/audit_math02_opus/a1_finite_horizon.py)
#           report: review_packet/astra/audit_math02_sections_2_4_6_2026-09-10.md
# author  : Claude (math_02 sections 2/4/6 auditor)
# date    : 2026-09-10
# seed    : 20260910
# command : venv/bin/python3 code/audits/math02_sections_2_4_6/a1_finite_horizon.py
# ---------------------------------------------------------------------------
"""
AUDIT (A1) -- finite-horizon loss-attribution bound
    int_0^T ||grad_theta L||^2 dt / (L(0)-L(T))  <=  a/(a+kappa).

INDEPENDENT of astra/math_02_checks.py, which only tests CONSTANT kernels with
an exact spectral integral for the squared loss.  Here we test the three
strengthenings Astra explicitly claims:

  (1) time-varying (genuinely nonlinear, hence non-commuting and non-constant)
      logit kernels,
  (2) an arbitrary differentiable loss (multiclass softmax cross-entropy, and a
      Huber-like loss),
  (3) the inequalities need hold only on the ACTUAL residual vectors r=grad_z L
      along the trajectory -- i.e. no common invariant subspace hypothesis.

We also (4) check the sharpness claim (scalar kernels give equality) and
(5) compare against the OLD finite-T bound  a||e0||^2 / (2 kappa (L0-LT))
that we asserted in claude_math_reply_01.md (A1-E2).

Seed: 20260910
Run:  venv/bin/python3 a1_finite_horizon.py
"""
import numpy as np
from scipy.integrate import solve_ivp

rng = np.random.default_rng(20260910)
FAIL = []


def check(name, cond, info=""):
    if not cond:
        FAIL.append(f"{name}: {info}")
        print(f"  FAIL {name}  {info}")


# ----------------------------------------------------------------------
# A genuinely nonlinear two-block model:   z(theta, phi) in R^n
#     z = G tanh(P theta)  +  Q softplus(S phi)
# theta in R^p (small block, small Jacobian), phi in R^q with q >> n so that
# K_phi = J_phi J_phi^T is coercive.  Both blocks unit-rate Euclidean flow.
# ----------------------------------------------------------------------
def make_model(n, p, q, scale_theta, scale_phi, rng):
    G = rng.normal(size=(n, p)) * scale_theta
    P = rng.normal(size=(p, p))
    Q = rng.normal(size=(n, q)) * scale_phi / np.sqrt(q)
    S = rng.normal(size=(q, q)) / np.sqrt(q)

    def z_of(theta, phi):
        return G @ np.tanh(P @ theta) + Q @ np.logaddexp(0.0, S @ phi)

    def jacs(theta, phi):
        dt = (1.0 - np.tanh(P @ theta) ** 2)          # (p,)
        Jth = G * dt[None, :] @ P                      # (n,p)
        sg = 1.0 / (1.0 + np.exp(-(S @ phi)))          # (q,)
        Jph = (Q * sg[None, :]) @ S                    # (n,q)
        return Jth, Jph

    return z_of, jacs


def losses(kind, n, C, rng):
    """Return (L(z), grad_z L(z)) for a differentiable loss on z in R^n."""
    if kind == "ce":
        assert n % C == 0
        B = n // C
        y = rng.integers(0, C, size=B)

        def L(z):
            Z = z.reshape(B, C)
            lse = np.logaddexp.reduce(Z, axis=1)
            return float(np.mean(lse - Z[np.arange(B), y]))

        def gL(z):
            Z = z.reshape(B, C)
            Pm = np.exp(Z - Z.max(1, keepdims=True))
            Pm /= Pm.sum(1, keepdims=True)
            Pm[np.arange(B), y] -= 1.0
            return (Pm / B).ravel()

        return L, gL
    if kind == "huber":
        t = rng.normal(size=n)
        d = 0.8

        def L(z):
            r = z - t
            ab = np.abs(r)
            return float(np.sum(np.where(ab <= d, 0.5 * r**2, d * (ab - 0.5 * d))))

        def gL(z):
            r = z - t
            return np.clip(r, -d, d)

        return L, gL
    raise ValueError(kind)


def run_case(kind, n, C, p, q, scale_theta, scale_phi, T, rng):
    z_of, jacs = make_model(n, p, q, scale_theta, scale_phi, rng)
    L, gL = losses(kind, n, C, rng)
    th0 = rng.normal(size=p)
    ph0 = rng.normal(size=q)

    def rhs(t, w):
        th, ph = w[:p], w[p : p + q]
        Jth, Jph = jacs(th, ph)
        r = gL(z_of(th, ph))
        gth = Jth.T @ r
        gph = Jph.T @ r
        # extra coordinate: running integral of ||grad_theta L||^2
        return np.r_[-gth, -gph, gth @ gth]

    w0 = np.r_[th0, ph0, 0.0]
    sol = solve_ivp(rhs, [0.0, T], w0, rtol=1e-11, atol=1e-13,
                    method="DOP853", dense_output=True)
    assert sol.success

    # sup / inf Rayleigh quotients on the ACTUAL residual, densely sampled
    ts = np.linspace(0.0, T, 4001)
    a_hat, k_hat = 0.0, np.inf
    for t in ts:
        w = sol.sol(t)
        th, ph = w[:p], w[p : p + q]
        Jth, Jph = jacs(th, ph)
        r = gL(z_of(th, ph))
        nr2 = r @ r
        if nr2 <= 0:
            continue
        a_hat = max(a_hat, (Jth.T @ r) @ (Jth.T @ r) / nr2)
        k_hat = min(k_hat, (Jph.T @ r) @ (Jph.T @ r) / nr2)

    wT = sol.sol(T)
    L0 = L(z_of(th0, ph0))
    LT = L(z_of(wT[:p], wT[p : p + q]))
    num = wT[-1]
    share = num / (L0 - LT)

    # energy identity sanity: L0-LT == int (r'K_th r + r'K_ph r)
    def rhs2(t, y):
        w = sol.sol(t)
        th, ph = w[:p], w[p : p + q]
        Jth, Jph = jacs(th, ph)
        r = gL(z_of(th, ph))
        return [(Jth.T @ r) @ (Jth.T @ r) + (Jph.T @ r) @ (Jph.T @ r)]

    tot = solve_ivp(rhs2, [0, T], [0.0], rtol=1e-11, atol=1e-14,
                    method="DOP853").y[0, -1]
    return dict(share=share, a=a_hat, k=k_hat, bound=a_hat / (a_hat + k_hat),
                L0=L0, LT=LT, energy_rel=abs(tot - (L0 - LT)) / max(L0 - LT, 1e-30),
                num=num)


print("=" * 78)
print("(A1) finite-horizon attribution, NONLINEAR time-varying kernels")
print("=" * 78)
print(f"{'loss':>6} {'T':>7} {'a':>10} {'kappa':>10} {'share':>10} "
      f"{'a/(a+k)':>10} {'ratio':>8} {'energyId':>10}")
worst_ratio = 0.0
for kind, n, C in [("ce", 24, 3), ("huber", 20, 1)]:
    for scale_theta, scale_phi in [(0.3, 3.0), (1.0, 1.0), (2.5, 0.6)]:
        for T in [0.05, 0.5, 3.0, 30.0]:
            r = run_case(kind, n, C, 5, 60, scale_theta, scale_phi, T, rng)
            ratio = r["share"] / r["bound"]
            worst_ratio = max(worst_ratio, ratio)
            print(f"{kind:>6} {T:7.2f} {r['a']:10.4f} {r['k']:10.4f} "
                  f"{r['share']:10.6f} {r['bound']:10.6f} {ratio:8.4f} "
                  f"{r['energy_rel']:10.2e}")
            check("A1 bound", r["share"] <= r["bound"] + 1e-9,
                  f"share={r['share']} > {r['bound']}")
            check("A1 energy identity", r["energy_rel"] < 1e-6,
                  f"rel={r['energy_rel']}")
print(f"worst share/bound ratio over 24 nonlinear runs: {worst_ratio:.4f}")

# ----------------------------------------------------------------------
# Sharpness: scalar kernels K_theta = a I, K_phi = kappa I  ->  equality.
# Realise exactly with a linear model z = z0 + sqrt(a) theta + sqrt(k) phi
# (n-dim blocks) and squared loss.
# ----------------------------------------------------------------------
print()
print("Sharpness (scalar kernels): share should EQUAL a/(a+kappa) at every T")
n = 6
for (a, kap) in [(1.0, 1.0), (0.25, 4.0), (9.0, 0.5), (1e-3, 1.0)]:
    A = np.sqrt(a) * np.eye(n)
    B = np.sqrt(kap) * np.eye(n)
    e0 = rng.normal(size=n)
    for T in [0.01, 0.7, 12.0]:
        # e(t) = exp(-(a+kap) t) e0 ; num = int a||e||^2 ; denom = (||e0||^2-||eT||^2)/2
        g = a + kap
        num = a * (e0 @ e0) * (1 - np.exp(-2 * g * T)) / (2 * g)
        den = (e0 @ e0) * (1 - np.exp(-2 * g * T)) / 2
        share = num / den
        check("A1 sharpness", abs(share - a / (a + kap)) < 1e-14,
              f"share={share} vs {a/(a+kap)}")
    print(f"  a={a:<8g} kappa={kap:<8g} share={share:.12f} "
          f"a/(a+k)={a/(a+kap):.12f}  (equality)")

# ----------------------------------------------------------------------
# (item i) A1 vs the OLD finite-T bound  a||e0||^2 / (2 kappa (L0-LT)).
# Show A1 is uniformly tighter, and that the old bound is vacuous (>1) for
# small T while A1 is never vacuous.
# ----------------------------------------------------------------------
print()
print("(item i) A1 vs the old finite-T bound a||e0||^2/(2 kappa (L0-LT))")
print(f"{'a':>8} {'kappa':>8} {'T':>8} {'share':>10} {'A1':>10} {'old':>12}")
for (a, kap) in [(1.0, 4.0), (0.2, 20.0)]:
    for T in [1e-3, 0.05, 1.0, 50.0]:
        g = a + kap
        E0 = 1.0
        num = a * E0 * (1 - np.exp(-2 * g * T)) / (2 * g)
        den = E0 * (1 - np.exp(-2 * g * T)) / 2
        share = num / den
        a1 = a / (a + kap)
        old = a * E0 / (2 * kap * den)
        print(f"{a:8g} {kap:8g} {T:8g} {share:10.6f} {a1:10.6f} {old:12.4g}")
        check("A1 tighter than old", a1 <= old + 1e-12, f"{a1} > {old}")

print()
print("FAILURES:", FAIL if FAIL else "none")
