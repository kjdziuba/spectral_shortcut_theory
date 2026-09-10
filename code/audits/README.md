# `code/audits/` — durable numerical verification record for the math packet

These are the numerical audit scripts written while checking the review packet's two
mathematics notes:

- `review_packet/astra/math_01.md` — items **P1**–**P5** (Theorems 1.1, 1.3, 3.2, 5.1,
  Lemmas 3.1, 4.1, 4.3, 4.4, Counterexamples 1.2, 1.4, 2.2, 2.3, 3.3, 3.4);
- `review_packet/astra/math_02.md` — the reviewer's reply (finite-horizon attribution,
  the nonlinear shallow-ReLU bridge, and the isotropic-initialization section).

They lived in a session scratchpad and were re-run and persisted here on **2026-09-10**
at the reviewer's request, so that every claim in the packet has a script that can be
re-executed. Each file is **byte-identical to the original** apart from a header comment
recording origin, author, date, seed and the exact command. None of them read or write
repo files, and none of them depend on scratchpad-relative paths, so no path edits were
needed.

Layout: the scripts in this directory cover math_01.md, the paper propositions and
math_02.md §3 (Theorem N); `math02_sections_2_4_6/` holds the nine scripts behind the
audit of math_02.md §2, §4 and §6.

## How to run

From the repository root:

```bash
cd /home/u37314kd/Projects/spectral_shortcut_theory
VENV=/home/u37314kd/Projects/spectral_tokenization/venv/bin/python3
$VENV code/audits/audit_A1.py
$VENV code/audits/audit_a2.py
$VENV code/audits/audit_a2_bn.py
$VENV code/audits/check_p5.py
$VENV code/audits/audit_blocks.py
$VENV code/audits/check_theoremN.py
$VENV code/audits/astra_math_02_checks.py
```

Environment of the recorded run: Python 3.12.13, numpy 2.0.2, scipy 1.16.3,
torch 2.9.1+cu128, CPU only.

## Summary table

| script | verifies | seed | runtime | result |
|---|---|---|---|---|
| `audit_A1.py` | math_01 **P4** Lemmas 4.1/4.3/4.4 and eqs (4.2)–(4.7); **P1** Theorem 1.1 eqs (1.1)/(1.3), Counterexample 1.2, Theorem 1.3 eqs (1.5)–(1.7) with non-commuting kernels, Counterexample 1.4, and the P1-corollary constants (ReLU moments, `K_phi^V` identity, `E||J_theta||_F^2`, `E||e0||^2`, kernel concentration, `kappa_*`, `M_0`) | `np.random.default_rng(0)` | 14.6 s | **PASS on 18 of 19 checks; 1 FAIL** — the `share <= min(1, a/kappa)` sub-check of Theorem 1.3. Diagnosed below: it is a **quadrature artefact of the audit script**, not a failure of Theorem 1.3. |
| `audit_a2.py` | math_01 **P3** Lemma 3.1 witness identity, Theorem 3.2 probability/curvature-floor events, eq (3.6), Counterexamples 3.3 and 3.4; **P2** eq (2.2) CE counterexample, eq (2.5), Counterexamples 2.2/2.3 | `np.random.default_rng(0)` | 2.7 s | **PARTIAL (exit 1)** — every check up to line 168 passes; the script then aborts inside `torch.func.jacrev` over an `nn.BatchNorm1d` **module**. Environmental, not mathematical; see below. |
| `audit_a2_bn.py` | math_01 Counterexample 3.4 using the real `torch.nn.BatchNorm1d` in train mode (both `eps` regimes), the `S_h^p ~ M` companion statement, the mean-direction witness under non-constant features, plus **P2** eq (2.5) and Counterexamples 2.2/2.3 | `torch.manual_seed(1)` | 1.7 s | **PASS** |
| `check_p5.py` | math_01 **P5** Theorem 5.1: invariants (5.3)/(5.4), fitting-time bounds (5.5), residual envelope (5.8), GGN constants (5.9), gradient-norm bound (5.10), joint-vs-frozen gap (5.11), the (5.12) displacement table, and the `1/sqrt(M)` readout control | none (deterministic) | 0.4 s | **PASS** |
| `audit_blocks.py` | paper propositions `cor:attribution` and `prop:ntk_classprior` in the paper's own normalization (`03_setup.tex`): the exact gradient identity `grad_p L = J^T r`, the head-weight block `J^T w = c hbar^T` for a shared-`W` `1x1`-conv head fed by a pixel-mixing `3x3` conv, and the degenerate stationary-point counterexample to `cor:attribution` | `torch.manual_seed(0)`, then `(1)` | 1.0 s | **PASS** |
| `check_theoremN.py` | math_02 **§3 Theorem N**: the `Z` inequality, the three-term Jacobian decomposition identity, gate-flip containment and the encoder-Jacobian / feature / preactivation perturbation bounds, the (N5) moment bounds, the (N7) boundary mass, `E[ReLU^4] = 3 s^4/2` with the Frobenius/Chebyshev constant 42, and (A1) along the model's **nonlinear** flow | `np.random.default_rng(0)` plus per-block `default_rng(2..7)` | 2.6 s | **PASS** |
| `astra_math_02_checks.py` | math_02 §2.1 finite-horizon attribution with non-commuting PSD matrices (exact spectral integral); §4 (I1)–(I3) with general `a0`, signed context and unequal zero-sum readouts; §3 BN derivative + co-scaled-`eps` curvature law; §3 ReLU feature/gate tube bounds | `np.random.default_rng(902)` | 0.3 s | **PASS** (all `assert`s; 800 + 12 + 200 checks) |

### `math02_sections_2_4_6/` — the §2/§4/§6 audit

Nine scripts backing the ledger of
`review_packet/astra/audit_math02_sections_2_4_6_2026-09-10.md`, which audits
math_02.md **§2.1 (A1)**, **§2.2 (A2)**, **§2.3 (the six bullets)**, **§4 (I1)–(I6)** and
**§6 (B1)–(B4)**. Item labels (`A1.1`, `A2.4`, `b3`, `I2b`, `B2c`, …) refer to that
report's ledger tables. Each prints `FAILURES: none` on success.

```bash
cd /home/u37314kd/Projects/spectral_shortcut_theory
VENV=/home/u37314kd/Projects/spectral_tokenization/venv/bin/python3
for f in a1_finite_horizon a2_chernoff a2_chernoff_biting a2_chernoff_constant \
         c23_bullets i_toy_ode i6_reversal_mc b_bn_torch b_bn_crosschannel; do
  $VENV code/audits/math02_sections_2_4_6/$f.py
done
```

| script | verifies | seed | runtime | result |
|---|---|---|---|---|
| `a1_finite_horizon.py` | §2.1 (A1) finite-horizon loss attribution, items **A1.1–A1.7**: 24 nonlinear runs + sharpness + old-vs-new bound | 20260910 | 16.8 s | **PASS** (`FAILURES: none`) |
| `a2_chernoff.py` | §2.2 (A2) truncated matrix Chernoff, items **A2.1–A2.4, A2.6, A2.7**: 4e6-draw MC, 16 000 Chernoff trials | 424242 | 5.9 s | **PASS** (`FAILURES: none`) |
| `a2_chernoff_biting.py` | §2.2 items **A2.2, A2.3** in a second parameterisation: 8e6-draw MC, 120 000 trials | 909090 | 2.9 s | **PASS** (`FAILURES: none`) |
| `a2_chernoff_constant.py` | §2.2 item **A2.5** (constant stress on the extremal balls-in-bins family): 8 configs × 200 000 trials | 5150 | 1.8 s | **PASS** (`FAILURES: none`) |
| `c23_bullets.py` | §2.3 items **b3, b5, b6**: 12 ODE reductions, 1600 softmax draws, 10 000 clipping trials, PyTorch check | 246810 | 4.1 s | **PASS** (`FAILURES: none`) |
| `i_toy_ode.py` | §4 items **I1, I2, I2b, I3, I4, I5, I5b**: 24 full `(2+M)` DOP853 solves + 18 high-`M` roots | 31337 | 0.7 s | **PASS** (`FAILURES: none`) |
| `i6_reversal_mc.py` | §4 items **I6, I6b, I6c, I6d**: 3×5×40 000 MC + 20 000 per-draw + 3960 grid + 4e6 tail draws | 77007 | 2.0 s | **PASS** (`FAILURES: none`) |
| `b_bn_torch.py` | §6 items **B1, B1b, B2, B2b, B2c, B2d, B3, B3b, B3c, B3d, B4**: `torch.func.jvp` against the real `nn.BatchNorm2d` | 13579 | 7.2 s | **PASS** (`FAILURES: none`) |
| `b_bn_crosschannel.py` | §6 item **B2** channelwise summation: exact block-diagonality of `D_BN` across channels | 13579 | 1.6 s | **PASS** (prints its result directly rather than a `FAILURES:` line) |

`b_bn_torch.py` and `b_bn_crosschannel.py` emit a benign torch `UserWarning`
("Converting a tensor with requires_grad=True to a scalar") on stderr; both exit 0.

## Key printed numbers

### `audit_A1.py` — PASS on 18/19

```
[PASS] (4.2) discrete sum bound                        max sum/bound = 1.0000
[PASS] (4.4) left-Riemann bound with eta_max           max lhs/rhs   = 0.9969
[PASS] (4.7) Adam bound                                max ratio     = 0.6112 (loose by design)
[PASS] (1.1)/(1.3) Gronwall comparison and psi_alpha bound
[PASS] CE1.2 linearized expansion factor at t=1        numeric = 44.37709, formula = 44.37709
[FAIL] Thm 1.3 (1.5)-(1.7) + share<=a/kappa            max ratios: (1.5) 0.775, (1.6) 0.870, (1.7) 0.226
[PASS] CE1.4 M in {1, 10, 1000}: |zJ-zF| = 1-e^-t      (independent of M)
[PASS] E ReLU(N(0,s^2))^4 = 3 s^4/2                    4.2813 vs 4.2842
[PASS] K_phi^V == (HH^T/N) kron I_C
[PASS] E||J_theta||_F^2 <= C s_w^4 tr(Sigma_X)         MC = 7.65, bound = 15.51
[PASS] E||HH^T/(NM) - K_*||_F^2 <= 3 s_max^4/(2M)      MC = 5.56e-01, bound = 2.06e+01
kappa_* = 3.065e-03;  note's M_0 with rho = 0.1 : 1.31e+10
```

**The one FAIL, diagnosed.** The printed max ratios are all `< 1`, i.e. bounds (1.5),
(1.6) and (1.7) themselves all hold; the ratios simply do not cover the sub-check that
fails. Instrumenting the loop (200 trials) shows:

```
SUBCHECK FAIL COUNTS: {'1.5 pointwise': 0, '1.5 uniform': 0, '(1.6)': 0, '(1.7)': 0, 'share': 16}
max share = 1.7309   (trials with a/kappa > 1: 177)
```

So the failing assertion is `share <= min(1, a/kappa)`, and specifically its `share <= 1`
half. That half is an **exact** consequence of the energy identity
`d/dt (1/2 ||e_J||^2) = -e_J^T (K_phi + K_theta) e_J`, which forces
`int_0^inf e_J^T K_theta e_J dt <= 1/2 ||e0||^2`. The script computes that integral with
a rectangle rule, `np.sum(||A^T e_J||^2) * dt` on a 4000-point grid over `[0, 40/kappa]`,
which over-estimates a decaying integrand. Recomputing the same quantity three ways on
300 fresh trials:

| quadrature | trials with `share > 1` | max share |
|---|---|---|
| rectangle rule (as in `audit_A1.py`) | 17 / 300 | 1.9350 |
| trapezoid, same grid | 1 / 300 | 1.1207 |
| **exact spectral integral** | **0 / 300** | **0.9189** |

The `[FAIL]` is therefore an artefact of the audit script's own quadrature, not a defect
in Theorem 1.3. The same attribution claim is verified **exactly** (closed-form spectral
integral, no quadrature) by `astra_math_02_checks.py`, whose first block passes 800
non-commuting-kernel checks of `E/total <= a/(a+k)`. The script is kept unmodified so
that the record matches what was actually run; anyone re-running it should expect this
one line to read `[FAIL]`.

### `audit_a2.py` — PARTIAL

Everything up to line 168 passes:

```
P2 CE (2.2): |grad|^2 = 0.2500 > G |r|^2 = 0.1250; ratio grows to 50.51 at p = 0.01
Lemma 3.1: logit-perturbation formula max abs err 1.11e-16; witness quadform 0.020637
           <= lambda_max(G_WW) = 0.521144; lambda_max(S_h,a) = 0.029927
Theorem 3.2 (M = 16/64/256): Pr(Q >= A/4) = 0.708/0.690/0.725 (>= 1/3),
           Pr(U > A/8) = 0.000 (<= 1/12), Pr(joint) = 0.708/0.690/0.725 (>= 1/4);
           curvature floor on the event: min quadform / (c_p A/32) ~ 1e33 (>= 1 required)
eq (3.6):  MC E[v1^2 Q] = 0.7638 +- 0.0253 vs s_v^2/(2q) lambda_max(S_h) = 0.7320 (M = 16)
CE 3.3:    Pr(all preacts <= 0) = 0.493 (theory 1/2) -> G_WW = 0 exactly
CE 3.4:    eps = 1e-5: max |d logits/dW| = 1.42e-13; BN output == beta at every site
```

It then aborts with

```
RuntimeError: During a grad (vjp, jvp, grad, etc) transform, the function provided
attempted to call in-place operation (aten::add_.Tensor) that would mutate a captured
Tensor.
```

raised by `nn.BatchNorm1d.forward` updating `num_batches_tracked` in place under
`torch.func.jacrev` (torch 2.9.1 forbids this; the module is constructed with the default
`track_running_stats=True`). This is a **torch-version incompatibility in a redundant
re-check**, not a mathematical failure: the very next lines re-verify Counterexample 3.4
with the `nn.BatchNorm1d` *module*, which `audit_a2_bn.py` does correctly (with
`track_running_stats=False`) and which passes there —
`max |d logits/dW| = 3.55e-14` at `eps = 1e-5` and `1.53e-16` at `eps = 0.1`.

The checks after the abort — eq (2.5) and Counterexample 2.3 — are likewise duplicated
verbatim in `audit_a2_bn.py`, which passes them:
`numeric ratio = 0.360968, formula (2.5) = 0.360968, rel err = 1.5e-08`, and
`CE2.3: E_v/E_u = 43.667 / 5.517 / 1.157 / 1.000` at `T = 0.01 / 0.1 / 1 / 10` for a
curvature ratio `L/l = 100`. **Nothing in `audit_a2.py` is left unverified by the abort.**
The file is kept unmodified rather than patched, so the record matches what was run.

### `audit_a2_bn.py` — PASS

```
nn.BatchNorm1d(train, eps=1e-05): max |d logits/dW| = 3.55e-14; BN out == beta: True
nn.BatchNorm1d(train, eps=0.1)  : max |d logits/dW| = 1.53e-16; BN out == beta: True
d logits/d gamma max = 0.00e+00  (gamma is dead too)
lambda_max(S_h^p) = 9.651 = pmin * M   (proportional to M, while G_WW = 0 exactly)
non-constant features: channel-0 witness energy 32.47 (~M=32) incoming -> 0.400 post-BN
(2.5) numeric ratio = 0.360968 vs formula 0.360968, rel err 1.5e-08
CE2.2 (Euler, T=2): E_v = 0.00e+00, E_u = 0.4909  (theory 0 and (1-e^-4)/2 = 0.4908)
```

### `check_p5.py` — PASS

```
m = 2.944438979 = log 19,  B_m = 2.198831610,  Psi(m) = 20.944439
    M      a_M       v-1      disp   note-table    bound  note-table   ratio    M*a_M
   16  0.142325  0.166474  0.219021    0.219021  1.370601   1.370601   6.258   2.2772
   64  0.042094  0.057238  0.071050    0.071050  0.358465   0.358465   5.045   2.6940
  256  0.011216  0.016144  0.019658    0.019658  0.090662   0.090662   4.612   2.8712
 1024  0.002857  0.004181  0.005064    0.005064  0.022732   0.022732   4.489   2.9253
asymptotic bound/disp ratio -> 4.4464
Full (M+2) flow, M=16: v - cosh(sqrtM a) = -7.9e-13, b - sqrtM sinh = -2.1e-12, q - m = 8.9e-16
   (5.5) 6.099e-01 <= T_m = 9.078e-01 <= 1.232e+00; max r/envelope (5.8) = 1.000000 (<= 1)
   (5.11) max (q - q_F) = 0.382316 <= uniform bound 11.297333; min diff = 0.00e+00
1/sqrt(M) control: max |a_ctrl - a_M1| = 1.29e-11, max |b_ctrl - b_M1| = 8.45e-12
GGN (5.9): lam_theta / lam_phi / D / R / ||grad_theta|| all match their closed forms exactly
```

Every table entry reproduces the value printed in math_01.md.

### `audit_blocks.py` — PASS

```
N_cls = 3 and 2, bias True and False:
  head-block vs c hbar^T          max |diff| = 1.1e-16 .. 5.6e-17
  ||J^T w||^2 >= ||hbar||^2       2.32 .. 3.55  vs  1.316009      -> OK
  grad identity grad L = J^T r    max |diff| = 2.8e-17 .. 5.6e-17
cor:attribution stationary counterexample (tanh head, zero init, balanced classes):
  max |grad| over ALL params = 0.000e+00 (exact critical point), ||r|| = 0.816497 != 0
  => L(0) - L(T) = 0 for every T, so A_theta(T) = 0/0 is undefined,
     while ass:residual still holds on [0,T] with C = max(1, ||r||(1 + mu T)).
```

### `check_theoremN.py` — PASS

```
[1] Z inequality: max(lhs - rhs) = -2.706e-08                       (must be <= 0)
[2] three-term decomposition identity: max abs error = 1.332e-15
[3-5] M = 64, (r_theta, r_phi) = (0.05,0.2) / (0.5,0.5) / (1.0,1.0):
      gate flips outside the boundary set = 0 in all three;
      max(jac lhs - rhs) = max(feat lhs - rhs) = max(preact lhs - rhs) = 0.000e+00;
      boundary units = 64/64, B_M = 13.989 / 13.956 / 13.471
[6] (N5), M = 128:  E||H0||_F^2/M = 0.6056 <= s_max^2/2 = 1.3017
                    E||W0||_F^2/M = 1.2090 vs s_w^2 = 1.2100
                    E||V0||_F^2   = 1.2815 vs C s_v^2 = 1.2800
                    E||J_Theta,0||_F^2 = 2.4889 <= C s_v^2 s_w^2 X^2 = 10.0978
                    E||e0||^2 = 2.5493 <= 6.8827
[7] (N7) boundary mass:  M = 256  -> E B_M = 16.2521 <= C_bd = 273.4899
                         M = 1024 -> E B_M = 31.7319 <= C_bd = 273.4899
[8] E[ReLU(g)^4]/s^4 = 1.4985 (theory 1.5); E[ReLU(g)^2]/s^2 = 0.4994 (theory 0.5);
    the Frobenius/Chebyshev threshold constant is 6*7 = 42
[9] (A1) along the nonlinear flow (M = 40): encoder share = 0.0332, max over horizons
    0.0331, against a/(a+kappa) = 0.4162; L0 = 1.856 -> L(T) = 1.15e-02
```

Note that the widths actually exercised are `M = 64` for the perturbation bounds,
`M = 128` for the (N5) moments, `M = 256` and `M = 1024` for the (N7) boundary mass, and
`M = 40` for the (A1) nonlinear-flow instance. None of the checks relies on the very
large `M` thresholds that Theorem N's proof quotes.

### `astra_math_02_checks.py` — PASS

```
finite-horizon attribution: 800 checks passed
general-initialization toy: 12 full-parameter solves passed; max a/v error 1.145e-10
BN derivative and exact co-scaled-epsilon curvature law passed
ReLU feature/gate tube bounds: 200 random perturbations passed
```

### `math02_sections_2_4_6/` — all nine PASS

```
a1_finite_horizon.py  worst share/bound over 24 nonlinear runs (CE + Huber, time-varying
                      non-commuting kernels, T in {0.05,0.5,3,30}, DOP853 rtol 1e-11)
                      = 0.9992; the scalar-kernel sharpness check prints "(equality)" at
                      every T for all four (a, kappa) pairs (12 printed decimals);
                      (A1) = 0.200000 = the exact share where the old finite-T bound is
                      25.13 (vacuous) at a=1, kappa=4, T=1e-3.        FAILURES: none

a2_chernoff.py        max ||X||/max_n g_n^2 = 0.9889 <= 1; chosen R = 27.85 s_max^2 gives
                      lambda_min(E[X 1_R]) = 6.781e-4 >= kappa_*/2 = 3.391e-4;
                      PSD-monotonicity residual -5.4e-13; Tropp's exact delta=1/2 exponent
                      coefficient 0.076713 = 1/13.036 vs Astra's 1/16 (conservative by
                      1.2274x); threshold M >= 2.404e6 at eta=0.1 vs Cor 1.5's 3.124e8
                      (ratio 7.7e-3).                                  FAILURES: none

a2_chernoff_biting.py second parameterisation: tail bound 6.0693e-4 <= kappa_*/2 = 6.1218e-4,
                      MC E[||X|| 1_{>R}] = 1.5973e-5 (slack 38.0x); empirical failure
                      frequency <= Astra's bound at every M in {40..1280}. FAILURES: none

a2_chernoff_constant.py  extremal balls-in-bins family, 8 configs x 200 000 trials:
                      empirical / exact-Tropp in [0.099, 0.227], never exceeded; the
                      simplified 1/16 form never violated.              FAILURES: none

c23_bullets.py        b3: full (2+M) flow vs the 3-variable reduction, max discrepancy
                      7.5e-12, beta_i - beta_j drift <= 5.3e-15, closure also holds for
                      NONZERO sum(beta_0); b5: max|H 1| = 3.8e-16, max|sum r| = 5.3e-16,
                      max ratio to ||r||^2/lambda_min^+ = 1.000000; b6: PyTorch
                      clip_grad_norm_ + SGD(momentum) == form A to 4.441e-16, and 10 000
                      random trials find no form-B violation of (4.6)-with-a_i
                      (max ratio 1.000000).                             FAILURES: none

i_toy_ode.py          24 full (2+M) DOP853 solves (rtol 3e-12), M in {1,16,256,4096},
                      6 cases incl. a0=0, v0<0 and |v0|=0.03, unequal zero-sum beta_0:
                      (I2), (I3) and the (I5) bracket held in every solve; u_M strictly
                      decreasing in M in all 6 cases; at a0=1.7, m=2.5, M=1e10 the (I4)
                      table gives a(T)/a0 = 1.000000, a(T)/m = 0.680000 = a0/m (NOT 0)
                      and (a-a0)/Delta = 6.25e-10.                      FAILURES: none

i6_reversal_mc.py     |MC - Phi(m/2 sigma)| <= 0.0025 (<= 1.4 s.e.) at m in {0.5,2,5},
                      M up to 1e14; 0 of 20 000 per-draw disagreements with 1{a0<m/2} at
                      M=1e16; sufficient condition gave error 1 in 2160/2160 combinations
                      and I6b correctness in 1800/1800; the |v0| tail bound is exactly
                      tight (P(|v0|<0.03133) = 0.02499591 vs rho/2 = 0.025).
                                                                        FAILURES: none

b_bn_torch.py         torch.func.jvp through the real nn.BatchNorm2d (train mode, incl.
                      eps=0 and gamma<0): max|jvp - D_BN d| = 2.7e-15; spectrum
                      {0, 1, eps/(s^2+eps)} to 1.7e-16; (B2c) lambda_max(G_WW)/M =
                      0.168881741275 at every M in {2,...,8192} (12 digits) with the
                      centred Gram top eigenvalue 1.000000000000; (B3) outputs preserved
                      to 4.0e-15 with ||J(cW)||/||J(W)|| x c = 1.00000000 and
                      lambda(cW)/lambda(W) x c^2 = 1.00000000; (B3b) BN_{c^2 eps}(ca) =
                      BN_eps(a) to 1.1e-15;
                      (B3d) uniform rescaling leaves ||W||^2 lambda_max invariant
                      but CHANNELWISE rescaling moves it 235.3x.        FAILURES: none

b_bn_crosschannel.py  max |d(output)| in channels other than the perturbed one = 0.000e+00
                      -> D_BN is block diagonal across channels, so the channelwise sum
                      in (B2) is exact.
```

## Related verification scripts already in the repository

These are not copies; they are the standing numerical checks for the same packet and live
under `code/experiments/`:

| script | verifies | outputs |
|---|---|---|
| `code/experiments/toy_serial_ce.py` | math_01 Theorem 5.1 (P5) with the special initialization `a(0)=0, v(0)=1, beta(0)=0`: eqs (5.3)–(5.12), autograd gradient check, plain-GD convergence, GGN blocks, `1/sqrt(M)` control | `results/toy_serial_ce.csv` |
| `code/experiments/toy_serial_ce_isotropic.py` | math_02 §4 eqs (I1)–(I6): the same serial-CE model with **arbitrary** `a0`, `v0` and **unequal zero-sum** readouts; suppression (I4) vs the false pair of limits, displacement (I3), fitting time, the frozen-comparator gap (I5), the reversal test, and the Gaussian average (I6) | `results/toy_serial_ce_isotropic.csv`, `results/toy_serial_ce_isotropic_gaussian.csv`, `results/toy_serial_ce_isotropic_REPORT.md` |
| `code/experiments/toy_interior_witness.py` | math_01 P3 interior-witness results (Lemma 3.1, Theorem 3.2, eq (3.6)) | `results/toy_interior_witness.csv` |

## Provenance

Origin of the copies is recorded in each file's header. The originals remain in the
session scratchpad
`/tmp/claude-1008/-home-u37314kd-Projects-spectral-tokenization/887fb000-54db-4ab4-82e6-4eeab4fb6c48/scratchpad/`
(nothing there was deleted). `astra_math_02_checks.py` is the reviewer's own script,
copied from `review_packet/astra/math_02_checks.py` (author: Astra, 2026-09-10) and left
in place there as well.
