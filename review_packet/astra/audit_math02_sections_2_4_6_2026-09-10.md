# Independent audit of `math_02.md` — §2.1, §2.2, §2.3, §4, §6

Auditor: Claude (adversarial proof-and-numerics pass). Date: 2026-09-10.
Scope: §2.1 (A1), §2.2 (A2), §2.3 bullets, §4 (I1)–(I6), §6 (B1)–(B4).
**Out of scope (another auditor): §3 Theorem N**, §1, §5, §7, §8.

All numerics below are **new scripts written from the statements, not re-runs of
`review_packet/astra/math_02_checks.py`**. Astra's script tests constant-kernel
attribution by exact spectral integral, 12 CE ODE solves, a numpy BN identity and
200 ReLU tube perturbations; none of the checks below duplicate those. Scripts live in

```
/tmp/claude-1008/-home-u37314kd-Projects-spectral-tokenization/887fb000-54db-4ab4-82e6-4eeab4fb6c48/scratchpad/audit_math02_opus/
```

run with `/home/u37314kd/Projects/spectral_tokenization/venv/bin/python3`
(numpy 2.0.2, scipy 1.16.3, torch 2.9.1). Each script prints `FAILURES: none`.

---

## Verdict

**Every mathematical assertion in §2.1, §2.2, §2.3, §4 and §6 that I was asked to
audit is CORRECT. I found no false statement.** (A1) is a genuine strengthening and
does supersede our A1-E2 clause; (A2) is a valid alternative certificate and does
remove one power of κ\*; (I1)–(I6) are exactly right including every constant and
the corrected limit; (B1)–(B3) are exact and the necessity counterexample works
verbatim as written.

What I return instead is **three exposition gaps** (all one-line repairs, none
affecting truth), **five sharpenings that are free**, and **four retractions of our
own earlier statements** — the three Astra names, plus one more he was too generous
about (our A1-E4 clipping correction).

Two things a reviewer will attack that Astra should pre-empt rather than defend:

1. **The (B2c) necessity counterexample is (B3) in disguise.** Its Θ(M) curvature
   comes entirely from `‖w_M‖ = M^{-1/2}`, i.e. it is exactly the
   function-preserving rescaling of (B3) with `c = M^{-1/2}`. It refutes the
   *unqualified* necessity claim, which is all it is asked to do, but stated as-is
   it invites "that's just a reparameterisation, not a mechanism." Say so first.
2. **The ρ-uniform isotropic corollary degrades as ρ⁻⁴, not ρ⁻².** The (I2) limit's
   leading error is `1/(M v₀²) + (2/3)Δ²/(M v₀⁴)`; the second term dominates for
   small |v₀|. "Constants depending poorly on rho" understates a fourth power.

Neither is an error. Both are places where the current wording is weaker than the
mathematics and a reviewer will get there first.

---

## Ledger

Legend: **VERIFIED** = derived by hand and confirmed numerically. **GAP** = the
claim is true but the written proof skips a step. No item came back FALSE.

### §2.1 — (A1) finite-horizon loss attribution

| # | Item | Verdict | One-line derivation | Numeric check |
|---|---|---|---|---|
| A1.1 | `∫₀ᵀ‖∇_θL‖²dt / (L(0)−L(T)) ≤ a/(a+κ)` | **VERIFIED** | Energy identity `L(0)−L(T)=N_θ+N_φ` with `N_θ=∫rᵀK_θr`, `N_φ=∫rᵀK_φr`; pointwise `κ rᵀK_θr ≤ κa‖r‖² ≤ a rᵀK_φr` integrates to `κN_θ ≤ aN_φ`; `x↦x/(x+c)` increasing gives `N_θ/(N_θ+N_φ) ≤ 1/(1+κ/a) = a/(a+κ)`. `a=0` ⇒ `N_θ=0`, bound `0`, still correct. | `a1_finite_horizon.py` (seed 20260910): **24 runs**, nonlinear two-block model `z = G tanh(Pθ) + Q softplus(Sφ)`, softmax-CE and Huber losses, `T∈{0.05,0.5,3,30}`, DOP853 rtol 1e-11. Bound held every time; worst share/bound = **0.9992**. Energy identity rel. error ≤ 5.8e-7. |
| A1.2 | Proof needs no commutation / infinite horizon / exponential solution | **VERIFIED** | The derivation above uses only the energy identity and a pointwise Rayleigh comparison. Nothing else enters. | Same script — the kernels are genuinely time-varying and non-commuting (nonlinear model), and neither loss is quadratic. |
| A1.3 | Sharpness: equality for `K_θ=a`, `K_φ=κ` | **VERIFIED** | Scalar kernels give `share = a‖e‖²/((a+κ)‖e‖²) = a/(a+κ)` at *every* `T`. Sharp within the class of hypotheses `(a,κ)`. | Same script, 4 `(a,κ)` pairs × 3 horizons: equality to **1e-14**. |
| A1.4 | Holds for time-varying kernels and any differentiable loss when the two inequalities hold on the actual residuals, under unit Euclidean gradient flow | **VERIFIED**, hypotheses need naming | With `r=∇_zL`, `K_θ=J_θJ_θᵀ`, `dL/dt = −rᵀ(K_θ+K_φ)r` and `‖∇_θL‖²=rᵀK_θr`. Identical algebra. **Silently required:** (i) `L(0)−L(T)>0`; (ii) both blocks at *unit* rate (per-block rates `η_θ≠η_φ` break the identification of the numerator with the θ-share); (iii) the Rayleigh bounds hold on the realised `r(t)` for a.e. `t` — a trajectory hypothesis, not an initialisation one; (iv) `K_θ⪰0` (automatic for a Gram). | Covered by A1.1's runs; `a`,`κ` were measured as sup/inf Rayleigh quotients **on the realised residual**, densely sampled (4001 points). |
| A1.5 | Drops Theorem 1.3's common-invariant-subspace hypothesis | **VERIFIED** (unstated strengthening) | Theorem 1.3 needed `V` invariant only to guarantee `e(t)∈V` so the Rayleigh bounds apply. (A1) assumes the bounds on the realised residual directly, so invariance is not used. Worth saying. | — |
| A1.6 | "does not rescue an attribution claim based only on a **top** spatial eigenvalue" | **VERIFIED** | `κ` is a *lower* Rayleigh bound on the visited residual; `λ_max(K_φ)` bounds nothing from below. Counterexample 1.4 is the witness. | — |
| A1.7 | (A1) supersedes our A1-E2 bound | **VERIFIED** | `a/(a+κ) < a/κ` always, and `a/(a+κ) < 1` always, so (A1) beats `min{1,a/κ}`; and `a/(a+κ) ≤ a‖e₀‖²/(2κ(L₀−L_T))` since `L₀−L_T ≤ ‖e₀‖²/2` and `a/(a+κ)<a/κ`. | Same script, "item i" table: at `a=1,κ=4,T=1e-3` the old finite-T bound is **25.13** (vacuous); (A1) gives **0.200000**, which is the exact share. |

### §2.2 — (A2) truncated matrix Chernoff

| # | Item | Verdict | One-line derivation | Numeric check |
|---|---|---|---|---|
| A2.1 | `‖X‖` bounded by the max squared Gaussian preactivation | **VERIFIED** | `X=ψψᵀ/N` is rank one so `‖X‖=‖ψ‖²/N = (1/N)Σ ReLU(g_n)² ≤ max_n g_n²`. | `a2_chernoff.py` (seed 424242): 4,000,000 draws, `N=6,K=3,s_w=1.1,s_b=0.7`; max ratio `‖X‖/max g_n²` = **0.9889** ≤ 1. |
| A2.2 | `E[‖X‖1{‖X‖>R}] ≤ 2N(R+2s_max²)e^{−R/(2s_max²)}` | **VERIFIED** | `E[Z1{Z>R}] = R P(Z>R) + ∫_R^∞P(Z>u)du` with `Z=max_n g_n²`; union bound `P(Z>u) ≤ 2N e^{−u/(2s_max²)}`; the integral contributes `4Ns_max²e^{−R/(2s²)}`; sum is exactly the display. | Same script: holds at `R/s_max² ∈ {0.5,1,2,4,8,16}`, slack 55.6× → ≫1e3. `a2_chernoff_biting.py` (seed 909090) repeats in a second parameterisation, slack 38×. |
| A2.3 | Choice of `R` gives `E[X1{‖X‖≤R}] ⪰ (κ\*/2)I` | **VERIFIED** | `E[X1_{≤R}] = K_* − E[X1_{>R}]`, and `‖E[X1_{>R}]‖ ≤ E[‖X‖1_{>R}] ≤ κ_*/2` by construction, so `⪰ κ_*I − (κ_*/2)I`. Truncated summands are PSD and `‖·‖≤R`. | `a2_chernoff.py`: chosen `R = 27.85 s_max²`; `λ_min(E[X1_R]) = 6.781e-4 ≥ κ_*/2 = 3.391e-4`. |
| A2.4 | `P{λ_min(ΣX_i) < Mκ_*/4} ≤ N exp[−Mκ_*/(16R)]` | **VERIFIED**, one **GAP** in the write-up | Tropp lower tail with `δ=1/2`: `P{λ_min(ΣY_i) ≤ μ_min/2} ≤ N[e^{−δ}/(1−δ)^{1−δ}]^{μ_min/R}`; `μ_min ≥ Mκ_*/2` so `Mκ_*/4 ≤ μ_min/2`; the standard `[e^{−δ}/(1−δ)^{1−δ}] ≤ e^{−δ²/2}` gives exponent `μ_min/(8R) ≥ Mκ_*/(16R)`. **GAP:** the display is written for `ΣX_i` but only `ΣY_i` is justified. The transfer is one line — `X_i⪰0 ⇒ ΣX_i ⪰ ΣY_i ⇒ λ_min(ΣX_i) ≥ λ_min(ΣY_i)` — and is currently missing. | `a2_chernoff.py`: PSD-monotonicity residual `−5.4e-13` (≥0 to machine precision). Failure frequency 0/4000 at `M∈{200,1000,5000,20000}` vs a bound of ≈5.8–6.0 (vacuous, see A2.6). `a2_chernoff_constant.py` (seed 5150) stresses the Chernoff step on the **extremal** balls-in-bins family (8 configs, 200 000 trials each): empirical/exact-Tropp ∈ **[0.099, 0.227]**, never exceeded; the simplified form never violated. |
| A2.5 | Is Astra's `1/16` conservative? | **YES, by 1.2274×** — valid, not tight | Tropp's exact `δ=1/2` constant is `−log(e^{−1/2}/√(1/2)) = 0.1534264`; with `μ_min ≥ Mκ_*/2` the exact exponent coefficient is `0.0767132 = 1/13.036`. Astra's `0.0625 = 1/16` is exactly the textbook simplification `e^{−δ}/(1−δ)^{1−δ} ≤ e^{−δ²/2}` (`0.857764 ≤ 0.882497`). | `a2_chernoff.py` and `a2_chernoff_constant.py` print both constants and verify `Astra ≥ Tropp` at every tested `M`. |
| A2.6 | `M ≥ (16R/κ_*)log(N/η)` suffices; "may still be useless numerically" | **VERIFIED**, and the uselessness is confirmed | Substituting gives `N e^{−log(N/η)} = η`. `R = 2s_max²·L` with `(L+1)e^{−L} ≤ κ_*/(8Ns_max²)`, so `R = O(s_max² log(Ns_max²/κ_*))`. | `a2_chernoff.py`: at `N=6,K=3,κ_*=6.78e-4`, `η=0.1` needs `M ≥ 2.404e6` (vs Cor 1.5's `M₀ = 3.124e8`); `η=0.01` needs `3.755e6` (vs `3.124e9`). Non-vacuous only above ~2e6 — Astra's caveat is correct. |
| A2.7 | "the inverse-square dependence belongs to the certificate, not to a universal necessity" | **VERIFIED** | New threshold is `O((s_max²/κ_*)·log(Ns_max²/κ_*)·log(N/η))` versus `18s_max⁴/(ρκ_*²)`: one power of `κ_*` traded for a log, and `1/ρ` traded for `log(1/η)`. | Ratios above: **7.7e-3** at ρ=η=0.1, **1.2e-3** at 0.01. |

### §2.3 — the six bullets

| # | Item | Verdict | Derivation / finding | Numeric check |
|---|---|---|---|---|
| b1 | Mini-batch residual has no universal positive noise floor | **VERIFIED** | On an interpolated finite dataset every sampled residual is exactly 0. Our A1 hidden-hypothesis wording ("has a noise floor") asserted a universal that does not hold; Astra's replacement — the envelope must hold for the actual sampled sequence, and an *observed* floor limits the empirical envelope — is the correct scope. | (analytic; no simulation needed) |
| b2 | CE curvature can have a positive constant on a fixed finite fitting horizon | **VERIFIED** | On a bounded-logit region `|q|≤Q`, `σ'(q)=σ(q)(1−σ(q)) ≥ σ(Q)(1−σ(Q)) > 0`, depending on the target margin. The asymptotic vanishing as margins diverge does not contradict this. **Note to keep:** this is curvature in *logit* space; Theorem 1.1's `α` is strong monotonicity of the *parameter* gradient map, which can still fail through the Jacobian. Astra's "the obstacle is proving and controlling it" already concedes this — keep that clause attached. | (analytic) |
| b3 | Unequal replicated readouts keep `(a,v,b)` closed | **VERIFIED** | `β̇_j = vϱ` is the same for every `j` ⇒ `d(β_i−β_j)/dt = 0`; the loss enters only through `q = a + bv`, `b = Σβ_j`, so `ḃ = Mvϱ` closes. Equal init is needed only for `β_j = b/M`. With `Σβ_j(0)=0` the aggregate formulas hold verbatim. | `c23_bullets.py` (seed 246810): full `(2+M)` flow vs the 3-variable reduction at `M∈{1,5,64,500}` and `Σβ₀ ∈ {0, ±0.08…, ±52.7}` — max discrepancy **7.5e-12**; `β_i−β_j` drift ≤ **5.3e-15**. Closure verified for **nonzero** `Σβ₀` too, which is stronger than the note claims. |
| b4 | Constant-direction ReLU witness need not be informative — an informativeness limit, not a validity hypothesis of Lemma 3.1 | **VERIFIED** | Lemma 3.1 reads `λ_max(G_WW) ≥ λ_max(S_{h,a})` for *any* unit `a`. A bad `a` makes the right side small; the inequality stays true. Nothing in the proof requires `a` to be informative. Astra's classification is right. | (logical, no numeric check possible) |
| b5 | `r ∈ range(H)`; uniform curvature suffices; "diverges as predictions saturate" not universal | **VERIFIED** | `H1 = p − p(pᵀ1) = 0` and `rank H = C−1` for `p>0`, so `range(H) = 1^⊥ = {v:Σv=0}`, and `Σr = Σp − Σy = 0`. Then `rᵀH†r ≤ ‖r‖²/λ_min⁺(H)`. Binary saturation: confidently **correct** `p=(1−ε,ε)` gives `rᵀH†r = ε/(1−ε) → 0`; confidently **wrong** gives `(1−ε)/ε → ∞`. | `c23_bullets.py`: 1600 draws, `C∈{2,3,5,12}`, logit scales to 8σ — `max|H1| = 3.8e-16`, `max|Σr| = 5.3e-16`, `max` ratio to `‖r‖²/λ_min⁺` = **1.000000** (attained at `C=2`). Saturation table: correct `1.0e-8` vs wrong `1.0e+8` at `ε=1e-8`. **(Computed on an orthonormal basis of `1^⊥`; `np.linalg.pinv` with any fixed `rcond` silently truncates the true smallest positive eigenvalue at saturated logits and gives spurious violations — worth a footnote if this is ever re-run.)** |
| b6 | Distinguish clipping before momentum from clipping the velocity; global clipping is the former | **VERIFIED — and it lands on us** | PyTorch `clip_grad_norm_` writes the clipped gradient into `.grad`; `SGD(momentum, dampening=0)` then does `v ← βv + ĝ` with `ĝ = a_k g_k`. That is Lemma 4.3's stated recurrence, so **(4.6) holds with the `a_i` retained** for our arm. Our A1-E4 ("the clipping factor is dropped") describes the *other* algorithm. **Extra finding:** even for the velocity-clipping form, `‖v_{k+1}‖ ≤ min(c, β‖v_k‖+‖g_k‖) ≤ Σ_i β^{k−i}a_i‖g_i‖` by a one-line induction (if `‖g_k‖≤c` use the second argument; else `min = c = a_k‖g_k‖`), so (4.6) **with** the `a_i` is true there as well. Our A1-E4 was a statement about what the coarse recurrence *proves*, not about the truth of (4.6). | `c23_bullets.py`: PyTorch vs form A agrees to **4.4e-16** once torch's `clamp(max_norm/(total+1e-6),max=1)` epsilon is mirrored (that epsilon is exactly what `code/experiments/exp1_7_train.py:963-967` already documents). 10 000 random `(K,β,c,d,η,g)` trials: **no** form-B violation of (4.6)-with-`a_i`, max ratio 1.000000. **Repo confirmation:** `exp1_7_train.py:960` clips, `:970` steps — form A. |

### §4 — (I1)–(I6), isotropic serial CE toy

| # | Item | Verdict | One-line derivation | Numeric check |
|---|---|---|---|---|
| I1 | `v = v₀cosh(√M u)`, `b = √M v₀ sinh(√M u)`, `q = a₀+u+ (√M v₀²/2) sinh(2√M u)` | **VERIFIED** | `ȧ = ϱ > 0` so `a` is a valid clock; `dv/da = b`, `db/da = Mv` ⇒ `v'' = Mv` ⇒ the cosh/sinh pair with `b₀=0`; then `q = a + bv` and `2 sinh x cosh x = sinh 2x`. Strictly increasing since `dq/du = 1 + Mv₀²cosh(2√Mu) > 0`. | `i_toy_ode.py` (seed 31337): **24** full `(2+M)` DOP853 solves (rtol 3e-12), `M∈{1,16,256,4096}`, 6 cases (`a₀<0`, `a₀>0`, `a₀=0`, `v₀<0`, `|v₀|=0.03`), unequal zero-sum `β₀`. `a`,`v`,`b` match the closed form to **<1e-7**, `q(T_m)=m` to 1e-8. |
| I2 | `0 < u_M ≤ Δ/(1+Mv₀²)`; `M u_M → Δ/v₀²`; `u_M` strictly decreasing in `M` | **VERIFIED** | `sinh x ≥ x` gives `Δ ≥ u_M(1+Mv₀²)`. Hence `√M u_M = O(M^{-1/2}) → 0`, and the Taylor substitution gives the limit. Monotonicity: `d/ds[(s/2)sinh(2su)] = ½sinh(2su) + su cosh(2su) > 0` for `u>0`, `s=√M`. | Same script: bound held in all 24 solves; `u_M` strictly decreasing in `M` in all 6 cases; limit checked at `M∈{1e6,1e8,1e10}`. **Rate derived and confirmed** — see I2b. |
| I2b | *(new)* rate of the limit | **SHARPENING** | `Δ = u(1+Mv₀²) + (2/3)M²v₀²u³ + O(M⁴u⁵)` ⇒ `M u_M = (Δ/v₀²)(1−ε)` with `ε = 1/(Mv₀²) + (2/3)Δ²/(Mv₀⁴) + O(M^{-2})`. **The second term dominates for small \|v₀\|, so uniformity needs `M v₀⁴ ≫ Δ²`.** | Predicted vs observed `ε` at `M=1e10`, all six cases: `2.27e-9/2.27e-9`, `1.57e-10/1.57e-10`, `2.29e-9/2.29e-9`, `6.67e-5/6.67e-5`, `3.63e-4/3.63e-4`, `3.67e-10/3.67e-10`. Exact to 3 s.f. |
| I3 | `\|v(T_m)−v₀\| ≤ Δ²/(2M\|v₀\|³)` and the `W`-displacement bound | **VERIFIED** (and the proof is tighter than it looks) | With `x=√Mu_M`, `x̄ := Δ/(√M v₀²)`: from `bv ≤ Δ` we get `sinh x cosh x ≤ x̄`, hence `sinh²x ≤ x̄²/cosh²x`, hence `cosh x − 1 = sinh²x/(cosh x+1) ≤ x̄²/(cosh²x(cosh x+1)) ≤ x̄²/2` since `cosh²x(cosh x+1) ≥ 2`. Multiply by `\|v₀\|`. **Note this is exact at every `M`, not asymptotic** — the naive `cosh x − 1 ≥ x²/2` route would *not* give it. The `W` bound is `√(u_M² + (v−v₀)²)` with the two component bounds. Invariant `b² = M(v²−v₀²)` follows from `cosh²−sinh²=1`; `0 ≤ bv ≤ Δ` since `bv = q−a` and `bv(T_m)=Δ−u_M`. | Same script: both bounds held in all 24 solves, including `M=1` where `x = O(1)`; invariant residual ≤ 1e-7 relative, `bv∈[0,Δ]` on 400 sampled times per solve. |
| I4 | Our proposed pair is false; `(a(T_m)−a₀)/(m−a₀) → 0` is correct | **VERIFIED** | `a(T_m) = a₀+u_M → a₀`, so `a(T_m)/a₀ → 1` (for `a₀≠0`) **but** `a(T_m)/m → a₀/m ≠ 0` unless `a₀=0` — and at `a₀=0` the first ratio is `0/0`. The two proposed limits are jointly unattainable. `u_M/Δ ≤ 1/(1+Mv₀²) → 0` is the correct vanishing fraction. | Same script, (I4) table: at `a₀=1.7, m=2.5, M=1e10`: `a(T)/a₀ = 1.000000`, `a(T)/m = 0.680000 = a₀/m` (**not 0**), `(a−a₀)/Δ = 6.25e-10`. |
| I5 | `Ψ_{a₀}(m)/(1+Mv₀²+2Δ²/v₀²) ≤ T_m ≤ Ψ_{a₀}(m)/(1+Mv₀²)` | **VERIFIED** | `q̇ = ϱ(1+Mv²+b²) = ϱ(1+Mv₀²+2b²)` using the invariant; `Ψ'(q)=1+e^q=1/ϱ` gives `dΨ/dt = 1+Mv₀²+2b²`, with `Ψ(a₀)=0`. Lower/upper bound the integrand by `1+Mv₀²` and `1+Mv₀²+2Δ²/v₀²` (from `\|b\| ≤ bv/\|v₀\| ≤ Δ/\|v₀\|`, using `sign b = sign v` and `\|v\|≥\|v₀\|`). | Same script: bracket held in all 24 solves. |
| I5b | Frozen comparator `Ψ_{a₀}(q_F) = Mv₀²t`; gap `0 ≤ q_J−q_F ≤ C₀t/(1+e^{a₀}+p₀Mv₀²t) ≤ C₀/(p₀Mv₀²)` | **VERIFIED**, incl. the denominator step | Frozen: `q̇_F = v₀ḃ = Mv₀²ϱ_F` ⇒ `dΨ/dt = Mv₀²`. Then `Ψ(q_J)−Ψ(q_F) ≤ C₀t` with `C₀=1+2Δ²/v₀²`, and `≥0` so `q_J ≥ q_F`. Convexity of `Ψ` (`Ψ'`增) gives `q_J−q_F ≤ [Ψ(q_J)−Ψ(q_F)]/Ψ'(q_F) = C₀t/(1+e^{q_F})`. The denominator step: `e^{q_F}−e^{a₀} ≥ e^{a₀}(q_F−a₀)` (convexity of exp) is exactly Astra's `q_F−a₀ ≤ (e^{q_F}−e^{a₀})/e^{a₀}`, so `Mv₀²t = Ψ(q_F) ≤ (e^{q_F}−e^{a₀})/p₀` ⇒ `1+e^{q_F} ≥ 1+e^{a₀}+p₀Mv₀²t`. Sup over `t` is `1/(p₀Mv₀²)`. | Same script: frozen closed form matched to <1e-7 relative; `gap ≥ 0` and both gap bounds held at 600 sampled times × 24 solves. |
| I6 | Test margin `2a(T_m)−m → 2a₀−m`; sufficient condition `Mv₀² > m/(m−2a₀)`; `lim E[Err] = Φ(m/(2σ))` | **VERIFIED** | Reversal flips the contextual sign: `q_test = a − bv = a − (m−a) = 2a(T_m)−m`. Error ⟺ `u_M ≤ m/2 − a₀`; substituting `u_M ≤ Δ/(1+Mv₀²)` and clearing denominators gives **exactly** `Mv₀² > m/(m−2a₀)`. Limit indicator is `1{a₀<m/2}`; dominated convergence over bounded indicators gives `P(a₀<m/2) = Φ(m/(2σ))`; no atom at `m/2`. | `i6_reversal_mc.py` (seed 77007): MC at `m∈{0.5,2,5}`, 40 000 draws, `M` up to 1e14 — `\|MC − Φ(m/2σ)\| ≤ 0.0025` (≤1.4 s.e.). **Per-draw**: at `M=1e16`, **0 of 20 000** disagreements with `1{a₀<m/2}`. Sufficient condition: **2160/2160** combinations gave error 1. |
| I6b | "For `a₀ > m/2`, the **large-width** classifier is correct" | **VERIFIED but understated** | `u_M > 0` always, so `2a(T_m)−m > 2a₀−m > 0` at **every** `M`, not just large `M`. Free strengthening. | Same script: **1800/1800** combinations correct at `M∈{1,3,17,1000,1e7}`. |
| I6c | Gaussian tails `\|a₀\| ≤ σ√(2log(4/ρ))`, `\|v₀\| ≥ ρσ√(π/2)/2` | **VERIFIED**; the second is *exactly tight* | `P(\|a₀\|>t) ≤ 2e^{−t²/2σ²} = ρ/2` at that `t`. `P(\|v₀\|<s) ≤ 2sφ(0) = 2s/(σ√(2π))`, and `s = ρσ√(π/2)/2` makes that **exactly** `ρ/2`. So the true probability sits just below `ρ/2` with essentially no slack. | Same script, exact values: at `ρ=0.05`, `P(\|a₀\|>2.960) = 0.00307` (vs 0.025 — lots of slack) but `P(\|v₀\|<0.03133) = 0.02499591` (vs 0.025 — **1.6e-6 of slack**). 4e6-draw MC straddles it, as expected. Correct, but do not weaken the constant. |
| I6d | ρ-uniform substitution "depending poorly on rho" | **VERIFIED but understated** — see I2b | Substituting `\|v₀\| ≥ ρσ√(π/2)/2` into I2b's `ε` gives a **ρ⁻⁴** width requirement, not ρ⁻². | Same script: `M` needed for `ε ≤ 0.01` at `Δ=3`: `6.3e4` (ρ=0.5), `3.9e7` (ρ=0.1), `6.2e8` (ρ=0.05). |

### §6 — (B1)–(B4)

| # | Item | Verdict | One-line derivation | Numeric check |
|---|---|---|---|---|
| B1 | `D_BN = (γ/σ̃)(P − ttᵀ/(L(s²+ε)))` | **VERIFIED** | `dt/da = P`, `ds²/da = (2/L)tᵀ`, `dσ̃/da = tᵀ/(Lσ̃)`; quotient rule on `γt/σ̃` gives `(γ/σ̃)[P − ttᵀ/(Lσ̃²)]`. | `b_bn_torch.py` (seed 13579): **`torch.func.jvp` through the real `torch.nn.BatchNorm2d` in train mode**, 3 configs incl. `ε=0` and `γ<0`, group sizes `L=54,72,100`; `max\|jvp − D_BN d\| = 2.7e-15`. |
| B1b | Spectrum `{0 on constants, 1 on centred ⊥t, ε/(s²+ε) on t}`; contraction | **VERIFIED** | `Q1 = 0` since `P1=0` and `tᵀ1=0`; `Qd = d` for centred `d⊥t`; `Qt = t(1 − s²/(s²+ε)) = t·ε/(s²+ε)` using `‖t‖² = Ls²`. All in `[0,1]` ⇒ `‖Q‖ ≤ 1`. | Same script: eigen-residuals ≤ **1.7e-16**; full `eigvalsh` spectrum matches `{0,1,ε/(s²+ε)}` in every config. |
| B2 | `‖D_BN d‖² ≤ (γ²/(s²+ε))‖Pd‖²`; sum channelwise | **VERIFIED** | `Q = QP`, so `‖D_BN d‖ = (\|γ\|/σ̃)‖Q(Pd)‖ ≤ (\|γ\|/σ̃)‖Pd‖`. | Same script, `ε∈{0,1e-8,1e-5,1e-2,1}`: ratio to the cap is **exactly 1** on centred `⊥t`, `(ε/(s²+ε))²` on `t` (`2.6e-32` at `ε=0`, `0.785` at `ε=1`), `≤1` on 500 random directions each. `b_bn_crosschannel.py`: perturbing one channel changes **no** other channel's output (`0.000e+00`), so "sum these channelwise" is exact. |
| B2b | Upper bound only; a large centred Gram can align with the erased radial direction | **VERIFIED** | The `t` eigenvalue `ε/(s²+ε) → 0` as `ε/s² → 0`, so a centred perturbation parallel to `t` is annihilated regardless of its size. | Same table, `ratio(d=t)` column. |
| B2c | Necessity counterexample: rows `(1,0),(0,1),(−1,−1)` padded to `M`, `w_M=e₁/√M`, `γ=1`, `ε=0`, logits `(BN(w_Mᵀh),0)` — weight GGN top eigenvalue **exactly** `M×const`, centred input Gram constant | **VERIFIED exactly as written** | Batch mean is zero, so the centred feature covariance `(1/3)ΣhhᵀT` has top eigenvalue `1` for every `M`. `BN(H(cw)) = BN(Hw)` for `ε=0` ⇒ `J(e₁/√M) = √M·J(e₁)` ⇒ `G_WW(w_M) = M·G_WW(e₁)`. `J(e₁) ≠ 0` (with `u=(1,−2,1)/√6` spanning `span{1,t}^⊥`, `uᵀ`col₂`= −3/√6 ≠ 0`), and the binary-logit Hessian is `p(1−p)>0` on that direction, so the constant is positive. | Same script, `M ∈ {2,4,16,128,1024,8192}`: centred Gram top eigenvalue `1.000000000000` at every `M`; `λ_max(G_WW)/M = 0.168881741275` at **every** `M` (12 digits); BN output `(1.2247,0,−1.2247)` `M`-independent. |
| B2d | With fixed `ε>0` the mechanism has a stabilisation cutoff | **VERIFIED** | `s² = 2/(3M) → 0`, so `σ̃ → √ε` and `D_BN → P/√ε`; `J(w_M) → PH/√ε`, `M`-independent. | Same script: at `ε=1e-2`, `λ_max(G_WW)` at `M=2,128,8192,1e6` is `0.3288, 9.727, 24.44, 25.00` — saturated. At `ε=0`: `0.3378, 21.62, 1383, 1.689e5` — linear. |
| B3 | `J_W(cW) = c^{-1}J_W(W)`, `G_WW(cW) = c^{-2}G_WW(W)`; other Jacobians unchanged | **VERIFIED** | `g(cW)=g(W)` ∀`c>0` ⇒ differentiating in `W`: `c J_g(cW)[δ] = J_g(W)[δ]`. `G = JᵀHJ` with `H` unchanged (outputs unchanged) ⇒ `c^{-2}`. Upstream Jacobians: `D_BN(cWh) = c^{-1}D_BN(Wh)` cancels the explicit `c`. **Requires (stated):** the block's only downstream use is that BN, positive variance, `ε=0` (or co-scaled), bias co-scaled or absent. | Same script, 3×3 conv → BN → 1×1 head, `c∈{0.1,0.5,2,7}`: outputs preserved to `4.0e-15`; `‖J(cW)‖/‖J(W)‖ × c = 1.00000000`; `λ(cW)/λ(W) × c² = 1.00000000`. |
| B3b | `BN_{c²ε}(ca) = BN_ε(a)` | **VERIFIED** | Input `ca` ⇒ `t→ct`, `s²→c²s²`, so `γct/√(c²s²+c²ε) = γt/√(s²+ε)`. | Same script, `ε∈{1e-5,1e-2,1}`, `c∈{0.03,0.4,3,25}`: max deviation `1.1e-15`. |
| B3c | "(B3) is approximate only when the relevant variances dominate eps; verify that condition" | **VERIFIED — now quantified** | At fixed `ε`, the departure is governed by `s²/ε`. | Same script, `ε=1e-5`: relative deviation from the exact `c^{-2}` law is `6.3e-6` at `s²/ε = 4.2e5`, `6.3e-4` at `4.2e3`, `5.8e-2` at `42`, `0.36` at `3.8`, and `0.67–1.02` at `0.42`. **Rule of thumb: the law is good to 1% only when `s² ≳ 10³ε`.** |
| B3d | `‖W‖_F² λ_max(G_WW)` invariant under uniform rescaling; "not invariant under every channelwise reparameterization" | **VERIFIED — and the caveat is large** | `‖cW‖² = c²‖W‖²` cancels `c^{-2}`. Per-output-channel scaling `diag(c_i)W` is equally function-preserving (each BN channel normalises separately) but does **not** cancel. | Same script: uniform `c` — invariant ratio `1.0000000000` at all four `c`. Channelwise `c=(0.2,1,3,8)` — outputs preserved to `2.7e-15` but the invariant moves from `10 569.9` to `2 487 296.2`, a **235.3×** change. |
| B4 | `γ²(M+K)/(c₀+ε(M+K))` from `s² = c₀/(M+K)` | **VERIFIED as algebra** | `γ²/(s²+ε) = γ²/(c₀/(M+K)+ε)`; multiply through. Increasing in `M`, plateau at `γ²/ε`. | Same script, `c₀=2, K=64, γ=1.1`: at `ε=1e-2` the gain² goes `43.4, 67.9, 83.7, 120.8` for `M=48,192,384,1e5` against a plateau of `121`. Astra's "diagnostic model, not a theorem" label is the right one — the `c₀/(M+K)` premise needs **total branch energy held fixed** as channels grow; standard Kaiming with per-channel energy fixed gives an `M`-independent `s²` and no growth at all. |

---

## Required fixes (exact wording)

**F1 — §2.1, name the hypotheses of (A1).** After "The same argument holds for
time-varying logit kernels and any differentiable loss whenever these inequalities
hold on the actual residual vectors and both blocks follow unit Euclidean gradient
flow.", add:

> Explicitly: `L` is differentiable, the parameter path is absolutely continuous, the
> residual is `r = ∇_z L` with `K_θ = J_θ J_θᵀ`, `K_φ = J_φ J_φᵀ`, the two Rayleigh
> inequalities hold at almost every time along the realised trajectory, and
> `L(0) − L(T) > 0`. The unit rate is not cosmetic: with per-block rates `η_θ ≠ η_φ`
> the numerator `∫‖∇_θ L‖² dt` is no longer the θ-share of the decrease. No common
> invariant subspace is required — that hypothesis of Theorem 1.3 was only a device
> for guaranteeing the Rayleigh bounds along the path, and (A1) assumes them directly.

**F2 — §2.2, close the truncation gap.** After "The independent PSD matrices
`X_i 1{||X_i||<=R}` are bounded by R and their expectation is at least
`(kappa_*/2) I`.", insert:

> Since each `X_i` is positive semidefinite, `Σ_i X_i ⪰ Σ_i X_i 1{‖X_i‖≤R}` and
> therefore `λ_min(Σ_i X_i) ≥ λ_min(Σ_i X_i 1{‖X_i‖≤R})`; the lower-tail bound for
> the truncated sum applies verbatim to the untruncated sum below.

**F3 — §2.2, state the provenance of `1/16`.** After the (A2) display, add:

> The constant is Tropp's `δ = 1/2` lower tail after the standard simplification
> `[e^{−δ}(1−δ)^{−(1−δ)}]^{μ_min/R} ≤ e^{−δ²μ_min/(2R)}` together with
> `μ_min ≥ Mκ_*/2`. Keeping Tropp's exact constant `−log(e^{−1/2}√2) = 0.15343`
> replaces `1/16` by `1/13.04`; the difference is immaterial at the widths involved.

Optional sharpening, same paragraph: because `P(ReLU(g)² > u) = P(g > √u) ≤
e^{−u/(2s_max²)}` is **one-sided**, the leading `2N` in the tail bound may be
replaced by `N`.

**F4 — §4, make the ρ-dependence explicit.** Replace "gives width-uniform constants
depending poorly on rho" with:

> gives width-uniform constants. The dependence is severe: the relative error of
> `M u_M → Δ/v₀²` is `1/(M v₀²) + (2/3)Δ²/(M v₀⁴) + O(M^{-2})`, so the second term
> dominates for small `|v₀|` and the ρ-uniform statement needs `M ≳ Δ²ρ^{-4}σ^{-4}`,
> a fourth power of `1/ρ`, not a second.

**F5 — §4, strengthen (I6) for `a₀ > m/2`.** Replace "For `a0>m/2`, the large-width
classifier is correct under this reversal." with:

> For `a₀ > m/2` the classifier is correct under this reversal at **every** width,
> since `u_M > 0` gives `2a(T_m) − m > 2a₀ − m > 0`.

**F6 — §6, disclose that the counterexample is (B3).** After "The latter has a
nonzero contrast direction, so the weight GGN top eigenvalue is exactly M times a
positive constant.", insert:

> This growth is the function-preserving rescaling (B3) with `c = M^{-1/2}`: the
> normalisation `‖w_M‖ = M^{-1/2}` is the whole mechanism. That is sufficient to
> refute the *unqualified* necessity claim — a Θ(M) centred Gram is demonstrably not
> needed for Θ(M) weight curvature — but it is not an example of a normalisation-free
> mechanism, and it should not be offered as one.

**F7 — §6, quantify the ε condition.** Replace "With fixed eps, (B3) is approximate
only when the relevant variances dominate eps; verify that condition rather than
asserting exact invariance." with:

> With fixed `ε`, (B3) is approximate, controlled by the ratio `s²/ε` of pre-BN
> centred variance to the stabiliser. Measured on a conv→BN→head block at
> `ε = 10⁻⁵`, the departure from the exact `c^{-2}` law is `6×10⁻⁶` at `s²/ε ≈ 4×10⁵`,
> `6×10⁻²` at `s²/ε ≈ 40`, and order one at `s²/ε ≲ 1`. Report `s²/ε` per channel
> alongside the scale control; the law is usable at roughly `s² ≳ 10³ ε`.

**F8 — §6, quantify the invariant's caveat.** After "It is not invariant under every
channelwise or network reparameterization", add:

> Per-output-channel rescaling `W → diag(c) W` is equally function-preserving under
> the same hypotheses, and moves `‖W‖_F² λ_max(G_WW)` freely: with
> `c = (0.2, 1, 3, 8)` on a four-channel block the scalar changed by a factor 235
> at identical logits.

**F9 — §6, state (B4)'s premise.** After "(B4) is a diagnostic model, not a theorem
about the supplied network", add:

> Its premise is that **total** incoming branch energy is held fixed as channels are
> added, so per-channel energy falls like `1/(M+K)`. Standard fan-in initialisation
> with per-channel energy held fixed gives an `M`-independent pre-BN variance and no
> width trend at all. Which of the two the decoder realises is exactly what
> diagnostic 1 must measure.

---

## Our own retractions

### (i) "a useful share bound is only infinite-horizon" — **RETRACTED. (A1) supersedes it.**

Located at `claude_math_reply_01.md:45` (audit item A1-E2, verbatim quote) and echoed
in the candidate framing at `claude_math_reply_01.md:266` ("the encoder's share of
total loss decrease **over the infinite horizon** is at most `min{1, a/κ}`").

Our A1-E2 said the correct finite-`T` bound is `a‖e₀‖²/(2κ(L(0)−L(T)))`, "which
exceeds `a/κ`" — implying no useful finite-horizon statement exists. It does:
`a/(a+κ)` holds at **every** `T` with positive decrease, is strictly better than both
`a/κ` and `1`, and needs neither the horizon nor the invariant subspace. Our bound is
valid but vacuous at short horizons: at `a=1, κ=4, T=10⁻³` it returns **25.13** where
the true share is **0.200000 = a/(a+κ)** exactly
(`a1_finite_horizon.py`, seed 20260910).

**Replace `claude_math_reply_01.md:266`'s clause with:** "…and the encoder's share of
total loss decrease, at **every** horizon on which the loss decreases, is at most
`a/(a+κ)` (Theorem 1.3 with (A1))." Delete "over the infinite horizon" and
`min{1, a/κ}` wherever they appear in the packet's `cor:attribution`.

### (ii) `a(T_m)/a₀ → 1` **and** `a(T_m)/m → 0` at fixed `m` — **RETRACTED as jointly impossible.**

Located at `claude_math_reply_01.md:307` (Q2).

`a(T_m) = a₀ + u_M` with `u_M → 0`, so at fixed `m`: if `a₀ ≠ 0` then
`a(T_m)/a₀ → 1` **and** `a(T_m)/m → a₀/m ≠ 0`; if `a₀ = 0` the first ratio is `0/0`.
There is no `a₀` for which both proposed limits hold. Confirmed numerically
(`i_toy_ode.py`, seed 31337): at `a₀ = 1.7, m = 2.5, M = 10¹⁰`,
`a(T_m)/a₀ = 1.000000` and `a(T_m)/m = 0.680000 = a₀/m`, not 0.

**(I4) is the right statement**, and it says what we actually wanted: the *update*
vanishes as a fraction of the fitting task, `(a(T_m)−a₀)/(m−a₀) ≤ 1/(1+Mv₀²) → 0`.
Use that in the main-text toy description, phrased as "learning of the spectral
coefficient is suppressed", never as "the spectral coefficient is small".

### (iii) matrix concentration "cannot remove the `κ_*^{-2}` factor" — **RETRACTED as too strong.**

Located at `claude_math_reply_01.md:70`: *"ρ-dependence 1/ρ and 1/√ρ comes from
Markov on Frobenius norms — matrix Bernstein would give log(N/ρ) but cannot remove
the `κ_*^{-2}` factor."*

The clause is defensible about **Bernstein specifically** — an additive bound with
variance proxy `ν ≤ MRλ_max(K_*)` and deviation `Mκ_*/2` gives
`M ≳ R λ_max/κ_*² · log(N/η)`, which does keep `κ_*^{-2}`. But we used it to close
the question for concentration in general, and that is wrong: the **multiplicative**
(Chernoff) route on PSD summands depends on `μ_min/R = Mκ_*/(2R)` and so keeps only
**one** power. Astra's (A2) gives
`M = O((s_max²/κ_*)·log(N s_max²/κ_*)·log(N/η))`, i.e. `κ_*^{-1} log(1/κ_*)` in place
of `κ_*^{-2}`, and `log(1/η)` in place of `1/ρ`. Verified end to end
(`a2_chernoff.py`, seed 424242): at `N=6, K=3, s_w=1.1, s_b=0.7`, `κ_* = 6.78×10⁻⁴`,
the A2 threshold is **2.40×10⁶** versus Corollary 1.5's **3.12×10⁸** at `ρ=η=0.1`
(ratio 7.7×10⁻³), and **3.76×10⁶** versus **3.12×10⁹** at `0.01`.

**Replace with:** "the `1/ρ` and `1/√ρ` come from Markov on Frobenius norms; matrix
Bernstein would give `log(N/ρ)` but retains a `κ_*^{-2}` variance term. A truncated
**Chernoff** bound on the PSD summands keeps only `κ_*^{-1} log(1/κ_*)` — see (A2) —
so the inverse-square dependence belongs to this certificate, not to a necessity
result. The threshold remains numerically unusable either way."

### (iv) *(not on Astra's list — ours to retract anyway)* A1-E4, clipping — **SCOPE ERROR, and the correction was too weak.**

Located at `claude_math_reply_01.md:56-58`, where we wrote *"This is the arm that
matters for us — our SGD arm is momentum 0.9 with global clipping."*

Astra's §2.3 clipping bullet is right and it lands on us twice.

1. **Wrong arm.** `code/experiments/exp1_7_train.py:960` calls
   `nn.utils.clip_grad_norm_` and `:970` calls `optimizer.step()`. The momentum buffer
   therefore accumulates the **already-clipped** gradient: `v_{k+1} = βv_k + a_k g_k`,
   which is Lemma 4.3's stated recurrence. **(4.6) holds for our arm with the `a_i`
   retained**; A1-E4 describes velocity-clipping, which we do not run. Verified against
   PyTorch to `4.4×10⁻¹⁶` once torch's `clamp(max_norm/(total+1e-6), max=1)` epsilon is
   mirrored — the same epsilon our own trainer already documents at `:964-967`
   (`c23_bullets.py`, seed 246810).
2. **Over-correction.** Even for the velocity-clipping form, (4.6) **with** the `a_i`
   is true, by induction on `‖v_{k+1}‖ ≤ min(c, β‖v_k‖ + ‖g_k‖)`: if `‖g_k‖ ≤ c` use
   the second argument, otherwise `min = c = a_k‖g_k‖`. So `math_01.md`'s original
   "yields the same coarse bound" reached the correct conclusion by an insufficient
   argument, and our A1-E4 correctly flagged the argument but wrongly weakened the
   conclusion. 10 000 randomised trials found no violation; max displacement/bound
   ratio 1.000000.

**Replace A1-E4's follow-up sentence with:** "Our arm clips gradients before the
momentum accumulation (`exp1_7_train.py:960` precedes `:970`), so (4.6) applies with
the `a_i` factors intact. For the velocity-clipping variant the coarse recurrence
`‖v_{k+1}‖ ≤ β‖v_k‖ + ‖g_k‖` proves only the `a_i ≡ 1` form, but the sharper
`‖v_{k+1}‖ ≤ min(c, β‖v_k‖+‖g_k‖)` recovers (4.6) with the `a_i` there as well.
`math_01.md`'s conclusion stands; only its stated justification needs the sharper
step."

---

## Scripts, seeds, commands

| Script | Seed | Covers | Result |
|---|---|---|---|
| `a1_finite_horizon.py` | 20260910 | A1.1–A1.7 | 24 nonlinear runs + sharpness + old-vs-new; FAILURES: none |
| `a2_chernoff.py` | 424242 | A2.1–A2.4, A2.6, A2.7 | 4e6-draw MC, 16 000 Chernoff trials; FAILURES: none |
| `a2_chernoff_biting.py` | 909090 | A2.2, A2.3 second parameterisation | 8e6-draw MC, 120 000 trials; FAILURES: none |
| `a2_chernoff_constant.py` | 5150 | A2.5 (constant stress on the extremal family) | 8 configs × 200 000 trials; FAILURES: none |
| `c23_bullets.py` | 246810 | b3, b5, b6 | 12 ODE reductions, 1600 softmax draws, 10 000 clipping trials, PyTorch check; FAILURES: none |
| `i_toy_ode.py` | 31337 | I1, I2, I2b, I3, I4, I5, I5b | 24 full `(2+M)` DOP853 solves + 18 high-`M` roots; FAILURES: none |
| `i6_reversal_mc.py` | 77007 | I6, I6b, I6c, I6d | 3×5×40 000 MC + 20 000 per-draw + 3960 grid + 4e6 tail draws; FAILURES: none |
| `b_bn_torch.py` | 13579 | B1, B1b, B2, B2b, B2c, B2d, B3, B3b, B3c, B3d, B4 | `torch.func.jvp` against real `BatchNorm2d`; FAILURES: none |
| `b_bn_crosschannel.py` | 13579 | B2 channelwise summation | exact block-diagonality (`0.000e+00`); FAILURES: none |

```
cd /tmp/claude-1008/-home-u37314kd-Projects-spectral-tokenization/887fb000-54db-4ab4-82e6-4eeab4fb6c48/scratchpad/audit_math02_opus
for f in a1_finite_horizon a2_chernoff a2_chernoff_biting a2_chernoff_constant \
         c23_bullets i_toy_ode i6_reversal_mc b_bn_torch b_bn_crosschannel; do
  /home/u37314kd/Projects/spectral_tokenization/venv/bin/python3 $f.py
done
```

These are session-scratch. Astra's §7 asks for audit scripts to be persisted into
`code/audits/` with seeds and commands; that move is a separate, explicitly-requested
edit and was **not** performed here (this audit created exactly one file, the present
report).

## Two things I could not settle

1. **b2 (CE curvature on a finite fitting horizon).** I verified the logit-space
   statement — `σ'(q) ≥ σ(Q)(1−σ(Q)) > 0` on `|q| ≤ Q`. What Theorem 1.1 needs is a
   positive `α` for the *parameter* gradient map, which additionally requires the
   Jacobian not to degenerate on the horizon. Astra's own "the obstacle is proving and
   controlling it" is the honest position; I could not close the stronger statement and
   do not believe the note claims it.
2. **b4 (ReLU witness informativeness).** The logical classification is right and
   checkable by inspection; whether an adaptive output-channel direction actually helps
   on the production decoder is an empirical question this audit does not touch.
