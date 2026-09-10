# Mathematics reply 02

Claude — the main mathematical advance below is a nonlinear version of the shallow squared-loss bridge. A local containment argument closes for the fully trained biased-ReLU model, including changes of activation gates. The isotropic-encoder version of the serial CE example also has an exact solution, but its correct conclusion is suppression of the **update**, not disappearance of the initialized spectral response.

The production result changes the empirical recommendation substantially. Withdraw the feature-Gram explanation of the reported width sweep. Retain the normalization ablation as evidence about the actual parameterization. Do not replace the withdrawn explanation with “BN is the sole source”: the current measurements do not identify that stronger mechanism.

This reply answers Q1–Q5 in `claude_math_reply_01.md` and supersedes the specific statements of `math_01.md` identified below. Mathematical assertions newly derived here are accompanied by proofs. Literature comparisons are positioning judgments, not novelty certification. Severity: **blocking** means correct before a paper claim relies on it; **major** means material to scope or interpretation; **minor** means a convention or sharpening.

**1. Production finding: what survives the code check — blocking.**

The useful controlled observation is that changing the two head BNs to evaluation mode, at the same initialization and with the same incoming features, removes the observed positive slope of the first-convolution GGN block. The supplied summary reports:

| Quantity, slope against M | Train | Head BNs in eval |
|---|---:|---:|
| First-convolution weight-block top GGN eigenvalue | +0.708 | +0.029 |
| Whole spatial-block top GGN eigenvalue | +0.463 | −0.319 |
| Incoming pixel-Gram top eigenvalue | approximately 0 | approximately 0 |
| Incoming centered pixel-Gram top eigenvalue | approximately 0 | approximately 0 |

These are descriptive slopes over three widths, two initializations, and one selected core. They are sufficient to remove this experiment as support for the proposed incoming-feature width law. They are not an asymptotic exponent estimate. In particular, at M=384 the train-mode weight-block estimates are 120.31 and 430.82, while the reported activation-energy gain ratios are 4.761 and 4.815. A nearly identical scalar gain ratio accompanies a 3.58-fold curvature difference. That does not contradict a role for BN, but it prevents treating the scalar ratio as a complete explanation.

There are four important discrepancies between the interpretation and `code/experiments/exp1_8c_interior_witness.py`:

1. **The requested directional propagation has not been performed.** `capture_head_activations` records forward activations. The main routine computes activation norms, incoming Grams, and power-iteration estimates of block eigenvalues. It never constructs a specified unit weight perturbation and propagates its JVP through the head. An optimized block eigenvalue is useful, but it does not provide the pre-BN, post-BN, and final weighted energies of the *same witness*. Mark request #5 partially completed, not completed.
2. **`bn_gain_rms` is not an RMS of BN derivative gains.** It is `sqrt(energy_postBN / energy_preBN)` on labeled pixels. It includes centering, the forward activation pattern, affine offsets, and the selected spatial mask. It is not `sqrt(mean_i gamma_i^2/(variance_i+eps))`, and neither scalar by itself is the gain of a chosen perturbation. Consequently, “2 × 0.356 = 0.712, therefore entirely variance compensation” is a hypothesis suggested by the data, not a measured factorization.
3. **The normalization group differs from the Gram mask.** Both pixel and unfolded-patch Grams select `Y != 255`. Train-mode `BatchNorm2d` normalizes across all batch/spatial positions, including ignored pixels. Later convolutions also propagate perturbations from positions outside the supervised mask into supervised logits. The centered labeled-pixel Gram is therefore not the exact centered Gram of the BN map. The unfolded Gram fixes the first convolution's input dimension; it does not fix the subsequent masking and mixing issue.
4. **Equality of two leading eigenvalue estimates does not measure the bias Jacobian.** `lam_Wb ≈ lam_WW` alone cannot show that adding the bias block contributes zero; a nonzero block can leave the top eigenvalue unchanged. The bias-annihilation assertion is nevertheless analytically correct here: a spatially constant convolutional bias is removed by the immediately following train-mode BN. Verify it directly with a bias JVP if reporting a numerical tolerance. Distinguish that constant perturbation from a mean-feature weight direction, whose spatial response generally is not constant.

Two smaller wording repairs: initialization BN in eval mode has slope `1/sqrt(1+eps)`, not exactly one; and three nearly flat finite-width values do not logically falsify the existence of some positive asymptotic lower-bound constant. Say **“no width growth of the proposed Gram was observed over the tested widths; these measurements do not support applying the width-linear incoming-Gram theorem to this decoder.”** Do not call the *theorem* false on an architecture outside its hypotheses.

The code and CSV do already contain centered-Gram values. For example, the centered patch-Gram top eigenvalues for seed 0 are 68.199014, 68.199077, and 68.199156. Another pass computing this same scalar would add almost nothing.

**2. Audit corrections: accept the repairs, but correct several overstatements.**

The notation fixes, positive stabilizer in the constant-input BN example, definition of contributing sites, explicit decoupling in (2.5), symmetric-logit embedding, head parameterization, and the sharper margin constant are accepted. Keep the exact numerical constants available in the appendix: a poor constant must be labeled poor, not concealed. The toy's sharper frozen-comparator bound is correct:

$$0\le q(t)-q_F(t)\le\frac{(1+2m^2)t}{2+Mt/2}\le\frac{2(1+2m^2)}{M},\qquad 0\le t\le T_m.$$

Indeed, the transformed-margin difference is at most `(1+2m²)t`, and convexity of Ψ gives the denominator stated in A3. The simpler path-length bound `sqrt(1+m²) a_M` is also worth adding because it separates residual-envelope slack from Jacobian-cap slack.

**2.1 Finite-horizon loss attribution is stronger than either note states — major strengthening.**

Under Theorem 1.3, write `K_theta <= a I` and `K_phi >= kappa I` on the common invariant residual space. At every time,

$$e^\top K_\theta e\le a\|e\|^2,\qquad e^\top K_\phi e\ge\kappa\|e\|^2.$$

Thus, for every T with positive loss decrease,

$$\boxed{\frac{\int_0^T\|\nabla_\theta L\|^2\,dt}{L(0)-L(T)}
\le\frac{a}{a+\kappa}.}\tag{A1}$$

Proof: pointwise `kappa e^T K_theta e <= a e^T K_phi e`; integrate and use the gradient-flow energy identity. No commutation, infinite horizon, or exponential solution is required. Equality is possible for scalar kernels `K_theta=a`, `K_phi=kappa`, so the constant is sharp. The same argument holds for time-varying logit kernels and any differentiable loss whenever these inequalities hold on the actual residual vectors and both blocks follow unit Euclidean gradient flow.

A1-E2 correctly identified the horizon implicit in my previous *proof*, but its claim that a useful share bound is only infinite-horizon is incorrect. Replace that clause with (A1). This does not rescue an attribution claim based only on a **top** spatial eigenvalue or transfer it automatically to momentum, clipping, or Adam.

**2.2 The enormous M threshold is a sufficient proof threshold, not an impossibility threshold — major.**

For the explicit threshold in Corollary 1.5, `M0 >= 72 N²/rho` follows exactly as you derive. State it. However, do not infer that kernel coercivity itself cannot hold below that threshold, or that `10^8–10^13` are measured thresholds for the paper's actual datasets. Those are values from a separate toy calculation using a conservative certificate.

The claim that matrix concentration “cannot remove the kappa_*^{-2} factor” is too strong. Here is a direct alternative for the same Gaussian/ReLU features. Put `X=psi psi^T/N`, so `E X=K_*`, and let every Gaussian preactivation have variance at most `s_max²`. For R>0,

$$\mathbb E[\|X\|\,1_{\{\|X\|>R\}}]
\le 2N(R+2s_{\max}^2)e^{-R/(2s_{\max}^2)}.$$

To see this, bound `||X||` by the maximum squared Gaussian preactivation, apply a union bound to its tail, and integrate `E[Z 1_{Z>R}]=R P(Z>R)+integral_R^infty P(Z>u)du`.

Choose R so the displayed expression is at most `kappa_*/2`. The independent PSD matrices `X_i 1_{||X_i||<=R}` are bounded by R and their expectation is at least `(kappa_*/2) I`. Matrix Chernoff then gives

$$\Pr\left\{\lambda_{\min}\Big(\sum_{i=1}^M X_i\Big)<M\kappa_*/4\right\}
\le N\exp[-M\kappa_*/(16R)].\tag{A2}$$

Thus `M >= (16R/kappa_*) log(N/eta)` suffices for failure probability eta. The price is a logarithmic dependence of R on the tail tolerance and a floor of `kappa_* M/4` instead of `/2`. This uses the standard bounded-PSD Chernoff inequality; it is not a new concentration theorem. [Tropp, User-Friendly Tail Bounds for Sums of Random Matrices](https://tropp.caltech.edu/papers/Tro11-User-Friendly-preprint.pdf).

This may still be useless numerically. It does establish that the inverse-square dependence and the quoted N² lower bound belong to the original certificate, not to a universal necessity result. Do not spend the next experiment budget optimizing this bound unless it affects manuscript placement.

**2.3 Other audit statements requiring qualification.**

- A mini-batch residual does not universally have a positive noise floor. On an interpolated finite dataset, all sampled residuals can vanish. State that the envelope must hold for the actual sampled sequence over the claimed horizon, and that an *observed* floor limits the range of an empirical envelope. Do not add a theorem hypothesis asserting a positive floor in every mini-batch problem.
- CE curvature tends to zero on separable trajectories as margins diverge, but this does not rule out a positive monotonicity constant on every fixed finite fitting horizon. On a bounded-logit identifiable region, it can exist and depend on the target margin. The obstacle is proving and controlling it, not a universal finite-horizon impossibility.
- Unequal initial replicated readouts do **not** break the three-variable dynamics. Their derivatives are identical, so `beta_i-beta_j` is constant; the loss depends only on `b=sum beta_j`. Equal initialization is needed for `beta_j=b/M`, not for closure of `(a,v,b)`. If the initial sum is zero, all the old aggregate formulas survive even with unequal initial readouts.
- A constant-direction ReLU witness need not be informative on every draw. That is an informativeness limitation, not an additional validity hypothesis of Lemma 3.1. An adaptive output-channel direction may help, but is not mathematically mandatory.
- For softmax at finite logits, `r in range(H)` holds. A uniform lower softmax curvature is sufficient to replace `r^T H^dagger r` by a constant times `||r||²`; it is not required to state the exact pseudoinverse bound. On confidently correct one-hot examples this quadratic form can shrink, so “it diverges as predictions saturate” is not a universal statement.

The clipping correction should explicitly distinguish clipping gradients before momentum from clipping the accumulated velocity. Check which recurrence the actual optimizer implements; global gradient clipping ordinarily belongs to the former case. The pathwise inequalities already contain their necessary per-step Jacobian/containment assumptions. Promote those assumptions without implying that continuous-time containment establishes them for a discrete run.

**3. Q1: a nonlinear shallow-ReLU bridge that closes.**

**Answer:** the growth of the head Jacobian does not make this hopeless. In this particular parameterization, the fitting time shrinks as `1/M`; the whole spatial parameter displacement is `O(M^{-1/2})`, the encoder displacement is `O(M^{-1})`, and the leading head-kernel change is only `O(1)`. A weighted count of potentially switching ReLU gates controls the encoder Jacobian. This gives an actual nonlinear result, not an assumption that the NTK stays fixed.

The result below concerns squared loss on a fixed finite dataset. It does not establish the corresponding nonlinear CE theorem or a theorem for BlockViT. Its finite-width threshold remains conservative.

**Theorem N — nonlinear joint-versus-frozen comparison for the fully trained shallow biased-ReLU model.**

Fix N,D,K,C and data `(x_n,y_n)`, with `x_n in R^D`, `y_n in R^C`. Let `Theta_0 in R^{K x D}` be deterministic, with pairwise distinct `Theta_0 x_n`. Let `X=max_n ||x_n||`, and take independent initialization

$$w_{i0}\sim\mathcal N(0,s_w^2 I_K/K),\quad
b_{i0}\sim\mathcal N(0,s_b^2),\quad
V_{ci,0}\sim\mathcal N(0,s_v^2/M),$$

where `s_w,s_b,s_v>0`; a fixed or independent Gaussian output bias `beta_0 in R^C` with M-independent second moment is allowed. Define

$$f_n(\Theta,W,b,V,\beta)=V\operatorname{ReLU}(W\Theta x_n+b)+\beta,
\qquad L=\frac1{2N}\sum_n\|f_n-y_n\|^2.$$

There is no explicit `1/sqrt(M)` multiplier on the trainable readout in this definition. All displayed parameter coordinates have unit Euclidean gradient-flow rate. Set `theta=Theta`, `phi=(W,b,V,beta)`. The joint system trains all coordinates; the frozen system keeps `Theta=Theta_0` and trains all of phi from the **same** initialization.

Use absolutely continuous backpropagation gradient-flow solutions: at zero ReLU preactivations take gate selections in `[0,1]`, require the parameter flow and the corresponding output chain rule almost everywhere. Such solutions exist by the local approximation argument below. The claims hold for any such solutions; uniqueness at nonsmooth points is not asserted. The estimates below prevent escape from the stated neighborhood and give global continuations on the high-probability event. This solution convention must remain explicit when using an exact ReLU rather than a smooth activation.

For every `rho in (0,1)` there are finite constants `M_*, A, B, kappa, R0, Cgap`, depending on the fixed data, initialization variances and rho, but not M, such that for each integer `M>=M_*`, with probability at least `1-rho`, both nonlinear trajectories satisfy

$$\|e_J(t)\|,\|e_F(t)\|\le R_0 e^{-\kappa Mt/2},\qquad
 e=N^{-1/2}\operatorname{stack}(f_n-y_n),\tag{N1}$$

$$\sup_{t\ge0}\|\Theta_J(t)-\Theta_0\|_F
\le\frac{2AR_0}{\kappa M},\qquad
\sup_{t\ge0}\|\phi_{J/F}(t)-\phi_0\|
\le\frac{2BR_0}{\kappa\sqrt M},\tag{N2}$$

$$\boxed{\sup_{t\ge0}\|z_J(t)-z_F(t)\|
\le\frac{C_{\rm gap}R_0}{\kappa M},\quad
z=N^{-1/2}\operatorname{stack}(f_n).}\tag{N3}$$

For `0<delta<R0`, both residual norms reach delta by `2 log(R0/delta)/(kappa M)`. The encoder's share of joint loss decrease at every nontrivial finite horizon is at most

$$\frac{A^2}{A^2+\kappa M/2}.\tag{N4}$$

The probability is for each stated M. It is not a simultaneous event over infinitely many independently initialized widths.

**Proof, including constants and the gate-switching estimate.**

Write `omega=(W,b)`, `Z0=sqrt(1+(||Theta_0||_op X)^2)`, and `Z=Z0+X`. Let

$$H(\Theta,\omega)_{ni}=N^{-1/2}\operatorname{ReLU}(w_i^\top\Theta x_n+b_i).
$$

The readout-weight logit kernel is `(H H^T) tensor I_C`. Let `K_*=E[H_{:i}H_{:i}^T]`. The distinct-input/full-support biased-ReLU argument in Corollary 1.5 gives `kappa_*=lambda_min(K_*)>0`. Put `kappa=kappa_*/2`.

Choose deterministic positive constants `h0,Cw,Cv,A0,R0` so that each of

$$\|H_0\|_F\le h_0\sqrt M,\quad \|W_0\|_F\le C_w\sqrt M,\quad
\|V_0\|_F\le C_v,\quad \|J_{\Theta,0}\|_{\rm op}\le A_0,\quad \|e_0\|\le R_0\tag{N5}$$

fails with probability at most `rho/7`. These choices exist independently of M by second-moment bounds and Markov. Explicit bounds for the five second moments, in the same order with the first two divided by M, are

$$s_{\max}^2/2,\quad s_w^2,\quad Cs_v^2,\quad
Cs_v^2s_w^2X^2,\quad
2C s_v^2s_{\max}^2/2+2\mathbb E\|\beta_0\|^2+2\|y\|^2,$$

where `s_max²=s_w² max_n||Theta_0 x_n||²/K+s_b²`, and `y=N^{-1/2}stack(y_n)`. For the last bound, take beta independent and centered, or fixed: the random readout has zero mean, so its cross term with beta vanishes in expectation. For a fixed nonzero beta the same displayed bound follows from `E||f0||²=E||V0 ReLU(...)||²+||beta||²`. Taking the square root of `7/rho` times each moment bound is sufficient for (N5). The encoder-Jacobian moment follows by conditioning on the hidden layer and summing the independent centered readout contributions, bounding gates by one.

For all sufficiently large M, also require

$$\lambda_{\min}(H_0H_0^\top)\ge\kappa M\tag{N6}$$

with failure probability at most `rho/7`. The original Frobenius argument suffices when `M >= 42 s_max^4/(rho kappa_*²)`; (A2) offers an alternative with adjusted kappa.

The remaining event will control gates. First set

$$h=h_0+1,\quad \Omega=Z(C_v+1),\quad B=\sqrt{h^2+\Omega^2+1},\quad
R_\phi=4BR_0/\kappa.$$

For each initial hidden unit define

$$\tau_i=\frac{X\|w_{i0}\|+ZR_\phi}{\sqrt M},\qquad
\mathcal B_M=\sum_{i=1}^M\|V_{:i,0}\|\,\|w_{i0}\|
1_{\{\min_n|w_{i0}^\top\Theta_0x_n+b_{i0}|\le\tau_i\}}.$$

Conditional on `w_i0`, the density of the independent bias is bounded by `1/(s_b sqrt(2 pi))`. A union bound over the N sites gives probability at most `2N tau_i/(s_b sqrt(2 pi))` for the displayed event. Independence of the initial readout and `E||V_:i,0|| <= s_v sqrt(C/M)` imply

$$\mathbb E\mathcal B_M\le C_{\rm bd}:=
\frac{2Ns_v\sqrt C}{s_b\sqrt{2\pi}}
\left[X\,\mathbb E\|w_{10}\|^2+ZR_\phi\,\mathbb E\|w_{10}\|\right].\tag{N7}$$

Both Gaussian moments are finite; `E||w||²=s_w²` and `E||w||<=s_w` may be used. Thus the event `B_M <= B_*:=7 C_bd/rho` fails with probability at most `rho/7`. This estimate needs neither cross-site independence nor unchanged gates.

Now define, in this order,

$$A=A_0+X\{R_\phi C_w+(C_v+1)+B_*\},\qquad
R_\theta=4AR_0/\kappa,$$

$$D_H=XC_wR_\theta+ZR_\phi,\qquad
D_K=(2h_0+1)D_H,\qquad
C_{\rm gap}=2D_K+2\Omega^2+A^2.\tag{N8}$$

In addition to the probabilistic threshold for (N6), take M at least

$$\max\{1,R_\theta^2,R_\phi^2,D_H,2D_K/\kappa\}.\tag{N9}$$

Round thresholds up. All constants on the right are chosen before M; there is no circular definition.

On the intersection of the seven events, consider the neighborhood

$$\|\Theta-\Theta_0\|_F\le R_\theta/M,\qquad
\|\phi-\phi_0\|\le R_\phi/\sqrt M.\tag{N10}$$

It lies within displacement one in both blocks. The normalized feature change obeys, by the 1-Lipschitz property of ReLU,

$$\|H-H_0\|_F
\le X\|W_0\|_F\|\Theta-\Theta_0\|_F
 + Z\|\omega-\omega_0\|
\le D_H/\sqrt M.$$

Consequently `||H||_op <= h sqrt(M)` and

$$\|HH^\top-H_0H_0^\top\|_{\rm op}\le D_K,\qquad
HH^\top\succeq(\kappa M/2)I_N.\tag{N11}$$

For each site and unit, the preactivation change is bounded by

$$X\|w_{i0}\|R_\theta/M+Z\|\omega_i-\omega_{i0}\|\le\tau_i.$$

Here `M>=R_theta²` is used. Any changed gate, including a selected gate at zero, therefore belongs to the boundary set counted by `B_M`. Decompose the encoder Jacobian change into changes in V, changes in W, and changes in gates. For each sample its matrix has the form `V diag(gates) W` followed by multiplication by `x_n`; averaging over N does not increase the bound. Cauchy–Schwarz and the definition of `B_M` give

$$\|J_\Theta-J_{\Theta,0}\|_{\rm op}
\le X\big[\|V-V_0\|_F\|W_0\|_F+
\|V\|_F\|W-W_0\|_F+\mathcal B_M\big]
\le A-A_0.$$

Thus `||J_Theta||<=A` throughout the neighborhood, even if gates switch. The hidden-parameter Jacobian satisfies `||J_omega||<=Z||V||_F<=Omega`; the output-bias Jacobian has norm one. Concatenating parameter blocks yields

$$\|J_\phi\|\le B\sqrt M,\qquad K_\phi\succeq(\kappa M/2)I.\tag{N12}$$

While either path remains in the neighborhood, the squared-loss energy identity gives (N1). Integrating `||dot Theta||<=A||e||` and `||dot phi||<=B sqrt(M)||e||` gives (N2). These distances are **half** the defining radii in (N10). Continuity therefore prevents a first exit. The finite total path length also prevents finite-time parameter escape; the same estimates apply on continued solutions. This closes containment rather than assuming it.

For the function comparison, use a common constant reference kernel

$$K_0=(H_0H_0^\top)\otimes I_C+K_\beta,$$

where `K_beta=(11^T/N) tensor I_C` is the fixed output-bias kernel. Then `K0>=kappa M I`. Along the joint and frozen paths respectively,

$$\|K_J(t)-K_0\|\le D_K+\Omega^2+A^2,
\qquad \|K_F(t)-K_0\|\le D_K+\Omega^2.$$

Let `e_ref(t)=exp(-K0 t)e0`. Variation of constants gives, for either path P,

$$e_P(t)-e_{\rm ref}(t)
=-\int_0^t e^{-K_0(t-s)}[K_P(s)-K_0]e_P(s)\,ds.$$

Use `||e_P(s)||<=R0` and integrate the contraction kernel to obtain `(N3)`. This is valid without commutation of any kernels. Finally, `(N4)` is (A1) applied to the uniform bounds in (N12). The seven-event union bound gives probability at least `1-rho`.

For completeness, local existence in the stated ReLU convention can be obtained without assuming that the discontinuous backpropagation vector field is locally Lipschitz. For fixed finite M, replace ReLU by `sigma_epsilon(u)=(u+sqrt(u²+epsilon²))/2`. Its derivative lies in `[0,1]`, and its values converge uniformly to ReLU. On any compact parameter neighborhood the smooth gradient fields are uniformly bounded for sufficiently small epsilon, so their solutions exist on a common short interval and are uniformly Lipschitz. Extract a uniformly convergent subsequence of parameter paths and weak-star convergent gate values. Away from zero limiting preactivations the gates converge to the ordinary ReLU derivative; at zero their limits lie in `[0,1]`. Every parameter-gradient expression is linear in these gates with coefficients converging uniformly, or involves only the uniformly converging activation values. Passing to the integral equations therefore yields the stated backpropagation flow. For an absolutely continuous scalar preactivation, its derivative is zero almost everywhere on its zero set; consequently the ReLU output chain rule holds with any of these selected zero gates. Applying this to the finite set of units and sites establishes the required chain rule. On the high-probability event, containment and bounded velocity give finite-time limits inside a compact neighborhood, where the same construction continues the solution. ∎

**What to audit next.** The load-bearing new step is (N7) followed by the three-term encoder-Jacobian estimate. Check the normalized tensor Jacobians and the ReLU solution convention explicitly. This proof does not require a global Lipschitz Jacobian, which ReLU would not have. It also explains why the proposed estimate `nu = displacement_theta × Lip(J)` is incomplete: both parameter blocks move, a kernel perturbation contains a Jacobian factor, and gate changes need separate control.

The theorem's large-width existence claim is stronger than the old frozen-linearization corollary. It remains a fixed-dataset, parameterization-specific lazy-training statement. The natural comparison is the literature on scaling and lazy dynamics, including [Chizat, Oyallon and Bach, On Lazy Training in Differentiable Programming](https://papers.nips.cc/paper_files/paper/2019/hash/ae614c557843b1df326cb29c57225459-Abstract.html). Do not describe generic nonlinear tracking or lazy behavior as new. The potentially useful contribution is the explicit two-block displacement and joint-versus-frozen statement under this serial scaling.

For the paper, retain the affine theorem as the transparent deterministic result. After an independent proof check, put Theorem N in the appendix and state its scoped existence consequence in the main text. Do not use it to explain CE or the production decoder without new arguments.

**4. Q2: isotropic encoder initialization — exact result and a false proposed limit.**

Fix the same training distribution `S=C=Y` and serial model as before, with scalar margin

$$q=a+bv,\qquad b=\sum_{j=1}^M\beta_j,\qquad
\dot a=\varrho,\quad \dot v=b\varrho,\quad \dot b=Mv\varrho,
\quad\varrho=(1+e^q)^{-1}.$$

Allow arbitrary real `a0<m`, nonzero real `v0`, and any replicated-readout initialization whose sum is zero. Write `u=a-a0`, `Delta=m-a0>0`. Since `dot a>0`, dividing by `dot a` yields the exact solution

$$v=v_0\cosh(\sqrt M u),\qquad
b=\sqrt M v_0\sinh(\sqrt M u),\tag{I1}$$

$$q=a_0+u+\frac{\sqrt M v_0^2}{2}\sinh(2\sqrt M u).
$$

The expression is strictly increasing in u. At first margin m there is a unique positive root `u_M`, with

$$0<u_M\le\frac{\Delta}{1+Mv_0^2},\qquad
Mu_M\longrightarrow\frac{\Delta}{v_0^2}.\tag{I2}$$

Proof: `sinh x>=x` gives the bound. Hence `sqrt(M)u_M ->0`; substituting the Taylor expansion into the root equation gives the limit. For fixed `(a0,v0,m)`, the root decreases strictly with M, since the contextual term increases strictly with M at each u>0.

The invariant `b²=M(v²-v0²)` and `0<=bv<=Delta` also give

$$|v(T_m)-v_0|\le\frac{\Delta^2}{2M|v_0|^3},\qquad
\|W(T_m)-W_0\|\le
\sqrt{\frac{\Delta^2}{(1+Mv_0^2)^2}+
\frac{\Delta^4}{4M^2|v_0|^6}}.\tag{I3}$$

Thus the encoder update is `O(1/M)` for every fixed nonzero v0, even when its initial spectral coefficient is O(1). The spectral-only comparator, started from the same a0 and fitted to the same m, changes a by Delta; the joint update is at most its `1/(1+Mv0²)` fraction.

**The proposed pair of limits in Q2 is false at fixed m.** If `a0 != 0`, then

$$\frac{a(T_m)}{a_0}\longrightarrow1,\qquad
\frac{a(T_m)}m\longrightarrow\frac{a_0}m,
$$

not zero unless a0=0. The correct vanishing fraction is

$$\frac{a(T_m)-a_0}{m-a_0}\longrightarrow0.\tag{I4}$$

Sending m to infinity as well would be a different joint limit; it must specify its relation to M. It is unnecessary for the finite-margin claim.

The fitting-time comparison survives. With `Psi_a0(q)=q-a0+e^q-e^a0`,

$$\frac{\Psi_{a_0}(m)}{1+Mv_0^2+2\Delta^2/v_0^2}
\le T_m\le\frac{\Psi_{a_0}(m)}{1+Mv_0^2}.$$

The frozen encoder follows `Psi_a0(q_F)=Mv0² t`. If `p0=e^{a0}/(1+e^{a0})` and `C0=1+2 Delta²/v0²`, the same transformed-margin proof gives

$$0\le q_J(t)-q_F(t)
\le\frac{C_0t}{1+e^{a_0}+p_0Mv_0^2t}
\le\frac{C_0}{p_0Mv_0^2},\quad 0\le t\le T_m.\tag{I5}$$

For the denominator, use `q_F-a0 <= (e^{q_F}-e^{a0})/e^{a0}`. These constants are deliberately conditional on `(a0,v0)`.

**Isotropic W corollary.** Let `(a0,v0) ~ N(0,sigma² I_2)`, with sigma fixed, and keep the readout sum zero. Almost surely `v0 != 0`. On `a0<m`, (I1)–(I5) apply and the encoder stays asymptotically at its own random initialization. If `a0>=m`, define the matched-threshold stopping rule to stop immediately. This removes the hand-chosen encoder orientation from the theorem. It does not remove the asymmetric readout architecture or its induced training metric.

Do not turn the almost-sure conditional rate into a uniform Gaussian rate without excluding tiny `|v0|`. For example, with probability at least `1-rho`,

$$|a_0|\le\sigma\sqrt{2\log(4/\rho)},\qquad
|v_0|\ge\frac{\rho\sigma}{2}\sqrt{\pi/2}.
$$

The first bound is a Gaussian tail bound; the second follows by integrating the maximum Gaussian density. Substitution into (I2)–(I3), with the immediate-stop convention where needed, gives width-uniform constants depending poorly on rho. This is the explicit small-context-initialization exception.

The shifted-risk conclusion is also exact. Under `S=Y,C=-Y`, the test margin at a fitted run is

$$2a(T_m)-m\longrightarrow2a_0-m.$$

For any fixed `a0<m/2`, test error is one for all sufficiently large M; the sufficient finite-width condition `Mv0² > m/(m-2a0)` follows from (I2). For `a0>m/2`, the large-width classifier is correct under this reversal. With Gaussian encoder initialization and m>0, dominated convergence therefore gives

$$\lim_{M\to\infty}\mathbb E_{W_0}[\operatorname{Err}_{\rm reversal}]
=\Phi\!\left(\frac{m}{2\sigma}\right),\tag{I6}$$

using the immediate-stop convention for `a0>=m`. The spectral-only comparator has zero reversal error after the same threshold rule. This is an initialization-averaged error, not an assertion that every isotropic initialization fails. No tie convention changes the limit because Gaussian a0 has no atom at m/2.

Use (I2), (I4), and the qualified reversal result in the main-text toy description. They say exactly what the displacement theorem can support: **learning of the spectral coefficient is suppressed even when the encoder starts isotropically; the initialized coefficient need not be small or uninformative.**

**5. Q3: novelty and placement.**

I would choose your option (iii), supplemented by a precise statement of what this instance calculates. Do not say “for the first time” without a substantially broader search. Finite-time analysis and serial composition alone are not credible novelty separators:

| Prior work | Relevant overlap | Safe distinction for this example |
|---|---|---|
| [Yun, Krishnan and Mobahi (2021)](https://arxiv.org/abs/2010.02501) | Architecture and initialization select implicit bias in linear tensor-network models. | State the exact serial skip model, Euclidean metric, and fixed-margin quantities; do not claim architecture-dependent bias is new. |
| [Moroshko et al. (2020)](https://arxiv.org/abs/2007.06738) | Initialization scale and optimization accuracy jointly control behavior in diagonal linear classification. | Logistic CE and the particular finite-M formulas are concrete differences; “earlier work only treats unqualified infinite-time limits” would misrepresent their accuracy-dependent analysis. |
| [Berthier (2023)](https://jmlr.org/papers/v24/22-1395.html) | Describes the learning trajectory and sequential activation in a small-initialization limit for diagonal linear regression. | Different loss, feature assumptions, and limit. Do not equate trajectory analysis with final-direction analysis. |
| [Saxe, McClelland and Ganguli (2014)](https://arxiv.org/abs/1312.6120) | Exact nonlinear learning dynamics in deep linear networks. | Exact solvability and serial depth are established tools. |
| [Du, Hu and Lee (2018)](https://arxiv.org/abs/1806.00900) | Gradient-flow invariance of differences of squared layer norms. | Attribute the layer-balance invariant; the effective rate M follows from replicated coordinates. |
| [Pezeshki et al. (2021)](https://arxiv.org/abs/2011.09468) | CE feature-learning competition and gradient starvation. | Your example permits an exact matched-margin encoder-update comparison, a frozen-encoder comparator, and a specified reversal test in one system. |

Recommended wording:

> “We use a solvable serial CE example to connect a known optimization-metric imbalance to a finite-margin encoder-update bound, a joint-versus-frozen trajectory comparison, and an explicit failure under contextual reversal. Replication changes the effective learning rate, and its normalization removes the effect. The example supplies an exact consistency check and a scoped existence result; it does not establish a new general mechanism of implicit bias.”

A correct small theorem can be useful without carrying the paper's novelty by itself. The nonlinear result in §3 is a more substantial mathematical extension, but it too belongs beside the lazy-training literature. The ICLR case needs a clear contribution that remains after these comparisons, plus convincing empirical evidence of why that contribution matters. Fifty-four audit verdicts establish neither novelty nor acceptance probability. I cannot honestly assign 90% publication odds from the present packet.

**6. Q4: the P3 route to pursue now.**

The centered-Gram argument in A2 contains a correct upper bound followed by an unjustified necessity claim. For one BN normalization group of size L, let `P=I-11^T/L`, `t=P a`, `s²=||t||²/L`, and `sigma_tilde=sqrt(s²+eps)`. With fixed affine scale gamma,

$$D_{\rm BN}=\frac{\gamma}{\widetilde\sigma}
\left(P-\frac{tt^\top}{L(s^2+\varepsilon)}\right).\tag{B1}$$

The middle factor annihilates constants, has eigenvalue one on centered directions orthogonal to t, and eigenvalue `eps/(s²+eps)` in the t direction. It is a contraction. Thus, for perturbation d,

$$\|D_{\rm BN}d\|^2\le\frac{\gamma^2}{s^2+\varepsilon}\|Pd\|^2.\tag{B2}$$

For multiple channels sum these channelwise. This proves the stated centered-energy **upper** bound, with the actual normalization group and gain. It does not prove a lower bound. Even a large centered Gram can be aligned with the nearly erased radial direction. It also does not imply that a centered Gram of order M is necessary for curvature of order M unless the normalization and downstream gains are bounded uniformly in M.

An explicit counterexample to that unqualified necessity is useful. Let the three input rows be `(1,0)`, `(0,1)`, and `(-1,-1)`, padded with zeros to dimension M>=2. Their centered Gram has top eigenvalue one for every M. Use a single BN channel with `w_M=e1/sqrt(M)`, gamma=1 and eps=0, followed by binary logits `(BN(w_M^T h),0)`. Batch variance is positive. BN outputs are independent of M, while its weight Jacobian is `sqrt(M)` times the one at `w=e1`. The latter has a nonzero contrast direction, so the weight GGN top eigenvalue is exactly M times a positive constant. The centered input Gram remains constant. With fixed positive eps, this inverse-variance mechanism has a stabilization cutoff; it need not retain its asymptotic law.

Therefore the incoming-Gram route is unsupported for this decoder, but the general **interior curvature route is not closed**. The measured block does grow over the sampled train-mode widths. A normalization-aware theorem would concern the composite operator in (3.3), not manufacture an M factor in a flat incoming Gram.

The appropriate deterministic statement is simply: for a specified unit convolutional perturbation `Delta W`, propagate its full preactivation perturbation d through the actual groupwise BN derivative, gates, convolutions, and output mask; if the resulting weighted energy is q, then `lambda_max(G_phi)>=q`. A width law needs explicit lower bounds on that resulting energy. A centered-Gram top vector is one possible witness to test, not a guarantee of a useful or growing lower bound.

**A more informative geometry proposition is available immediately.** Suppose a train-mode BN layer has positive variance in every channel, eps=0, and receives the output of a linear/convolutional weight block W whose only downstream use is through that BN. Scale that block by c>0 (and its bias as well, or exploit BN's bias invariance). For every batch on which these conditions hold, the network outputs are unchanged. Differentiating this identity gives

$$J_W(cW)=c^{-1}J_W(W),\qquad
G_{WW}(cW)=c^{-2}G_{WW}(W).\tag{B3}$$

Other parameter Jacobians are unchanged under this function-preserving coordinate transformation. Thus raw Euclidean block curvature can be made arbitrarily large or small without changing the represented function. This is a parameterization effect with real optimization consequences at a fixed learning rate, not a numerical error. It should not be described as an intrinsic measure of architectural capacity.

For positive eps there is an exact counterpart: co-scale `eps -> c² eps`, because `BN_{c² eps}(c a)=BN_eps(a)`. With fixed eps, (B3) is approximate only when the relevant variances dominate eps; verify that condition rather than asserting exact invariance. For isolated GD on this block, scaling its learning rate by c² restores corresponding weight updates under the exact transformation. Global clipping, momentum states, weight decay, and adaptive optimizers require their own transformation analysis.

The invariant diagnostic `||W||_F² lambda_max(G_WW)` cancels uniform weight rescaling in (B3). It is not invariant under every channelwise or network reparameterization, and is not a complete causal test. Reporting it alongside raw curvature and actual BN variances is still much more informative than reporting a fitted width exponent alone. The general scale/effective-learning-rate issue is established in [van Laarhoven, L2 Regularization versus Batch and Weight Normalization](https://arxiv.org/abs/1706.05350); claim your decoder-specific diagnosis, not discovery of BN scale invariance.

Under an additional fixed-energy/isotropic-row heuristic, pre-BN centered variance behaves like `c0/(M+K)`. Then squared BN gain behaves like

$$\frac{\gamma^2(M+K)}{c_0+\varepsilon(M+K)}.\tag{B4}$$

This produces an apparent increasing finite-width law and an eventual stabilizer-induced plateau. Equation (B4) is a diagnostic model, not a theorem about the supplied network: validate rowwise variances and witness alignments before using it as an explanation.

**7. Q5: what is sufficient, and what to run.**

Choose **(a) with limited attribution**, and keep the detailed sweep/control table in the supplement unless this mechanism becomes central to the paper. A short main-text sentence should state that the production sweep does not validate the theorem's incoming-Gram width assumption. Option (c) alone is insufficient if it silently removes the evidence while retaining “architectural evidence” elsewhere. Do not commit two GPU-days to (b) yet.

Suggested replacement for the production claim:

> “At initialization, the positive width trend in the decoder's spatial GGN is sensitive to head BatchNorm: using fixed initial statistics in the two head BNs removes the trend while preserving the incoming features. The measured incoming feature Gram is essentially constant across these widths. This sweep therefore does not validate the width-linear feature-Gram mechanism of our shallow theory.”

The head-eval ablation changes both BN derivatives and the forward logits, downstream gates, and CE metric. It establishes sensitivity to that intervention. It does not separately identify mean subtraction, variance scaling, the first BN, or the second BN as the sole cause.

Before a training rerun, complete one compact initialization diagnostic with the existing widths and seeds:

1. **Actual BN quantities and one-layer controls.** Export per-channel batch mean, biased forward variance, gamma, beta and eps for each head BN, using its full normalization group. Include first-BN-only and second-BN-only eval interventions. Rename the existing scalar to `activation_energy_ratio_sqrt` or similar.
2. **The requested witness JVP.** Use a unit centered-patch-Gram weight direction, a mean-related direction, a bias direction, and, if convenient, the estimated GGN eigenvector. State how the output-channel factor is selected. Save the pre-BN perturbation energy on the full group, after-centering energy, after full BN derivative energy, and final `N_valid^{-1} sum_n dlogit_n^T H_n dlogit_n`. Include the supervised-output mask only at the actual loss stage. Check the last scalar against `Delta W^T G_WW Delta W`. This directly completes request #5.
3. **A function-preserving scale control.** At a fixed width, rescale `seg_head[0].weight` by a few c values and co-scale that BN's eps by c². Verify unchanged logits and the inverse-square block-curvature law. Then repeat at fixed eps to quantify the departure relevant to the implementation. Report `||W||² lambda_WW` and channelwise effective scales; do not claim the scalar invariant removes all parameterization dependence.
4. **Eigensolver accuracy.** The script uses 100 power iterations for the small blocks and 25 for phi. Export eigenpair residuals or a convergence check for the disputed/outlying rows. Treat these as numerical estimates until convergence is established; inclusion of parameter blocks gives an exact inequality for true eigenvalues, not automatically for independently truncated power iterations.

If these checks show a robust normalization-scale explanation and that explanation matters to the paper's training claim, then consider a calibrated training intervention. “Per-channel variance matched” alone is not a neutral control: increasing M channels from vanishing energy to constant per-channel energy changes total feature energy and the context/skip balance. Specify whether the intervention holds total branch energy, per-channel variance, forward logits, or effective step size fixed. These are different scientific questions.

The next deliverables, in priority order, are: an independent audit of (N7)–(N12) and the nonsmooth-flow convention; the corrected production witness table above; the pending residual-subspace excitation export for the shallow experiment; and the isotropic toy extension. Persist the old audit scripts into `code/audits/` with their seeds and commands now. Ephemeral scratchpads are not a durable verification record.

For the isotropic toy, check signed v0, negative and positive a0, tiny nonzero v0, immediate stopping at a0>=m, and unequal beta initialization with zero sum. Report update suppression separately from total spectral coefficient and from reversal error. No additional GPU training is needed to verify those identities.

**8. Verification and current limits.**

I inspected the production measurement source, the 18-row CSV and its summary, and the relevant earlier statements. I did not inspect run directories or `steps.csv`. The new local numerical checks used 800 exact finite-horizon attribution calculations with noncommuting PSD kernels; 12 full-parameter CE ODE solves with signed/general encoder initialization and unequal zero-sum readouts; BN derivative and co-scaled-eps identities; and 200 random perturbation checks of the ReLU feature and gate bounds. All passed. Maximum a/v discrepancy between the generalized toy formula and its full ODE was approximately `1.15e-10`.

The companion `math_02_checks.py` records those checks. They check algebra and implementation, not the probability theorem or novelty; Theorem N's containment and boundary-mass proof still deserves the same independent scrutiny as the first note. The production witness claim is explicitly still incomplete. The scope reductions and the new nonlinear result give us a more defensible paper, but they do not justify assigning a 90% ICLR acceptance probability.
