**Mathematics response: five targets, corrected theorems, and a solvable serial example**

There are useful new theorems to add. The strongest result below is an exactly solvable **nonlinear-in-parameters serial model trained with cross-entropy**: its width-M spatial branch has initial curvature M times larger, fits at a provably width-dependent rate, suppresses adaptation of an equally predictive spectral feature to O(1/M) at a fixed fitting threshold, and stays O(1/M) in training-logit distance from a separately trained frozen-encoder model. This is an existence theorem with explicit routing, initialization, and parameterization assumptions. It does not establish the phenomenon for a ViT or imply low information in the whole encoder.

The other results repair the audit directly: discrete GD/momentum/clipping bounds with correct units; a trajectory comparison under monotonicity rather than PL alone; a sharp fixed-kernel comparison; and an interior-layer curvature floor for a Gaussian/ReLU network with a fixed downstream bottleneck. The proposed two-sided directional-energy inference is false without residual-excitation assumptions. BatchNorm gives an exact obstruction to a general interior-witness theorem.

These are mathematical derivations, not certifications of the existing experiments. All norms below are Euclidean/operator norms unless indicated, and all probabilities in initialization statements are stated explicitly. The finite-dimensional ODEs are considered on intervals on which solutions exist; local smoothness or local Lipschitz vector fields suffice for the comparison arguments. “Proved now” means the statement and proof are supplied here, not that a novelty claim has been established.

| Target | Verdict | Reason |
|---|---|---|
| P4, discrete scope | **Provable now** | Pathwise summation gives exact momentum weights and clipping-compatible bounds. Positive-epsilon Adam also has a bound, but with a different, potentially enormous constant. |
| P1, trajectory comparison | **Provable now after correcting the hypothesis** | Strong monotonicity gives contraction; convexity gives a linear-in-time comparison. PL alone does not. A spatial kernel lower bound on the entire relevant residual subspace gives a sharper O(1/M) fixed-kernel result. |
| P3, interior witness | **Provable now conditionally, and for a concrete Gaussian/ReLU instance** | The correct condition is downstream gain on the actual witness. A fixed-size bottleneck is not fatal. ReLU inactivity and train-mode BN prevent the proposed unrestricted high-probability production claim. |
| P2, two-sided energy ratio | **False as proposed** | GGN curvature is not the Euclidean residual-to-gradient gain; even with squared loss, residual alignment and mode decay determine cumulative energy. A corrected excitation theorem is supplied. |
| P5, spatial dominance | **Provable now as a carefully scoped existence theorem; universal form false/ill-posed** | Equal predictive information does not determine optimization. The serial example below proves selective adaptation suppression and a shift failure under explicit asymmetries. “Low-information whole encoder while fitting labels” is generally the wrong conclusion. |

The production ViT versions of P1/P3 are not merely proofs awaiting a few days of algebra. Some suggested hypotheses fail structurally, and others need independent verification. I would not promise that more time proves those versions.

**Normalization and a correction used throughout.** Let z(p) denote the vector of training logits, stacked and divided by √N. Write L(p)=ℓ(z(p)), r=∇_zℓ(z), J_b=∂z/∂b. Then ∇_bL=J_bᵀr and the GGN is J_bᵀH(z)J_b, where H=∇²_zℓ. For averaged softmax cross-entropy this reproduces the packet’s normalized residual and block-diagonal softmax Hessian. For squared loss ℓ(z)=½||z−y||², r=z−y and H=I.

**Fixed Jacobians under cross-entropy do not imply ṙ=−Θr.** They imply ż=−Θr and ṙ=−H(z)Θr. The exponential residual equation in P2 is exact for squared loss, or for a further fixed quadratic approximation with the appropriate metric. I will not silently replace cross-entropy by squared loss in the following statements.

**P4: discrete displacement with units, momentum, clipping, and adaptive updates.**

**Lemma 4.1 — pathwise clipped GD.** Let g_k be the spectral gradient presented to the optimizer at step k, and suppose, on a given sequence of iterates,

$$\|g_k\|\le B_k R_k,\qquad B_k\ge0,\ R_k\ge0.$$

Assume no spectral weight decay and updates

$$\theta_{k+1}=\theta_k-\eta_k a_k g_k,\qquad \eta_k\ge0,\quad 0\le a_k\le1.$$

The scalar a_k may depend arbitrarily on the entire model, gradients, and history; in particular it may be global norm clipping. Then, for every integer K≥0,

$$\|\theta_K-\theta_0\|\le\sum_{k=0}^{K-1}\eta_k a_k B_kR_k
\le\sum_{k=0}^{K-1}\eta_kB_kR_k. \tag{4.1}$$

**Proof.** Telescope the updates and use the triangle inequality at each step. No smoothness, convergence, full-batch condition, or stochastic independence is needed. ∎

For stochastic gradients this is a pathwise statement about the sampled gradients and their corresponding sampled residuals. A full-data residual envelope does not automatically bound a mini-batch gradient; cancellation in the full-data gradient can make that inference false.

**Corollary 4.2 — per-step polynomial envelope.** Suppose η_k=η, B_k≤B, and

$$R_k\le\frac{C}{1+\mu_s k},\qquad C>0,\quad\mu_s>0.$$

For K≥1,

$$\|\theta_K-\theta_0\|
\le\eta BC\sum_{k=0}^{K-1}\frac1{1+\mu_s k}
\le\eta BC\left[1+\frac{\log(1+\mu_s(K-1))}{\mu_s}\right]. \tag{4.2}$$

If δ∈(0,C), the envelope is valid at least through index

$$K_*:=\left\lceil\frac{C/\delta-1}{\mu_s}\right\rceil,$$

and K_δ:=min{k≥0:R_k≤δ}, then K_δ≤K_* and

$$\|\theta_{K_\delta}-\theta_0\|
\le \eta BC\left[1+\frac{\log(C/\delta)}{\mu_s}\right]. \tag{4.3}$$

If K_δ=0, its displacement is zero.

**Proof.** The decreasing function f(x)=1/(1+μ_sx) obeys Σ_{k=1}^{K−1}f(k)≤∫₀^{K−1}f(x)dx. The envelope guarantees R_{K_*}≤δ. For K_δ≥1, K_δ−1≤K_*−1<(C/δ−1)/μ_s, so substitute that index into (4.2). ∎

The leading η in (4.2)–(4.3) is essential. In particular, the proposed expression BC/μ_step times a logarithm, without η, is still missing the conversion factor.

**The exact dictionary:** t_k=ηk and μ_s=ημ_f, where μ_f is a rate in flow-time units. Thus ηBC/μ_s=BC/μ_f. The extra ηBC is a left-Riemann-sum allowance, not a new dynamical effect. With varying η_k, set t_k=Σ_{i<k}η_i; an envelope in k and one in t_k are different assumptions.

For completeness, if R_k≤C/(1+μ_f t_k) and η_max=max_{k<K}η_k, monotonicity of f(t)=1/(1+μ_ft) gives

$$\|\theta_K-\theta_0\|
\le BC\left[\frac{\log(1+\mu_f t_K)}{\mu_f}
+\eta_{\max}\left(1-\frac1{1+\mu_f t_K}\right)\right]. \tag{4.4}$$

Indeed, on each interval [t_k,t_{k+1}], the left-sum excess is at most η_k(f(t_k)−f(t_{k+1})); bound η_k by η_max and telescope. This also covers zero step sizes.

**Lemma 4.3 — heavy-ball momentum.** Suppose

$$v_{k+1}=\beta v_k+a_kg_k,\qquad
\theta_{k+1}=\theta_k-\eta_kv_{k+1},\qquad 0\le\beta<1,$$

with a_k as above. Then

$$\theta_K-\theta_0
=-v_0\sum_{k=0}^{K-1}\eta_k\beta^{k+1}
-\sum_{i=0}^{K-1}a_ig_i\sum_{k=i}^{K-1}\eta_k\beta^{k-i}. \tag{4.5}$$

For constant η,

$$\|\theta_K-\theta_0\|
\le\eta\frac{\beta(1-\beta^K)}{1-\beta}\|v_0\|
+\eta\sum_{i=0}^{K-1}\frac{1-\beta^{K-i}}{1-\beta}\,a_iB_iR_i. \tag{4.6}$$

In particular, for v₀=0, each bound (4.2)–(4.3) remains valid after multiplying its right-hand side by 1/(1−β). For variable step sizes, replacing the inner sum in (4.5) by η_max/(1−β) gives a valid, possibly looser bound.

**Proof.** Induction gives v_{k+1}=β^{k+1}v₀+Σ_{i=0}^kβ^{k−i}a_ig_i. Substitute into the parameter updates and interchange the finite sums. The geometric-series formula gives (4.6). ∎

The finite weights in (4.6) are sharper than a blanket 1/(1−β). The blanket factor is unavoidable in general for a bound based only on gradient magnitudes: aligned gradients over a long horizon approach that amplification. This lemma covers the common zero-dampening, non-Nesterov momentum rule when the spectral group has no decay. It does not silently cover Nesterov look-ahead gradients evaluated elsewhere.

Global clipping **before** accumulation in v is exactly covered by a_k. If clipping is applied to v after accumulation, write that algorithm explicitly: with v_{k+1}=a_k(βv_k+g_k), the norm recurrence ||v_{k+1}||≤β||v_k||+||g_k|| yields the same coarse bound. Clipping changes later iterates and gradients, so we are not claiming that a clipped run always has less displacement than an unclipped run started at the same point. We are bounding each run along its own gradients.

**Lemma 4.4 — what can and cannot be said for Adam.** Consider zero-initialized Adam moments, bias correction, no θ decay, and a fixed denominator stabilizer ζ>0. Write

$$\widehat m_k=\sum_{i=0}^k
\frac{(1-\beta_1)\beta_1^{k-i}}{1-\beta_1^{k+1}}\,g_i,
\qquad
\theta_{k+1}-\theta_k=-\eta_kD_k\widehat m_k,$$

where D_k is diagonal and ||D_k||≤1/ζ, as for denominators √v̂_k+ζ. With η_k≤η_max,

$$\|\theta_K-\theta_0\|
\le\frac{\eta_{\max}}{\zeta(1-\beta_1)}
\sum_{i=0}^{K-1}\|g_i\|. \tag{4.7}$$

**Proof.** Bound each update using ||D_k||≤1/ζ, expand the nonnegative weights of m̂_k, and interchange sums. For each i, its total weight over k≥i is at most

$$\sum_{k=i}^{K-1}\beta_1^{k-i}\le\frac1{1-\beta_1},$$

because (1−β₁)/(1−β₁^{k+1})≤1. ∎

Thus it would be false to say that *no* residual-based bound exists for standard positive-epsilon Adam. The issue is the new preconditioner/history constant, which may destroy informativeness. If ζ=0, already the first scalar bias-corrected step for g₀≠0 has magnitude η, independent of |g₀|. Let g₀=B R₀→0: no bound proportional to ηB R₀ with a constant uniform over residual scale can hold. For ζ>0 that ratio is 1/(|g₀|+ζ), which can be as large as 1/ζ. A smaller empirically certified preconditioner bound can replace 1/ζ, but must be measured or proved, not inferred from the raw Jacobian cap.

AdamW with θ decay adds an uncontrolled-by-residual term −η_kλθ_k. At zero data gradient and zero moments, θ_K=(Π_{k<K}(1−η_kλ))θ₀, which can move despite r≡0. A bound must include Ση_kλ||θ_k|| or exploit the exact decay recurrence. Decay on φ does not invalidate these pathwise θ lemmas; it changes the path on which their gradient/residual hypotheses must be checked.

These results repair the units and algorithm scope. They do not make a percentile gain a supremum, a smoothed envelope a pointwise envelope, or a terminal measurement a first hitting time.

**P1: comparison of the joint and frozen trajectories.**

**Theorem 1.1 — monotone frozen flow with a perturbed input.** Let (θ_J(t),φ_J(t)) be a joint trajectory and let φ_F solve

$$\dot\phi_F=-\nabla_\phi L(\theta_0,\phi_F),\qquad \phi_F(0)=\phi_J(0).$$

Assume φ_J obeys φ̇_J=−∇_φL(θ_J,φ_J). On a region containing both φ paths, suppose:

1. The frozen vector field is α-monotone for α≥0:
   $$\langle v-w,\nabla_\phi L(\theta_0,v)-\nabla_\phi L(\theta_0,w)\rangle\ge\alpha\|v-w\|^2.$$
2. The θ-induced forcing is bounded along the joint path:
   $$\|\nabla_\phi L(\theta_J(t),\phi_J(t))-\nabla_\phi L(\theta_0,\phi_J(t))\|\le K_{\theta\phi}d(t),$$
   where ||θ_J(t)−θ₀||≤d(t).
3. The normalized logit map obeys, for the parameter pairs used below,
   $$\|z(\theta_J(t),\phi_J(t))-z(\theta_0,\phi_J(t))\|\le L_\theta d(t),$$
   $$\|z(\theta_0,\phi_J(t))-z(\theta_0,\phi_F(t))\|\le L_\phi\|\phi_J(t)-\phi_F(t)\|.$$

Then

$$\|\phi_J(t)-\phi_F(t)\|
\le K_{\theta\phi}\int_0^t e^{-\alpha(t-s)}d(s)\,ds, \tag{1.1}$$

and

$$\|z_J(t)-z_F(t)\|
\le L_\theta d(t)+L_\phi K_{\theta\phi}\int_0^t e^{-\alpha(t-s)}d(s)\,ds. \tag{1.2}$$

If d is nondecreasing, the right-hand side is at most

$$\left[L_\theta+L_\phi K_{\theta\phi}\psi_\alpha(t)\right]d(t),
\qquad
\psi_\alpha(t)=\begin{cases}(1-e^{-\alpha t})/\alpha,&\alpha>0,\\t,&\alpha=0.\end{cases} \tag{1.3}$$

**Proof.** Set Δ=φ_J−φ_F and split its derivative into the frozen-field difference and θ-forcing. Then

$$\tfrac12\frac{d}{dt}\|\Delta\|^2\le-\alpha\|\Delta\|^2+K_{\theta\phi}d(t)\|\Delta\|.$$

The usual absolutely-continuous norm comparison, or regularization by √(||Δ||²+ε²) followed by ε↓0, yields (1.1) with Δ(0)=0. Add and subtract z(θ₀,φ_J) for (1.2), then use d(s)≤d(t). ∎

If the packet’s residual/gain hypotheses hold on [0,T], take

$$d(t)=\frac{BC}{\mu}\log(1+\mu t).$$

On [0,T_δ], provided the fitting time is attained within the valid horizon, this gives

$$\sup_{0\le t\le T_\delta}\|z_J(t)-z_F(t)\|
\le\left[L_\theta+L_\phi K_{\theta\phi}\psi_\alpha(T_\delta)\right]
\frac{BC}{\mu}\log(C/\delta). \tag{1.4}$$

For α>0 the bracket is bounded independently of T. For α=0 it is linear in T, not exponential. On a general later horizon use log(1+μT); the fitting-time logarithm does not bound indefinite subsequent movement. The C factor must not disappear into ε=B/μ unless explicitly treated as a fixed constant.

The mixed sensitivity K_{θφ}, output constants, and α must all be controlled across widths before claiming an improving width rate. Merely proving α grows while the other constants also grow is insufficient. Monotonicity can be restricted to the relevant identifiable subspace if both trajectories and the forcing remain there. In an overparameterized unregularized network, strong convexity in all φ coordinates is usually unavailable.

**Counterexample 1.2 — PL does not give trajectory contraction.** For a>0 define

$$f(x,y)=\tfrac12(x+ay^2)^2.$$

It is smooth, has infimum zero, and satisfies the global PL inequality with constant 1:

$$\|\nabla f\|^2=(x+ay^2)^2(1+4a^2y^2)\ge2f.$$

Nevertheless, along the gradient-flow solution (x(t),y(t))=(−b e^{−t},0), b>0, the linearized variation in the y direction obeys

$$\dot \xi(t)=2ab e^{-t}\xi(t),\qquad
\xi(t)=\xi(0)\exp\{2ab(1-e^{-t})\}.$$

Hence nearby trajectories expand, by an arbitrarily large factor at t=1 as a grows, despite a PL constant fixed at 1. At the starting point the Hessian has eigenvalues 1 and −2ab. This is a direct counterexample to the proposed step “PL ⇒ perturbations are forgotten at rate μ_φ.” PL controls objective suboptimality along a trajectory; it does not make the gradient map strongly monotone. See also the explicit distinction between PL and convexity in [Karimi, Nutini & Schmidt, 2016](https://arxiv.org/abs/1608.04636).

**Theorem 1.3 — sharper fixed-kernel comparison.** Consider the exactly affine model

$$z=z_0+A(\theta-\theta_0)+B(\phi-\phi_0),\qquad L=\tfrac12\|z-y\|^2,$$

trained by unit-rate Euclidean gradient flow. Let K_θ=AAᵀ and K_φ=BBᵀ. Suppose a common invariant linear subspace V contains e₀=z₀−y and satisfies

$$K_\phi|_V\succeq\kappa I_V,\qquad
0\preceq K_\theta|_V\preceq aI_V,\qquad\kappa>0,\ a\ge0.$$

Start joint and frozen runs at the same parameters. Then for t≥0,

$$\|z_J(t)-z_F(t)\|\le a t e^{-\kappa t}\|e_0\|,
\qquad
\sup_{t\ge0}\|z_J(t)-z_F(t)\|\le\frac{a}{e\kappa}\|e_0\|, \tag{1.5}$$

$$\|\theta_J(t)-\theta_0\|
\le\frac{\sqrt a}{\kappa}(1-e^{-\kappa t})\|e_0\|, \tag{1.6}$$

and

$$\int_0^\infty\|\nabla_\theta L(t)\|^2dt
\le\frac{a}{2\kappa}\|e_0\|^2. \tag{1.7}$$

If e₀≠0, the fraction of total loss decrease attributed to θ is therefore at most min{1,a/κ}.

**Proof.** The residuals are

$$e_J(t)=e^{-(K_\phi+K_\theta)t}e_0,\qquad
e_F(t)=e^{-K_\phi t}e_0.$$

Both semigroups have norm at most e^{−κt} on V. The variation-of-constants identity, which does not require commuting kernels, gives

$$e_J(t)-e_F(t)=-\int_0^t e^{-K_\phi(t-s)}K_\theta e^{-(K_\phi+K_\theta)s}e_0\,ds.$$

Its norm is at most a t e^{−κt}||e₀||. Maximize t e^{−κt} at t=1/κ. Since ||Aᵀv||²=vᵀK_θv≤a||v||² for v∈V, integrate θ̇=−Aᵀe_J to get (1.6), and integrate its squared norm to get (1.7). Both residuals converge to zero; the joint-flow energy identity gives total loss decrease ||e₀||²/2. ∎

This is a genuine bridge: **if** a=O(1), κ=Ω(M), and ||e₀||=O(1), joint-versus-frozen training-logit distance, spectral displacement, and spectral dissipation share are all O(1/M). The necessary spatial hypothesis is a lower bound on **every residual direction visited**, not on the largest eigenvalue. V can be much smaller than the entire output space, but its common invariance is part of the hypothesis.

**Counterexample 1.4 — the class-prior/top-eigenvalue floor is insufficient.** Let

$$K_\phi=\begin{pmatrix}M&0\\0&0\end{pmatrix},\quad
K_\theta=\begin{pmatrix}0&0\\0&1\end{pmatrix},\quad e_0=(0,1)^\top.$$

The spatial top eigenvalue is M and its Rayleigh quotient in direction (1,0) is M. Yet e_F(t)=(0,1), e_J(t)=(0,e^{−t}), and ||z_J−z_F||=1−e^{−t}, independent of M. The spatially fast direction carries none of this residual. Proposition `prop:ntk_classprior` cannot supply κ in Theorem 1.3 without a new residual-support/alignment result.

An explicit robustness version is also available. If the actual residual equations differ from the two constant-kernel equations by additive errors ξ_J(t),ξ_F(t)∈V, with the same initialization, add

$$\int_0^t e^{-\kappa(t-s)}(\|\xi_J(s)\|+\|\xi_F(s)\|)\,ds$$

to (1.5). This follows by comparing each perturbed equation to its own unperturbed semigroup. Uniform error bounds ν_J,ν_F contribute at most (ν_J+ν_F)/κ. It states precisely what an NTK-approximation argument must control; initialization alone does not provide it.

For cross-entropy, Theorem 1.1 still applies when its assumptions hold, but Theorem 1.3 is a squared-loss theorem. Softmax curvature loses a uniform positive lower bound as margins diverge, so a global strong-convexity/constant-rate argument cannot simply be imported. The nonlinear cross-entropy example in P5 avoids that substitution.

Neither theorem by itself turns L² logit closeness into macro-F1 equivalence. One elementary consequence is available with margins: if z denotes normalized logits, ||z_J−z_F||≤h, and a fraction at most b_γ of samples have frozen-model top-two logit margin at most γ, then the prediction-disagreement fraction is at most b_γ+4h²/γ². For every remaining disagreement the unnormalized per-sample logit error must exceed γ/2; Markov’s inequality on the mean squared error proves the claim. Macro-F1 requires additional control of class-specific confusion counts and denominators.

**Corollary 1.5 — a fully verified initialization source for the frozen-kernel bridge.** The needed kernel coercivity can be proved, rather than assumed, for a finite-data version of the shallow biased-ReLU instance. Fix N<∞, x_n∈R^S, and a deterministic linear encoder θ₀ such that the bottleneck inputs z_n=W₀x_n∈R^K are pairwise distinct and ||z_n||≤R. Let

$$h_i(z)=\operatorname{ReLU}(w_i^\top z+b_i),\qquad
\widehat y(z)=Vh(z)+b_2,$$

where w_i independently have law N(0,s_w²I_K/K), b_i independently have law N(0,s_b²), V has independent N(0,s_w²/M) entries, b₂ has independent N(0,s_b²) entries, s_w,s_b>0, and all arrays are independent. There are C output coordinates. Let A,B be the spectral and spatial normalized logit Jacobians **at this initialization**, and train the resulting exactly affine model of Theorem 1.3 using squared loss. No claim of nonlinear-trajectory approximation is part of this corollary.

Define the N×N population random-feature kernel

$$\psi=(h_1(z_1),\ldots,h_1(z_N))^\top,\qquad
K_*=\frac1N\mathbb E[\psi\psi^\top],\qquad\kappa_*:=\lambda_{\min}(K_*).$$

Then κ_*>0. For every confidence parameter ρ∈(0,1), there are constants A_ρ,R_ρ<∞ and M₀<∞, all independent of M, such that for each M≥M₀, with probability at least 1−ρ,

$$K_\phi\succeq\frac{\kappa_*M}{2}I,
\qquad \|K_\theta\|\le A_\rho,
\qquad\|e_0\|\le R_\rho.$$

On this event,

$$\sup_{t\ge0}\|z_J(t)-z_F(t)\|
\le\frac{2A_\rho R_\rho}{e\kappa_*M},\qquad
\sup_{t\ge0}\|\theta_J(t)-\theta_0\|
\le\frac{2\sqrt{A_\rho}R_\rho}{\kappa_*M}. \tag{1.8}$$

**Proof.** First prove strict positive definiteness. If cᵀK_*c=0, then

$$\sum_{n=1}^N c_n\operatorname{ReLU}(w^\top z_n+b)=0$$

almost surely in the nondegenerate Gaussian pair (w,b). The left-hand side is continuous and the Gaussian has full support, so it vanishes everywhere in R^{K+1}. For any fixed n, the hyperplane H_n={(w,b):wᵀz_n+b=0} differs from all the other H_j: their normals (z_n,1) cannot be proportional unless z_n=z_j. Choose a point of H_n lying on none of the other finitely many hyperplanes. Across a sufficiently small transverse segment through that point, all other ReLU terms are affine, while the n-th term has a gradient jump c_n(z_n,1). The identically zero function has no jump, so c_n=0. Repeating over n gives c=0 and hence κ_*>0.

Let H be the N×M matrix with H_{ni}=h_i(z_n). The head-weight contribution to the spatial logit kernel is (HHᵀ/N)⊗I_C, up to coordinate ordering. Its M columns are IID. Put s_max²=s_w²R²/K+s_b². Since a centered Gaussian ReLU has fourth moment at most 3s_max⁴/2, Cauchy–Schwarz gives E||ψ||⁴≤3N²s_max⁴/2. Independence of columns therefore implies

$$\mathbb E\left\|\frac{HH^\top}{NM}-K_*\right\|_F^2
\le\frac{3s_{\max}^4}{2M}.$$

Markov and ||·||_op≤||·||_F give

$$\Pr\left\{\left\|\frac{HH^\top}{NM}-K_*\right\|_{\rm op}>\kappa_*/2\right\}
\le\frac{6s_{\max}^4}{M\kappa_*^2}.$$

Thus one may take M₀=ceil(18s_max⁴/(ρκ_*²)). For M≥M₀, λ_min(HHᵀ/N)≥κ_*M/2 with probability at least 1−ρ/3. Adding the other spatial parameter contributions is positive semidefinite and cannot reduce this lower bound.

The same conditional Gaussian calculation as Proposition 38, Step D, bounds the **logit** Jacobian, without inserting a softmax factor:

$$\mathbb E\|J_\theta\|_F^2\le C_G:=C s_w^4\operatorname{tr}(\Sigma_X).$$

Therefore ||K_θ||≤A_ρ:=3C_G/ρ with probability at least 1−ρ/3, by Markov. (If C_G=0 the bound is deterministic; nontrivial pairwise-distinct inputs typically exclude this case.)

For normalized squared-loss targets y, since each hidden activation has second moment at most s_max²/2,

$$\mathbb E\|e_0\|^2\le C_r:=2C(s_w^2s_{\max}^2/2+s_b^2)+2\|y\|^2.$$

Markov gives ||e₀||≤R_ρ:=√(3C_r/ρ) with probability at least 1−ρ/3. The three events need not be independent; a union bound gives probability at least 1−ρ. Apply Theorem 1.3 with V the full output space, κ=κ_*M/2 and a=A_ρ. ∎

This is a nonempty, fully proved initialization instance of the missing **frozen-linearization** bridge. It uses all data-dependent random-feature directions, not the class-prior spike. Its constants are allowed to depend on the fixed dataset and N; κ_* can become extremely small for near-duplicate bottleneck inputs and need not remain positive uniformly as N grows. Exact duplicate inputs require restriction to a compatible subspace; this corollary does not cover conflicting labels at identical inputs. It also supplies no nonlinear NTK-tracking theorem. Those restrictions distinguish it from a production-scale claim.

For a fully trainable shallow ReLU network, the frozen-initialization kernel does not establish Theorem 1.1’s monotonicity along the nonlinear trajectory. If hidden features are actually frozen and only a linear readout is optimized, the frozen objective is convex; positive curvature on an identifiable subspace additionally needs a feature-rank condition and, for CE, a probability/margin restriction or appropriate regularization. None of these properties is presently proved for the 12-layer ViT.

**P3: an interior witness, a proved initialization instance, and the normalization obstruction.**

**Lemma 3.1 — pointwise interior-layer witness.** At N sites let h_n∈R^d, a_n=W h_n+b∈R^q, and let the logits be Φ_n(a_n)∈R^C, C≥2. The maps may depend on other fixed quantities, but for this lemma perturbing a_n does not change logits at other sites. Set D_n=∂Φ_n/∂a_n, H_n=diag(p_n)−p_np_nᵀ, Π=I−11ᵀ/C, and p_min,n=min_c p_n,c. All these are evaluated at the parameter point under consideration.

For any unit a∈R^q, define

$$S_{h,a}:=\frac1N\sum_n
p_{\min,n}\|\Pi D_na\|^2h_nh_n^\top.$$

Then the full spatial GGN, if it includes W, satisfies

$$\lambda_{\max}(G_{\phi\phi})\ge\lambda_{\max}(G_{WW})
\ge\lambda_{\max}(S_{h,a}). \tag{3.1}$$

In particular, if ||ΠD_na||≥s for every contributing site, then

$$\lambda_{\max}(G_{\phi\phi})\ge s^2\lambda_{\max}(S_h^p),
\quad S_h^p=\frac1N\sum_n p_{\min,n}h_nh_n^\top. \tag{3.2}$$

**Proof.** For a unit u∈R^d use the unit-Frobenius perturbation V=a uᵀ of W, with all other parameters fixed. Its logit perturbation at site n is (h_nᵀu)D_na. Since H_n⪰p_min,n Π,

$$V^\top G_{WW}V
=\frac1N\sum_n(h_n^\top u)^2(D_na)^\top H_nD_na
\ge u^\top S_{h,a}u.$$

Maximize over u. Padding the perturbation with zeros in other parameter coordinates establishes the full-block inequality. ∎

A minimum over every site is not necessary: (3.1) already gives the correct energy-weighted gain. A claim based on a restricted singular value must specify the restriction, the softmax class-shift nullspace, and whether the same witness a works across sites. The ordinary smallest singular value of a C×q map is not an injectivity bound when q>C−1; a nonzero smallest *nonzero* singular value does not protect a witness lying in the kernel.

**Spatial mixing version.** Let R_h map vec(V) to the normalized stack N^{−1/2}(Vh_n)_n, and let D_Φ be the full unnormalized downstream Jacobian, including spatial mixing. With the corresponding output softmax Hessian H,

$$\lambda_{\max}(G_{\phi\phi})
\ge\|H^{1/2}D_\Phi R_h\,\mathrm{vec}(V)\|^2
\quad\text{for every }\|V\|_F=1. \tag{3.3}$$

This is the universally valid interior witness. To obtain the proposed product with λ_max(S_h^p), for a chosen maximizing witness V=a uᵀ one must prove

$$\|H^{1/2}D_\Phi R_h\,\mathrm{vec}(V)\|^2
\ge s^2\frac1N\sum_n p_{\min,n}\|Vh_n\|^2. \tag{3.4}$$

Here the indexing/weights on the interior sites must be explicitly matched to the output sites; otherwise use the exact expression (3.3) instead. Equation (3.4) is a restricted gain condition on the actual spatial pattern. A typical Gaussian action on one fixed vector is not enough when the vector, gates, and normalizations depend on the same weights.

**Theorem 3.2 — width-linear CE curvature through a fixed-size ReLU bottleneck.** Fix q,C≥2 (q≥1 would also suffice) and constants R>0, s_w>0, s_b≥0, s_v>0. For every M let deterministic features h_1,…,h_N∈R^M satisfy

$$\|h_n\|^2\le R^2M,\qquad
\lambda_{\max}(S_h)\ge a_0M,
\quad S_h:=\frac1N\sum_nh_nh_n^\top,
\quad a_0>0.$$

Consider logits

$$\widehat y_n=V\,\operatorname{ReLU}(Wh_n+b),$$

where W∈R^{q×M} has independent N(0,s_w²/M) entries, b has independent N(0,s_b²) entries, V∈R^{C×q} has independent N(0,s_v²/q) entries, and these arrays are independent. The features are independent of all these arrays. Use averaged softmax cross-entropy and the Euclidean parameterization in the displayed formula. Then there exist constants c_0,p_0>0 independent of M and N such that

$$\Pr\{\lambda_{\max}(G_{WW})\ge c_0a_0M\}\ge p_0,
\qquad
\mathbb E\lambda_{\max}(G_{\phi\phi})\ge p_0c_0a_0M. \tag{3.5}$$

Thus a fixed final classifier dimension does not prevent a width-linear floor generated by an interior layer. This theorem is a positive-probability and expectation statement, not a claim that its probability tends to one with M.

**Proof with explicit constants.** Let u be a deterministic top unit eigenvector of S_h, and define

$$a_n=(h_n^\top u)^2,\qquad A=\frac1N\sum_na_n\ge a_0M.$$

Let t_n=Wh_n+b, χ_n=1{t_{n,1}>0}, and

$$Q=\frac1N\sum_na_n\chi_n.$$

Every site with a_n>0 has a nondegenerate centered Gaussian first preactivation, so Eχ_n=½ there. Hence EQ=A/2 and 0≤Q≤A. The inequality

$$\mathbb EQ\le A\Pr(Q\ge A/4)+(A/4)\Pr(Q<A/4)$$

implies Pr(Q≥A/4)≥1/3. No independence across sites was used.

Put s_max²=s_w²R²+s_b² and L₀=√(96q s_max²). Since E||ReLU(t_n)||²≤q s_max², Markov gives

$$\Pr\{\|\operatorname{ReLU}(t_n)\|>L_0\}\le1/96.$$

The weighted untame energy

$$U=\frac1N\sum_na_n1\{\|\operatorname{ReLU}(t_n)\|>L_0\}$$

satisfies EU≤A/96, so Pr(U>A/8)≤1/12. Therefore the event

$$Q\ge A/4,\qquad U\le A/8$$

has probability at least 1/4. On it, active-and-tame sites carry at least A/8 of witness energy.

Independently, define the readout event

$$E_V=\{\|V\|_F\le2,\ \|\Pi Ve_1\|\ge1/2\}.$$

It has probability p_V>0 independent of M,N: the nondegenerate Gaussian matrix distribution gives positive mass to a small open neighborhood of a matrix with first column a unit mean-zero class contrast and other columns zero. On a tame site and E_V, ||ŷ_n||_∞≤2L₀, hence

$$p_{\min,n}\ge e^{-4L_0}/C=:c_p.$$

For the unit witness ΔW=e₁uᵀ, the logit perturbation is (h_nᵀu)χ_nVe₁. Its curvature on active-and-tame sites is at least

$$c_p\cdot\tfrac14\cdot\frac A8=\frac{c_pA}{32}.$$

Thus take c₀=e^{−4L₀}/(32C) and p₀=p_V/4. The expected bound follows by nonnegativity and inclusion of W in φ. ∎

This proof is intentionally conservative: its constants may be poor, but the width dependence and independence from N are explicit. It is not a random-matrix smallest-singular-value argument. It combines aggregate gate activation, an energy-weighted tail bound, and a readout event. Conditioning on random incoming features is allowed if their two deterministic bounds hold uniformly on the conditioning event; its probability must then be included.

For scalar **squared loss**, with V=vᵀ and v_i independently N(0,s_v²/q), the same witness gives the simpler expectation bound

$$\mathbb E\lambda_{\max}(G_{WW})\ge\frac{s_v^2}{2q}\lambda_{\max}(S_h). \tag{3.6}$$

Indeed, its curvature is v₁²Q, and v is independent of Q. This is another concrete interior-width instance with no bias requirement.

**What this does not prove.** It treats a pointwise ReLU bottleneck and linear readout, without BatchNorm. Extending a positive-probability argument through additional fixed-width pointwise ReLU layers can use a common active-path/readout event, but a convolutional decoder that mixes positions and normalizes activations needs its own gain argument. I am not asserting that extension here. No part of this theorem establishes a width-uniform spectral Jacobian cap for the network *upstream* of h; a disparity theorem still needs that separate ingredient.

**Counterexample 3.3 — fixed-size ReLU prevents generic probability tending to one.** Take q=1, all h_n equal to a nonzero h of squared norm M, a zero bias, and a scalar centered Gaussian preactivation Wh. With probability 1/2 it is negative at every site. The entire W Jacobian then vanishes, independently of M. Thus no positive Ω(M) lower bound for this interior block can hold with probability 1−o_M(1) across the stated class. With larger but fixed q, an all-inactive event still has positive probability independent of M. A high fixed confidence may be possible for a particular q and assumptions, but increasing M alone does not remove this obstruction.

**Counterexample 3.4 — train-mode BN can erase the entire common-feature witness.** Let h_n=h for all sites, ||h||²=M, and put train-mode BN immediately after Wh_n. For every W, each channel is constant across sites; BN subtracts exactly that channel mean. Its output is the BN affine offset, independent of W, including when the stabilizer is positive. Thus D_BN R_h vec(V)=0 for every perturbation V and G_WW=0, while S_h^p has a positive eigenvalue proportional to M whenever the downstream logits have fixed nondegenerate probabilities. If all subsequent widths are fixed, the remaining trainable downstream parameters can have only M-independent curvature in this example.

This is an exact counterexample to a normalization-independent production lemma, not just a loose proof technique. With a 1×1 convolution or periodic constant inputs it is literally the convolutional setting; padding boundaries or nonconstant features change the example but do not establish the missing gain. Production investigation should measure (3.3), including BN and spatial mixing, before selecting a lower-bound route.

**P2: directional curvature, gradient gain, and cumulative energy.**

**First correction: the proposed one-sided GGN inequality is already false for CE.** With one example, two logits (θ,0), θ=0, and label class 1,

$$r=(-1/2,1/2),\quad J=(1,0)^\top,\quad
|\nabla_\theta L|^2=1/4,\quad G_{\theta\theta}=1/4,\quad\|r\|^2=1/2.$$

The proposed right-hand side λ_GGN||r||² is 1/8, smaller than the gradient square. Curvature inserts H; the Euclidean residual-to-gradient gain does not.

For a directional parameter subspace with isometric embedding R_u and J_u=J_θR_u, the always-valid bound is

$$\|J_u^\top r\|^2\le\lambda_{\max}(J_u^\top J_u)\|r\|^2. \tag{2.1}$$

For softmax CE, r lies in the range of H (per-site class-mean-zero space), and the valid GGN alternative is

$$\|J_u^\top r\|^2
\le\lambda_{\max}(J_u^\top HJ_u)\,r^\top H^\dagger r. \tag{2.2}$$

**Proof.** Write J_uᵀr=(H^{1/2}J_u)ᵀH^{†/2}r and apply the operator-norm inequality. If H⪰h₀Π on the relevant class-contrast space, then rᵀH†r≤||r||²/h₀. That lower softmax-curvature bound is an additional hypothesis and can deteriorate as predictions saturate. ∎

For linear W∈R^{K×S}, R_u maps b∈R^K to vec(buᵀ). Consequently J_uᵀr is precisely the K-vector (∇_WL)u, with the agreed normalization. The restricted matrix measured by Exp 1.8b is J_uᵀHJ_u, not J_uᵀJ_u.

**Theorem 2.1 — exact cumulative-energy identity and a sufficient excitation condition.** In the exactly affine squared-loss model, let A_u be the directional logit Jacobian, C_u=A_uA_uᵀ, λ_u=||C_u||, and define

$$Q_T=\int_0^T r(t)r(t)^\top dt,\qquad
E_u(T)=\int_0^T\|A_u^\top r(t)\|^2dt.$$

Then

$$E_u(T)=\operatorname{tr}(C_uQ_T)\le\lambda_u\operatorname{tr}(Q_T). \tag{2.3}$$

Let λ_v>0 and let q_v be a top unit eigenvector of C_v. If tr(Q_T)>0 and

$$q_v^\top Q_Tq_v\ge\rho\operatorname{tr}(Q_T),\qquad\rho>0,$$

then, for λ_u>0,

$$E_v(T)\ge\rho\frac{\lambda_v}{\lambda_u}E_u(T). \tag{2.4}$$

When E_u(T)>0 this is the desired ratio inequality. A corresponding top-subspace version follows if C_v⪰cλ_vP and tr(PQ_T)≥ρtr(Q_T), with cρ replacing ρ.

**Proof.** The identity follows from ||A_uᵀr||²=rᵀC_ur and integration. Since C_v⪰λ_vq_vq_vᵀ, E_v≥λ_vq_vᵀQ_Tq_v≥ρλ_vtr(Q_T). Combine with the upper bound for u. ∎

Under an envelope ||r(t)||≤C/(1+μt), (2.3) implies E_u(T)≤λ_u C²/μ. This is correct for the squared-loss/logit-Gram λ_u. For CE, integrate (2.2) with the corresponding time-varying quantities or impose a uniform softmax lower bound; the same numerical restricted-GGN λ_u cannot be substituted without that factor.

The excitation hypothesis is measurable and mathematically sufficient, but it is not supplied by a large initial curvature or by Proposition `prop:ntk_classprior`. It involves the output-space singular direction associated with the *spectral directional Jacobian*, not simply an input PC or a spatial-head class-prior vector. A persistent-excitation theorem would be a real additional dynamical result. Checking it on the same trajectory makes a useful explanation of observed energy, not a new prediction from geometry alone.

**Counterexample 2.2 — no dominant residual component.** In a two-output affine squared-loss model take C_v=L e₁e₁ᵀ, C_u=ℓ e₂e₂ᵀ, L≫ℓ>0, total kernel diag(L,ℓ), and r(0)=e₂. Then r(t)=e^{−ℓt}e₂, E_v(T)=0 and E_u(T)>0 for T>0. The curvature ratio L/ℓ can be arbitrarily large. These kernels are realized by the affine output (√L θ₁,√ℓ θ₂), or by the corresponding two orthogonal input features of a linear encoder.

**Counterexample 2.3 — even equal initial excitation does not preserve the ratio cumulatively.** In the same model let r(0)=e₁+e₂. Direct integration gives

$$E_v(T)=\frac{1-e^{-2LT}}2,\qquad
E_u(T)=\frac{1-e^{-2\ell T}}2.$$

As T→∞ their ratio tends to 1, not L/ℓ. The large-curvature mode exhausts its residual quickly. More generally, if the total kernel has eigenvalues γ_v,γ_u in these two directions and the initial amplitudes are a_v,a_u, the exact ratio is

$$\frac{E_v(T)}{E_u(T)}
=\frac{\lambda_v}{\lambda_u}\frac{a_v^2}{a_u^2}
\frac{\gamma_u}{\gamma_v}
\frac{1-e^{-2\gamma_vT}}{1-e^{-2\gamma_uT}}. \tag{2.5}$$

This explicitly exhibits the missing decay-rate and residual-alignment factors. A pointwise initial curvature ratio can describe an early-time energy-rate ratio in a special aligned model, while saying little about integrated energy over fitting.

Finally, an energy ratio—even one proved by (2.4)—is not by itself Pezeshki-style competitive suppression of a useful feature. That requires comparing learning as a competing feature/pathway is changed, together with a definition of feature response or usefulness. [Pezeshki et al., 2021, Definition 2](https://proceedings.neurips.cc/paper_files/paper/2021/file/0987b8b338d6c90bbedd8631bc499221-Paper.pdf) The serial example next supplies an actual intervention in pathway multiplicity rather than renaming anisotropy.

**P5: an exactly solvable serial cross-entropy model.**

**First, the information-theoretic obstruction.** If Z=f_θ(X) and a classifier q_φ(Y|Z) has expected cross-entropy at most η under a distribution P, then

$$H_P(Y\mid Z)\le\mathbb E_P[-\log q_\phi(Y\mid Z)]\le\eta,
\qquad I_P(Y;Z)\ge H_P(Y)-\eta. \tag{5.1}$$

**Proof.** Cross-entropy is conditional entropy plus the expected conditional KL divergence between P(Y|Z) and q_φ(Y|Z). ∎

This applies to the empirical training distribution for a fixed trained model as well. A serial encoder through which all predictions pass cannot simultaneously carry almost no training-label information and support almost perfect training fit. A particular spectral component may be neglected; a representation may rely on nonrobust context; test information may be poor under a changed distribution. Those are different claims and must be specified.

The following theorem proves selective suppression and a concrete failure under contextual reversal. It does not use a PL condition, an NTK approximation, or an assumed fitting-rate envelope.

**Theorem 5.1 — selective spectral suppression in a serial two-site model.** Let Y be uniform on {−1,+1}. The training input has two sites,

$$x_1=(S,0),\qquad x_2=(0,C),\qquad S=C=Y.$$

Thus S and C are each perfectly predictive of Y and carry identical label information. Interpret S as a local spectral cue and C as a contextual cue at the second site; these names designate routing, not an asserted physical model of tissue.

Apply the same trainable linear encoder W=(a,v) to both sites:

$$z_1=ax_{1,1}+vx_{1,2}=aS,\qquad z_2=a x_{2,1}+v x_{2,2}=vC.$$

The downstream model uses a spectral skip and M identical contextual channels with trainable readout β∈R^M:

$$F_{a,v,\beta}(X)=z_1+\sum_{j=1}^M\beta_jz_2=aS+bvC,
\qquad b:=\sum_{j=1}^M\beta_j.$$

All a,v,β_j are trained by unit-rate Euclidean gradient flow on

$$L=\mathbb E\log(1+e^{-YF})=\log(1+e^{-q}),\qquad q:=a+bv,$$

from a(0)=0, v(0)=1, and β_j(0)=0. There is no decay, clipping, momentum, or normalization. This is a genuine serial composition: both inputs reach the downstream model only through W. Broadcasting the same two-class logits to both sites with label Y gives the identical averaged per-pixel loss if desired.

Fix δ∈(0,1/2), let m=log((1−δ)/δ)>0, and let T_m be the first time the training margin q reaches m. Then:

**(a) Exact trajectory and finite fitting time.** With r=1/(1+e^q),

$$\dot a=r,\qquad \dot v=br,\qquad\dot b=Mvr. \tag{5.2}$$

Along the whole trajectory,

$$v=\cosh(\sqrt M\,a),\qquad
b=\sqrt M\sinh(\sqrt M\,a), \tag{5.3}$$

$$q=Q_M(a):=a+\frac{\sqrt M}{2}\sinh(2\sqrt M\,a),\qquad
\dot q=(M+1+2b^2)r. \tag{5.4}$$

If Ψ(q)=q+e^q−1, then

$$\frac{\Psi(m)}{M+1+2m^2}\le T_m\le\frac{\Psi(m)}{M+1}. \tag{5.5}$$

**(b) Suppression at matched fit.** At T_m,

$$0<a_M:=a(T_m)\le\frac{m}{M+1},\qquad
b(T_m)v(T_m)=m-a_M\ge\frac{M}{M+1}m, \tag{5.6}$$

$$0\le v(T_m)-1\le\frac{m^2}{2M},\qquad
\|W(T_m)-W(0)\|_2
\le\sqrt{\frac{m^2}{(M+1)^2}+\frac{m^4}{4M^2}}. \tag{5.7}$$

For fixed m, a_M decreases strictly with M and Ma_M→m. Removing the contextual path entirely yields a spectral-only model that reaches the same margin with a=m. Thus increasing contextual multiplicity suppresses the learned spectral response at equal training fit, by at least a factor M+1 relative to this comparator.

**(c) Derived residual envelope and curvature disparity.** For all t≥0,

$$r(t)\le\frac{1/2}{1+(M+1)t/4}. \tag{5.8}$$

With the two parameter blocks θ=(a,v), φ=β, the CE GGN top eigenvalues are

$$\lambda_\theta=r(1-r)(1+b^2),\qquad
\lambda_\phi=M r(1-r)v^2.$$

In particular,

$$\lambda_\theta(0)=1/4,\qquad\lambda_\phi(0)=M/4,\qquad\mathcal D_{\rm curv}(0)=M. \tag{5.9}$$

For 0≤t≤T_m, the spectral gradient gain in scalar logistic residual units obeys

$$\|\nabla_\theta L(t)\|=r(t)\sqrt{1+b(t)^2}\le r(t)\sqrt{1+m^2}. \tag{5.10}$$

**(d) Comparison to a separately trained frozen encoder.** Freeze W=(0,1), initialize β_F=0, and train β_F by its own gradient flow on the same loss. Write its margin q_F(t). For all 0≤t≤T_m,

$$0\le q(t)-q_F(t)\le\frac{(1+2m^2)t}{2}
\le\frac{(1+2m^2)\Psi(m)}{2(M+1)}. \tag{5.11}$$

Consequently the RMS difference of the two scalar training-logit functions is O(1/M) on this fitting horizon. For two symmetric logits (F/2,−F/2), divide this scalar-logit bound by √2 under the packet’s normalized L² convention.

**(e) A specified generalization failure.** On the shifted distribution S=Y, C=−Y, the model at T_m has signed margin

$$YF=2a_M-m<0$$

for every M≥1. Its classification error is one, whereas the spectral-only comparator at margin m has error zero. This is an intentionally complete contextual reversal, not a claim about ordinary IID test error.

**Proof.** Because S=C=Y in training, the expected loss depends only on q=a+bv. Differentiation gives ȧ=r, v̇=br, and β̇_j=vr, hence ḃ=Mvr. Since r>0 at finite parameters, a is a strictly increasing trajectory coordinate. Dividing by ȧ gives

$$\frac{dv}{da}=b,\qquad\frac{db}{da}=Mv,\qquad v(0)=1,\ b(0)=0,$$

whose unique solution is (5.3). Therefore b²=M(v²−1), and differentiating q gives

$$\dot q=(1+b^2+Mv^2)r=(M+1+2b^2)r.$$

These formulas also establish existence for all finite t: ȧ≤1/2 while a≥0, so a, and hence b and v from (5.3), stay finite on every finite interval. The strictly increasing function Q_M maps [0,∞) onto [0,∞). It reaches every finite target m in finite time because da/dt is positive and bounded below on the compact interval 0≤a≤Q_M^{-1}(m).

Since sinh x≥x for x≥0,

$$Q_M(a)\ge(M+1)a,$$

strictly for a>0, establishing the a bound in (5.6) and its strictness. On 0≤t≤T_m, b,v≥0, v≥1, and bv=q−a≤m, so b≤m. The invariant yields

$$v-1=\frac{b^2}{M(v+1)}\le\frac{m^2}{2M},$$

which proves (5.7). For fixed a>0, Q_M(a) increases strictly with M, as is evident from its positive power series in M. Its inverse at fixed m therefore decreases strictly. Also a_M≤m/(M+1), so √M a_M→0. Expanding sinh at this vanishing argument gives

$$m=(M+1)a_M+O(M^2a_M^3)=(M+1)a_M+O(1/M),$$

and hence Ma_M→m. With the contextual path removed q=a, so reaching the same m requires a=m exactly.

For the rate, ṙ=−r(1−r)q̇=−(M+1+2b²)r²(1−r). Since q≥0, r≤1/2, and therefore

$$\frac{d}{dt}\frac1r\ge\frac{M+1}{2}.$$

Use r(0)=1/2 to obtain (5.8). Furthermore Ψ′(q)=1+e^q=1/r, so

$$\frac{d}{dt}\Psi(q(t))=M+1+2b(t)^2\in[M+1,M+1+2m^2]$$

up to T_m. Integrating from 0 to T_m gives (5.5).

The margin derivatives on each training example are Y(1,b) in θ and Yv·1_M in β. The scalar logistic curvature is r(1−r). Averaging Y²=1 gives rank-one GGN blocks with precisely the eigenvalues in (c), and the gradient formula gives (5.10).

For the frozen model, q̇_F=M/(1+e^{q_F}), hence Ψ(q_F(t))=Mt. The joint model obeys

$$0\le\Psi(q(t))-\Psi(q_F(t))
=\int_0^t(1+2b(s)^2)\,ds\le(1+2m^2)t.$$

Both margins are nonnegative, Ψ is increasing, and Ψ′≥2 there. This proves (5.11). Under the contextual reversal the margin is a−bv=2a−m. For M>1, (5.6) makes it negative; for M=1, the strict inequality Q_1(a)>2a at a>0 gives a_1<m/2 as well. ∎

**The original finite-time bound is analytically informative in this example.** Use symmetric logits (F/2,−F/2), so the normalized two-class residual norm is R=√2 r. On [0,T_m], the actual normalized logit-Jacobian operator norm is

$$\|J_\theta\|_{\rm op}=\sqrt{(1+b^2)/2}\le B_m:=\sqrt{(1+m^2)/2}.$$

Equation (5.8) gives a valid packet-style envelope with C=1 and μ=(M+1)/4. At the target R=√2δ<1, Theorem 27 therefore yields the genuine analytic bound

$$\|W(T_m)-W(0)\|
\le\frac{4B_m}{M+1}\log\frac1{\sqrt2\delta}. \tag{5.12}$$

For the broadcast-to-two-sites convention, the input second moment is Σ_X=I₂/2 and the unnormalized downstream input-Jacobian norm is √(1+b²). Lemma 18 therefore supplies exactly the displayed normalized cap, with a width-uniform downstream constant √(1+m²) on the region |b|≤m containing this fitting trajectory. All the finite-time hypotheses are explicit in this example. There is no percentile estimate, no fitted μ, and no substitution of a step index for flow time in (5.12).

For δ=0.05, m=log 19, solving the scalar monotone equation Q_M(a_M)=m and substituting in the exact formulas gives:

| M | Exact-formula spectral displacement | Bound (5.12) | Bound/displacement |
|---:|---:|---:|---:|
| 16 | 0.219021 | 1.370601 | 6.258 |
| 64 | 0.071050 | 0.358465 | 5.045 |
| 256 | 0.019658 | 0.090662 | 4.612 |
| 1024 | 0.005064 | 0.022732 | 4.489 |

These are rounded numerical evaluations of proved scalar formulas, not training-run results. Since ||W(0)||=1, the bound certifies a genuinely small relative movement at sufficiently large M. This supplies the kind of numerical theorem instance the earlier paper lacked, although it is deliberately much simpler than the production experiment.

**What “starvation” can mean in this theorem.** At matched fit, the response to the independently usable spectral cue decreases strictly with contextual multiplicity, and removing the contextual path allows that response to reach m. This is a genuine comparative suppression statement. The suppression is not due to an initially zero spectral gradient: ȧ(0)=1/2, and every β̇_j(0)=1/2. No dead-ReLU argument is involved.

It would still be wrong to call a_M S a low-information noiseless representation: for every finite M, a_M>0 and sign(a_M S)=Y perfectly. Its margin under a unit readout shrinks, and its robustness to fixed observation noise shrinks. To make the latter precise, observe Z_S=a_MY+σξ with ξ∼N(0,1) independent and fixed σ>0. Its optimal balanced binary accuracy is Φ(a_M/σ)→1/2, and

$$I(Y;Z_S)\le\frac{a_M^2}{2\sigma^2}=O(M^{-2}). \tag{5.13}$$

For the information bound, compare each conditional Gaussian to Q=N(0,σ²): E_Y KL(P_{Z_S|Y}||Q)=a_M²/(2σ²)=I(Y;Z_S)+KL(P_{Z_S}||Q)≥I(Y;Z_S). This noise is a specified observation/test perturbation, not part of the training theorem. The full encoder retains the contextual signal and therefore still carries label information on the training distribution, consistently with (5.1).

**The parameterization control is essential.** Replace the downstream readout by

$$F=z_1+\frac1{\sqrt M}\sum_j\gamma_jz_2,$$

with γ_j(0)=0 and the same unit-rate flow. Define b=M^{-1/2}Σγ_j. Then ḃ=vr, not Mvr. Every reduced equation above becomes the M=1 equation, regardless of the number of γ parameters. More generally, multiplying each copy by α_M replaces the effective factor M by Mα_M². This is an exact counterfactual: **the effect is caused by the training metric induced by replication/readout scaling, not by a larger function class.** All M versions already represent the same functions of these two cues.

The initialization also matters: W(0) passes context and suppresses the designated spectral cue; the contextual readouts start at zero. This is neither isotropic Kaiming initialization of the entire pipeline nor a theorem about arbitrary encoders. If a useful spectral coefficient is already O(1) at initialization, small displacement need not remove it. These facts must appear in the theorem’s main-text description, not only in a limitations footnote.

**What to prove instead of forcing P1+P3 into a ViT claim.** I recommend two linked, scoped statements:

1. **Residual-subspace dominance implies selective movement and joint/frozen closeness** — Theorem 1.3 in the exactly affine setting, with a perturbation-error term when a linearization can actually be certified. The required condition is K_φ⪰κI on the active residual subspace, not λ_max(K_φ) large. This is the clean general principle.
2. **A serial cross-entropy instance verifies the complete mechanism** — Theorem 5.1 derives the rate, movement, functional comparison, and comparative feature suppression without invoking an unproved architecture-to-rate bridge. It shows the mechanism can occur, while its normalization control explains why it is parameterization sensitive.

The interior-layer result can accompany these as a separate geometry theorem. It does not supply the residual-subspace coercivity needed for the dynamics theorem, even when it proves a large λ_max perfectly. P2’s counterexamples make that separation mathematically unavoidable.

**Publication judgment.** These results improve correctness and give the theory a coherent constructive example. They do not by themselves establish high novelty. P4 is elementary but useful; the monotone comparison and fixed-kernel Duhamel argument are standard techniques; the interior CE argument is a scoped extension of the existing witness proof; the serial example is a parameterization-induced implicit-bias calculation. A reviewer can fairly ask whether the last theorem is simply duplicated features acting as a larger effective learning rate. Its answer is yes—and the theorem should explain that explicitly, then ask whether the same effective mechanism explains an independently verified empirical observation.

Before promoting Theorem 5.1 to the headline, position it against initialization-dependent linear-network implicit bias and finite-training-accuracy effects, not only multimodal competition. Relevant primary starting points are [Yun, Krishnan & Mobahi, “A unifying view on implicit bias in training linear neural networks,” ICLR 2021](https://research.google/pubs/a-unifying-view-on-implicit-bias-in-training-linear-neural-networks/), [Moroshko et al., “Implicit Bias in Deep Linear Classification: Initialization Scale vs Training Accuracy,” NeurIPS 2020](https://arxiv.org/abs/2007.06738), and [Berthier, “Incremental Learning in Diagonal Linear Networks,” JMLR 2023](https://jmlr.org/papers/v24/22-1395.html). These works establish substantial prior art on architecture/initialization-dependent linear-network dynamics; the exact finite-time serial CE formulas here still need a focused novelty comparison. I have not established that this particular proposition is new.

**Concrete requests for Claude’s mathematics reply.** No cluster-scale experiment is needed to check the new toy theorem.

- Implement the full (M+2)-parameter CE model independently and verify (5.2)–(5.12), with decreasing GD steps and an adaptive ODE solver as separate numerical checks. Compare β replication with the 1/√M readout control. Persist formula values, applied-step units, and relative discrepancies in one small CSV.
- Check the exact two-block GGN analytically and by your dense operator on the toy model. At initialization it must give λ_φ=M/4, λ_θ=1/4 under the specified loss normalization. Confirm normalized two-logit residual R=√2r, rather than fitting a new constant.
- On the interior Gaussian/ReLU instance, separately test the squared-loss expectation formula (3.6) and the CE floor. Include the constant-feature BN counterexample: the W block should vanish to numerical tolerance while the incoming Gram has an eigenvalue proportional to M.
- For the existing shallow verified instance, export the actual residual overlap with the fast spatial kernel subspace and the restricted spectral output singular directions. If the residual lies mostly outside that subspace, stop treating a large top eigenvalue as a rate certificate. An instantaneous measurement is a diagnostic, not proof of persistence.
- If testing production P3, compute the *full downstream directional derivative* in (3.3) through normalization and spatial mixing. Report the chosen witness, its incoming energy, post-BN energy, and final softmax-weighted energy. A positive result can motivate a hypothesis; it cannot by itself prove a width-uniform high-probability bound.
- Independently audit the proofs and constants in this note, especially the residual normalization in (5.12), the aggregate probability argument in Theorem 3.2, and the noncommuting-kernel comparison in Theorem 1.3. Return exact objections or corrected equations, not a general endorsement.

For the manuscript I would add P4’s lemmas to the appendix immediately after checking implementation conventions, retain the counterexamples as scope tests, and give the serial CE example a small self-contained section only if its novelty and empirical relevance survive that independent check. I would not spend the remaining days trying to extract a general ViT learning theorem from a top-curvature witness alone.

**Source inventory.** The targets and notation come from the private workspace brief `review_packet/astra/claude_math_brief_01.md` and the frozen `paper/sections/{03_setup,04_theorem1_hessian,05_theorem2_twoscale,supplement}.tex` previously reviewed. The claims and proofs above are derived in this note; literature is cited for positioning rather than as a substitute for their hypotheses. External primary sources consulted for that positioning are Karimi, Nutini & Schmidt (2016), Pezeshki et al. (2021), Yun, Krishnan & Mobahi (2021), Moroshko et al. (2020), and Berthier (2023), linked at their points of use. This is not an exhaustive novelty search.
