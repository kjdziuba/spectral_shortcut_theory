# Mathematics 03 — decisions, proofs, and integration notes

Astra → Claude and Krzysztof, 2026-09-10.

`reply_02.md` contains Q8 first, then Q6/Q7/Q9. `appendix_math.tex` supplies the full mathematics requested in task B. The result is a conditional mathematics package with explicit model and optimizer boundaries. It supports the revised contribution; it does not recover a practical freezing recommendation.

## Q9 — a phase theorem exists, with one additional scaling requirement

**Decision: include the proposition and its short scaling corollary in the appendix.** Refer to the phase result in one main-text sentence. It generalizes finite-horizon attribution, not joint/frozen trajectory stability.

Let `q_b(t)=r(t)^T K_b(t) r(t)`, `D=q_theta+q_phi`, and assume unit Euclidean gradient flow, absolute continuity, the a.e. chain rule, and positive total decrease `ΔL=∫D`. For a measurable phase E let `τ_E=|E|` and `w_E=∫_E D/ΔL`. Suppose on E the residual puts at least fraction α>0 of its squared norm into a measurable top-k projector of K_phi, its kth eigenvalue is at least ℓ>0, and `q_theta≤a||r||²`. Then

$$ S_E\le\frac{a}{a+\alpha\ell},\qquad
S_{[0,T]}\le w_E\frac{a}{a+\alpha\ell}+(1-w_E). $$

The first ratio requires positive phase decrease. With a verified complementary share cap b, the second term becomes `(1−w_E)b`. The proof is one spectral inequality, `q_phi≥αℓ||r||²`, followed by the exact flow energy identity. All hypotheses and the proof are in `prop:astra_phase`.

**The additional point matters for your claimed approximately 1/D cumulative scaling.** If the early cap is O(1/M), the late instantaneous encoder share is bounded between fixed positive constants `b_-` and `b_+`, and `S_M` denotes the total share, then

$$ b_-(1-w_E)\le S_M\le w_E\frac{a_*}{a_*+cM}+b_+(1-w_E). $$

Thus, under these hypotheses, `S_M=O(1/M)` **if and only if** the complementary phase carries `1−w_E=O(1/M)` of the relevant energy/decrease. “Most learning is early” is not quantitatively enough: a fixed 90% early fraction leaves a constant late contribution when the late share is about 0.4. A short early phase does not determine its energy weight. This is proved in `cor:astra_phase_width`. Neither this equivalence nor the present three-width table establishes an asymptotic law for the actual runs.

For a phase E defined by a threshold that can be crossed repeatedly, retain the set definition and report re-entry. If using an initial interval `[0,τ]`, say so explicitly; its flow loss-decrease fraction is `(L(0)−L(τ))/(L(0)−L(T))`. Do not silently replace an instantaneous threshold set by the first-crossing interval.

**What to export now, with no new large sweep:**

- On one consistently defined sequence, export the phase membership, threshold α and k, duration/number of included steps, `q_theta`, `q_phi`, and weights used in every sum; report `w_E`, the two phase shares, and the resulting bound separately for every width and seed.
- Compute `λ_k`, not only `λ_max`, if you want the top-k spectral certificate. Report the direct effective Rayleigh floor alongside `α λ_k`: the latter may be much weaker. A realized-path supremum/infimum is a retrospective certificate, not an independently predictive hypothesis.
- For the existing Adam runs call these **gradient-energy fractions**. The algebraic weighted-sum version holds for any optimizer on the same sequence. It is not a decomposition of actual Adam loss decrease. Your fixed-probe checkpoints and training-minibatch cumulative energies are presently different sequences; report probe results as probe results. Eleven checkpoints alone cannot certify an a.e. trajectory condition or all intervening training steps.
- If an actual flow loss-decrease illustration is wanted, use a small full-batch flow/controlled-GD instance and report the energy-identity error. This is optional for the paper: the existing run can support the diagnostic with correct terminology.

**Blocking corrections before using the residual plot:**

1. For CE, `r^T K_b r=||∇_b L||²` already holds exactly. Multiplying by the CE Hessian changes the diagnostic; it is not the correction to a squared-loss approximation. Squared loss is needed for the constant-kernel exponential residual formulas, not for the gradient-energy identity.
2. The appropriate isotropic reference is in the per-site class-contrast space. If `Π=I_N⊗(I_C−11^T/C)`, a random unit residual in this space has expected projected mass `tr(PΠ)/[N(C−1)]`, not generally `rank(P)/(NC)`. Recompute the reference for the projectors you actually used, or recompute top subspaces of the class-centered kernel and use their matching reference. For C=2 this is particularly consequential. The current “below isotropic” statement is unsupported until corrected; the measured projection masses themselves remain valid.
3. Replace “Counterexample 1.4 realised on our instance” by “the diagnostic limitation illustrated by Counterexample 1.4 is observed.” The experiment is nonlinear CE/Adam with an ill-conditioned kernel; the counterexample is an exactly singular affine squared-loss system.

## Q10 — number a conditional gain identity, reject the unrestricted witness conclusion

**Decision: a small appendix proposition is worthwhile for clarity, not as a novel normalization theorem.** This is `prop:astra_bn_gain`, followed by the diagnostic-model remark. It is not a replacement for the full downstream witness condition of Lemma 3.1.

Your proposed statement is false as written in two ways. Fixed total *uncentered* second moment need not imply a positive centered variance: a constant feature over the normalization group is a counterexample. And “any witness with width-independent pre-BN energy inherits that factor” is false: a perturbation parallel to the centered activation is annihilated by zero-stabilizer BN despite having positive energy. At positive stabilizer its radial gain is suppressed by an additional squared factor `(ε/(s²+ε))²`. A downstream projection, gate pattern, or loss mask can also kill a surviving witness.

Here is the valid chain, with d the actual fan-in (9(M+K) for a 3×3 convolution):

$$ s^2=w^TS_cw,\qquad
\mathbb E_w s^2=\sigma_w^2\operatorname{tr}(S_c)/d $$

for a centered covariance S_c and an independent isotropic row with covariance `σ_w²I/d`. To infer a realized gain law assume explicitly `c_-/d≤s²≤c_+/d`, with positive constants, and a bounded nonzero BN affine scale. Then

$$\frac{\gamma_-^2d}{c_++\varepsilon d}
\le\frac{\gamma^2}{s^2+\varepsilon}
\le\frac{\gamma_+^2d}{c_-+\varepsilon d}.$$

A unit parameter witness inherits a lower fraction of this gain only if a positive fraction of its pre-BN perturbation is centered and transverse to the centered activation. A final GGN lower bound additionally needs a positive gain of the actual downstream, softmax-weighted, masked Jacobian on that same propagated perturbation. All normalization-group sites must enter the BN calculation, even those excluded from the loss.

Expectation of variance cannot be inverted into concentration of gain. A single Gaussian row over a low-rank covariance need not concentrate as fan-in increases; fixing the trace does not fix its relative fluctuations. Also, fixed ε>0 gives a plateau asymptotically. Linear growth means the range `εd` small relative to the variance constants, not arbitrarily large width with ε held fixed.

**What the production data now support:** the measured median BN1 variances 0.151/0.064/0.035, their near-constant products with M+64, the flat incoming Gram, and the actual directional JVP stage energies together support the finite-range normalization-gain explanation. This is materially stronger than the earlier forward activation-energy ratio. It remains a three-width initialization observation; medians do not certify per-row uniform bounds, and changing BN statistics also changes downstream forward activations and gates. Either single BN intervention preserving a trend is evidence against a unique first-BN attribution. The fixed-statistics double intervention and the function-preserving co-scaled-ε control are the cleanest complementary diagnostics.

I agree with the revised §8.3's limited attribution and with declining another GPU-day intervention. Keep the full-group versus labeled-group discrepancy and the direct bias JVP in the appendix; they explain why a superficially plausible witness calculation can be wrong. Do not combine the raw full-group perturbation energy with a labeled-only covariance normalization as if they were the same quantity.

The novelty is the diagnosis of this particular model, together with a precise boundary for applying the geometric theorem. The gain identity and normalization sensitivity are established mathematics. Cite [van Laarhoven (2017)](https://arxiv.org/abs/1706.05350) for the parameterization/effective-rate issue; do not sell the algebra as a new BN mechanism.

## Q11 — yes, retain the factors in both recurrences

**Decision: give adjacent statements with a shared finite-weight bound.** The appendix separates the exact momentum identity from the velocity-clipping norm comparison so that readers cannot transfer the wrong vector identity. This is the same mathematical content as a single two-case lemma, with clearer hypotheses.

For clipping before accumulation,

$$v_{k+1}=\beta v_k+a_kg_k,$$

unrolling gives the exact identity and the usual weight `(1−β^{K−i})/(1−β)` for gradient i at constant step size. The actual global clipping coefficient belongs here, even if it depends on all parameter groups. This is your implemented ordering.

For ideal radial clipping of velocity, define the raw-gradient comparison coefficient `a_k=min(1,c_k/||g_k||)`, with its zero-gradient convention. Then

$$\|v_{k+1}\|\le\min(c_k,\beta\|v_k\|+\|g_k\|)
\le\beta\|v_k\|+a_k\|g_k\|.$$

Induction proves the same finite-weight **norm bound**, including nonzero initial velocity and variable steps. I accept the auditor's sharpening.

There is also a useful second version. Let `t_k` be the *actual* velocity-clipping multiplier. Since `0≤t_k≤1`,

$$\|v_{k+1}\|\le\beta\|v_k\|+t_k\|g_k\|.$$

So the bound remains valid with these actual multipliers too. Exact unrolling has coefficients `β^{k−i}∏_{j=i}^k t_j`; dropping subsequent factors gives the `t_i` bound. This argument works for radial stabilizer-based clipping as well. The raw-gradient comparison coefficient and the actual velocity multiplier need not be ordered because of cancellation. State the convention; neither algorithm has the other's unweighted vector identity.

The stronger bounds remain pathwise statements about each run's own gradients. They do not compare trajectories of separately evolved clipped and unclipped runs. The appendix retains `μ_s=ημ_f`, every per-step containment/Jacobian assumption, Adam's positive stabilizer and potentially vacuous `1/ζ`, and the encoder-decay term when present.

## Audit dispositions and two qualifications I do not accept verbatim

The audit is useful, but “no false statement found” applies to its checked note, not automatically to every new sentence suggested by the audit. I accept the mathematical corrections and provenance without treating the verdict count as a proof of the integrated appendix.

**Theorem N F7 — replace the proposed Clarke identification (important mathematical qualification).** Arbitrary zero-gate backpropagation selections need not be Clarke subgradients of the represented function. For example, the graph `ReLU(x)−ReLU(−x)` represents x; taking both zero gates to be zero gives backpropagation derivative zero at x=0, whereas its Clarke subdifferential is `{1}`. This graph example does not establish failure for every special architecture, but it invalidates the unrestricted justification. The appendix therefore uses its explicit consistent-gate flow convention and proves local existence by smooth approximation, uniform path compactness, and weak-* gate limits. It proves the chain rule directly from the fact that an absolutely continuous preactivation has derivative zero a.e. on its zero level set. It then applies the containment argument to **every** solution, without asserting uniqueness.

This qualification is consistent with the cited literature: [Bolte–Pauwels, Corollary 6 and Remark 13](https://arxiv.org/html/1909.10300) distinguish the conservative field generated by backpropagation from subdifferential identities; [Davis–Drusvyatskiy–Kakade–Lee, Theorem 5.8](https://sites.math.washington.edu/~ddrusv/stoc_subgrad_semi.pdf) supplies a relevant nonsmooth chain-rule framework. Neither citation replaces the local existence argument we need for this model.

**Theorem N F10 — retain the threshold chain, qualify the coarse asymptotic description.** A schematic term `N²/(ρ³κ_*²)` is a useful illustration of severity when the relevant input and boundary constants are nonzero and other scales are fixed. It is not a universal necessary width, and the displayed constants have additional nested dependence on ρ and κ_*. The appendix prints every constant and the complete maximum defining M_*. It labels this construction existence-only and potentially numerically vacuous. Degenerate inputs can remove some terms; almost duplicated encoded data can make the remaining threshold much worse.

The other Theorem N fixes are implemented: the exact three-term Jacobian decomposition uses current V in the W-change term; the Z inequality is proved; the tube radii are explicitly at most one; A is the encoder-Jacobian cap; output bias need not be centered; all solutions are covered; gate-change control and lazy/linearized dynamics are attributed; and the union bound lists exactly seven charged events. N3 is explicitly comparison of **both paths with one common linear reference**.

For the sections audit, A1 now states unit rates, a.e. realized-path Rayleigh inequalities and positive finite-horizon decrease, without the affine theorem's invariant-subspace requirement. A2 includes PSD transfer to the untruncated sum and the exact versus simplified Tropp constant. The isotropic statement uses the update fraction, includes the finite-width correct-classification result for `a_0>m/2` (also the equality case), and identifies the fourth-power small-`|v_0|` crossover. The `ρ^-4` expansion is pointwise unless its remainder is made uniform; it is not stated as a universal necessary width. The BN necessity counterexample is explicitly B3 at `c=M^-1/2`. B4 explicitly requires fixed positive total **centered** energy and realized variance control.

**The proposed `s²≥10³ε` “usable regime” is not a universal error guarantee.** The departure depends on c and downstream sensitivity as well as variance. Your own c=1/4 control has minimum ratio about 216, which does not literally satisfy a ≥1000 condition, although its measured error is small. Keep the actual departure and variance/stabilizer ratios; do not promote that heuristic cutoff into a theorem. The four-channel factor-235 example is likewise an example of scalar non-invariance, not a universal constant.

**Readout normalization needs its training-coordinate convention.** In Theorem N write `f=M^-1/2 U h+β`, initialize U with O(1) entry variance, and train U at unit rate. The initial function law is preserved and the head kernel loses its factor M; the encoder kernel stays O(1). This removes a forced order-M disparity, not all anisotropy. Merely multiplying the old variance-1/M readout by another `M^-1/2` changes the initial function law. The appendix says exactly which control is meant. It also cites the directly relevant standard-parameterization analysis of [Sohl-Dickstein et al. (2020)](https://arxiv.org/abs/2001.07301), in addition to the seven requested Theorem N references.

## Integration and validation

- Input `appendix_math.tex` after `\appendix`, using the existing environments and macros. The file uses `\thetap`, `\phip`, the GGN block notation, and existing main labels `thm:hessian`, `lem:phi_scaling`, `thm:twoscale`, `cor:attribution`. All new local labels have an `astra_` component to avoid collisions.
- Replace the main statement at `cor:attribution` by A1, referring to `cor:astra_attribution` for its appendix version/proof. Preserve the existing main labels; do not include duplicate old and new proofs as if they were different results. Review-note numbers are mnemonic titles, not hard-coded manuscript theorem counters.
- The file includes P4, all requested bridge/interior/excitation counterexamples and proofs, BN statements, the serial theorem and isotropic extension, the new phase result, and full Theorem N. The provenance paragraph replaces the novelty table. Every cited source has a BibTeX entry in the final comment block, with namespaced keys that Claude can deduplicate against `references.bib`.
- Validation: compiled in a standalone harness using the actual `paper/main.tex` preamble and placeholders for the four main-text references; ran BibTeX and repeated LaTeX to resolution. The final harness has no undefined citations/references, duplicate-destination warnings, or overfull boxes. This checks integration syntax in that preamble, not the final ICLR template's page layout, and is not an independent proof audit of this newly integrated file.
- `paper/main_iclr.pdf` was not present at the check during this response. The requested ICLR-style review remains the next task when that draft is supplied. No reviewer score is assigned to an unseen draft.

The most consequential next action is to make the main text's claim hierarchy match these hypotheses and to correct the residual figure's reference space before drawing a mechanistic conclusion from it. The new mathematics strengthens a scoped paper; the central review question remains what the combined mathematical and diagnostic distinction adds beyond known optimization-metric and lazy-training results.
