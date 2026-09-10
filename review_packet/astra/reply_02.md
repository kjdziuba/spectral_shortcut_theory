# Reply 02: nine-page plan and empirical interpretation

**Q8 — allocation for Claude's skeleton (ready to use).**

| Section | Pages, including its figures |
|---|---:|
| 1. Introduction and precise contribution | 0.80 |
| 2. Related work: optimization geometry, lazy training, feature competition | 0.60 |
| 3. Setup, normalization, and the distinction between curvature and gradient gain | 0.60 |
| 4. Conditional geometry: Theorem 1 and the hypothesis boundary | 1.10 |
| 5. Dynamics: Theorem 2, affine bridge, finite-horizon attribution, scoped nonlinear consequence | 1.30 |
| 6. Solvable serial CE instance, isotropic extension, normalization control | 1.10 |
| 7. Diagnostic limits and matched-protocol evidence | 3.00 |
| 8. Limitations and conclusion | 0.50 |
| **Total main text** | **9.00** |

Within §7, budget approximately 1.15 pages for production input/directional/BN diagnostics, 0.65 for the shallow residual-subspace experiment, and 1.20 for the matched comparison. Keep the shallow model and production model visibly distinct. These are writing budgets, not instructions to shrink fonts when the first draft overruns.

**Exactly three main-text figures:**

1. **Serial CE example:** encoder update against width with exact bound; joint/frozen margin gap; isotropic extension and normalized-readout control. A small architecture inset can replace a separate schematic. The shift result can be a compact annotation rather than another large panel.
2. **Diagnostic limits:** a compact composite showing production input/directional anisotropy, production BN intervention with a fixed incoming Gram, and shallow-model residual mass/effective Rayleigh quotient over training. Label the model and optimizer on each panel. Prefer measured directions and actual seed points to another decorative spectrum.
3. **Matched protocol:** paired arm differences with intervals, with the original versus matched fine-tune/frozen contrast identified as a *bundled protocol intervention*. Show the pre-specified final-five endpoint prominently; separate exploratory best-validation selection. No claim of equivalence from n=3.

The existing `fig8a_geometry.png` and `fig8c_matched.png` can supply panels, but should be rebuilt around this allocation; the historical width-only curve does not get its own figure. Full witness stages, all seeds, recalibration, and learning-rate arms belong in the appendix.

**Main-text statements:** Theorems 1 and 2 with their conditional hypotheses; the transparent affine squared-loss bridge (Theorem 1.3); the sharp finite-horizon share corollary (A1, using `cor:attribution`); one compact serial-CE theorem with the isotropic update conclusion immediately following. State Theorem N's nonlinear existence consequence in a short paragraph with a reference, not another full formal theorem block: fixed dataset, this parameterization, squared loss, sufficiently large width, encoder/head displacement split and share. Its full hypotheses and all constants are in the appendix.

**Appendix only:** every proof; P4; Theorem 1.1 and its PL counterexample; the detailed robustness and margin statements; Corollary 1.5 and concentration thresholds; all interior-layer and BN propositions; residual-excitation theorem and counterexamples; full isotropic theorem and constants; Theorem N; EGR; historical corrections and complete empirical tables. The new two-phase result can be one sentence in §7 with a reference to its appendix proposition. It concerns attribution, not a substitute trajectory-comparison theorem.



**Q6 — fair in its numerical reporting, but the causal conclusion still overreaches (blocking).**

“No resolvable freezing effect at this width and seed count” is acceptable if it names the two within-initialization comparisons, the endpoint, and the validation split. It is not evidence of equivalence. I reproduced the paired intervals from `results/e3c_analysis_runs.csv`: joint-linear minus frozen-random is +0.01057 [−0.14192,+0.16305] on final-five and +0.02321 [−0.06588,+0.11230] on best-validation; fine-tune minus frozen-pretrained is +0.02057 [−0.02181,+0.06295] and −0.00420 [−0.04782,+0.03942], respectively.

The current ending of §8.4 is too strong: “the measurable shortcut ... is a property of the training protocol, not of the architecture” and “sufficient to manufacture a 0.11 ... advantage” assert a causal identification that the comparison does not deliver. The original deficit was itself unresolved. Moreover, the paired change in the final-five joint/frozen gap between protocols is +0.12006 with a descriptive 90% paired-t interval [−0.23031,+0.47043], using the same three seed identifiers. The corresponding best-validation gap change is −0.00995 [−0.10503,+0.08513]. These intervals are conditional seed summaries, not a population or multi-factor causal analysis. Saving a checkpoint changes what is available to evaluate or deploy; it does not itself change the final-five training trajectory.

Use instead:

> “The original protocol confounded freezing with clipping scope and normalization-parameter treatment. Its nominal final-window freezing advantage was absent after these differences were removed, while checkpoint selection reversed the ordering within the original runs. At width 48 and three seeds, neither the within-protocol freezing contrasts nor the change in their gap establishes a population effect. These results demonstrate that the original contrast cannot identify an architectural shortcut and that checkpoint policy changes its descriptive conclusion.”

This supports the decision to drop a freezing prescription. It supports a *protocol-confounded apparent effect*, not a proof that architecture has no effect or that the named changes uniquely caused the old difference. Frame the contribution decision accordingly.

Additional corrections to §8.4:

- Say “the same clipping rule and scope,” not “clipping ... now identically in every arm.” The clipping coefficients and gradients differ by arm; that is an outcome of the matched rule.
- The matched **pairs** are the clean freezing comparisons. The four core arms do not all differ only in a single factor: `build_model` uses a linear reduction with BN for joint-linear/frozen-random and an MLP without reduction-stage BN for the pretrained pair. Comparing frozen-pretrained to frozen-random therefore changes architecture, initialization/training history, and validation selection, not just informativeness. Name this explicitly. Reduction BN affine parameters are trained wherever that BN exists; they do not exist in every arm.
- “Verified gradient flow” should read “verified nonzero applied encoder updates” for AdamW. “Spectral share of squared gradient flow” should read “share of summed squared gradient norms.” Neither is an AdamW loss-decrease attribution.
- The pretraining selection/weight-decay disclosures, augmentation disclosure, single validation fold, lack of test set, exploratory best-validation endpoint, and final-five endpoint are appropriate. Also disclose the pretraining crop/augmentation difference recorded by `resolve_pretrained`: pixel pretraining uses the unaugmented/center-crop protocol, while subsequent training uses random crop plus augmentation where applicable. This affects cross-pair pretraining comparisons, not the frozen/fine-tuned pair starting from the same checkpoint.
- One failed cosine-schedule intervention does not rule out an excessive late learning rate. Say that this particular schedule did not improve either mean endpoint. Similarly, three multiplier means do not establish a systematic causal relation between displacement and peak score; the intervals are wide.
- Train-only BN recalibration explains part of the observed endpoint variation; “explains the variance ... not the degradation itself” should be restricted to this intervention and these checkpoints, since it recovers a nonzero part of the degradation too.

I found these issues by checking the summary CSV and runner's model/group construction, not by reading run directories. The summary CSV does not expose all per-run configuration hashes or data-order pairing. Before declaring the pairs completely matched, include a compact configuration/provenance manifest in results: architecture, initial checkpoint hash, parameter-group membership, clipping/decay, BN mode and state, augmentation/data-order seed, scheduler, and selection rule. This is a documentation check, not a request for another training sweep. Same seed labels alone do not prove bit-identical minibatch order across different model constructions.

**Q7 — contribution paragraph and weakest link.**

> “We distinguish three ingredients of selective encoder adaptation in serial models: block curvature, residual excitation, and optimizer-dependent displacement. Conditional bounds identify when a fast downstream module limits encoder movement, with a transparent affine comparison and a scoped nonlinear ReLU existence result; an exactly solvable serial cross-entropy instance exposes the role of the induced training metric at a matched fitting threshold. Measurements on a production spectral-spatial model show why scalar curvature can be misleading: input anisotropy and normalization alter its interpretation, while a separate shallow-model trajectory experiment reveals loss of residual overlap with fast directions. Matched retraining comparisons and checkpoint-policy analysis show why the original apparent freezing advantage cannot be attributed to architecture from that protocol.”

**Weakest link:** contribution beyond established lazy-training, optimization-metric and feature-competition results, given that the strongest constructive guarantees apply to deliberately scoped models and the production study does not establish their mechanism. Lead with the precise mathematical/diagnostic distinction the paper adds, not the volume of audits or a stronger freezing claim. The nonlinear existence theorem strengthens the mathematical package; it does not by itself close the theory-to-production gap.

**Q9 — yes for phase-restricted attribution; no automatic replacement of the affine bridge.**

Let `r=∇_z L`, `K_theta=J_theta J_theta^T`, `K_phi=J_phi J_phi^T`, and let both blocks follow unit Euclidean gradient flow with the a.e. chain rule. On `[0,T]` set `D(t)=||∇theta L||²+||∇phi L||²`, and assume `ΔL=∫D=L(0)−L(T)>0`. For a measurable phase E define its duration `τ_E=|E|` and its loss-decrease fraction `w_E=∫_E D/ΔL`.

Let P_k(t) be a measurable top-k spectral projector of K_phi(t), with a tie convention, and suppose on E that `||P_k r||² >= alpha ||r||²`, the kth eigenvalue is at least `ell>0`, and `r^T K_theta r <= a ||r||²`. Then `r^T K_phi r >= alpha ell ||r||²`, so

$$\frac{\int_E\|\nabla_\theta L\|^2dt}{\int_E D(t)dt}\le\frac{a}{a+\alpha\ell},\qquad
\frac{\int_0^T\|\nabla_\theta L\|^2dt}{\Delta L}
\le w_E\frac{a}{a+\alpha\ell}+(1-w_E).$$

The first ratio requires positive phase decrease. If the complement has a separately verified share cap b, replace the final `(1-w_E)` by `(1-w_E)b`. Time-varying pointwise bounds give the sharper loss-decrease-weighted average of `a(t)/(a(t)+alpha(t)ell(t))`. Duration is explicit but does not determine w_E. If E is the initial interval `[0,τ]`, then its decrease is `L(0)−L(τ)`; under squared loss its residual contracts during this phase at rate at least `alpha ell`. For a threshold-defined set with re-entry, use the integral definition rather than pretending there is one boundary.

This is useful: an early fast phase can account for most decrease even if it is brief. It does not compare a joint trajectory with a separately trained frozen trajectory after that phase. A small share or small movement alone supplies no missing stability hypothesis.

**Critical export distinction:** the supplied runs use Adam and the kernel diagnostics use a fixed 256-site probe, while cumulative gradient energy uses training minibatches. Their raw energy ratio is not the share of actual Adam loss decrease, and a probe phase cannot simply be spliced into a minibatch numerator. Export a same-sequence discrete *gradient-energy* version with E, phase duration, phase energy fraction and numerator evaluated on that same sequence; its bound is algebraic and valid for any optimizer. Call it gradient energy. A flow loss-decrease verification needs an appropriate full-batch gradient-flow or controlled GD experiment, or a separately justified optimizer-specific energy identity.

Two corrections to the residual report before plotting it:

1. `r^T K_b r=||∇_b L||²` is exact for CE, not a squared-loss approximation. H-weighting produces a different curvature quantity; it is not the correction needed to make the share valid under CE. Only the exponential *residual dynamics* require squared loss in that argument.
2. CE residuals lie in the per-site class-contrast subspace of dimension `N(C−1)`. For an isotropic reference in that admissible subspace, the expected mass in a projector P is `tr(P P_contrast)/(N(C−1))`, not generally `rank(P)/(NC)`. Thus the “below isotropic” interpretation needs that reference recomputed, preferably using class-centered output kernels. The measured overlaps themselves remain valid. Also replace “Counterexample 1.4 realised” with “the diagnostic limitation illustrated by Counterexample 1.4 is observed”: the actual run is nonlinear CE/Adam and its kernel is ill-conditioned, not the exact singular affine squared-loss construction.

The full proposition and proof will be included in the appendix and discussed in `math_03.md`. A final-draft reviewer report is pending the nine-page PDF; the allocation above is not a review of a PDF that has not yet been supplied.
