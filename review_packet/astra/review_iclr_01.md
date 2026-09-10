# ICLR reviewer pass 01

Astra → Krzysztof and Claude, 2026-09-10.

Reviewed `paper/main_iclr.pdf`, 56 pages, PDF SHA-256 `d8560123c5073cb1ff89c4736647eb2de77308063ae40bedf6168ccefa022b05`. Page and line references below refer to that PDF's printed numbering. I read the nine main pages first, then the appendices and supporting sources/results. Appendix B differs from the supplied mathematics note only by citation-key substitutions. I checked selected numerical claims directly against the summary CSVs; I did not rerun training, inspect E3c run directories, or read `steps.csv`. No paper files were edited.

## I. Reviewer report

### Summary

The paper studies a small spectral encoder followed by a larger spatial model. It separates parameter-block curvature, gradient gain along the residual, and optimizer-dependent encoder displacement. Its mathematical package includes a conditional width-dependent curvature ratio, a displacement bound obtained from a residual envelope, a sharp gradient-flow attribution inequality, affine and scoped nonlinear joint/frozen comparisons, and an exactly solvable serial logistic model. Experiments show several ways scalar curvature can mislead: directional concentration on infrared inputs, BatchNorm-dependent curvature scaling, residual migration away from fast output directions, and protocol-confounded freezing comparisons. The paper proposes no validated mitigation.

My recommendation on this version is **lean reject**. There is useful mathematics and unusually informative diagnostic work, but the main text contains repairable statements stronger than its appendix, and the contribution beyond established optimization-metric/lazy-training analysis is not yet demonstrated sharply enough.

### Strengths

1. **The separation of curvature, residual gain, and actual optimizer movement is useful and mostly well maintained.** Equations (1)–(2), Corollary 1, and the Adam qualification in §7.2 distinguish quantities frequently conflated in informal explanations (pp. 3, 5, 7). The affine counterexample and phase bound make the limitations concrete.
2. **There are constructive results, not only impossibility warnings.** The finite-horizon share constant is sharp. The serial CE example gives exact update, fitting-time, reversal, and frozen-comparator calculations, with a normalization control that exposes the effective learning rate (pp. 5–6; Appendix B.6). The nonlinear theorem supplies a properly scoped existence result with the joint/frozen reference made explicit (Appendix B.7).
3. **The production BN measurements are the strongest empirical part.** Actual directional JVPs, the normalization-group correction, the constant-bias null direction, both single-BN interventions, and the co-scaled-stabilizer control support a precise diagnostic account (p. 7, lines 327–345; Appendix E.3). These are more informative than another width-versus-accuracy sweep.
4. **The matched study is responsibly interpreted in its final paragraph.** It reports paired differences, broad intervals, checkpoint selection, architecture differences across pairs, and lack of test evaluation (pp. 8–9). I independently recomputed the CNN/ViT paired endpoint and peak intervals from `results/exp1_2v4_summary.csv`; every interval includes zero at both 90% and 95%. The “unresolved” conclusion for those comparisons is supported.

### Weaknesses, ranked

**W1 — Major, submission-blocking until corrected: main statements lose hypotheses that the appendix explicitly needs.** Locations: p. 5, lines 216–269; p. 6, Figure 1 and lines 284–294.

Theorem 4 states `D_curv(0)=M` before moving to arbitrary `a_0<m, v_0≠0`; in that general setting the exact ratio is `Mv_0²`. Its special residual-envelope constants and the isotropic conclusion need separate initialization/stopping conventions. Figure 1 writes a supremum over all time although the cited gap guarantee is only through `T_m`. The phase paragraph asserts a necessary O(1/M) late-phase energy fraction without its positive late-phase share floor. The nonlinear summary omits the distinct encoded-input requirement needed even to fit the frozen system. The detailed list below gives corrections and counterexamples. These are principally errors of compression; they do not invalidate the correctly scoped Appendix B results. Nevertheless a reader must be able to rely on the main theorem statements.

**W2 — Major, and the main reason for my score: the central scientific payoff remains incomplete.** Locations: p. 2, lines 85–104; p. 4, lines 165–174 and 202–215; p. 5, lines 236–252; p. 9, lines 482–485.

Theorem 1 combines an assumed width-linear feature-Gram floor with a Jacobian cap. Theorem 2 integrates an assumed residual envelope. Corollary 1 is a useful sharp inequality, but follows directly from its two Rayleigh inequalities. The nonlinear result instantiates known lazy-training techniques, and the serial instance's width effect is exactly a replicated-coordinate learning rate. The paper acknowledges these precedents appropriately. What is still missing is a sufficiently substantial consequence of combining the distinctions: a quantitatively useful certificate, a prediction that a simpler diagnostic gets wrong, or a controlled phenomenon that this analysis uniquely resolves. The last sentence identifies a same-sequence phase certificate as an *open problem*. It reads like the natural central experiment that has not yet been done.

A synthesis/diagnostic paper can merit acceptance without a new optimizer or a universal theorem. My concern is the demonstrated value of this particular synthesis, not the absence of a method or of state-of-the-art accuracy. More theorem labels or additional numerical proof audits would not resolve it.

**W3 — Major empirical limitation: the pieces do not yet form one tested explanation.** Locations: pp. 6–7, lines 306–369; p. 9, lines 470–478; Appendix E.7, p. 55.

Production observations concern initialization and a model outside the head hypothesis. The residual experiment uses a different shallow model, Adam, a fixed probe for spectra, and minibatches for cumulative energy. The production movement certificate is explicitly withdrawn in the appendix. These scopes are now disclosed, which is a strength, but disclosure does not itself demonstrate that the refined diagnostic predicts encoder adaptation or failure. In particular, lower curvature along a class-mean contrast is not a measurement of impaired learning of that contrast. I would retain this as motivation and refrain from calling directional curvature alone the informative learning quantity.

**W4 — Moderate to major: the protocol study supports a narrow internal validity lesson and occupies too much of the argument.** Locations: p. 2, lines 62–64; p. 3, lines 108–109; pp. 8–9, Table 1, Figure 3, lines 424–462.

The cleanest supported conclusion is that the original comparison could not identify an architectural effect. Neither the original freezing gap nor its change across protocols is resolved. The final paragraph says this correctly, while the related-work sentence still attributes the apparent benefit to the protocol as a fact. Moreover the appendix itself leaves the per-run matching manifest to be supplied. A study correcting its own confounded comparison is valuable evidence of methodological care, but it is a limited standalone research contribution without a general or independently replicated diagnostic lesson. I would shorten this material enough to make room for W2's experiment, rather than expand the number of poorly resolved arms.

**W5 — Moderate: the initialization interpretation is stronger than the measurements and the implementation justify.** Location: p. 4, lines 178–188; Appendix D.1–D.2, pp. 40–41; `code/synthetic/models.py`.

The finite-range slopes are reported honestly, but the statement that default fan-in initialization makes both relevant variance terms vanish with width is not justified. The actual first layer has fixed fan-in K; at fixed depth, default initialization attenuates signal by a constant per layer, which is different from forcing its per-channel second moment to vanish with M. This matters to the interpretation of the almost-flat finite-range deep-head curve. A positive output-bias curvature floor can also dominate a small width-dependent feature contribution over the reported range. Report the measured feature moments/head witness or describe the slopes without the vanishing-mechanism assertion.

**W6 — Moderate presentation/reliability: several factual and numerical statements remain inconsistent, and the appendix contains unreconciled old claims.** Locations: p. 3, lines 121–143; p. 4, lines 174–176; p. 6, lines 292–319; p. 9, lines 480–481; Appendix A.5, pp. 19–20, and Appendix C, pp. 39–40.

Examples include breast FTIR versus the actual QCL measurement, “within six” versus 6.258 at M=16 and 9.897 at M=1, a universal `10^-11` verification claim that does not match the recorded relative-error summaries, and “mitigations ... are not evaluated” despite the freezing/LR arms. Appendix A's claim that a logit penalty shifts parameter curvature by a multiple of the identity is false: even for an affine logit map its added Hessian is `λJᵀJ`, not `λI`. Appendix A also says no input transformation can affect spatial width scaling, although that conclusion requires preserving its feature-Gram hypothesis. Appendix C calls a run-level partial correlation an inferential statement after correctly noting shared seeds across cells; pooling the runs does not remove that dependence. These do not all drive the score separately, but together undermine the otherwise careful claim hierarchy.

The main figures are readable. Figure 1 would be more useful with the margin m stated, the time horizon in the gap label, and a panel showing the advertised displacement certificate rather than only the scalar update bound. Figure 2 should define its directional curvature maximization and make seed versus batch replication explicit. The appendix should be a curated supplement, not a record of every earlier framing and audit outcome.

### Questions for the authors

1. Can you split Theorem 4 into its special initialization and general/isotropic cases, with the immediate-stop convention, the expectation over initialization in the reversal limit, and the finite gap horizon stated in the main text?
2. Can you produce one same-sequence phase certificate, including phase membership, phase energy/decrease weight, and bound slack, and demonstrate a useful prediction beyond recomputing `q_theta/(q_theta+q_phi)`? Which constants and thresholds are fixed before inspecting the evaluation trajectories?
3. For the initialization isolation, what are the width curves of the actual penultimate per-channel second moment, common-feature energy, and `λ_max(S_h^p)`? What observation distinguishes a vanishing signal term from a small nonzero term masked by width-independent curvature?
4. Which diagnostic outcome is expected to generalize beyond the production architecture and the shallow probe? Is a second model/normalization family needed for that claim, or will the contribution explicitly remain a case study plus sufficient conditions?
5. Can you provide the compact matching manifest and a timestamped artifact for the predictions called pre-registered? The scientific issue is what was fixed before observing these runs, not use of a particular registration service. This is a request for provenance, not an allegation that registration or matching failed.
6. Will you reconcile the stale Appendix A/C interpretations, verify the Papyan citation's exact claim, and make the actual measured breast modality consistent throughout?

### Score and confidence

**4/10 — lean reject on a conventional ten-point recommendation scale. Confidence: 4/5 on the technical/empirical scope assessment, lower on predicting other reviewers' novelty judgments.** This is an internal reviewer-style score, not a claim about the final 2027 review form.

The positive case is a careful conditional framework, a transparent exact example, and strong production diagnostics. The negative case is that several main statements currently overreach, while the central new payoff is mostly an articulated distinction whose natural operational demonstration remains future work. Correcting the statements is necessary, and would remove a substantial reliability objection, but would not by itself make me an accept. I could move toward the acceptance boundary if the paper demonstrates a nonvacuous consequence of the combined framework on one controlled experiment. I would not demand production-wide validity of the shallow theorem or penalize the paper simply for reporting negative results.

### The single change most likely to move the score

**Replace the open-problem ending with one controlled, same-sequence residual-excitation experiment that demonstrates a useful phase certificate.** Make this the empirical center of the paper, using space recovered from the protocol discussion.

A suitable small experiment would use a tractable full-batch model and unit gradient flow or carefully controlled GD, keeping the data, initialization and optimization metric specified. Vary residual excitation while keeping the initial scalar curvature information comparable; square-loss targets allow an especially clean fixed-initialization residual-direction control. Fix k, the phase rule, and the proposed bounds before examining the evaluation runs. On every evaluated step compute the same residual, kernels, energy terms, and loss; show the energy-identity/discretization error, phase duration, `w_E`, predicted share cap, actual share, and slack. Include both a regime where the certificate is informative and one where it becomes uninformative. If using squared loss, label that scope rather than presenting it as production CE validation.

The success criterion is not that an inequality evaluated with its own exact numerator and denominator holds. It is that the **coarser spectral/excitation certificate gives a useful quantitative distinction that the scalar curvature ratio misses**, with thresholds/constants not retrospectively chosen to force that distinction. This is one bounded experiment, not a new architecture sweep. A positive result would change what the paper delivers to the reader; a vacuous result would tell us that the diagnostic contribution still needs a different centerpiece.

## II. Separate sentence-level overclaim and contradiction list

The quotations below retain the wording of the main text, with line-break hyphenation removed and mathematical notation transcribed. “Unsupported” means the named proof/data do not establish the sentence; it does not automatically mean its opposite is true. Severity: **blocking** = formal result or central interpretation must change before submission; **important** = materially misleading scope/evidence; **minor** = bounded wording, numerical, or citation repair.

### 1. General initialization inherits a special exact disparity — blocking

**p. 5, lines 261–262:**

> “For this model, D_curv(0)=M exactly and the residual envelope of Assumption 5 holds with μ=(M+1)/4.”

**Contradicting/restricting file:** `paper/sections_iclr/appendix_math.tex`, Theorem B.5 and Corollary B.6 (`thm:astra_serial`, `cor:astra_isotropic`). The first has `(a_0,v_0)=(0,1)`; the second allows general initialization. With `b_0=0`, the general encoder and head GGN eigenvalues are `ρ_0(1−ρ_0)` and `M v_0²ρ_0(1−ρ_0)`, hence the disparity is `Mv_0²`. For instance `v_0=0.2` gives `0.04M`.

**Repair:** separate the two cases explicitly, including in Figure 1's annotation. Do not call the special envelope constant an initialization-independent fitting rate. For general initialization a directly proved scalar-residual envelope is

$$\varrho(t)\le\frac{\varrho_0}{1+(1+Mv_0^2)p_0\varrho_0t},\qquad
\varrho_0=(1+e^{a_0})^{-1},\quad p_0=1-\varrho_0.$$

This follows from `−ρ̇≥(1+Mv_0²)p_0ρ²`. Translate to the normalized two-logit residual if used. Specifying μ alone with an unspecified adjustable C is insufficient to communicate the claimed rate; I am not asserting that no larger-C envelope could use that μ.

### 2. A finite-width reversal probability replaces a limit — important

**pp. 1–2, lines 53–55:**

> “A serial cross-entropy model with replicated readouts has disparity exactly M, fitting rate (M+1)/4, a spectral update suppressed as 1/(1+Mv_0²), and a failure under contextual reversal with probability Φ(m/2σ) (Theorem 4, Figure 1); the displacement bound is within a factor of six of the truth.”

**Restricting files:** `paper/sections_iclr/appendix_math.tex`, Corollary B.6; `results/toy_serial_ce_isotropic_gaussian.csv`; `results/toy_serial_ce_isotropic_REPORT.md`, §1 and reversal table. The CDF is the **limit of expected error over initialization** as M grows, not the finite-width probability. Appendix D reports 0.8207/0.8380/0.8397 at M=64/1024/16384 versus the limit 0.8413 for m=2, σ=1.

**Repair:** say “expected reversal error tends to ... as M→∞,” and identify which initialization supports each preceding constant. In Theorem 4 itself, write `E_init Err→Φ(...)`; include `m>0`, `σ>0`, and immediate stopping when `a_0≥m`, since a full Gaussian initialization is not supported on `a_0<m`.

### 3. The margin-gap figure loses its time horizon — blocking

**p. 6, lines 286–287:**

> “(b) Joint-minus-frozen margin gap sup_t(q_J−q_F) with its bound C_0/(p_0 Mv_0²).”

**Contradicting file:** `paper/sections_iclr/appendix_math.tex`, equation `eq:astra_iso_gap`, explicitly restricted to `0≤t≤T_m`; the same restriction is in the main theorem. `results/toy_serial_ce_isotropic.csv` records a gap on the fitting interval.

**Repair:** write `sup_{0≤t≤T_m}` in the caption, axis and appendix tables. This is substantive, not just notation: in the special serial model continued indefinitely, `q̇_J~2√M q_J e^{-q_J}` while `q̇_F~M e^{-q_F}`. Consequently `q_J−q_F~log q_J+log(2/√M)→∞` at fixed M. There is no finite all-time uniform gap bound of the displayed kind.

### 4. The phase implication drops the late-phase share floor — blocking

**p. 5, lines 249–252:**

> “If on a phase E the residual keeps a fraction α of its mass in the top-k eigenspace of K_ϕ whose kth eigenvalue is at least ℓ, the encoder’s share of the decrease accrued on E is at most a/(a+αℓ), and the total share is at most w_E a/(a+αℓ)+(1−w_E) with w_E the phase’s fraction of the decrease; a cumulative share of order 1/M then requires the complementary phase to carry order 1/M of the energy (Proposition B.12, Corollary B.4).”

**Restricting file:** `paper/sections_iclr/appendix_math.tex`, Proposition B.12 and Corollary B.4, especially pp. 31, lines 1647–1660. The phase cap also needs `q_theta≤a||r||²`, positive phase decrease, and the unit-flow energy identity. The O(1/M) **necessity** needs a width-uniform positive late-phase share floor `b_->0`; the sufficient direction also needs `a≤a_*` and `αℓ≥cM`.

**Counterexample to the unqualified necessity:** let the encoder Jacobian be zero, let the decoder be `diag(M,1)`, and choose the residual in its second direction. The phase defined by top-direction overlap can be empty, all decrease is complementary, and encoder share is zero for every M. A variant with encoder kernel `diag(0,1/M)` gives share `1/(M+1)` and the same completely complementary decrease.

**Repair:** restore the hypotheses and say “squared residual norm.” The parallel sentence on p. 7, lines 365–367, and Appendix D's reading require the same qualifier.

### 5. Gaussian nonlinear fitting omits distinct encoded inputs — important

**p. 5, lines 236–240:**

> “For the fully trained shallow biased-ReLU model under squared loss in the standard parameterization, for M≥M_* and with probability at least 1−ρ over the initialization, every joint and every frozen solution fits at rate Ω(M), the encoder moves O(1/M) while the head moves O(M^{-1/2}), both paths track one common linear reference to O(1/M), and the encoder’s share of the loss decrease is at most A²/(A²+κM/2) (Appendix B, Theorem B.6).”

**Restricting file:** `paper/sections_iclr/appendix_math.tex`, Theorem B.6. Fixed finite data, a fixed-dimensional linear encoder with **pairwise distinct encoded inputs**, positive initialization scales and the specified unit-rate dynamics are essential. Equal encoded inputs with unequal targets make exact fitting by the frozen decoder impossible at every M.

**Repair:** add the fixed-data/distinct-encoded-input qualifier and refer to the theorem for the remaining initialization details. In the preceding Corollary B.3 summary (lines 229–231), also state “with probability at least 1−ρ” and “at initialization/in the frozen linearization.”

### 6. The serial model supposedly checks all earlier hypotheses — important

**p. 5, line 257:**

> “The hypotheses above can all be checked in one model.”

**Contradicting file:** `paper/sections_iclr/appendix_math.tex`: the affine bridge assumes affine outputs and squared loss; Theorem N assumes its Gaussian biased-ReLU architecture and squared loss. The serial example has a bilinear margin and logistic CE.

**Repair:** “The displacement and initialization-curvature statements can be instantiated analytically in the following serial CE model; it also admits a direct frozen-comparator calculation.” Do not imply that the example simultaneously realizes all preceding theorems.

### 7. Normalization supposedly removes all dataset-size dependence — important

**p. 3, line 121:**

> “We stack with N^{-1/2} so that no constant depends on the dataset size.”

**Contradicting file:** `paper/sections_iclr/appendix_math.tex`, B.1, explicitly says normalization does not remove data dependence of minimum kernel eigenvalues; Corollary B.3 has `M_0≥72N²/ρ`, and Theorem B.6's constants depend on the fixed dataset.

**Repair:** “We use N^{-1/2} normalization to remove extraneous factors from averaging; data-dependent constants can still depend on N.” Likewise qualify the N-independent cap in Lemma 1 by a genuinely uniform input/Jacobian assumption across the datasets under consideration, rather than deriving uniformity from notation.

### 8. An expectation cap is presented as the uniform hypothesis, with the wrong norm power — important/minor

**p. 3, lines 139–143:**

> “This is a modelling assumption: it is proved in expectation for a two-layer ReLU head at initialization (Appendix A), where the naive product of layer norms would give Θ(M/K) instead.”

**Restricting/contradicting file:** `paper/sections_iclr/appendix.tex`, Proposition A.1, Step D. It proves a pointwise expected squared Frobenius-norm cap and a trace-form encoder bound. It explicitly leaves the uniform-over-inputs/during-training assumption unproved. Its naive product of **norms** is `Θ(√(M/K))`; `Θ(M/K)` is the squared product.

**Repair:** call it “an initialization expectation analogue” rather than “it” being proved. Specify the squared norm if retaining M/K. The instance-level disparity proof remains valid through its separate probabilistic trace-cap argument; that argument is not a proof of Assumption 3 as written.

### 9. CNN/ViT applicability is asserted as a verified hypothesis — important

**p. 4, lines 174–176:**

> “The head hypothesis is a cross-input correlation condition on Θ(M) penultimate features: met by dense per-pixel heads on CNN and ViT backbones, and structurally unmet by the production model of Section 7, whose classifier input has a fixed dimension at every width, so that the theorem has no premise there rather than a counterexample.”

**Contradicting file:** `paper/sections_iclr/appendix.tex`, Remark A.1 (`rem:meanfield_scope`), says for CNN and ViT heads it is an explicit assumption to be checked. A dense head of the right dimension does not itself guarantee a width-linear weighted feature Gram.

**Repair:** “potentially applicable to dense per-pixel CNN/ViT heads, provided their weighted feature Gram satisfies the stated hypothesis.” Retain the production structural failure.

### 10. Default initialization is claimed to force signal variance to vanish — important

**p. 4, lines 184–188:**

> “A deeper head separates the two regimes sharply: over M=16→1024 its spatial slope is 0.105 under default initialization and 0.989 under the theorem’s, and isolation arms (fixed bias variance alone 0.924; He weights with scaled or zero bias 0.696, 0.606) show that the floor survives when either variance term of the head hypothesis is width-independent, while default fan-in initialization makes both vanish with width.”

**Restricting files:** `results/exp1_1_v3_ggn_aggregated.csv`, `results/exp1_1_v3_ggn_theoreminit_aggregated.csv`, `results/exp1_1_v3_ggn_sigmab_isolation_aggregated.csv` record finite-range GGN slopes, not vanishing feature moments. **Contradicting implementation:** `code/synthetic/models.py`, `SpatialDeepMLP`, is fixed depth `K→M→M→M→C`, with fixed K and ordinary `nn.Linear` defaults.

For a fixed input z, those symmetric defaults give exactly, in expectation,

$$q_1=\frac{||z||^2+1}{6K},\qquad
q_2=\frac{q_1}{6}+\frac1{6M},\qquad
q_3=\frac{q_2}{6}+\frac1{6M},$$

where q_l is one hidden coordinate's post-ReLU second moment. Therefore `q_3→q_1/36>0`, not zero. This uses the weight/bias variance `1/(3 fan_in)` in the [PyTorch 2.9 Linear specification](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Linear.html), plus symmetry of each preactivation. It does **not** by itself prove the weighted head-Gram condition, which is exactly why that quantity should be measured.

**Repair:** retain the slopes, remove the asymptotic/variance explanation unless directly established, and export penultimate moments and the actual weighted witness. “A small signal contribution is hidden by width-independent terms” is an alternative to investigate, not a result to assert now.

### 11. “Within six” and the later tightness statement omit the tested regime — minor but direct

**p. 1, lines 21–23:**

> “An exactly solvable serial cross-entropy instance exposes the role of the induced training metric at a matched fitting threshold; there the displacement bound is within a factor of six of the truth.”

The same phrase ends the contribution sentence quoted in item 2. **p. 6, lines 292–294**, then says the bound is within 4.5–6.3 “here,” adjacent to the 579 general-initialization cases.

**Contradicting/restricting file:** `results/toy_serial_ce.csv`, quantity `(5.12) bound / displacement`: M=1 gives **9.897155**, M=16 **6.257862**, M=64 **5.045234**, M=256 **4.612046**, M=1024 **4.489008**. The 4.5–6.3 range describes four selected widths at `(a_0,v_0)=(0,1)`, m=log 19; it is not a uniform theorem or a conclusion over the isotropic grid.

**Repair:** state that exact regime, or say “within a small constant factor in the reported special-initialization sweep,” with the full range attached. In the abstract, omit the tightness number unless its qualification fits.

### 12. A universal numerical discrepancy claim exceeds the recorded precision — minor

**p. 6, lines 292–294:**

> “Theorem 2’s displacement bound is within a factor 4.5–6.3 of the realised displacement here, and every displayed quantity agrees with the closed form to 10^{-11} against a full (2+M)-parameter integration over 579 cases (Appendix D).”

**Restricting evidence:** `results/toy_serial_ce_isotropic.csv` and `_REPORT.md`. The sentence does not specify absolute versus relative error; the report uses relative discrepancies. In that convention, the maximum endpoint discrepancy over the four specified endpoint quantities is `1.2602e−11` over **525 nontrivial solves**, plus 54 immediate-stop cases. The maximum trajectory identity discrepancy is `1.5180e−10`; `rel_gap_sup_cf_vs_ode` reaches **7.6147e−9** at M=4096, a_0=0.4, v_0=−1.3, m=4.

**Repair:** name the tested quantities, error convention and corresponding maximum, not “every displayed quantity.” If an absolute-error claim is intended, compute and report that separately. Figure 1's last sentence needs the same check. These errors are tiny and do not undermine the inequalities; the issue is accurate scope of the verification claim.

### 13. The primary measured modality is misidentified — important factual correction

**p. 6, lines 309–311:**

> “Data are breast-tissue FTIR micro-array cores (fold 0 of a five-fold patient-level split) with a prostate QCL replication (Appendix E).”

**Contradicting files:** `review_packet/ASTRA_BRIEF_2026-09-09.md`, §9.1 and dataset definitions, explicitly identifies the breast and prostate geometry data as **QCL**, both S=942=3×314 wavenumbers, and states that there is no FTIR geometry run. `paper/sections_iclr/appendix_experiments.tex`, E.1, also calls the acquisition QCL.

**Repair:** “breast-tissue QCL ... with prostate QCL replication,” after confirming the manifest. This changes the reader's understanding of what was replicated.

### 14. Four initialization slopes are called verification of the cap — important

**p. 6, lines 315–316:**

> “The cap holds: the log–log slope of λ_max(G_θθ) against M∈{48,96,192,384} is −0.003 (breast) and −0.012 (prostate).”

**Restricting files:** `results/exp1_8_real_dcurv.csv`, `results/exp1_8_real_dcurv_prostate_qcl.csv` measure finite-width initialization curvature; `paper/sections_iclr/appendix_experiments.tex`, E.7, explicitly records unverified containment/Jacobian hypotheses and large training-time gain growth. Flat measured GGN eigenvalues neither certify a Jacobian supremum nor an asymptotic width-uniform bound.

**Repair:** “The measured spectral-block curvature is approximately flat over the initialization width sweep.”

### 15. Low effective rank is made a sufficient/causal explanation — important

**p. 6, lines 317–319:**

> “The reason is the input: Σ_X has effective rank 1.07 (breast) and 1.02 (prostate) out of 942, and 1.21 after the module’s input normalization, so the cap C_θ∝λ_max(Σ_X) is enormous and the top-ratio is uninformative.”

Related stronger summaries are **p. 1, lines 24–26**, “rank-one inputs invert it,” and **p. 2, lines 68–70**, “its inputs are effectively rank one, so the top-eigenvalue ratio is uninformative and a directional picture must replace it.”

**Restricting files:** `results/exp1_8b_directions.csv`, `results/exp1_8b_summary.csv` support strong anisotropy and alignment; `paper/sections_iclr/appendix.tex`, the cap proof, gives an **upper bound**, not an equality. Effective rank alone does not determine λ_max: rescaling a rank-one input changes its magnitude without changing its effective rank. A large cap does not force large realized encoder curvature or inversion of the block ratio. The normalization controls themselves change the ratio.

**Repair:** report that the inverted ratio **coexists with** strong input/spectral-block concentration, then state what the directional measurements add. The measured alignment is evidence for an interpretation, not a general implication from rank or proof of its unique cause. No extra whitening experiment is required if the causal wording is removed.

### 16. A protocol-confounded contrast becomes a demonstrated protocol effect — important

**p. 3, lines 108–109:**

> “Our matched-protocol study is a caution for this literature: the apparent benefit of freezing in our production model was a property of an unmatched protocol.”

The contribution summary on **p. 2, lines 62–64** also says the advantage “is absent once ... equalized,” without “nominal” or the seed-count qualification.

**Restricting files:** `results/e3c_analysis_runs.csv`; `paper/sections_iclr/appendix_experiments.tex`, E.6; and the main text's own lines 456–462. The paired change in final-five gap is +0.120 with 90% interval [−0.230,+0.470]. That does not establish that the protocol caused the earlier gap. Saving a best checkpoint also does not change the final-window training trajectory.

**Repair:** “The original apparent benefit came from a protocol-confounded comparison and cannot be attributed to architecture from that comparison.” Qualify the introductory disappearance as that of the **nominal mean difference**. Preserve the appropriately cautious §8 ending.

### 17. The paper denies evaluating interventions that it reports — minor factual/scope correction

**p. 9, lines 480–482:**

> “Mitigations (per-module learning rates, gradient modulation, freezing) are not evaluated here, and the matched study is a reason to distrust untuned comparisons of them.”

**Contradicting files:** `results/e3c_analysis_runs.csv` and Table 1 contain per-module learning-rate multipliers and freezing comparisons. `paper/sections_iclr/appendix_experiments.tex`, D.5, additionally reports a logit-penalty intervention. Gradient modulation itself is not evaluated, but the blanket sentence is false.

**Repair:** “These experiments do not establish a practical mitigation recommendation; gradient-modulation methods are not evaluated.”

### 18. The cited Papyan paper does not substantiate the specific formula as presented — important citation repair

**p. 2, lines 91–95:**

> “The width-linear growth of the top Gauss–Newton eigenvalue in proportional-width networks is the mean-field law of Karakida et al. (2019); block heterogeneity of the Hessian is the optimizer-side antecedent of Zhang et al. (2024), and the top-eigenvector structure u⊗E[x] appears in Papyan et al. (2020).”

**Source mismatch:** `paper/references.bib`/`paper/main_iclr.bbl` resolves Papyan et al. (2020) to *Prevalence of neural collapse during the terminal phase of deep learning training*. Its [primary text](https://arxiv.org/html/2008.08186) concerns terminal within-class collapse, class-mean geometry and classifier alignment, and discusses earlier Hessian papers as related work. I did not locate the asserted input-mean Kronecker eigenvector result there. Papyan's separate [*Traces of Class/Cross-Class Structure Pervade Deep Learning Spectra*](https://jmlr2020.csail.mit.edu/papers/v21/20-933.html) is more directly relevant to spectral outliers, but do not substitute it without locating the precise result and assumptions.

**Repair:** provide the exact equation/proposition for this formula, or replace it with a narrower claim that the cited source actually supports. This is a citation-support finding, not a claim that such a formula never appears anywhere in the literature.

### Additional formal and appendix repairs

These are not additional claims that the whole theory fails; they are explicitness/consistency requirements for integration.

- **Main Corollary 1 and Theorem 3, p. 5, lines 216–231:** state `a≥0`, `κ>0`; define T as a finite positive horizon where needed. Without positive κ the denominators and the claimed bound are not valid in general. The phase paragraph needs `0<α≤1`, `ℓ>0` and its encoder Rayleigh cap. Assumption 5 should specify `0<δ<C` and the interval on which its claimed hitting guarantee is ensured.
- **Main Theorem 1/Appendix A.4:** put `E[D_curv]` in the expectation display and clarify that the random initialization is in the regime covered by the deterministic cap. An expectation cap alone cannot simply divide inside a ratio. Proposition A.1's separate event-intersection proof correctly handles its own probabilistic cap.
- **Appendix A.5, p. 20, lines 1026–1032:** a logit-magnitude penalty does not shift both parameter blocks by `2αI`. For an affine normalized logit map z(w), the Hessian of `(λ/2)||z||²` is `λJᵀJ`; for nonlinear z there is additionally `λΣ_i z_i∇²z_i`. Remove that inclusion and the unproved assertion that any width-scaled decay necessarily prevents fitting.
- **Appendix A.5, p. 19, lines 1016–1025:** input transformations can destroy the spatial head hypothesis. “No input transformation can touch” its scaling needs explicit preservation of the feature-Gram/nondegeneracy assumptions. For example collapsing all inputs with zero biases can eliminate activations entirely.
- **Appendix C, pp. 39–40:** the partial correlation over 144 runs is still descriptive when the same seeds generate dependent observations across cells. It is not made inferential by pooling instead of testing 24 cell correlations. Keep the numerator result as a decreasing upper envelope; that result is valid, but it should not be read as monotonicity of the realized gradient norm.
- **Appendix E.3, p. 50:** zero channels below `10ε` does not imply that every channel's variance exceeds `1000ε`. The proof needs the actual ratios, not this logical substitution. The co-scaled-control evidence remains valid.
- **Appendix F, pp. 55–56:** scope the audit record to the statements/versions actually checked; it does not independently certify new main-text summaries or all later integrations. Retaining reproducible scripts is useful, but the audit count is not evidence that a newly shortened theorem has kept its hypotheses.

## III. Acceptance estimate

**For this draft as it stands: approximately 20% acceptance probability, with a subjective plausible range of 10–30%.** This is a judgment under substantial reviewer/area-chair uncertainty, not a calibrated statistical forecast and not a conversion formula from the score. The paper has a credible positive case, but the correctness-of-summary repairs and the unresolved contribution payoff make rejection more likely than acceptance. I cannot justify 90% from the current evidence. Fixing the ledger is necessary; demonstrating the useful same-sequence certificate is the change most likely to improve the scientific case.
