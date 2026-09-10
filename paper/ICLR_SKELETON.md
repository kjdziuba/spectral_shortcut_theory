# ICLR 2027 main text — skeleton, budgets, claims ledger (v2, 2026-09-10 13:30)

Authoritative spec for `paper/main_iclr.tex` + `paper/sections_iclr/*.tex`.
v2 incorporates Astra's `review_packet/astra/reply_02.md` (Q8 allocation, Q6
blocking corrections, Q7 contribution paragraph, Q9) and `math_03.md`.
Decisions (author, 2026-09-10): practical freezing prescription DROPPED;
target ICLR 2027 (abstract Sep 18, paper Sep 25); 9 pages main text;
unlimited appendix; AI-use statement required (outside the count);
double-blind.

## 0. Contribution paragraph (Astra Q7 — the spine of abstract, intro and conclusion; use its content, may rephrase lightly)

> "We distinguish three ingredients of selective encoder adaptation in serial
> models: block curvature, residual excitation, and optimizer-dependent
> displacement. Conditional bounds identify when a fast downstream module
> limits encoder movement, with a transparent affine comparison and a scoped
> nonlinear ReLU existence result; an exactly solvable serial cross-entropy
> instance exposes the role of the induced training metric at a matched
> fitting threshold. Measurements on a production spectral–spatial model show
> why scalar curvature can be misleading: input anisotropy and normalization
> alter its interpretation, while a separate shallow-model trajectory
> experiment reveals loss of residual overlap with fast directions. Matched
> retraining comparisons and checkpoint-policy analysis show why the original
> apparent freezing advantage cannot be attributed to architecture from that
> protocol."

Weakest link (state it ourselves in §9): the contribution beyond established
lazy-training, optimization-metric and feature-competition results, given that
the strongest constructive guarantees apply to deliberately scoped models and
the production study does not establish their mechanism. Lead with the
precise mathematical/diagnostic distinction, never with audit volume.

## 1. Page budget (Astra Q8; 9.00 pages including figures)

| # | Section (file in `paper/sections_iclr/`) | Pages | Words (prose) | Displays | Figures/tables |
|---|---|---|---|---|---|
| — | Title + abstract (main_iclr.tex; `00_abstract.tex`) | 0.30 | 170 | — | — |
| 1 | Introduction and precise contribution (`01_intro.tex`) | 0.80 | 520 | 0 | — |
| 2 | Related work (`02_related.tex`) | 0.60 | 400 | 0 | — |
| 3 | Setup, normalization, curvature vs gradient gain (`03_setup.tex`) | 0.60 | 380 | 3 | — |
| 4 | Conditional geometry: Theorem 1 and the hypothesis boundary (`04_geometry.tex`) | 1.10 | 600 | Thm 1, Lemma cap | — (Exp 1.1 numbers in text) |
| 5 | Dynamics: Thm 2, affine bridge, attribution, scoped nonlinear consequence (`05_dynamics.tex`) | 1.30 | 750 | Thm 2, Cor (A1), Thm 3; Theorem N as a PARAGRAPH | — |
| 6 | Solvable serial CE instance, isotropic extension, normalization control (`06_instance.tex`) | 1.10 | 450 | Thm 4 | **Fig 1** (serial example) |
| 7 | Diagnostic limits (`07_diagnostics.tex`): 7.1 production input/directional/BN diagnostics (1.15 pp, 650 w); 7.2 shallow residual-subspace experiment (0.65 pp, 400 w) | 1.80 | 1050 | 0 | **Fig 2** (composite) |
| 8 | Matched-protocol evidence (`08_matched.tex`) | 1.20 | 600 | 0 | **Fig 3**, **Table 1** |
| 9 | Limitations and conclusion (`09_limitations.tex`) | 0.50 | 330 | 0 | — |
| — | AI-use + reproducibility statements (`statement_ai.tex`, `statement_repro.tex`; outside the count) | — | 200 | — | — |
| **Σ** | | **9.00** | **≈ 5,250** | | 3 figs, 1 table |

Hard rules: ≤ +10% over the word budget; every number traceable to a file
under `results/` or a statement in `review_packet/astra/appendix_math.tex`;
no `\TODO`, no `??`, no author-identifying text; the §6 forbidden list.
Budgets are writing budgets: if the first compile overruns, cut prose, never
fonts or margins.

## 2. Section specs

### Abstract (≤ 170 words, `00_abstract.tex`, body only)
One paragraph built from §0: problem (serial pipelines with capacity
asymmetry); the three ingredients; conditional bounds (Thm 1 disparity ∝ M
under a head hypothesis; Thm 2 displacement O(ε log 1/δ) under a residual
envelope; share ≤ a/(a+κ); affine bridge; scoped nonlinear existence);
the solvable instance (disparity M, bound within 4–6×, update suppressed as
1/(1+Mv₀²), reversal failure Φ(m/2σ)); production diagnostics (head
hypothesis unmet; rank-one inputs invert the scalar; anisotropy 3.7×/12–28×;
width trend carried by BatchNorm gains; residual loses overlap with fast
directions); matched protocol (the original apparent freezing advantage is
not attributable to architecture from that protocol). No prescription.

### 1. Introduction (520 w) — `01_intro.tex`
- Hook: compositional pipelines with capacity asymmetry — spectral–spatial
  imaging (FTIR, QCL, hyperspectral remote sensing; `berisha2019deep`,
  `oleary2026spatial`), frozen image encoders under temporal models,
  multimodal fusion. Pretrain-and-freeze is widespread; whether it is
  necessary is what a fair comparison must decide.
- The question and the three ingredients (curvature, residual excitation,
  optimizer-dependent displacement). Theory stated as CONDITIONAL at first
  mention.
- Contributions: four bullets ≤ 45 words each with \ref pointers:
  (i) conditional theory → thm:hessian, thm:twoscale, cor:attribution,
  thm:bridge, and the nonlinear paragraph (label par:nonlinear via
  \phantomsection\label if needed, else refer to App. B);
  (ii) solvable instance → thm:instance, fig:serial;
  (iii) diagnostic limits → sec:diagnostics, fig:diagnostics;
  (iv) matched protocol → sec:matched, fig:matched, tab:matched.
- One paragraph on what the production measurements teach (scalar disparity
  wrong lens on rank-one inputs; curvature growth can be a normalization
  artefact; residual subspace matters; protocol hygiene decides the verdict).
- FORBIDDEN here: the frozen-random 0.70 / joint 0.675 paradox; "first
  rigorous explanation"; any prescription.

### 2. Related work (400 w) — `02_related.tex`
Groups: modality competition and laziness (Wang 2020; Peng 2022; Du et al.
ICML 2023; Huang et al. 2022 provable); shortcut/starvation theory (Pezeshki
2021; Lim–Kim–Moon NeurIPS 2025); curvature, width, normalization (Karakida
2019; Zhang et al. 2024 "Why Transformers need Adam"; Papyan 2020; van
Laarhoven 2017); implicit bias in serial/diagonal models and lazy training
(Saxe 2014; Du–Hu–Lee 2018; Yun–Krishnan–Mobahi 2021; Moroshko 2020;
Berthier 2023; Chizat–Oyallon–Bach 2019; Sohl-Dickstein et al. 2020;
Yang–Hu 2021; Du–Zhai–Póczos–Singh 2019); frozen features in practice
(Kirichenko et al. DFR; Zhang–Bengio–Singer 2022; Coil & Cheney 2025
"hypothesize"). One-sentence delta each; never "first". New bib entries only
when verified (see §6 rules); Astra's appendix bib block
(`paper/bib_additions_astra.bib`) already contains verified entries for the
math references — reuse its keys where they exist.

### 3. Setup (380 w + 3 displays) — `03_setup.tex`
From `sections/03_setup.tex`: model f_θ (linear S→K per pixel) ∘ g_φ (width
M); per-pixel CE; N-normalized residual r and logit Jacobians J_θ, J_φ with
∇_bL = J_bᵀr; GGN G = JᵀH_τJ and its two diagonal blocks; the output-space
kernels K_b = J_bJ_bᵀ (Astra §1 of appendix: K_b is NOT the parameter block
G_bb — state the distinction: curvature (G_bb, parameter space) versus
gradient gain (rᵀK_br = ‖∇_bL‖², exact for CE)); Definition \Dcurv =
λ_max(G_φφ)/λ_max(G_θθ) at initialization. Assumptions in ONE compact list:
(i) linear f_θ (ass:linear); (ii) bounded mean input Gram Σ_X (ass:subg);
(iii) width-uniform input-Jacobian cap L̃ on Φ_reg × Z_reg (ass:inputlip) —
a modelling assumption, proved for a two-layer ReLU head in App. A;
(iv) a.e. regularity for ReLU (ass:reg). Capacity gap in one clause
("informative when C_g ≫ C_f; the production ratio is 299"). No EGR.

### 4. Conditional geometry (600 w) — `04_geometry.tex`
Theorem 1 (thm:hessian) statement verbatim from `sections/04_theorem1_hessian.tex`
with the head hypothesis INSIDE the statement; Lemma (lem:opcap)
λ_max(G_θθ) ≤ L̃² λ_max(Σ_X) =: C_θ; two-sentence proof sketch (witness
c⊗u; chain rule) → App. A. "The hypothesis boundary" paragraph: readiness
not usefulness; not a dynamical claim; top-of-spectrum only; Kaiming/standard
parameterization, not μP; the head hypothesis is a cross-input correlation
condition on Θ(M) penultimate features — say now that the production model
does not satisfy it structurally (§7). "Empirical check" paragraph (Exp 1.1
v3, 5 seeds; numbers from `results/exp1_1_v3_ggn_*aggregated.csv`): shallow
verified instance under default fan-in init: λ_φ 0.62 → 46.0 over M = 16 →
8192 (log–log slope 0.71), λ_θ ≈ 0.02 flat (slope −0.02), \Dcurv 40 → 2405;
under the theorem's initialization (He weights + fixed σ_b): λ_φ 2.1 → 442
(slope 0.90), λ_θ ≈ 0.6 flat. Deep head, M = 16 → 1024: default init slope
0.105 (λ_φ 0.68 → 1.11) vs theorem init 0.989 (3.6 → 243); isolation arms
default_fixedbias 0.924, he_scaledbias 0.696, he_zerobias 0.606. Reading
(PROGRESS 2026-09-01): the width-linear floor survives if EITHER the
weight-variance or the bias-variance term of the head hypothesis is
width-independent; default fan-in initialization makes both vanish with
width. Quote slope AND level; never use \Dcurv for cross-arm attribution;
finite-range fits. No figure (appendix table).

### 5. Dynamics (750 w) — `05_dynamics.tex`
- Theorem 2 (thm:twoscale) verbatim: ‖θ(T)−θ₀‖ ≤ (B_T C/μ) log(1+μT) and at
  T_δ ≤ (B_T C/μ) log(C/δ), ε = B_T/μ; hypotheses: containment, envelope
  ‖r(t)‖ ≤ C/(1+μt) (ass:residual), cap B_T ≤ L̃√λ_max(Σ_X). Two-line proof.
  One sentence: discrete versions (GD, momentum with clipping before
  accumulation — our trainer's ordering — velocity clipping, Adam with
  positive stabilizer, decoupled decay) hold with μ_s = ημ_f
  (App. B, lem:astra_gd … lem:astra_decay).
- Corollary (cor:attribution) = (A1): unit Euclidean rates, a.e. Rayleigh
  inequalities eᵀK_θe ≤ a‖e‖², eᵀK_φe ≥ κ‖e‖² on the realised residual,
  L(0) > L(T) ⇒ ∫₀ᵀ‖∇_θL‖²/(L(0)−L(T)) ≤ a/(a+κ); sharp; no invariant
  subspace; proof in App. B (cor:astra_attribution). Say: for CE, rᵀK_br =
  ‖∇_bL‖² exactly (no squared-loss approximation).
- Theorem 3 (thm:bridge) = affine squared-loss bridge (thm:astra_affine):
  K_φ ⪰ κI on the visited residual subspace, ‖K_θ‖ ≤ a ⇒ joint-vs-frozen
  logit gap, encoder displacement, θ loss-share all O(a/κ) (copy the exact
  orders from appendix eq:astra_affine_gap/disp); the biased-ReLU instance
  gives κ = κ_*M/2 with M-independent a (cor:astra_init), existence only,
  threshold labelled numerically vacuous; Counterexample 1.4 in one
  sentence (prop:astra_top_counter): a large TOP spatial eigenvalue is not a
  rate certificate when the residual lives elsewhere.
- Nonlinear consequence as ONE PARAGRAPH (not a theorem block; Astra Q8):
  "For the fully trained shallow biased-ReLU model under squared loss in the
  standard parameterization, for M ≥ M_* and with probability ≥ 1−ρ over
  initialization, every joint and frozen solution fits at rate Ω(M), the
  encoder moves O(1/M) while the head moves O(M^{-1/2}), both paths track
  one common linear reference to O(1/M), and the encoder's share is ≤
  A²/(A²+κM/2) (App. B, Theorem N). This is lazy training in the standard
  parameterization (Chizat–Oyallon–Bach; Du et al. 2019 gate-flip argument);
  the disparity disappears under a readout f = M^{-1/2}Uh + β with O(1)
  entries trained at unit rate; M_* is existence-only and numerically
  vacuous; the content is the two-block displacement split."
- Residual excitation paragraph (prop:excitation, informal; App. B
  residual-excitation theorem): curvature anisotropy ⇏ cumulative starvation
  without a hypothesis on the residual's projection onto the starved
  directions — what §7.2 measures. Phase-restricted attribution in one
  sentence: share over a phase E where the residual keeps mass ≥ α in the
  top-k eigenspace of K_φ (kth eigenvalue ≥ ℓ) is ≤ a/(a+αℓ), and the total
  share is ≤ w_E a/(a+αℓ) + (1−w_E); a cumulative O(1/M) share requires the
  complementary phase to carry O(1/M) of the energy (App. B,
  prop:astra_phase, cor:astra_phase_width).
- "Is the hypothesis circular?" two sentences (contrapositive).

### 6. Solvable instance (450 w + Thm 4 + Fig 1) — `06_instance.tex`
Theorem 4 (thm:instance) compact: model q = a + bv, b = Σ_j β_j over M
replicated readouts, logistic CE, unit-rate gradient flow; \Dcurv(0) = M;
envelope rate μ = (M+1)/4; for a₀ < m, v₀ ≠ 0, zero-sum readouts: update
fraction (a(T_m)−a₀)/(m−a₀) ≤ 1/(1+Mv₀²) → 0 (the coefficient itself need
not be small); joint-vs-frozen margin gap ≤ C₀/(p₀Mv₀²); contextual reversal:
error → Φ(m/2σ) over isotropic Gaussian (a₀,v₀), spectral-only comparator
error 0, and for a₀ > m/2 correct at every width; readout normalization
f = M^{-1/2}Uh (O(1) entries, unit rate) removes the M-dependence
(layer-balance invariant, Du–Hu–Lee 2018). Then: Theorem 2's bound is within
4.5–6.3× of the realised displacement (3.8× with the true amplitude);
closed form vs ODE to 1e-11 (App. B/D). Quote Astra's positioning sentence
verbatim (math_02 §5). Figure 1 (fig:serial) caption ≤ 70 w.

### 7. Diagnostic limits (1050 w + Fig 2) — `07_diagnostics.tex`
\section{Diagnostic limits on a production model and a shallow instance}
7.1 Production model (650 w). Setup: BlockViT-v2, C_f = 61,108, C_g =
18.27M, ratio 299; breast FTIR TMA fold 0 (numbers as in
`sections/08_real_data.tex` §8.1), prostate QCL replication; head
Conv(M+K→96)–BN–ReLU–Conv(96→48)–BN–ReLU–Conv(48→4): classifier input 432
at every width ⇒ Theorem 1's head hypothesis structurally unmet, not
contradicted. Geometry at init (Exp 1.8/1.8b): \Dcurv 0.46 ± 0.21 at M=192,
0.33 → 0.70 over 48 → 384, prostate 1.09 ± 0.52 at 384; λ_θ slopes
−0.003/−0.012 (cap holds); λ_φ 0.38/0.45; Σ_X effective rank 1.07/1.02
(1.21 post-BN); anisotropy: v₁ = mean spectrum (|cos| > 0.995); θ-block top
eigenvector 0.954 ± 0.017 in the v₁-induced subspace; top-40/61,108 hold
0.83 ± 0.12 of the trace; full CancerEpi–CAS contrast 3.7× (2.6–4.8) less
curvature than v₁; its v₁-orthogonal part (76–84% of squared norm) 12–28×;
"starvation is a hypothesis; the measurement is anisotropy". Width trend
(Exp 1.8c/1.8d, 3 seeds): incoming patch Gram flat to five figures; λ slope
+0.61 with batch statistics (per seed +0.35 to +0.79) vs +0.006 with both
head BNs at fixed initial statistics; either BN alone carries it
(+0.56/+0.57) — evidence AGAINST a unique first-BN attribution; pre-BN
variance ∝ 1/(M+K) (0.15/0.064/0.035; products with M+K near-constant),
gain 2.6/4.0/5.4; the directional JVP stage energies support a FINITE-RANGE
normalization-gain explanation; bias perturbation annihilated exactly; the
function-preserving rescaling (weights and bias by c, ε by c²) leaves
logits unchanged and scales curvature by exactly c⁻² — a parameterization
quantity (cite `vanlaarhoven2017l2`); conditional gain law in App. B
(prop:astra_bn_gain) — NOT a new BN mechanism. Astra's sentence verbatim:
"This sweep therefore does not validate the width-linear feature-Gram
mechanism of our shallow theory." Three-width slopes are descriptive;
interventions establish sensitivity, not a unique cause.
7.2 Shallow residual-subspace experiment (400 w). Model = verified instance
(linear S=64→K=16 + theorem-init ReLU head), Exp 1.2 protocol (Adam, 2400
steps, D ∈ {128, 512, 2048}, 3 seeds), probe N = 256, C = 2. Report as
PROBE results and GRADIENT-ENERGY fractions (Astra Q9): on the probe,
rᵀK_θr = ‖∇_θL‖² exactly; the cumulative quantity is the share of summed
squared minibatch gradient norms over the Adam run (not a decomposition of
Adam's loss decrease; probe and minibatch are different sequences). Numbers:
cumulative gradient-energy share ≈ 1/D (medians 0.29 / 0.062 / 0.012);
at t = 0 κ_eff (the residual's Rayleigh quotient under K_φ) grows with D
(0.95 / 7.1 / 64) and the instantaneous share is 0.43 / 0.061 / 0.0077;
after ~75 steps the residual's mass in the top-1 / top-20 / top-100
eigenspaces of K_φ is 8.6e-5 / 0.030 / 0.16 (medians) and κ_eff ≈ 0.0025
λ_max while the instantaneous share returns to ≈ 0.4 at every width; the
top-eigenvalue proxy λ_max(K_θ)/(λ_max(K_θ)+λ_max(K_φ)) understates the
share in 85% of checkpoints (median 2.7×) and overstates it by up to 11×;
the λ_min proxy is 0.9998–1.0000. DO NOT write "below isotropic" (the
reference must be recomputed in the class-contrast subspace of dimension
N(C−1); App. D says so). Write "the diagnostic limitation illustrated by
Counterexample 1.4 is observed", not "realised". Reading via the phase
result: the early phase (residual in the coercive top, κ_eff ∝ D) carries
most of the energy, but a cumulative O(1/D) law requires the late phase's
energy fraction to be O(1/D) — the three-width table does not establish an
asymptotic law. Two sentences on Exp 1.2 v4 (CNN/ViT heads, Adam): frozen
ahead at the endpoint by ≤ 0.04 at every width (unresolved at n = 3), peak
accuracies indistinguishable, joint test loss diverging at D = 1024 —
appendix. Figure 2 (fig:diagnostics) caption ≤ 90 w, model and optimizer
named per panel.

### 8. Matched-protocol evidence (600 w + Fig 3 + Table 1) — `08_matched.tex`
Compress `sections/08_real_data.tex` §8.4 with Astra's Q6 corrections
applied ON TOP (they override the long draft):
- Original protocol registered and confounded: joint arms trained the
  reduction-stage BN affine parameters and were clipped over θ+φ, frozen arms
  φ-only; no saved best checkpoints; final-5 endpoint −0.110 at h=48 (AdamW),
  unresolved itself.
- Momentum-SGD controls (paired best-val +0.002/+0.011; final-5
  +0.028/−0.057 at M=48/192); BN recalibration recovers 7–17% of the
  peak-to-final drop "for this intervention and these checkpoints" (it
  recovers a nonzero part of the degradation too).
- Matched protocol at h = 48, n = 3: "the same clipping rule and scope"
  (NOT "identically in every arm"); saved best-val checkpoints; Table 1
  (tab:matched): 7 rows × {best-val, final-5, θ drift}, mean ± sd from
  PROGRESS 2026-09-10 08:15. NAME that the four core arms are two matched
  PAIRS, not a single-factor design: joint_linear/frozen_random use a linear
  reduction with BN; frozen_pretrained/finetune_real use a pretrained MLP
  encoder without reduction-stage BN — comparing across pairs changes
  architecture, initialization/training history and validation selection.
- Verdicts with the paired 90% t-intervals (Astra reproduced them):
  fine-tune − frozen-pretrained: final-5 +0.021 [−0.022, +0.063], best-val
  −0.004 [−0.048, +0.039] → prediction of harmful fine-tuning not supported
  ("verified nonzero applied encoder updates" under AdamW, not "verified
  gradient flow"); joint − frozen-random: final-5 +0.011 [−0.142, +0.163],
  best-val +0.023 [−0.066, +0.112]; the paired change in the final-5
  joint/frozen gap between protocols is +0.120 [−0.230, +0.470] and in the
  best-val gap −0.010 [−0.105, +0.085] — descriptive seed summaries; LR
  multipliers: ×10 − ×1 best-val +0.048 [−0.016, +0.112] (3/3 seeds) with
  endpoint 0.561 — "three multiplier means do not establish a systematic
  relation between displacement and peak score"; cosine: "this particular
  schedule did not improve either mean endpoint" (does not rule out an
  excessive late learning rate).
- Disclosures: width 48 only; n = 3; single validation fold; no test set;
  best-val exploratory vs final-5 pre-specified; pretrained encoders selected
  by macro-F1 on the same fold-0 validation cores and pretrained with weight
  decay 0.01 on θ; pixel pretraining used the unaugmented/centre-crop
  protocol while subsequent training used random crop plus augmentation
  (affects cross-pair comparisons, not the frozen/fine-tuned pair from the
  same checkpoint); augmentation on in all arms; bound proxy withdrawn (one
  sentence).
- Reading paragraph = Astra's replacement text VERBATIM: "The original
  protocol confounded freezing with clipping scope and normalization-
  parameter treatment. Its nominal final-window freezing advantage was
  absent after these differences were removed, while checkpoint selection
  reversed the ordering within the original runs. At width 48 and three
  seeds, neither the within-protocol freezing contrasts nor the change in
  their gap establishes a population effect. These results demonstrate that
  the original contrast cannot identify an architectural shortcut and that
  checkpoint policy changes its descriptive conclusion." Then one sentence:
  this supports a protocol-confounded apparent effect, not a proof that
  architecture has no effect nor that the named changes uniquely caused the
  old difference. FORBIDDEN: "property of the training protocol, not of the
  architecture"; "sufficient to manufacture"; "identically in every arm";
  "verified gradient flow"; "share of squared gradient flow"; any claim of
  equivalence from n = 3.

### 9. Limitations and conclusion (330 w) — `09_limitations.tex`
Conclusion first (2 sentences from §0), then limitations: linear encoder in
the theorems; the strongest constructive guarantees hold for deliberately
scoped models (affine; shallow ReLU with vacuous thresholds; the serial
scalar instance) and the production study does not establish their
mechanism; head hypothesis unmet on the production model; three-width
exponent fits are descriptive; the residual experiment is a probe/Adam
diagnostic, its isotropic reference awaits recomputation in the
class-contrast subspace, and eleven checkpoints cannot certify an a.e.
condition; the matched study is width 48, n = 3, one validation fold, no
test set, two matched pairs; mitigation strategies (per-module learning
rates, gradient modulation, freezing) are not evaluated as prescriptions —
the matched study shows untuned comparisons are not evidence. End on the
open problem: a same-sequence phase certificate (prop:astra_phase) with
measured w_E, or a full-batch flow instance where the energy identity is
exact. No "concluding remarks" heading.

### Statements (outside the count)
- `statement_ai.tex` (≤ 120 w): generative AI tools (LLM assistants) were
  used for literature search, drafting and editing text, writing and
  checking code and numerical verification scripts, and adversarial review
  of proofs and claims; all proofs were independently re-derived and all
  numbers recomputed from the artifacts by the authors; AI-generated text and
  code were reviewed by the authors, who take responsibility for the final
  content.
- `statement_repro.tex` (≤ 90 w): code, audit scripts with seeds and
  commands (anonymized repository in supplementary material); every number
  traces to a results file named in the appendix; hypotheses and complete
  proofs in Appendices A–B; data description in Appendix E.

## 3. Figures and table (main text) — exactly three figures (Astra Q8)

- **Fig 1 (`figures/fig1_serial.pdf`, `fig:serial`, §6)** — the serial CE
  example. Panel (a): update fraction (a(T_m)−a₀)/(m−a₀) vs M (log–log)
  for several (a₀, v₀) with the exact bound 1/(1+Mv₀²) drawn, plus the
  normalized-readout control (flat); panel (b): joint/frozen margin gap
  sup_t(q_J−q_F) vs M with the bound C₀/(p₀Mv₀²); compact annotations:
  \Dcurv(0) = M, μ = (M+1)/4, reversal error → Φ(m/2σ). Data:
  `results/toy_serial_ce.csv`, `results/toy_serial_ce_isotropic.csv`,
  `_gaussian.csv`. Script `paper/figures_src/make_fig1_serial.py`.
- **Fig 2 (`figures/fig2_diagnostics.pdf`, `fig:diagnostics`, §7)** — a
  three-panel composite, model and optimizer labelled on each panel:
  (a) production directional anisotropy = panel (b) of
  `make_fig8a_geometry.py` (mean / full contrast / v₁-orthogonal; 9
  measurements; "BlockViT-v2, initialization"); (b) production BN
  intervention: λ_max of the first-head-convolution block vs M (log–log)
  for train / eval_bn1 / eval_bn2 / eval_head, per-seed points and pooled
  lines, with the incoming patch-Gram top eigenvalue drawn flat as reference
  (from `results/exp1_8d_witness_jvp.csv`, direction ggn_top; and
  lamS9_valid); (c) shallow residual experiment: κ_eff/λ_max(K_φ) and the
  residual's top-20 mass vs step (symlog x) for D ∈ {128, 512, 2048}, joint
  arm, medians thick and seeds thin ("shallow verified instance, Adam,
  probe N = 256"); NO isotropic reference line. Script
  `paper/figures_src/make_fig2_diagnostics.py`.
- **Fig 3 (`figures/fig3_matched.pdf`, `fig:matched`, §8)** — paired arm
  differences with 90% paired-t intervals (n = 3): rows = joint_linear −
  frozen_random (matched), finetune_real − frozen_pretrained (matched), and
  the original-protocol joint − frozen contrast at h = 48 labelled "bundled
  protocol intervention"; two panels: final-5 (pre-specified, left/prominent)
  and best-val (exploratory, right); seed points shown; zero line; no
  equivalence wording. Data: `results/e3c_analysis_runs.csv` (reuse the
  computations in `make_fig8c_matched.py`). Script
  `paper/figures_src/make_fig3_matched.py`.
- **Table 1 (`tab:matched`, §8)**: 7 arms × (best-val, final-5, θ drift),
  mean ± sd, booktabs, \small.

## 4. Appendix map (`sections_iclr/appendix.tex` + inputs; unlimited)
A. Proofs for §4 (witness argument; cap; prop:verified_instance; classical
   conditioning; not_pca) — from `sections/supplement.tex` and
   `sections/04_theorem1_hessian.tex`.
B. `\input{sections_iclr/appendix_math}` = Astra's file (copied verbatim;
   labels astra_*; includes P4 lemmas, bridges, (A1) as
   cor:astra_attribution, Cor 1.5 + Chernoff, interior/BN propositions incl.
   prop:astra_bn_gain, residual excitation, phase result prop:astra_phase +
   cor:astra_phase_width, Theorem 5.1 + isotropic, Theorem N). Its bib
   records are in `paper/bib_additions_astra.bib` (to be merged into
   references.bib; keys namespaced).
C. EGR (from `sections/06_egr.tex`, ≤ 700 w, descriptive statistics only,
   seed-clustered caveat; no Fisher-z p-value).
D. Synthetic details (`appendix_experiments.tex` part 1): generator and
   architectures; Exp 1.1 v3 five-arm table; Exp 1.2 v4 full table; Exp 1.2
   v5 tables (probe, validation identities, headline, cumulative) with the
   isotropic-reference caveat and the probe/minibatch distinction; Exp 1.3
   with both caveats; Exp 1.6 only if artifacts exist; solvable-instance
   verification tables.
E. Production details (`appendix_experiments.tex` part 2): architecture,
   capacity table, data/splits, channel convention, Exp 1.8/1.8b tables,
   Exp 1.8c/1.8d tables (incoming Gram, BN quantities, witness JVP stages,
   scale control, residuals, full-group vs labelled-group Gram discrepancy,
   bias JVP), E3c original protocol, SGD controls, BN recalibration, matched
   full table, verdicts, disclosures incl. the pretraining crop difference, a
   provenance-manifest note, the withdrawn bound proxy.
F. Verification record (`code/audits/README.md`).

## 5. Astra corrections that override the long draft (apply everywhere)
1. Residual export: no "below isotropic"; "the diagnostic limitation
   illustrated by Counterexample 1.4 is observed"; gradient-energy fractions
   on the same sequence; probe vs minibatch; H-weighting is a different
   quantity, not a correction; phase result via prop:astra_phase.
2. Matched lane: Astra's Q6 replacement paragraph verbatim; two matched
   pairs, architecture differs across pairs; "same clipping rule and scope";
   "verified nonzero applied encoder updates"; pretraining crop disclosure;
   cosine/LR/BN-recal wording as in §8 spec.
3. BN width trend: finite-range normalization-gain explanation; either single
   BN preserving the trend argues against a unique first-BN attribution; the
   "s² ≳ 10³ε" cutoff is a heuristic, not a guarantee; do not present the
   gain identity as a new mechanism.
4. Theorem N: paragraph in the main text; gate convention is the explicit
   consistent-gate flow (not "Clarke"); readout control f = M^{-1/2}Uh with
   O(1) entries trained at unit rate.
5. cor:attribution's main statement is (A1), proved in App. B
   (cor:astra_attribution); no duplicate old proof.

## 6. Forbidden (grep before submitting)
"first rigorous", "for the first time", "prescription", "freeze the spectral
module" (as advice), "frozen random ... outperforms", "0.70 test F1",
"absorbed into a spurious stationary point", "sole source", "several orders
of magnitude", "D_curv of order 325", "2e4–2e5", "strongest single
datapoint", "all frozen beat all joint", "stays near initialization" (as an
unconditional claim), "below isotropic", "Counterexample 1.4 realised",
"property of the training protocol, not of the architecture", "sufficient to
manufacture", "identically in every arm", "verified gradient flow", "squared
gradient flow", "Clarke" (for the gate convention), any author name or
affiliation, "our companion paper" (use "a companion tokenization study
(anonymous, under review)").

Citation rule: cite only keys present in `paper/references.bib` or
`paper/bib_additions_astra.bib`; a new key requires a verified BibTeX entry
(URL checked) written to `paper/bib_additions_<section>.bib`; otherwise omit.

## 7. Sources
`paper/sections/*.tex` (48-page draft; §8.4 wording superseded by §5 above),
`PROGRESS.md` (2026-09-01 → 2026-09-10), `REMAINING_WORK.md`,
`review_packet/astra/{math_01,math_02,math_03,reply_02}.md`,
`claude_math_reply_02.md`, `appendix_math.tex`, `results/*` named above.
