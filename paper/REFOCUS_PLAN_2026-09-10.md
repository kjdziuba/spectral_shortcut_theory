# Refocus plan — one theorem, three experiments (2026-09-10, 19:30)

Status: DECISION RECORD + PRE-REGISTRATION. Written after Astra's reviewer
pass `review_packet/astra/review_iclr_01.md` (4/10, lean reject, ~20%
acceptance) and the author–Astra conversation of 16:19–16:30 (author: "we
went so overcomplex ... we prove some [cases], especially the linear, and then
by experiments we can make claims that it probably holds further").

Versioning rule (author, 19:00): nothing is deleted. The reviewed draft is
frozen as git tag `iclr-v1-2026-09-10` (`paper/main_iclr.tex`,
`paper/sections_iclr/`, `paper/ICLR_SKELETON.md` untouched). The refocused
paper is built in NEW files: `paper/main_iclr_v2.tex`,
`paper/sections_iclr_v2/`, `paper/ICLR_SKELETON_v3.md`. Every "cut" below
means "not in the v2 submission"; the material stays in the repo and this
file records where it went.

## 1. The claim (one sentence, Astra's wording adopted)

> In a specified spectral–spatial model, increasing the effective training
> speed of the contextual pathway suppresses learning of an equally
> predictive spectral cue. At matched training fit, this produces greater
> reliance on context and failure when that context changes.

Three levels (Astra 16:30):

1. **What we prove.** In the serial linear-encoder cross-entropy model, under
   explicit initialization and stopping assumptions, contextual replication M
   suppresses the spectral update at the matched fitting margin
   (fraction ≤ 1/(1+Mv₀²)), causes error one under the specified contextual
   reversal for a₀ < m/2 and large M (expected error → Φ(m/2σ) over
   isotropic initialization), and the M-dependence disappears when the
   replicated readout is normalized as M^{-1/2} (Theorem B.5, Corollary B.6,
   Proposition B.7 of the v1 appendix; Astra to restate as ONE theorem with
   all hypotheses visible — see §7).
2. **What we predict.** The same competition occurs in richer models whenever
   the contextual pathway learns fast relative to the spectral encoder; the
   width of a standard-parameterized head is one such speed control, a
   readout learning-rate multiplier is another, and normalized readouts
   remove it.
3. **What we observe.** Controlled experiments test that prediction in a
   nonlinear ReLU CNN (Experiment 2, synthetic, this document §4) and with
   real spectra in assigned neighbourhoods (Experiment 3, §5), each with a
   control that removes the proposed cause (uninformative context).

Novelty (Astra 16:27, agreed): the broad "one feature suppresses another" is
Gradient Starvation (Pezeshki et al. 2021) and simplicity bias (Shah et al.
2020). Our specific contribution is an explicit finite-fit account of
selective *encoder* learning in a *serial* spectral–spatial system, coupled
to interventions that show when contextual training speed produces the
predicted failure. The experiment beyond replicated coordinates (Exp 2) is
what separates the strong version from the weak one ("duplicated features
behave like a larger learning rate").

## 2. Page allocation for v2 (Astra 16:27; 9.00 pp incl. figures)

| Content | Pages | Figure |
|---|---|---|
| Introduction and focused related work | 1.50 | — |
| Model, task, definition of failure | 0.75 | Fig 1 (model diagram + failure definition) |
| Main theorem and explanation | 2.00 | Fig 2 (exact mechanism: update fraction, reversal, normalization control) |
| Exact-model and synthetic experiments (Exp 1 + Exp 2) | 2.00 | Fig 3 (synthetic intervention) |
| Real-spectrum experiment (Exp 3) | 2.00 | Fig 4 (real-spectrum intervention) |
| Limitations and conclusion | 0.75 | — |

A reader must understand the claim after Fig 1 and must not need the
appendix to learn what "failure" means.

## 3. Disposition of v1 material (nothing deleted)

| v1 material | v2 disposition | Where it lives |
|---|---|---|
| Serial CE theorem + isotropic corollary + normalization control | MAIN theorem (restated with hypotheses) | `appendix_math.tex` B.5–B.7 → v2 §3 + App. A |
| 579-case closed-form/ODE verification, bound table | Supplement (one table) | `results/toy_serial_ce*.csv`, App. B of v2 |
| Theorem 1 (width-linear disparity), Lemma cap, Exp 1.1 v3 | Supplement paragraph: "curvature disparity at initialization is the geometric antecedent"; NOT in the main argument | `sections_iclr/04_geometry.tex`, App. of v2 (short) |
| Theorem 2 displacement bound, Corollary (A1), Theorem 3 affine bridge | Cut from the submission (kept as v1 appendix material; may return as one remark if Exp 2 needs the displacement language) | `sections_iclr/05_dynamics.tex`, `appendix_math.tex` B.2–B.4 |
| Theorem N (nonlinear lazy split) | Cut from the submission | `appendix_math.tex` B.7 |
| Momentum/clipping/Adam lemmas | Cut from the submission | `appendix_math.tex` B.2 |
| Residual-subspace export (Exp 1.2 v5), phase attribution | Separate future project; NOT in v2 | `results/exp1_2v5*`, `appendix_math.tex` B.6 |
| Production BN / anisotropy diagnostics (Exp 1.8*) | One scope-limitation paragraph + supplement | `sections_iclr/07_diagnostics.tex`, App. E of v1 |
| Matched freezing study (E3c) | Short limitation/control; details in supplement; NO headline | `sections_iclr/08_matched.tex`, Table 1 |
| Exp 1.2 v4 (CNN/ViT joint vs frozen), Exp 1.3 (equal-information) | Superseded by Exp 2 (which adds matched fit, reversal, rate controls); cite in supplement as earlier runs | `results/exp1_2v4*`, `results/exp1_3v1*` |
| Astra's 18 sentence-level corrections (review §II) | Apply wherever the sentence survives into v2; items 1–4, 11, 12 concern the theorem/figure and MUST be honoured in the v2 statement | `review_iclr_01.md` |

## 4. Experiment 2 — synthetic intervention beyond replicated coordinates (PRE-REGISTERED)

Code: `code/synthetic/data_v2.py` (new generator; `data.py` untouched),
`code/experiments/exp2_intervention.py`. Results: `results/exp2/`,
`results/exp2_summary.csv`. Calibration constants are fixed by
`results/exp2/calibration.json` BEFORE any training run.

### 4.1 Data (per-pixel binary classification, images H×W = 16×16, S = 256)

Labels y_p ∈ {−1,+1} i.i.d. per pixel (no spatial smoothness, so the spectral
cue cannot be denoised by averaging neighbours). Fixed orthonormal directions
u, v₁…v₈ ∈ R^S (⊥ each other). For pixel p:

    x_p = α y_p u  +  β Σ_{d=1..8} (y_{p−o_d} + τ η_{p,d}) v_d  +  σ ε_p

with o_d the eight 3×3 offsets (torus), η, ε standard Gaussian.

- Spectral cue: own spectrum, amplitude α, buried in isotropic noise σ.
  Reading it requires the encoder to align with u.
- Context cue: the eight neighbours of p carry y_p along eight "texture"
  directions with context-specific noise τ. Its information about y_p sits
  ONLY in the neighbours' spectra (p's own v-coordinates encode the
  neighbours' labels, which are independent of y_p). Amplitude β is large, so
  a random linear encoder preserves its signal-to-noise ratio (signal and
  context noise share the direction), whereas the spectral cue's ratio is
  attenuated by ≈ √(K/S) under random projection. This is the nonlinear
  analogue of (a₀ small, v₀ = O(1)) in the theorem.
- Equal information: α = 1.645σ (oracle spectral accuracy Φ(1.645) = 0.95);
  τ calibrated so that the oracle context accuracy (LDA on the 5×5 window of
  the eight v-projections) is 0.95 ± 0.005. β = 5, σ = 1, K = 12.
- Readiness at initialization (recorded, not tuned): LDA accuracy on the
  random encoder's outputs for the spectral cue alone (single pixel) and the
  context cue alone (encoded 5×5 window). Expected ≈ 0.64 vs ≈ 0.93.

Test conditions (64 images each, fresh seeds): `iid`; `reversed` (neighbours
carry −y_p: context contradicts the label); `ctx_random` (neighbours carry
an independent label field: context present but uninformative); `spec_only`
(β = 0); `ctx_only` (α = 0).

### 4.2 Model and training

Linear encoder W ∈ R^{K×S} (per pixel, no bias, N(0,1/S) init) → ReLU CNN
head: Conv3×3(K→M) – ReLU – Conv3×3(M→1), circular padding, single logit,
loss = mean_p log(1+e^{−y_p F_p}) (the theorem's margin loss). Full-batch
gradient descent, ONE global learning rate η for every parameter (the
theorem's unit rates), no momentum, no clipping, no weight decay,
N_train = 256 images (65,536 pixels). Matched fit: the first step at which
the training loss is below L* = 0.30 (primary); snapshots also at
{0.6, 0.5, 0.4, 0.2, 0.15, 0.10}; run ends at loss < 0.10 or 40,000 steps.
η is chosen once by a stability check at the largest width and is the same
in every arm (a run that diverges or never reaches L* is reported as such).

Arms (3 seeds each; seed sets directions, data and initialization):

| arm | widths M | readout | training context | purpose |
|---|---|---|---|---|
| `sp` | 8, 32, 128, 512, 2048 | standard | informative | the width intervention |
| `mup` | 8, 32, 128, 512, 2048 | M^{-1/2}·γ, γ = O(1), same function at init | informative | the theorem's normalization control |
| `ctxfree` | 8, 128, 2048 | standard | uninformative (`ctx_random`) | removes the proposed cause |
| `lrmult` | 32 | standard, readout lr × {1, 4, 16, 64} | informative | effective speed without parameter count (32×64 = 2048) |
| `frozen` | 8, 128, 2048 | standard | informative | encoder fixed at init: the context-reliant reference |

### 4.3 Measurements (every 25 steps and at each threshold crossing)

train loss and accuracy; encoder alignment a_u = ‖Wu‖²/‖W‖²_F (init ≈ 1/S),
gain ‖Wu‖, context alignment a_V = Σ_d‖Wv_d‖²/‖W‖²_F, displacement
‖W−W₀‖_F/‖W₀‖_F; spectral probe = LDA accuracy on the encoder output for
`spec_only` data (fit on one set, evaluated on another): "does the encoder
expose the spectral cue"; block gradient norms ‖∇_W L‖, ‖∇_φ L‖ (exact
gradient-gain identity for CE); accuracies on the five test conditions.

### 4.4 Pre-registered predictions (evaluated at L* = 0.30; fixed before running)

- **P1 (width, SP).** a_u(L*) and the spectral probe decrease with M, and
  `reversed` accuracy decreases with M, in every seed (Spearman over the five
  widths ≤ −0.8 for a_u and for `reversed` in each seed).
- **P2 (normalization control).** Under `mup` the same quantities show no
  monotone width trend (|Spearman| < 0.5 in at least two of three seeds) and
  a_u(L*) at M = 2048 exceeds the `sp` value at M = 2048 in every seed.
- **P3 (cause removed).** Under `ctxfree` a_u(L*) is high (> 0.25, i.e. the
  encoder puts a quarter of its energy on u against an initial 1/256) at
  every width and its width trend is flat (|Spearman| < 0.5).
- **P4 (speed without parameters).** Under `lrmult` at M = 32, a_u(L*) and
  `reversed` accuracy decrease with the multiplier in every seed.
- **P5 (reference).** `frozen` has `reversed` accuracy < 0.5 at every width.

Success = P1–P4. If P1 fails while P3 holds, the mechanism does not transfer
beyond replicated coordinates and the paper's empirical claim narrows to the
exact model (report as such). If P3 fails (ctxfree also shows a width trend),
the width effect is not context competition and P1 cannot be interpreted as
the theorem's mechanism.

Secondary (descriptive, not pre-registered): trajectories a_u(t) to show
whether suppression is transient (encoder aligns after the context fit) or
persistent; the ‖∇_W L‖²/‖∇_φ L‖² ratio at init versus M.

### 4.5 Known limitations to state
Linear encoder (as in the theorem); one head family; full-batch GD; a
constructed context cue; matched-fit stopping is the theorem's convention,
not a training recipe; K/S sets the initial readiness asymmetry.

## 5. Experiment 3 — real spectra, assigned neighbourhoods (design; run after Exp 2)

Data: breast QCL pixels (companion repo, `/mnt/hdd2/u37314kd/data_breast_v2_pca23`
or the no-denoising variant; fold-0 patient-level split). Construct 3×3 (or
5×5) patches whose CENTRE pixel is a real spectrum with its real label and
whose NEIGHBOURS are real spectra drawn from cores of a class chosen with
probability ρ_train ∈ {0.5 (uninformative), 0.9 (informative)} to match the
centre's class. Evaluate on held-out patients under ρ_test ∈ {0.9, 0.5,
0.1 (reversed)}. Model: linear (or MLP) spectral encoder → CNN head of width
M ∈ {48, 192, 768} (SP and normalized readout), plain SGD with the same
learning rate, matched training fit. Measurements as in Exp 2, with the
spectral probe = LDA on encoder outputs for centre pixels. Describe as "a
controlled contextual-shift benchmark built from real spectra"; it does not
establish that the same mechanism explains natural clinical shifts. Pilot at
one width before the full grid.

## 6. Schedule (abstract Sep 18, paper Sep 25)

- Sep 10–11: Exp 2 calibration + pilot (1 seed) + full grid; Astra reviews
  the design and restates the theorem (§7).
- Sep 12–13: Exp 2 v2 if needed (nonlinear encoder; SGD check); v2 main text
  §1–3; `ICLR_SKELETON_v3.md`.
- Sep 12–16: Exp 3 pilot then grid (GPU on this machine; ≤ 18 concurrent jobs).
- Sep 17: v2 abstract final → register Sep 18.
- Sep 18–22: full v2 draft, four figures, trimmed appendix; Astra review 02.
- Sep 23–24: corrections; Sep 25 submit.

## 7. For Astra (relayed by the author)

1. Restate the serial result as ONE main theorem with every hypothesis
   visible (special vs general initialization, immediate-stop convention,
   sup over 0 ≤ t ≤ T_m, expectation over initialization in the reversal
   limit, m > 0, σ > 0), with three labelled conclusions: suppression at
   matched fit, reversal failure, normalization control. Target ≤ 0.6 page
   in the main text; proofs in the appendix (reuse B.5–B.7).
2. Review §4 of this document before the full grid runs: is the context
   construction a fair nonlinear analogue of (a₀, v₀) in the theorem; are the
   pre-registered predictions the right operationalization of "suppressed
   spectral learning at matched fit"; is anything in the design a hidden
   confound (initial readiness asymmetry via K/S, the single global η)?
3. Draft the "definition of failure" paragraph for v2 §2 (what "reliance on
   context" and "failure under context change" mean operationally, in
   the theorem and in the experiments).
