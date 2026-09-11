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

### 4.4a Pilot amendment (recorded 20:00, after ONE seed of sp M∈{8,2048} and ctxfree M=8; before the grid)

Observed: at L* = 0.30 spectral learning is fully suppressed at BOTH widths
(align_u 0.0048 / 0.0045 against an initial 0.0044; probe 0.65; reversed
accuracy 0.12), while ctxfree at M = 8 aligns (0.127; probe 0.92; reversed
0.89). The context fit at M = 8 takes 2,633 steps against 10,714 for the
encoder-driven fit, so every width ≥ 8 sits in the saturated regime
Mv₀² ≫ 1 and P1's *trend* cannot be seen there. Amendments, made before any
further run and applied uniformly:

- widths 2 and 4 added to `sp` and `mup` (the O(1)-speed regime);
- `lrmult` gains fractional multipliers 1/16 and 1/4 (slowing the readout
  is the continuous version of the `mup` control; 1/32 at M = 32 IS `mup`);
- `ctxfree` and `frozen` run at widths {2, 8, 128, 512} (2048 dropped for
  cost; the pilot shows the effect saturates far below it);
- learning rate fixed at 1e-3 from the stability check;
- P1 is evaluated as pre-registered at L* = 0.30 AND, as a labelled
  secondary analysis, at the lowest threshold the context cue alone can
  reach (0.15) and at 0.10 (beyond it), because the pilot shows the
  saturation floor at 0.30. The predictions' directions are unchanged.
- Implementation: the head is computed as explicit matrix products with
  torus rolls (identical to `nn.Conv2d` with circular padding to 1e-15 in
  float64, `--selftest`), in full fp32; the pilot used cuDNN convolutions,
  which PyTorch runs in TF32 by default. Pilot outputs are preserved in
  `results/exp2_pilot/`; the grid overwrites `results/exp2/`.

### 4.4b Amendment after Astra's design review (`review_packet/astra/refocus_01.md`; recorded 21:30, before the grid completed)

Astra reviewed §4 and the code. The grid design is unchanged; the following
were added or corrected before relaunch (the first launch was stopped after
six width-2 runs because a leftover pilot process starved the GPU; those six
runs are re-run identically):

- **Whole-head rate arm `headlr`** (Astra's recommended single intervention):
  encoder rate η, first convolution (incl. bias) and readout rate κη, κ ∈
  {1, 1/16, 1/256} at M = 32. In the theorem this is ḃ = κMvϱ and the bound
  becomes (a(T_m)−a₀)/(m−a₀) ≤ 1/(1+κMv₀²): κ moves the competition parameter
  through one while leaving initial predictions and cue information fixed.
  Protocol: pilot on seed 0; extend the bracket downward (1/4096) only if no
  appreciable spectral learning appears before matched fit; check that the
  slow head still reaches L*; then freeze the bracket and run seeds 1, 2.
  κ is never chosen by held-out reversal accuracy. **Prediction P6:** at
  L* = 0.30, probe gain, h_u and reversal accuracy increase as κ decreases
  (Spearman ≤ −0.8 against κ in each seed); at κ = 1/256 the probe gain
  approaches the `ctxfree` value at M = 32... [no ctxfree at 32 in the grid;
  compare with ctxfree M = 8 and 128].
- **Measures.** Primary operational measure of spectral learning = LDA probe
  accuracy relative to its initial value (`probe_acc_gain`), with the exact
  population companion h_u(W) = ‖P_row(W) u‖² exported at every snapshot
  (optimal probe accuracy = Φ(α/σ·√h_u)); encoder energy fraction a_u is an
  explanatory secondary quantity, NOT a necessary condition (P3's literal 0.25
  cutoff is kept for the record but a failure of that cutoff is not evidence
  against learning). Encoder weights are saved at every snapshot
  (`results/exp2/enc_*.npz`).
- **Same-state gradient norms:** backward now precedes the snapshot, so
  ‖∇_W L‖, ‖∇_φ L‖ at a threshold are those of the snapshotted iterate.
- **Paired test conditions:** the five conditions of a seed now share labels,
  context noise and spectral noise (one seed per family); differences between
  conditions are responses to the intervention alone.
- **Interpretation rules (Astra):** flat P1 curves in an already suppressed
  regime do not falsify competition; P2 (no width trend under the normalized
  readout) is a hypothesis for this CNN, not a consequence of part (c);
  the normalized readout is algebraically the readout-only rate-1/M arm at
  the same width (do not count them as independent evidence; call it
  "normalized readout", not μP); a width trend under `ctxfree` can reflect
  ordinary width-dependent optimization, so compare informative vs random
  context at matched width/rate and report the interaction; matched mean
  loss is an analogue of matched margin, not the theorem's stopping rule
  (report loss and accuracy at crossing; confirm on representative cases that
  halving η does not change the interpretation); the context-oracle
  population CE is ≈ 0.126 nats (spectral ≈ 0.127), so 0.15 is "just above
  the context-oracle optimum", not a certified floor, and loss < 0.10 alone
  does not prove spectral learning — the probe and shifted accuracies do.
- **Documentation fixes:** the oracle/readiness probes use the 3×3 window
  (r = 1; it contains all eight label copies), not 5×5; the readout
  multipliers actually run are {1/16, 1/4, 1, 4, 16} (64 dropped in 4.4a);
  readiness values are the recorded 0.647/0.653/0.608 (spectral) vs
  0.910/0.914/0.905 (context), not "≈ 0.93".

### 4.4c κ pilot outcome and bracket extension (seed 0, M = 32; recorded 23:10)

At L* = 0.30: κ = 1 → step 1,671, probe gain +0.028, h_u 0.079, reversal
0.128; κ = 1/16 → step 7,611, +0.107, 0.177, 0.129; κ = 1/256 → step
22,708, +0.223, 0.475, 0.152 (context-random 0.51–0.52 throughout;
spec-only 0.54 → 0.57 → 0.64). Spectral learning at matched fit increases
monotonically as the head slows (P6's probe/h_u part), while reversal
accuracy does not respond within this bracket: the encoder exposes the cue,
the head still predicts from context — the theorem's regime a(T_m) < m/2,
where reversal flips only once the spectral coefficient carries half the
margin. [Correction, Astra review 02 §IV: κMv₀² > m/(m−2a₀) is a
SUFFICIENT condition for failure; its negation does not guarantee
recovery — the boundary is set by the exact fitting equation. The earlier
"i.e. κMv₀² < m/(m−2a₀)" is withdrawn and not carried into the paper.] The protocol's extension rule (extend
downward until the intervention reaches the regime of interest) is
therefore applied for the REVERSAL outcome: κ ∈ {1/4096, 1/32768} added on
seed 0 (runs may not reach L* within 40,000 steps; reported as such). The
bracket is then frozen for seeds 1, 2. κ is not chosen by held-out
reversal accuracy; the extension is fixed before those runs are read.

### 4.4d Registered verdicts at L* = 0.30 (Exp 2 grid, 85/86 runs; recorded 2026-09-11 02:30)

- **P1 (width, sp): NOT MET.** ρ(a_u, M) = −0.54 / −1.00 / −0.96 and
  ρ(rev, M) = −0.64 / −0.96 / −0.82 by seed; seed 0 misses the −0.8
  threshold and the magnitudes are negligible (probe gain 0.04 at M = 2 →
  0.002 at M = 2048; reversal 0.134 → 0.117): the threshold sits in the
  saturated regime at every width including M = 2.
- **P2 (normalized readout flat): NOT MET as stated** (final grid: ρ(a_u, M)
  = +0.25 / −0.96 / −0.54, only one seed below 0.5 in magnitude); **second
  clause MET** — a_u(2048) mup > sp in every seed (0.0059 vs 0.0045, 0.0062
  vs 0.0045, 0.0043 vs 0.0027).
- **P3 (ctxfree): literal cutoff NOT MET** (a_u 0.15/0.13/0.09/0.06 < 0.25;
  ρ = −1.0 in every seed) **while its substance HOLDS** (probe 0.89–0.94,
  reversal 0.88–0.90 at every width; the energy fraction spreads over more
  head directions with M — Astra's warning in refocus_01 §2.3).
- **P4 (readout multiplier): HALF MET** — a_u and probe decrease with the
  multiplier in every seed (ρ = −1.0); reversal accuracy *increases* by
  0.02 (ρ = +1.0), the opposite sign, negligible magnitude.
- **P5 (frozen): MET** (reversal 0.12–0.14 at every width).
- **P6 (κ, amendment 4.4b): probe/h_u part MET in every seed** (probe gain
  means 0.022/0.087/0.210 for κ = 1, 1/16, 1/256; h_u 0.067/0.141/0.406;
  ρ(probe gain, κ) = −1.0 in seeds 0, 1, 2); **reversal part NOT MET**
  (means 0.121/0.123/0.143; ρ = −1.0, −1.0, −0.5; extension 1/4096,
  1/32768 on seed 0 did not reach L*, final reversal 0.184/0.208 with h_u
  0.69/0.70).
- Secondary (0.15): width trend in probe gain in every seed (ρ = −0.86 /
  −0.96 / −0.86); mup less suppressed at large M (0.104 vs 0.032 at 2048);
  reversal flat 0.06–0.08. Reported as labelled secondary analysis.
- Reading (for the paper and for Astra): the contextual pathway's speed
  relative to the encoder controls how much spectral structure the encoder
  learns before the fit; reliance at matched fit is governed by the head's
  gain for the large-amplitude cue and is removed only by an uninformative
  context; the probe measures recoverability by readout retraining.

### 4.5 Known limitations to state
Linear encoder (as in the theorem); one head family; full-batch GD; a
constructed context cue; matched-fit stopping is the theorem's convention,
not a training recipe; K/S sets the initial readiness asymmetry.

## 5. Experiment 3 — real centre spectra with constructed context (design; run after Exp 2)

Decision (Astra refocus_01 §3, adopted): **3B primary, 3A secondary.**

- **3B (primary; "real-centre-spectrum experiment with constructed context").**
  Astra's sentence: "We test contextual competition using measured centre
  spectra and their recorded labels, with a synthetic class-associated
  spatial intensity pattern along a training-estimated dominant spectral
  direction in the neighbourhood." Data: breast QCL fold-0 patient split
  (`/mnt/hdd2/u37314kd/data_breast_v2_pca23/splits_fold0.json`: 115 train /
  28 val / 26 test cores; per-core NPZ with X_raw/X_d1/X_d2 (H, W, 314), y in
  0–4, tissue_mask). Proposed task: BINARY, the preselected pair Cancer
  Epithelium (label 3) vs Cancer-Associated Stroma (label 4) — the contrast
  used in the v1 anisotropy measurements [author to confirm; the 4-class
  variant with a stated assigned-class transition matrix is the extension].
  Construction (Astra's minimum requirements): preprocessing (per-channel
  centering/scaling on the 942-dim raw+d1+d2 vector) and the dominant
  direction v₁ estimated on TRAINING patients only and frozen; the pattern is
  added AFTER preprocessing so it survives it; each centre spectrum is
  preserved with its recorded label; the eight neighbour spectra are real
  donors sampled independently of the centre label (from training-side
  donors for training patches, held-out-side donors for evaluation patches;
  never training donors in evaluation patches); the constructed cue is
  γ(code(y_centre)·s + τη)v₁ per neighbour with s = +1 (informative), s = −1
  (reversed), and code of an independent random label (uninformative
  control, same marginal amplitude/noise); γ, τ calibrated on training-side
  calibration data so that the neighbourhood oracle matches the centre-only
  oracle (LDA on the preprocessed centre spectrum); if they cannot be matched,
  report the imbalance and weaken the "equally predictive" claim. Readiness
  table as in Exp 2 (raw centre oracle, random-encoder centre probe,
  random-encoder neighbourhood probe). Model: linear encoder 942→K, ReLU CNN
  3×3 head, single logit, plain full-batch GD (or large-batch SGD) with one
  global rate; arms sp widths, headlr κ at one width, normalized readout,
  ctxfree, frozen; paired initialization within width; matched training
  loss; paired context interventions on held-out patients (val and test
  cores reported separately). Failure = excess shifted risk relative to a
  separately trained spectral-only comparator at the same threshold (nonzero
  on real spectra), not the theorem's zero/one dichotomy. Patient/seed
  variation is the replication unit.
- **Readiness scan (run 2026-09-10 22:40, `results/exp3/readiness_scan.csv`,
  fold-0 training cores split by core into fit/eval halves, three random
  encoders):** with per-feature standardization, a random encoder already
  exposes the epithelium-versus-stroma contrasts — CancerEpi vs CAS: oracle
  0.969, random probe 0.73 (K=2), 0.80 (K=4), 0.955 (K=12), 0.968 (K=32);
  NormalStroma vs CancerEpi: 0.989 vs 0.78/0.87/0.978/0.990 — because the
  class contrast lies along the top principal directions (|cos(contrast,
  v₁)| 0.50 and 0.88; top-10 PC fraction 1.00). With PCA whitening
  (estimated on the training half) the random probe drops to chance while
  the oracle stays high — CancerEpi vs CAS: oracle 0.944, random probe
  0.50/0.49/0.54/0.59 for K = 2/4/12/32. NormalStroma vs CAS (standardized):
  oracle 0.81, random probe 0.58–0.74. NormalEpi vs CancerEpi: oracle 0.76
  (9k training pixels), weak task.
  **Design decision (two regimes, same real spectra and labels, pair
  CancerEpi vs CAS, K = 12):** *ready* = standardized inputs (random probe
  0.955): the theorem's a₀ ≥ m/2 case — prediction: no suppression-driven
  failure, the joint model's shifted risk does not exceed the spectral-only
  comparator's at matched fit; *unready* = PCA-whitened inputs (random probe
  0.54): the a₀ small case — prediction: suppression at matched fit,
  reversal-risk excess over the comparator, modulated by the whole-head rate
  κ. The constructed cue rides on the first principal coordinate in both
  regimes (γ = 3√λ₁ standardized; γ = 30 in whitened units), with τ
  calibrated per regime to match the centre-only oracle. Whitening is a
  standard preprocessing choice; the pair is the clinically decisive one
  used in the v1 anisotropy measurements. [Author to confirm the binary
  pair; Astra to check the two-regime logic.]
  **Refinement after the 300-step smoke runs (23:40):** eight cue
  directions (top eight principal coordinates, one per neighbour offset, as
  in Exp 2) replace the single direction, because a random head's responses
  to one shared direction add coherently and the untrained model already
  predicted from context (reversal 0.25 at step 0); with eight directions the
  initial loss is 0.66–0.68 and initial reversal 0.38–0.39. Recalibrated:
  ready γ_d = 3√λ_d (46.7 … 12.9), τ = 1.47, oracles 0.969/0.969, random-
  encoder centre probe 0.94/0.97/0.95; unready γ = 30, τ = 1.74, oracles
  0.948/0.948 (val 0.884), random-encoder centre probe 0.48/0.51/0.60,
  patch 0.95. **Interpretation of the ready regime, corrected:** the
  encoder there already exposes the spectral cue (probe 0.91 at
  initialization), so there is no encoder learning to suppress; the smoke
  run shows the head nevertheless predicts from context at loss 0.19
  (context-random accuracy at chance) — head-level feature competition
  between a large-amplitude cue and an O(1) cue, which the theorem does not
  address. The ready regime is therefore a CONTROL that separates the two
  levels: prediction = probe ≈ its initial value at every κ (no encoder
  suppression to remove); reversal failure may persist and is not evidence
  about the encoder mechanism. The unready (whitened) regime is the primary
  test of the theorem's encoder-level mechanism: prediction = probe gain at
  L* suppressed at κ = 1 relative to `ctxfree`, increasing as κ decreases.
  Grid (launched 23:45): per regime sp M ∈ {8, 32, 128, 512}; headlr κ ∈
  {1, 1/16, 1/256} at M = 32; ctxfree M ∈ {8, 128}; frozen M = 32; seeds
  0–2; full-batch GD, η = 1e-3, 24,000 balanced training patches, probe fit
  on 6,000 disjoint training pixels, evaluation on up to 6,000 val and
  6,000 test centres (paired conditions iid / reversed / ctx_random /
  spec_only).
- **Amendment (2026-09-11 03:00, before the runs): regime `unready10`.**
  The whitened regime at γ = 30 shows absolute encoder suppression at
  every width and every κ (probe gain ≤ +0.05 against an initial 0.56–0.60;
  reversal 0.07; fits in 69–185 steps even at κ = 1/256), because the cue's
  amplitude makes it dominate the encoder's own gradient. To test the
  amplitude-ratio reading on real spectra, a third regime repeats the
  whitened construction with γ = 10 (Experiment 2's amplitude scale; τ
  recalibrated to equal oracles), arms sp M ∈ {32, 512}, headlr κ ∈
  {1, 1/16, 1/256} at M = 32, ctxfree M = 8, frozen M = 32, three seeds.
  Prediction: with the smaller amplitude the whole-head rate recovers its
  effect on spectral learning (probe gain increasing as κ decreases, as in
  Experiment 2); reliance is not expected to change.
  **Outcome (2026-09-11 05:00, 18 runs at M = 32):** prediction MET — probe
  gain at L* increases with a slower head in every seed (κ = 1 / 1/16 /
  1/256: seed 0 0.021/0.046/0.051; seed 1 0.132/0.170/0.180; seed 2
  0.076/0.155/0.170; means 0.076/0.124/0.134), fits take 530–1,800 steps
  (vs ~100 at γ = 30); reversal 0.06–0.08 unchanged; ctxfree gain 0.35,
  reversal 0.91–0.92; frozen 0.07–0.09. Kept in the main text as one
  sentence (Sec 5) with its table in Appendix D.
- **3A (secondary, if resources permit; result retained whatever it shows).**
  Astra's sentence: "We construct a contextual-shift benchmark from measured
  centre spectra and measured neighbour spectra, assigning neighbour classes
  to control their association with the centre label; the resulting
  neighbourhoods are assembled rather than naturally observed tissue
  neighbourhoods." Measure the same readiness table first: if A already
  shows the accessibility difference, it is the more natural transfer test;
  if not, a null A result tests a different regime.

## 6. Schedule (abstract Sep 18, paper Sep 25)

- Sep 10–11: Exp 2 calibration + pilot (1 seed) + full grid; Astra reviews
  the design and restates the theorem (§7).
- Sep 12–13: Exp 2 v2 if needed (nonlinear encoder; SGD check); v2 main text
  §1–3; `ICLR_SKELETON_v3.md`.
- Sep 12–16: Exp 3 pilot then grid (GPU on this machine; ≤ 18 concurrent jobs).
- Sep 17: v2 abstract final → register Sep 18.
- Sep 18–22: full v2 draft, four figures, trimmed appendix; Astra review 02.
- Sep 23–24: corrections; Sep 25 submit.

## 6a. Adopted from Astra's refocus_01 (2026-09-10 21:30)

- §3 theorem: Astra's `thm:serial` statement (three parts + special-init
  remark; 0.44 page) verbatim; proof reuse as described; interpretation rules:
  the spectral-only comparator is trained to ITS OWN first hitting time; the
  frozen comparator is evaluated at the joint path's elapsed times through
  T_m; **normalization removes the width dependence, it does not promise
  recovery** (the normalized system is the M = 1 system, which still fails
  under reversal for (0,1) initialization) — never write that part (c)
  eliminates shortcut reliance; no phase-attribution implication; no "factor
  of six"; no "10⁻¹¹ for every quantity" — numerical verification in its own
  scoped table.
- §2 definition-of-failure paragraph: Astra's draft verbatim (refocus_01
  end of §3).
- Exp 2: §4.4b above. Exp 3: §5 above (B primary).

## 6b. Response to Astra's review 02 (2026-09-11 05:00–08:00)

- **Recovery experiment (the "single change"):** `code/experiments/
  exp2_recovery.py`, protocol fixed before any shifted result was read
  (frozen L* encoders of κ = 1 and κ = 1/256 plus the random initial
  encoder, M = 32, three seeds; fresh head with paired initialization;
  20,000 GD steps on newly sampled context-random images; final state
  evaluated on the original paired test family). Result: retrained
  reversal accuracy 0.84/0.77/0.75 from the κ = 1/256 encoders vs
  0.62/0.59/0.54 from κ = 1 and 0.60/0.59/0.53 from random initialization;
  original classifiers 0.12–0.15 in both arms; no retraining reached loss
  0.10 (final 0.33–0.47 vs 0.65–0.68). Real-spectra version
  `exp3_recovery.py` (three regimes) running; γ = 30 whitened encoders
  recover 0.45–0.56 for all three encoders.
- **Update projections** (`update_projections.py`): Exp 3 R_C
  0.99995–0.99998 (γ = 30), 0.9996–0.9998 (γ = 10), ctxfree 0.88–0.90; Exp 2
  R_V 0.95–0.98, R_u 0.013–0.08 (ctxfree R_u 0.88–0.92). Reported in §5 as
  the measured fact; the amplitude account is a supported hypothesis.
- **Registered verdicts** recomputed on the original five widths (Astra's
  table reproduced: P1 not met 1/3; P2 alignment/probe flatness met,
  reversal not met; P2b met; P3 both requirements fail; P4 alignment met,
  reversal opposite; P5 met; P6 probe/h_u met, reversal 2/3) → Appendix C
  criterion table; §4 paragraph = Astra's suggested text.
- **γ = 10 paired rate effect:** +0.030/+0.049/+0.095 (γ = 10) vs
  +0.002/+0.023/+0.016 (γ = 30) within seeds; reversal unchanged; kept as
  one sentence in §5 with its table in Appendix D.
- **All 32 ledger items** addressed in the text (finite-width reversal
  condition everywhere; abstract split by regime; no binary
  recoverability; base rate + declared block multipliers; matched
  calibration discriminant; comparator defined by its training
  intervention; synthetic reversal changes the field at every pixel; 5×5
  receptive field; "substantially less accessible"; per-seed initial
  values; baseline-only wording for the width summaries; budget endpoints
  separated and labelled in Fig 3; no toy margin decomposition for the CNN;
  probe = accessibility on clean spectral inputs; no theorem-case
  identification for the real regimes; patch readiness; heterogeneous
  gains with the 0.10 run named; γ = 10 wording; excess risk 0.83–0.84;
  production paragraph numbers corrected; appendix repairs).
- Page fit: main text ends exactly at the bottom of page 9 after trims to
  §1, §3, §4, §5, §6 and the figure captions.

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

## 8. Experiment 4 — natural context, usage test (design recorded 2026-09-11 ~11:30, BEFORE any Stage B run)

**Why.** The author's motivating question (O'Leary et al. 2026, Anal. Chem.
98:2743; Müller et al. 2023, Analyst 148:5022, Mosig group) is the
inference "in-distribution accuracy of a spatial–spectral model is
insensitive to spectral compression (to ~16 features), therefore the
spectral dimension is largely redundant". Exp 3 shows the premise of that
inference holds for a model that uses almost no spectral information
(random and trained encoders tie in distribution; the model is at chance
without context) but with CONSTRUCTED context. Exp 4 tests the same
inference with NATURAL neighbourhoods, four classes (standing rule), on
the full-rank spectra. It is a usage test, not a mechanism test: no cue
calibration, no matched-SNR construction, so it does not reopen the
design argument that led to constructed context in Exp 3.

**Data.** `/mnt/hdd2/u37314kd/data_breast_v2_nodenoising` (full rank;
the Exp 3 cache came from the PCA-23-denoised copy, whose per-core
singular values vanish beyond 23 — DISCLOSE in Appendix D). Fold 0,
patient-level split (115/28/26 cores), identical split file. Natural
3×3 patches: centre labelled, all nine pixels in the tissue mask,
neighbours as recorded (labelled or not). Per core per class caps
150/100/100 (train/val/test), seed 1234. Code
`code/experiments/exp4_natural_context.py` (cache; Stage A), results
`results/exp4/`.

### 8.1 Stage A — premise check (a GATE, run first; not a registered prediction)
Per-pixel classifiers on the CENTRE spectra as a function of the number
of principal components k ∈ {2,4,8,16,23,32,64,128,256,942} (PCA on
standardized training centres): shrinkage LDA, balanced logistic
regression, and a 256-unit ReLU MLP (fixed 40-epoch schedule, three
seeds, no selection on evaluation patients). Metrics: four-class accuracy,
macro-F1, per-class F1 on validation and on test patients. Also recorded:
neighbourhood homogeneity (fraction of patches whose eight neighbours all
share the centre label).

- **Gate G1:** the best per-pixel classifier at k = 942 exceeds the same
  classifier at k = 16 by ≥ 0.02 macro-F1 on validation AND on test. If G1
  passes, the components beyond sixteen carry per-pixel information a
  classifier can use, and Stage B's compression arms are meaningful.
- **If G1 fails:** report it as such — on breast QCL fold 0, per-pixel
  classifiers of these three families do not gain beyond sixteen
  components — which SUPPORTS the "sixteen suffices" reading for
  per-pixel information on this data. Stage B then runs only the usage
  arms (P8, P9), and the O'Leary/Müller paragraph must say the
  sixteen-component claim was not contradicted here.

### 8.2 Stage B — joint training on natural patches (pre-registered)
Model as Exp 3: linear encoder 942→12 (or k→12 in compression arms, the
projection applied to all nine pixels), PatchHead ReLU MLP over the nine
encoded spectra, width M = 32, four logits, class-weighted mean CE
(weights ∝ inverse class frequency, normalized to mean one; amended from
"mean CE" at 11:50, before any registered run, because the per-core caps
leave the classes unbalanced and Stage A's classifiers are balanced),
full-batch GD, one
global rate (Exp 3 defaults), budget 40,000 steps, paired initialization
within seed; three seeds. Matched fit at L* = 0.30 with a fallback: if any
arm does not reach 0.30 in budget, the primary comparison is at the
highest threshold in {0.30, 0.40, 0.50} all arms reach, and budget-end
values are reported separately and labelled (review-02 rule).

Arms (train side): `nat` natural context; `nat16` natural context, inputs
projected to the top 16 PCs; `shuf` comparator: neighbours replaced by
pixels drawn from other training patches independently of the centre
label (uninformative context, jointly trained); `shuf16`; `frozen`
random encoder, head trained on `nat`; `nat` with κ = 1/256 (head rate
multiplier, one extra arm, secondary).

Evaluation (validation patients; test reported separately), each centre
held fixed: `iid` natural patches; `ctx_random` neighbours replaced by
pixels from other validation cores with independent labels;
`homogeneous` all nine pixels = centre (secondary). Probe: LDA on the
encoder output of centre spectra (accessibility), at init and at L*.
Recovery: fresh paired-init head retrained on `ctx_random` TRAINING
patches from the frozen encoder (protocol of exp2_recovery.py, 20,000
steps), evaluated on `ctx_random` validation. Per-pixel oracle O(k) from
Stage A is the reference.

Predictions (macro-F1, validation; evaluated at matched fit and at budget
end):
- **P7 (their observation reproduced at the patch level):** |iid(`nat`)
  − iid(`nat16`)| < 0.02. Ceiling caveat registered now: if both exceed
  0.97 the comparison is uninformative and is reported as such.
- **P8 (usage):** ctx_random(`nat`) ≤ O(942) − 0.10 while ctx_random
  (`shuf`) ≥ O(942) − 0.03. (Natural neighbours are almost always
  same-class, so part of this drop is legitimate noise averaging; P8
  alone does not establish a shortcut. P9 and P10 carry the claim.)
- **P9 (where the information stopped):** two registered readings.
  (a) HEAD-LEVEL: the head retrained on the frozen `nat` encoder reaches
  ≥ ctx_random(`shuf`) − 0.03 → the encoder passed the centre
  information and the readout ignored it (Exp 3 high-accessibility
  pattern). (b) ENCODER-LEVEL: it falls short by > 0.03 while `shuf`
  itself reaches ≥ O(942) − 0.03 → the jointly trained encoder never
  learned to pass what the comparator's encoder learned (the theorem's
  suppression, now with natural context). Either is "unused, not
  unnecessary". The outcome that supports the redundancy reading is
  ctx_random(`shuf`) ≈ ctx_random(`shuf16`) together with a failed G1.
- **P10 (compression asymmetry, only if G1 passes):** the comparator is
  sensitive where the natural-context model is not: iid(`shuf`) −
  iid(`shuf16`) ≥ 0.6 × [O(942) − O(16)], while P7 holds for `nat`.
  Equivalently, the components beyond sixteen are exploited only when the
  context cannot be leaned on.
- **Secondary (κ = 1/256):** probe gain and recovered macro-F1 of `nat`
  at κ = 1/256 exceed those at κ = 1 in every seed (as Exp 2/3).

Disposition rules: P7+P10 met → the paper states, as an empirical claim on
real tissue, that a spatial–spectral model's insensitivity to spectral
compression coexists with usable, recoverable information beyond sixteen
components. P7 met, P10 not met, G1 passed → the joint model and the
comparator are both insensitive; report that the K = 12 bottleneck or the
head family does not exploit the fine components, and keep the paragraph
logical, not empirical. P7 not met → O'Leary's observation does not
reproduce at 3×3; report as a scale limitation. Nothing is dropped;
whatever the verdict, Stage A and B numbers go to Appendix G.

### 8.3 Stage A outcome (recorded 2026-09-11 ~12:10; Stage B grid launched at 12:00, before this was read)
Cache: 19,650 / 3,700 / 2,700 natural 3×3 patches (train/val/test) from
79 / 20 / 16 labelled cores (54 of the 169 cores carry no label of the four
classes in either data copy). Neighbourhood homogeneity: no 3×3 patch
straddles two labelled classes (0.0%); 88% of neighbours share the centre
label, 12% are unlabelled. Natural 3×3 context is therefore redundancy of
the centre's class signal, not a separate cue. Top 16 PCs carry 81.7% of
the standardized variance (23: 86.5%; 64: 97.7%).

**G1 PASSED.** Per-pixel macro-F1 at k = 16 → 942 (validation / test):
LDA 0.560→0.651 / 0.538→0.645; logistic 0.595→0.673 / 0.571→0.658;
MLP 0.640→0.705 / 0.594→0.677. Gains 0.065–0.107 on both sides, all
≥ 0.02. Also recorded: the best k is 128 (MLP val 0.717); k = 23 (the
field's PCA-denoising rank) gives 0.654 (MLP val), so the standard
denoising itself discards per-pixel-usable information here; and
components 9–16 reduce TEST macro-F1 for all three classifiers (k = 8 →
16: LDA 0.596→0.538, MLP 0.621→0.594) before later components recover
it — a non-monotonicity to report, not hide. Stage B's compression arms
are meaningful; P10 is live. Single fold, three classifier families,
patient-level split; no claim about other cohorts.

### 8.3a Correction to 8.3 (recorded 12:25): the denoising remark was wrong
The same per-pixel sweep on the PCA-23-DENOISED copy (Exp 3's four-class
cache, 20,000 training pixels; per-core denoising, so the pooled data are
NOT rank 23 — singular values 22–25 are 0.11–0.12 of the largest):
MLP macro-F1 (val / test) k = 8: 0.598 / 0.606; 16: 0.670 / 0.637;
23: 0.710 / 0.697; 64: 0.783 / 0.782; LDA 16 → 64: 0.550 → 0.690 /
0.541 → 0.692. So (i) per-core denoising IMPROVES per-pixel
classification (0.783 vs 0.705 at the best k) — the 8.3 sentence "the
standard denoising itself discards per-pixel-usable information" is
withdrawn; (ii) G1 holds on the denoised copy too, and more strongly
(+0.113 MLP val from 16 to 64 global components), because per-core
subspaces differ and the pooled data need more than 23 global
components. Consequence: a replication of Stage B on the denoised copy
(the field-standard preprocessing, closer to O'Leary's pipeline) is
worth running with the same arms and predictions — see 8.4 if launched.

### 8.4 Replication on the PCA-23-denoised copy (pre-registered 12:40, before its cache or any run)
Same cores, split, sampling caps, seed, model, arms, budget, evaluation
conditions, recovery protocol and predictions P7–P10 as 8.2, with the
data copy switched to `data_breast_v2_pca23` (the companion pipeline's
field-standard preprocessing; per-core PCA denoising to 23 components).
Rationale (8.3a): per-pixel classification is stronger on this copy and
the gain beyond 16 global components is larger, so it is the copy on
which the O'Leary/Müller inference is best tested. Code: `--copy pca23`
in `exp4_natural_context.py` and `exp4_train.py`; results
`results/exp4_pca23/`. The non-denoised results of 8.2 remain the
primary pre-registered set; agreement between the copies is reported
as such, disagreement is reported as a preprocessing dependence.
Stage A on this copy is re-run with the same script for the table
(the 8.3a numbers came from Exp 3's pixel cache with a 20,000-pixel
subsample and one MLP seed).

### 8.5 Amendment (recorded 12:55, before these arms run): what O'Leary et al. actually varied
Verified from the abstract (Manchester Research Explorer) and the authors'
code (`~/Projects/avpn/third_party_oleary/src/{models,utils}.py`): the
neural-network "spectral bottleneck of just 16 features" is a LEARNED
1×1 convolution (BatchNorm → Conv2d(input_dim, 16, 1) → BatchNorm)
trained end to end with the model, compared against "fixed" = no
reduction (BatchNorm only); PCA bottlenecks are used only for the
classical models. Their stated conclusion: "tissue classification itself
is characterised by only a small set of spectral features", plus a strong
correlation between spatial receptive field and performance. So the
author's paraphrase "16 PCA or less is enough" and the v1 triage note
"frozen compression, not joint training" are both inaccurate: it is a
learned, jointly trained bottleneck. Consequences for Exp 4:
- Stage A (PCA-k) tests the paraphrase, not their claim; P10's contrast
  (`nat16` vs `nat`) is a PCA contrast.
- The exact analogue of their comparison is a learned K-dim bottleneck
  (`nat`, K = 12) versus NO bottleneck. New arms `natfull` and `shuffull`:
  identity encoder (frozen), head over the nine raw standardized spectra
  (9 × 942 → M = 32 → 4). Three seeds, same budget and evaluation.
- **P11 (their observation in its own form):** |iid(`nat`) −
  iid(`natfull`)| < 0.02 at budget end, and |ctx_random(`shuf`) −
  ctx_random(`shuffull`)| < 0.03: a learned 12-dim linear bottleneck
  loses nothing relative to no bottleneck. If met together with G1
  (PCA-16 loses 0.07–0.11), the paper can state the distinction that
  answers the author's question: the class information is low-dimensional
  as a LEARNED linear subspace (which any discriminant is, for four
  classes), not as a small number of principal components; a small learned
  bottleneck draws on the whole spectrum, so "a small set of spectral
  features" is a compressibility statement, not evidence that the spectral
  dimension is redundant.

### 8.6 Registered verdicts, non-denoised copy (recorded 13:05; Stage B 18 runs + 13 retraining runs; validation macro-F1, mean over three seeds, budget end = 40,000 steps unless stated)
Matched-fit rule: no threshold in {0.30, 0.40, 0.50} is reached by every
arm (`frozen` floors at loss ≈ 0.61, `shuf16` at ≈ 0.53; capacity, not
budget), so the PRIMARY comparison is budget end, with matched values at
0.5 (nat, nat16, headlr, shuf) and 0.4 (nat, shuf) reported alongside.
Per-pixel reference O(942) = 0.705 (MLP), O(16) = 0.640.

| arm | loss | probe (init 0.52) | iid | ctx-random | homogeneous | retrained head, ctx-random |
|---|---|---|---|---|---|---|
| nat | 0.19 | 0.672 | 0.732 | 0.234 | 0.705 | 0.681 (from init 0.494) |
| nat16 | 0.42 | 0.574 | 0.657 | 0.201 | 0.620 | 0.594 |
| shuf | 0.34 | 0.670 | 0.652 | 0.680 | 0.648 | 0.688 |
| shuf16 | 0.53 | 0.557 | 0.541 | 0.583 | 0.540 | — |
| frozen | 0.63 | 0.521 | 0.576 | 0.263 | 0.540 | — |
| headlr κ=1/256 | 0.49 | 0.619 | 0.646 | 0.305 | 0.630 | 0.644 |

- **P7 NOT MET** at budget end: iid nat 0.732 vs nat16 0.657 (Δ +0.075;
  test 0.652 vs 0.611, +0.041): the natural-context model is SENSITIVE
  to PCA-16 compression. At matched loss 0.5 it is not (0.643 vs 0.636),
  because the full-spectrum model keeps fitting (0.19) where PCA-16
  plateaus (0.42). Ceiling caveat not triggered.
- **P8 MET**: ctx-random nat 0.234 ≤ 0.605; shuf 0.680 ≥ 0.675 (marginal).
- **P9 reading (a), HEAD-LEVEL, MET**: retrained head on the nat encoder
  0.681 vs shuf 0.680; probes equal (0.672 / 0.670). The encoder is not
  starved: with natural context it learns the class signal exactly as
  without. A random K = 12 encoder retains much less (0.494 after
  retraining), so the bottleneck must be LEARNED to keep the information.
- **P10 MET numerically** (shuf − shuf16 = 0.111 ≥ 0.039) but the
  asymmetry reading FAILS because P7 is not met: BOTH the comparator and
  the natural-context model lose from PCA-16.
- **Secondary (κ = 1/256) NOT MET**: probe 0.619 < 0.672, recovery 0.644
  < 0.681; at matched 0.5, 0.616 vs 0.632. Slowing the head buys nothing
  when the context is a redundant copy of the spectral cue.
- Disposition (8.2 rules): "P7 not met → O'Leary's observation does not
  reproduce at 3×3" — superseded in part by 8.5: their observation is
  about a LEARNED bottleneck, tested by P11 (`natfull`/`shuffull`,
  running). What stands: natural 3×3 context is redundancy of the centre's
  class signal (8.3), so the paper's competition mechanism does not
  operate here — the encoder learns equally, the model uses the full
  spectrum, and its collapse under context-random is the head's use of
  same-class neighbours. Report as a scope result: the competition needs a
  contextual cue that is not the spectral cue replicated (larger receptive
  fields, morphology), which Exp 4 does not test.

## 9. Public hyperspectral replication of Experiments 3 and 4 (pre-registered 2026-09-11 14:40, before any cache-dependent run)

**Why.** The closest accepted papers (Gradient Starvation, NeurIPS 2021;
Modality Competition, ICML 2022; DFR, ICLR 2023) each rest on a
nonlinear-network theorem, a method, or public benchmarks; our real data
are private. A public scene lets a reviewer rerun Exp 3 (constructed
context, mechanism) and Exp 4 (natural context, usage) end to end.

**Scene.** ROSIS Pavia University (610×340×103, nine classes, 42,776
labelled pixels; `.mat` files as distributed by the EHU scenes page,
fetched from the HybridSN GitHub mirror; SHA-256 prefixes PaviaU
28447fa8, PaviaU_gt 23f6a426). Indian Pines (145×145×200, 16 classes)
is downloaded and is a second scene only if time allows.

**Spatial split (patient analogue).** 20×20-pixel tiles, assigned at
random (seed 0, first feasible attempt = 0) to train/val/test in the
proportion 0.6/0.2/0.2 (208/70/70 labelled tiles), with every class
having ≥ 60 admissible pixels per split; a pixel is admissible only if
its whole 3×3 neighbourhood lies inside its tile. Per tile and class, up
to 400/200/200 pixels (train/val/test), seed 1234. Residual spatial
autocorrelation across tile borders is stated as a limitation. Code
`code/experiments/hsi_data.py`; caches `results/exp3_paviau/`,
`results/exp4_paviau/`.

**Preprocessing.** Counts cast to float; per-feature standardization and
PCA fitted on the training split by the experiment code (ready =
standardized, unready = PCA-whitened with the Exp 3 ridge); no per-pixel
normalization, denoising or derivatives (the IR pipeline is
domain-specific and has no analogue here).

**Exp 3 analogue.** Binary pair fixed now: **Meadows (+1) versus Trees
(−1)** (the two largest vegetation classes, a standard confusion in this
scene; 8,347 / 1,322 admissible training pixels). Secondary pair if time:
Gravel versus Self-Blocking Bricks. Sizes scaled to the pair: up to 2,400
balanced training patches, 400 probe-fit pixels, evaluation on all
validation and test pixels of the pair (≤ 6,000). Everything else as
§5 / Appendix D: readiness scan (descriptive, run first), two regimes if
the scan shows the same accessibility asymmetry (standardized random-
encoder probe high, whitened low; the rule is the §5 rule), eight cue
directions, τ calibrated per regime to the centre-only discriminant,
arms sp {8, 32, 128, 512}, headlr κ ∈ {1, 1/16, 1/256} at M = 32,
ctxfree {8, 128}, frozen 32, three seeds, L* = 0.30, paired conditions,
retraining protocol. Predictions: those of §4.4/§5 as amended (P1–P6 read
with Astra's interpretation rules), evaluated exactly as in §4.4d and
Appendix C/D. In particular: informative context suppresses the
encoder's probe gain relative to ctxfree at L* in the unready regime;
reversal accuracy of informative-context arms is far below ctxfree's;
slowing the head raises the probe gain monotonically in κ within seeds
but does not remove the reversal failure; retraining recovers more from
slow-head encoders than fast-head ones where the cue is accessible.

**Exp 4 analogue.** All nine classes, macro-F1; Stage A gate G1 and Stage
B arms, predictions P7–P11 and disposition rules exactly as §8.1, §8.2,
§8.5 (k ∈ {2,4,8,16,23,32,64,103}; no second data copy). The registered
expectation from Exp 4 on tissue is carried over as a prediction here:
natural 3×3 context is redundancy of the centre's class signal, so P7
is expected NOT to be met, P8 met, P9 head-level, the κ prediction not
met, and P11 met; if instead P7 is met and P10 holds, the scene differs
from the tissue result and both are reported.

Disposition: nothing is dropped; results go to Appendix H with two
sentences in the main text (one in §5, one in §6).

### 8.7 P11 verdict, non-denoised copy (recorded 15:05; `natfull`/`shuffull`, three seeds, budget end)
**P11 MET.** iid macro-F1: natural context with the learned 12-dim
bottleneck 0.732 (0.723–0.741) vs no bottleneck 0.744 (0.739–0.749),
|Δ| = 0.012 < 0.02 (test 0.652 vs 0.667); context-random: shuffled
comparator 0.680 vs no bottleneck 0.661, |Δ| = 0.019 < 0.03 (test 0.628
vs 0.616). The no-bottleneck model reaches training loss 0.10 by
29,000–31,000 steps where the bottleneck model is at 0.19 at budget end,
so the equality is not a fitting artefact. Together with G1 (PCA-16
loses 0.065–0.107 per pixel) and P7-not-met (the spatial model loses
0.04–0.08 from PCA-16), the distinction of 8.5 is established on this
data: the class signal is low-dimensional as a LEARNED linear subspace
and not as a small number of principal components; a learned bottleneck
draws on the whole spectrum, so compressibility is not redundancy.

### 9.1 Pavia descriptive stages (recorded 15:20, before any registered run)
Readiness scan (`results/exp3_paviau/readiness_scan.csv`), Meadows vs
Trees: standardized oracle 0.933, random K = 12 probe 0.850 [0.836,
0.875]; whitened oracle 0.955, random K = 12 probe 0.667 [0.623, 0.702];
|cos(contrast, v1)| = 0.91. The §5 rule gives the same two regimes as on
tissue (ready = standardized, unready = whitened), with a smaller
asymmetry than breast (0.955 vs 0.54 there). Other pairs scanned for the
record: Gravel–Bricks 0.780/0.612, Asphalt–Bricks 0.922/0.650,
Meadows–BareSoil 0.837/0.729 (standardized/whitened K = 12 random probe).

**Stage A, G1 FAILS on Pavia.** Per-pixel macro-F1 (nine classes) at
k = 16 → 103, validation / test: LDA 0.767→0.781 / 0.742→0.759;
logistic 0.816→0.830 / 0.831→0.841; MLP 0.891→0.911 / 0.876→0.885. The
best gain is 0.020 on validation and 0.009 on test, below the 0.02
threshold on the test side. On this scene sixteen principal components
carry essentially all per-pixel information available to these
classifiers — the OPPOSITE of the tissue result (0.065–0.107), to be
reported as such (ROSIS bands are strongly correlated; 16 PCs carry
most of the variance). Disposition per §8.1: the compression arms
`nat16`/`shuf16` are dropped on Pavia; Stage B runs the usage arms
`nat`, `shuf`, `frozen`, `headlr` and the no-bottleneck arms
`natfull`/`shuffull` (P8, P9, secondary κ, P11).

## 10. Experiment 2, nonlinear-encoder arm (pre-registered 2026-09-11 15:50, before any run)

**Why.** Every experiment so far has a linear encoder; the most predictable
reviewer question is whether the suppression, the reliance and the rate
response survive a nonlinear one.

**Design.** Encoder = two-layer ReLU network S → 64 → K (fc1 N(0, 1/S)
with zero bias, fc2 N(0, 1/64)); everything else as §4 (same generator,
calibration τ, heads, one global rate, L* = 0.30, paired conditions,
three seeds). Arms: `nlenc` informative context at M ∈ {8, 32, 128, 512};
`nlenc_ctxfree` uninformative context at M ∈ {8, 128}; `nlenc_headlr`
κ = 1/256 at M = 32; `nlenc_frozen` random encoder at M = 32. The linear
alignment measures (a_u, h_u) are undefined for this encoder and are
recorded as NaN; the probe (LDA on encoder output of spectral-only
data), the paired shifted accuracies and the retraining protocol
(exp2_recovery.py, encoders nl_init / nl_kappa1 / nl_kappa256, 20,000
steps on fresh context-random images, paired head initialization) carry
the claims. Readiness of the random nonlinear encoder is recorded at
initialization (not tuned). Code: `exp2_intervention.py` (MLPEncoder,
ENC_HIDDEN = 64, arms), `exp2_recovery.py`.

**Predictions (validation of the same claims as §4.4/§4.4b, evaluated at
L* unless stated):**
- **P12 (suppression persists):** probe gain of `nlenc` is below that of
  `nlenc_ctxfree` at M = 8 and M = 128 in every seed.
- **P13 (reliance persists):** reversal accuracy of `nlenc` ≤ 0.5 at
  every width and seed; `nlenc_ctxfree` ≥ 0.75.
- **P14 (rate response persists):** probe gain at κ = 1/256 exceeds that
  at κ = 1 (M = 32) in every seed, while mean reversal accuracy changes
  by less than 0.05.
- **P15 (recovery ordering persists):** retrained reversal accuracy from
  the κ = 1/256 encoder exceeds that from the κ = 1 encoder in every
  seed, and the κ = 1 encoder yields no more than the random encoder
  plus 0.05.
Disposition: nothing dropped; results go to Appendix C with one clause in
§4 ("with a two-layer ReLU encoder the same pattern holds / does not
hold"), stating whichever the verdicts support.

## 11. Seed extension 3 → 5 for Experiments 2 and 3 (pre-registered 2026-09-11 16:00, before any seed-3/4 run)
Seeds 3 and 4 are added to every registered cell of Exp 2 (sp, mup,
ctxfree, headlr, lrmult, frozen) and Exp 3 (all three regimes, all
arms). Rule: the REGISTERED verdicts (§4.4d, §5 amendments) remain those
computed on seeds 0–2 and are not recomputed; the five-seed means and
spreads are reported alongside as a robustness extension, and any
prediction whose direction fails to hold in the added seeds is reported
as such (per-seed tables). The three-seed summaries are preserved as
`results/exp2/PREDICTIONS_3seeds_v1.md`, `results/exp2_summary_3seeds_v1.csv`,
`results/exp3_summary_3seeds_v1.csv`. Recovery runs for the added seeds
follow the same protocols.

## 12. Five-fold extension of Experiments 3 and 4 on tissue (pre-registered 2026-09-11 16:10, before any fold-1..4 run)
Folds 1–4 of the companion's five-fold patient-level split
(`splits_fold{1..4}.json`, same cores, one core per patient) are run with
exactly the fold-0 protocols: Exp 3 (three regimes, all arms, seeds 0–2,
per-fold calibration of τ and per-fold preprocessing, per-fold readiness
recorded), Exp 4 Stage A and Stage B (non-denoised copy, usage arms and
compression arms, seeds 0–2) and both retraining protocols. Rule: fold 0
remains the registered fold; the registered verdicts are not recomputed.
The five-fold values give the patient-split variation of every reported
contrast (mean and range over folds), and any contrast whose sign is not
the same in all five folds is reported as fold-dependent. Code: the
`EXP_FOLD` environment switch in `exp3_real_context.py` and
`exp4_natural_context.py`; results `results/exp3_fold{f}/`,
`results/exp4_fold{f}/`. Nothing is dropped.

### 8.8 Registered verdicts, PCA-23-denoised copy (8.4 replication; recorded 17:10; 24 runs + 16 retraining runs; validation macro-F1, three seeds)
Stage A on the patch cache: G1 PASSES more strongly than on the
non-denoised copy: k = 16 → 942, validation / test: LDA 0.574→0.695 /
0.543→0.807; logistic 0.598→0.773 / 0.586→0.858; MLP 0.665→0.804 /
0.634→0.883 (test patients are easier than validation on this copy).
Stage B at budget end (nat, natfull, shuffull reach the 0.10 stop; nat16
and headlr plateau at 0.37, shuf16 at 0.52, frozen at 0.52):

| arm | probe (init 0.53) | iid | ctx-random | test iid | retrained head, ctx-random |
|---|---|---|---|---|---|
| nat | 0.671 | 0.727 | 0.274 | 0.839 | 0.706 (from init 0.503) |
| nat16 | 0.582 | 0.683 | 0.187 | 0.640 | 0.595 |
| natfull | 0.756* | 0.744 | 0.280 | 0.802 | — |
| shuf | 0.676 | 0.703 | 0.722 | 0.763 | 0.720 |
| shuf16 | 0.569 | 0.564 | 0.588 | 0.556 | — |
| shuffull | 0.756* | 0.717 | 0.715 | 0.755 | — |
| frozen | 0.529 | 0.594 | 0.261 | 0.638 | — |
| headlr κ=1/256 | 0.683 | 0.697 | 0.299 | 0.678 | 0.702 |
(* the identity encoder's probe is a 942-dim discriminant.)

- **P7 NOT MET** (as on the other copy): iid nat 0.727 vs nat16 0.683,
  Δ 0.044 in every seed (test 0.839 vs 0.640); at matched loss 0.5 and
  0.4 the two are within 0.011.
- **P8 first clause MET** (nat 0.274 ≤ 0.704), **second clause NOT MET**:
  shuf 0.722 < O(942) − 0.03 = 0.774. The jointly trained M = 32 head
  with plain GD stays 0.08 below the per-pixel MLP oracle on this copy
  (and so does the no-bottleneck comparator, 0.715), so this is the head
  family / optimizer, not the bottleneck.
- **P9 HEAD-LEVEL MET**: retrained 0.706–0.712 ≥ 0.722 − 0.03; probes
  equal (0.671 / 0.676); random encoder 0.503.
- **P10 numerically MET** (shuf − shuf16 = 0.139 ≥ 0.083) but the
  asymmetry reading FAILS again (P7 not met).
- **P11 MET on validation**: |0.727 − 0.744| = 0.017; |0.722 − 0.715| =
  0.007. On test the bottleneck model is 0.037 ABOVE the no-bottleneck
  one (0.839 vs 0.802) — reported, not registered.
- **Secondary κ NOT MET**: probe 0.683 vs 0.671 (2/3 seeds higher, one
  lower); retrained 0.702 vs 0.706.
Conclusion: the two preprocessing copies agree on every registered
verdict; the denoised copy shows larger per-pixel gains beyond sixteen
components and a larger test-side compression effect for the spatial
model (0.20).

### 9.2 Pavia Exp 4 analogue, Stage B verdicts (recorded 17:40; 18 runs; nine-class validation macro-F1, three seeds; retraining pending a rerun)
Neighbourhood homogeneity and per-pixel reference: O(103) = 0.911 (MLP),
O(16) = 0.891. Budget end (40,000 steps; no arm reaches the 0.10 stop;
`nat`/`natfull` reach 0.3, `shuf`/`shuffull`/`frozen` only 0.5, `headlr`
κ = 1/256 stays at loss 0.84, i.e. barely trained):

| arm | probe (init 0.715) | iid | ctx-random | homogeneous | test iid |
|---|---|---|---|---|---|
| nat | 0.732 | 0.840 | 0.144 | 0.728 | 0.847 |
| natfull | 0.790* | 0.836 | 0.152 | 0.740 | 0.871 |
| shuf | 0.722 | 0.594 | 0.627 | 0.599 | 0.688 |
| shuffull | 0.790* | 0.648 | 0.654 | 0.651 | 0.735 |
| frozen | 0.715 | 0.703 | 0.169 | 0.668 | 0.785 |
| headlr κ=1/256 | 0.702 | 0.512 | 0.090 | 0.532 | 0.547 |
(* identity encoder: 103-dim discriminant.)

- **P7**: not evaluated (compression arms dropped by the 8.1 rule after
  G1 failed).
- **P8 first clause MET** (nat 0.144 ≤ 0.811; nine-class chance ≈ 0.11),
  **second clause NOT MET**: shuf 0.627 ≪ 0.881. The M = 32 head with
  plain GD is a weak nine-class per-pixel classifier here (loss 0.49 at
  budget end; the no-bottleneck comparator 0.654), so the within-family
  comparison (P9) carries the usage claim, not the oracle comparison.
- **Natural context helps a lot on Pavia**: iid 0.840 (nat) vs 0.594
  (shuf): the same-class neighbours (homogeneity to be read from
  neighbourhood_stats.json) raise accuracy by 0.25 in distribution —
  larger than on tissue (0.08).
- **Encoder not starved** (as on tissue): probes 0.732 (nat) vs 0.722
  (shuf), from 0.715 at init — the standardized inputs put this scene in
  the high-accessibility regime, where the linear bottleneck barely needs
  to move.
- **P11 MET**: |0.840 − 0.836| = 0.004; |0.627 − 0.654| = 0.027 < 0.03.
- **Secondary κ NOT EVALUABLE at matched fit** (the κ = 1/256 arm never
  reaches 0.5 in budget) and NOT MET at budget end (probe 0.702 < 0.732).
- **P9**: pending the retraining rerun (the first run crashed on the
  dropped compression arm; loop now skips missing encoders).
- **P9 (Pavia), recorded 18:20 after the retraining rerun: HEAD-LEVEL
  MET.** Retrained head on the `nat` encoder: ctx-random 0.629
  (0.625–0.634; at L* 0.619) vs the `shuf` comparator's 0.627 at budget
  end (its own retrained head 0.615); random encoder 0.559; κ = 1/256
  encoder 0.620. Test side: 0.710 (nat) vs 0.699 (shuf), 0.609 (init).
  The natural-context encoder passes the class information as well as
  the comparator's; the collapse under context-random evaluation is the
  head's use of same-class neighbours, exactly as on tissue. Pavia
  summary: G1 fails (16 PCs suffice per pixel), P8 first clause met,
  P9 head-level, P11 met, κ not evaluable/not met, encoder not starved,
  natural context worth +0.25 in distribution.

### 10.1 Registered verdicts, nonlinear encoder (recorded 18:40; 24 runs, all reached L*; retraining for P15 running)
Readiness of the random two-layer ReLU encoder: spectral-only probe
0.635 at initialization (linear encoder: 0.64), so the accessibility
asymmetry is the same. At L* = 0.30 (validation of the paired test
family):
- **P12 MET** (6/6): probe gain with informative context +0.022 /
  +0.003 / +0.001 (M = 8) and +0.016 / +0.008 / +0.020 (M = 128) against
  the uninformative-context comparator's +0.286 / +0.264 / +0.271 and
  +0.276 / +0.256 / +0.258.
- **P13 MET**: reversal accuracy of every informative-context run
  ≤ 0.128 (all widths, all seeds); comparator ≥ 0.879.
- **P14 MET** (3/3): probe gain at M = 32 rises from +0.020 / +0.020 /
  +0.016 (κ = 1) to +0.189 / +0.210 / +0.142 (κ = 1/256); mean reversal
  change +0.018 (< 0.05).
- P15: pending (`exp2_recovery.py nl_init nl_kappa1 nl_kappa256`).
Note for the tables: the collect step now sees the seed-3/4 runs that
§11 is adding; the REGISTERED Exp 2 verdicts stay on seeds 0–2
(`PREDICTIONS_3seeds_v1.md`), and `make_tables_exp2.py` must restrict the
criterion table to seeds 0–2 and report five-seed means separately.
- **P15 MET** (recorded 19:20; 9 retraining runs, 20,000 steps, none
  reached the 0.10 target): retrained reversal accuracy from the
  κ = 1/256 encoders 0.681 / 0.699 / 0.650 vs κ = 1 0.572 / 0.566 / 0.568
  vs random initialization 0.579 / 0.580 / 0.580 (probe of the frozen
  encoders 0.78–0.85 vs 0.64–0.66 vs 0.62–0.64). Ordering holds in every
  seed; the κ = 1 encoders yield no more than random ones. Magnitudes are
  smaller than for the linear encoder (0.75–0.84 vs 0.54–0.62 there): the
  fresh linear-readout head extracts less from the ReLU encoder's
  features within the budget (retraining loss 0.56–0.61 vs 0.67). All
  four nonlinear-encoder predictions (P12–P15) are met.

## 13. Abstract v4 candidate (drafted 19:35; NOT applied to `00_abstract.tex`; for the author and Astra; registration deadline Sep 18)
Changes from v3: "linear or two-layer ReLU encoder" in the synthetic
sentence; one added sentence on natural neighbourhoods (tissue + public
scene, the compressibility/redundancy distinction); the real-spectra
sentence shortened to pay for it. Numbers as in the appendices.

> We study how relative module training rates affect spectral adaptation
> in serial spectral–spatial models, in which a small encoder compresses
> each pixel's spectrum and a wide spatial network reads the encoded
> neighbourhood. An exactly solvable serial logistic model bounds the
> spectral encoder's update at matched fit by a fraction 1/(1+Mv₀²) of
> the update a spectral-only model needs, establishes contextual-reversal
> failure under explicit finite-width conditions, and shows through a
> M^{-1/2}-normalized readout that replicated-readout width acts through
> the training metric; normalization removes the width dependence but
> need not prevent the failure. In a nonlinear synthetic model with two
> cues of matched signal-to-noise ratio, with a linear or a two-layer
> ReLU encoder, slowing the spatial head by 256× raises the linear
> accessibility of the spectral cue at matched training loss from 0.66
> to 0.85 while the trained classifier remains strongly context-reliant
> (reversal accuracy 0.12 to 0.14); retraining the head on decorrelated
> context from the frozen encoders then recovers 0.75–0.84 shifted
> accuracy from the slow-head encoders against 0.54–0.62 from the
> fast-head ones, no better than random encoders. With measured tissue
> spectra and constructed context, strong contextual reliance appears at
> high and at low initial spectral accessibility, with a weaker rate
> response at high contextual amplitude. With natural neighbourhoods, on
> tissue and on a public hyperspectral scene, no competition arises: the
> neighbours carry the centre's class, the encoder learns it equally
> with or without them, and a learned twelve-dimensional bottleneck loses
> nothing where sixteen principal components lose per-pixel information.
> The results separate adaptation of the encoder, accessibility of the
> spectral cue, reliance of the fitted classifier, and what a specified
> retraining recovers.

### 9.3 Pavia Exp 3 analogue, registered verdicts (recorded 19:50; 78 runs, ALL reached L* = 0.30; validation of the pair Meadows vs Trees; retraining pending)
Init probe (whitened, K = 12): 0.664 / 0.703 / 0.755 by seed (mean 0.707);
standardized 0.915 / 0.923 / 0.909.

| regime | arm | probe gain | reversal | ctx-random | steps to L* |
|---|---|---|---|---|---|
| unready γ=30 | sp M 8–512 | −0.002…+0.001 | 0.07–0.12 | 0.47–0.52 | 15–85 |
| | headlr κ 1 / 1/16 / 1/256 | −0.001 / 0.000 / +0.002 | 0.07–0.08 | 0.47–0.48 | 42–250 |
| | ctxfree M 8 / 128 | +0.239 / +0.223 | 0.88 / 0.85 | 0.87 / 0.86 | 9,100–10,200 |
| | frozen | 0 | 0.07 | 0.48 | 53 |
| unready10 γ=10 | sp | +0.001…+0.008 | 0.05–0.07 | 0.48–0.51 | 103–416 |
| | headlr κ 1 / 1/16 / 1/256 | +0.004 / +0.021 / +0.029 | 0.05–0.06 | 0.48 | 283–1,900 |
| | ctxfree M 8 / 128 | +0.240 / +0.231 | 0.84 / 0.90 | 0.87 / 0.90 | 8,900–10,100 |
| ready (standardized) | sp | ≈ 0 | 0.23–0.26 | 0.53–0.57 | 431–1,219 |
| | headlr κ 1 / 1/16 / 1/256 | ≈ 0 | 0.23 / 0.23 / 0.19 | 0.54 / 0.54 / 0.51 | 1,066–13,524 |
| | ctxfree M 8 / 128 | +0.004 / +0.003 | 0.85 / 0.88 | 0.86 / 0.88 | 4,800–7,500 |
| | frozen | 0 | 0.22 | 0.54 | 1,418 |

- **Suppression at matched fit (unready): MET** in every seed and width:
  informative-context probe gain ≈ 0 against +0.22–0.24 for the
  comparator, which needs 100× more steps to reach L*.
- **Reliance / failure: MET** in all three regimes: reversal 0.05–0.26
  for every informative-context arm against 0.84–0.90 for ctxfree.
- **Rate response: NOT MET at γ = 30** (within-seed change κ 1 → 1/256:
  +0.023 / −0.009 / −0.006; monotone in 1/3 seeds) and **PARTIAL at
  γ = 10** (+0.073 / −0.001 / +0.005; monotone in 2/3): only the seed
  with the lowest initial accessibility (0.664) responds; the seeds
  starting at 0.703 and 0.755 gain nothing. Weaker than tissue
  (+0.030 / +0.049 / +0.095 at γ = 10). Reversal unchanged (≤ 0.01).
- **High accessibility (ready): failure without a probe deficit**, as on
  tissue (gain ≈ 0 everywhere; reversal 0.19–0.26 vs 0.85–0.88).
- Recovery ordering: pending `exp3_recovery.py` (running).
- **Recovery (Pavia), recorded 20:25 (27 retraining runs, 20,000 steps):**
  ready regime: retrained reversal 0.885 (init) / 0.887 (κ=1) / 0.907
  (κ=1/256) — from EVERY encoder, random included, against 0.19–0.26 for
  the original classifiers (tissue: 0.90–0.96); κ=1/256 > κ=1 in every
  seed (+0.014 / +0.020 / +0.028) → ordering MET where the cue is
  accessible, with a small margin. Unready γ=30: 0.660 / 0.658 / 0.649 —
  no encoder yields more than the random one (per-seed κ256 − κ1: +0.012
  / −0.021 / −0.018) → ordering NOT met, as expected where the probe did
  not respond. Unready10 γ=10: 0.687 / 0.687 / 0.697 (per seed +0.060 /
  −0.032 / +0.001) → 1/3, the responsive seed only. Pavia Exp 3 summary:
  suppression MET, reliance MET, high-accessibility failure-without-
  deficit MET with full recoverability, rate response and recovery
  ordering NOT met at γ=30 and PARTIAL (one seed) at γ=10.

## 14. Astra review-03 prompt (paste-ready; drafted 20:35; PDF of commit 3596a9c, SHA-256 prefix 70502959; the seed/fold extensions will change only the added tables)

Astra — revision after your review 02 is in paper/main_iclr_v2.pdf (sources
paper/sections_iclr_v2/; plan paper/REFOCUS_PLAN_2026-09-10.md §8–§13 records
every pre-registration, amendment and verdict with timestamps). Since review 02:
(1) Your single change is done: exp2_recovery.py (protocol fixed before any shifted
result; init / κ=1 / κ=1/256 encoders at L*, paired head init, 20,000 GD steps on
fresh context-random images): retrained reversal 0.84/0.77/0.75 after κ=1/256 vs
0.62/0.59/0.54 after κ=1 and 0.60/0.59/0.53 from random init; the real-spectra
version (27 runs): high accessibility 0.90–0.96 from every encoder incl. random,
γ=30 0.45–0.56 for all, γ=10 0.55–0.70 (slow) vs 0.48–0.60 (fast). Update
projections exported and reported in §5 as a supported hypothesis; registered
verdicts recomputed on the original five widths (Appendix C criterion table); all
32 ledger items applied.
(2) NEW Experiment 4 (Appendix E; plan §8): natural 3×3 neighbourhoods on breast
QCL fold 0, four classes, two preprocessing copies. Gate: per-pixel classifiers gain
0.07–0.11 macro-F1 beyond 16 PCs (0.14–0.25 on the denoised copy). Registered
verdicts on both copies: P7 (spatial model insensitive to PCA-16) NOT met at budget
end (0.73 vs 0.66; insensitive only at matched loss); P8 reliance met; P9 head-level
(encoder probes equal with/without informative context, retrained head recovers the
comparator's level); P10 asymmetry fails; P11 MET (learned 12-dim bottleneck = no
bottleneck, 0.732 vs 0.744); κ prediction NOT met. Reading: natural 3×3 context is a
redundant copy of the centre's class (no patch straddles classes; 88% same-label), so
the competition mechanism does not operate there — stated as a scope result in §5/§6.
(3) O'Leary et al. 2026 verified from the abstract and their code: the "16-feature
bottleneck" is a LEARNED 1×1 conv trained end to end, not PCA; the intro now says
compressibility is not redundancy (Müller et al. 2023 added).
(4) NEW public replication on Pavia University (Appendix F; plan §9): tile-disjoint
split; readiness scan gives the same two regimes; Exp 3 analogue (78 runs, all at
L*): suppression and reliance replicate in full; rate response and recovery
ordering NOT met at γ=30, partial (one seed) at γ=10; high-accessibility regime
fails without a probe deficit and recovers 0.89–0.91 from every encoder. Exp 4
analogue: per-pixel gate FAILS (16 PCs suffice there), P8 first clause met, P9
head-level, P11 met; the M=32 plain-GD head is a weak nine-class classifier (stated).
(5) NEW nonlinear encoder in Exp 2 (Appendix C; plan §10): two-layer ReLU encoder;
P12–P15 ALL met in every seed (suppression, reliance, rate response, recovery
ordering; recovered levels 0.65–0.70 vs 0.57 vs 0.58).
(6) Running: seeds 3–4 for Exp 2/Exp 3 (registered verdicts stay on seeds 0–2; plan
§11) and folds 1–4 for Exp 3/Exp 4 (plan §12; Appendix D subsection wired).
(7) Main text re-fitted to nine pages; §6 replaces "we recommend no mitigation" with
the conditional statement the experiments support; abstract v4 candidate in plan §13
(not applied; registration Sep 18).
Please write review_packet/astra/review_iclr_03.md: the reviewer report as before
(score, ranked weaknesses with page/line, the single most valuable change), a
sentence-level ledger of anything still stronger than the files support, and four
specific checks: (a) is the natural-context scope result stated correctly in §5/§6
and Appendix E, and does it undercut the paper's claim more than we say; (b) is the
compressibility-vs-redundancy sentence in §1 the strongest defensible statement
against the "small set of spectral features" reading of O'Leary et al.; (c) does the
Pavia replication (Appendix F) support calling the mechanism reproducible, given the
failed rate prediction at γ=30; (d) should the abstract v4 candidate replace v3 for
registration. Do not edit paper files.

### 11.1 Seed extension outcome, grids (recorded 20:55; Exp 2 60/60 and Exp 3 60/60 added runs; retraining for the added seeds running)
Registered verdicts unchanged (`PREDICTIONS.md` regenerated on seeds
0–2 is byte-identical in its verdict section; all-seed evaluation in
`PREDICTIONS_allseeds.md` for the record). Directions on seeds 3 and 4:
- **Exp 3 unready, suppression:** probe gain sp (mean over widths) +0.026
  / +0.061 vs ctxfree +0.329 / +0.462 (seeds 0–2: −0.004 / +0.031 /
  +0.020 vs 0.35 / 0.34 / 0.34). Holds 5/5.
- **Exp 3 reliance:** reversal sp 0.069 / 0.065 vs ctxfree 0.906 / 0.919
  (unready); ready 0.108 / 0.088 vs 0.913 / 0.924. Holds 5/5.
- **Exp 3 κ response, paired 1 → 1/256:** γ = 30: +0.002 / +0.022 /
  +0.015 / +0.032 / +0.054 (positive 5/5; larger in the added seeds);
  γ = 10: +0.030 / +0.048 / +0.094 / +0.086 / +0.082 (5/5). The §5
  statement "small at γ = 30, larger at γ = 10" holds with five seeds
  (means +0.025 vs +0.068).
- **Exp 2 P1 (width criterion ρ ≤ −0.8 for a_u AND reversal):** a_u ρ =
  −0.7 / −1.0 / −1.0 / −0.9 / −0.7; reversal ρ = −0.6 / −0.9 / −0.6 /
  +0.7 / −0.4 → still not met (2/5 for a_u; 0/5 for both). **P2b** (a_u at
  2048, normalized > standard): 5/5. **P5** frozen reversal max 0.149
  (< 0.5). ctxfree probe 0.887–0.943 at every width. **P6 κ** (M = 32):
  probe gain 0.028→0.223, 0.025→0.203, 0.014→0.205, 0.008→0.225,
  0.005→0.221 (monotone 5/5). Five-seed mean κ = 1/256 probe gain 0.215
  (three-seed 0.211).
Main text keeps the registered three-seed numbers; one sentence each in
§4 and §5 will say the two added seeds agree (Appendices C, D).
- **Exp 2 retraining, added seeds (recorded 21:35):** retrained reversal
  from κ=1/256 encoders 0.808 / 0.774 (seeds 3 / 4) vs κ=1 0.600 / 0.567
  vs random init 0.605 / 0.561 → the ordering of §4 holds 5/5 (five-seed
  means 0.789 / 0.583 / 0.578). Exp 2 tables regenerated with five seeds
  (criterion table on seeds 0–2).
- **Exp 3 retraining, added seeds (recorded 21:55; 18 runs):** ready
  regime 0.888–0.928 from every encoder (seeds 3–4: init 0.888 / 0.925,
  κ=1 0.921 / 0.909, κ=1/256 0.926 / 0.928) → "recovers from every
  encoder" holds 5/5. Unready γ=30: 0.523→0.571→0.593 (s3), 0.472→0.471→
  0.483 (s4): little recovery from any encoder, κ256 ≥ κ1 in 5/5 but by
  ≤ 0.02. Unready10 γ=10: κ256 0.755 / 0.621 vs κ1 0.636 / 0.486 vs init
  0.517 / 0.472 → ordering holds 5/5 (five-seed ranges 0.55–0.76 vs
  0.48–0.64; the main text keeps the registered three-seed 0.55–0.70 vs
  0.48–0.60). Exp 3 tables regenerated with five seeds.
