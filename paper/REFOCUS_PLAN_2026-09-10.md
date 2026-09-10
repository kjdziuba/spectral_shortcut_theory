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
margin, i.e. κMv₀² < m/(m−2a₀). The protocol's extension rule (extend
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
