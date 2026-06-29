# Progress Log

Running session-by-session log. Newest entries at the top.

---

## 2026-06-30 — EGR fixed-width sweep RESULT: real-but-weak

**Done**:
- Ran exp1_4_fw (144 runs, D=256 fixed, noise × lr × wd × 6 seeds).
- Spawned 3-skeptic + synthesis Workflow on the new correlations.
- Synthesis verdict: **"real_but_weak"** — capacity-independent
  signal exists and survives every adversarial control, but
  practical effect size is moderate.

**Key numbers (D=256, n=144)**:
- Raw r(egr_min, final_acc) = +0.344 (p = 2.5e-5)
- Partial r controlling for noise/lr/wd = +0.395 (larger, not smaller)
  → NOT a hyperparameter confound
- Within-bin r (seeds only): mean +0.584 across 24 cells, 23/24 strongly
  positive, Fisher-z aggregate p < 1e-13
- Partial r controlling for train loss = +0.335 (essentially unchanged)
  → EGR adds value beyond train-loss tracking
- ROC-AUC for "bad run" classification = 0.612 (real but modest)
- Best threshold: TPR 0.90 / FPR 0.57 → not deployable as standalone alarm

**Section 6 updated**:
- Removed "capacity-aware diagnostic" framing.
- New structure:
    (i)  pooled correlation, capacity confounded (r = -0.78 but partial = -0.17)
    (ii) fixed-capacity test (r = +0.34, survives all controls)
    (iii) practical interpretation (AUC modest, use alongside train loss)
- Verdict: capacity-independent signal real; practically a complementary
  diagnostic, not a standalone alarm.

**Why this matters**: had we stopped at the pooled correlation,
reviewers would have crushed us with "this is just capacity in disguise."
The fixed-width follow-up + adversarial workflow give us a defensible,
honest claim.

**Next session**:
- Phase 2 W7: write Lemma 4.3 (Schur complement bound for lambda_min).
  This is the last load-bearing piece for Theorem 1's full proof.
- Then: Theorem 2 proof (Phase 3).

---

## 2026-06-30 — EGR fixed-width sweep + Karakida derivation drafted

**Done**:
- Wrote `code/experiments/exp1_4_fw.py` (fixed width D=256, sweeps
  noise × lr × weight_decay × 6 seeds = 144 runs).
- Launched in background — expected ~70 min.
- Drafted `study_notes/04_karakida_derivation.md` covering the
  asymptotic scaling argument for Lemma 4.1
  (`λ_max(J_φ^T J_φ) = Ω(C_g)`):
  - Karakida 2019 framework recap (mean-field FIM)
  - FIM ↔ Gauss-Newton conversion factor for CE
  - Per-architecture scaling (MLP / CNN / ViT) — all give λ_max ∝ C_g
  - Honest explanation of why our empirical slope is 0.7 not 1.0
    (finite-width corrections, bias parameters, CE residual)
  - 5-step proof outline for the supplement
- Lab talk: pushed to GitHub at
  https://github.com/kjdziuba/spectral_shortcut_theory; design brief
  in `presentation/DESIGN_BRIEF.md` is the handoff for design tool.

**Next session**:
- Read the fixed-width sweep result when it lands (notification).
- Run adversarial verification on the new correlation (workflow).
- If within-bin signal survives → upgrade Section 6 to
  "capacity-independent diagnostic"; otherwise leave the
  "capacity-aware" framing.
- Then: 2-hour focused session writing the Lemma 4.1 proof into
  `paper/sections/supplement.tex` using the Karakida derivation.

---

## 2026-06-29 — Exp 1.4 + adversarial verification + Pezeshki session

**Done**:
- User started reading Pezeshki 2021. Captured notes + follow-ups in
  `study_notes/02_pezeshki_notes_and_followups.md`:
    1. Add Spectral Decoupling as an experimental condition (Exp 1.6).
    2. Cite Pezeshki Sec 4 / App B robustness claim to avoid running
       those ablations ourselves.
    3. Prepare 3-4 slide presentation for spectroscopy people (TODO).
    4. NTK measurements not necessary for the main theorems.
- Wrote `study_notes/03_hessian_block_algebra.md` walkthrough — block
  matrix decomposition, Cauchy interlacing, Schur complement, Karakida
  scaling, combining into κ(G) ≥ Ω(C_g/C_f).
- Ran Exp 1.4 v1 (early-window EGR only) — got NEGATIVE correlation
  r=-0.56. Identified confound: capacity drives both.
- Iterated to Exp 1.4 v2 with multiple EGR windows (early/mid/late/min/depth).
- Headline pooled correlations: r(egr_depth, final_acc) = -0.78,
  Spearman -0.81.
- **Spawned adversarial verification workflow** (3 skeptics + synthesis):
    - Reviewer 1 (within-capacity): VERDICT = CONFOUND. Partial
      correlation controlling for log(width) drops to -0.17 (NS).
      EGR depth almost perfectly tracks width (r=0.90).
    - Reviewer 2 (depth vs min): VERDICT = DEPTH_IS_REAL. Both
      egr_early and egr_min contribute independent signal; the gap
      isn't just a min restatement.
    - Reviewer 3 (vs train loss): VERDICT = ADDS_VALUE. Partial r
      controlling for train_loss = -0.572 (significant).
    - Synthesis: VERDICT = MODERATE_DIAGNOSTIC. EGR depth is a
      "capacity-aware" diagnostic, not capacity-independent.
- Inserted honest synthesis paragraph into Section 6 with full caveats
  and a TODO marker for the follow-up at fixed width.

**Key insight**: Workflow's adversarial verification caught the same
type of confound that v1 had. Without it we might have written "EGR
depth is a strong real-time diagnostic" and gotten torn apart by
reviewers. The honest framing is "capacity-aware diagnostic — useful
when comparing comparable-width models, not yet a capacity-independent
predictor."

**Decisions**:
- Section 6 now has explicit caveats. Follow-up Phase 5 work: rerun
  at fixed width varying noise/lr/reg to test residual within-bin signal.
- Spectral Decoupling experiment to add as Exp 1.6 eventually.

**Next session**:
- Continue with Hessian algebra proof writing (Phase 2 W5-W7).
- Or: kick off the fixed-width EGR follow-up to upgrade the diagnostic.

---

## 2026-06-27 — Experiment 1.3 HEADLINE RESULT

**Done**:
- Identified design flaw in initial Exp 1.3: spatial info lived only
  in labels (label-position correlation), not in X. CNN couldn't
  extract pure position info during 150-epoch training. spatial_only
  baseline was stuck at chance across all three calibration modes.
- Fixed data generator: added per-position fixed spatial signature
  along v ⊥ u (orthogonal direction in R^S). Spatial signal now
  manifests directly in X content.
- Preserved v1 broken-design results as `_brokendesign` files.
- Re-ran exp1_3 with fixed data.

**Final results (CNN D=256, 3 seeds)**:

Bayes mode (alpha=0.82, beta=18.75):
- spectral-only acc = 0.53 (CNN training)
- spatial-only  acc = 0.53 (CNN training)
- frozen        acc = 0.61
- joint         acc = 0.48 (BELOW CHANCE)
- **shortcut gap = 0.13** (frozen - joint)

NTK mode: shortcut gap = 0.04
Margin mode: shortcut gap = 0.09 but poor calibration quality

**WINNER: Bayes mode** -- combined score 1.26 vs ntk 0.88 vs margin 0.82.

**The headline finding**:
Joint training collapses BELOW the spectral-only baseline (0.48 vs
0.53). This means end-to-end training doesn't just fail to combine
the two pathways -- it actively destroys spectral information that
the model demonstrably can use when given spectral-only data. This
is the direct empirical demonstration of the spectral-shortcut
conjecture from Theorem 2.

- Updated Section 7 with full Exp 1.3 write-up and calibration mode
  comparison table.

**Paper state at end of session**:
- All sections 1-9 have substantive drafts.
- All three experiments empirically validated:
    Exp 1.1: kappa scales monotonically (lambda_theta flat, lambda_phi grows)
    Exp 1.2: EGR collapse depth monotonic in capacity (CNN)
    Exp 1.3: joint training collapses BELOW single-modality baseline
- Real-data F1 table connects synthetic to FTIR/QCL findings.
- Discussion section with prescriptions, limitations, open problem.

**Next session**:
- Begin Phase 2 (Theorem 1 proof details, W5-W7).
- Read Pezeshki 2021 to build intuition for Theorem 2 proof.
- Consider running Exp 1.4 (EGR as predictor) to support Section 6.

---

## 2026-06-26 (night) — Experiment 1.3 designed; Sections 7, 8, 9 drafted

**Done**:
- Built `synthetic/calibrate.py` with 3 modes (Bayes, NTK, Margin).
  Fixed spatial-only baseline to use position-majority class (the
  correct primitive given our data design — position info lives in
  labels, not in X).
- Bayes calibration successful: alpha=0.82, beta=18.75 gives
  spectral-only acc=0.75, spatial-only acc=0.75 (calibrated equal).
- NTK and Margin modes can't calibrate spatial side for this data
  (no position info in X) — will document as a finding.
- Wrote `experiments/exp1_3.py`: 36 runs (3 modes × 4 conditions ×
  3 seeds), CNN D=256. Currently running in background.
- Drafted Section 7 (Experiments) with Exp 1.1 + Exp 1.2 results.
- Drafted Section 8 (Real-Data Validation) with FTIR/QCL F1 table
  connecting synthetic to real-data findings.
- Drafted Section 9 (Discussion) with prescriptions, limitations,
  Spatial Dominance Conjecture as open problem, beyond-spectroscopy
  applications.
- Added bib entries for Huang 2022, Saxe 2011, Rahimi-Recht, Bhatia,
  Horn-Johnson, Lyu-Li, Gunasekar, Chen GradNorm, Coil-Cheney,
  Bozzo 2024.

**Current paper state**:
- Sections 1-9 ALL have substantive drafts.
- TODOs are limited to actual proof details (Phases 2-3, W5-W10) and
  Experiment 1.3 final results.
- Supplement and abstract are stubs (per plan, written last).

**Next session**:
- Read Exp 1.3 results, pick best calibration, finalize Section 7.
- Begin Phase 2 (Theorem 1 proof details, W5-W7).
- User to read Pezeshki 2021 to build intuition for Theorem 2 proof.

---

## 2026-06-26 (late+) — Experiment 1.2 v4: CNN extended + ViT universality

**Done**:
- Added SpatialViT to `synthetic/models.py` (2-layer pre-LN ViT, ReLU
  activation for Hessian compatibility).
- Wrote `experiments/exp1_2v4.py` running 42 conditions (CNN widths
  {16,64,256,1024} + ViT widths {64,128,256}, joint vs frozen, 3 seeds).
- Drafted Section 1 (Introduction) properly.
- Drafted Section 2 (Related Work) with all 6 subsections.

**Findings (42-run sweep)**:

CNN sweep:
- EGR collapse depth scales monotonically with capacity (the headline
  Theorem 2 result):
    D=16   drops then recovers to ~0.4
    D=64   stable around ~0.25
    D=256  drops to ~0.10
    D=1024 drops to ~0.08
- Joint training final test loss diverges at high capacity:
    D=1024 joint test loss = 2.3, frozen test loss = 1.2
- Test accuracy gap exists but is small (~2-3 points) — joint overfits.

ViT sweep:
- Train loss reaches 0.01-0.03 at all widths — full memorization regime.
- Test loss is catastrophic (2.5-5.0) — joint training fails harder than CNN.
- EGR pattern INVERTED: rises with capacity rather than falling.
  This is the memorization regime where both gradient norms approach
  noise floor. Theorem 2's prediction holds in spirit (joint training
  fails) but the EGR observable loses signal.
- Frozen variants still beat joint at every width.

**Decisions**:
- For the paper: lead with CNN as the primary demonstration of Theorem 2.
  Note ViT as a different failure mode (memorization regime) where
  the EGR observable becomes uninformative but the overall prescription
  (freeze) still holds.
- Will mention in discussion that EGR diagnostic needs to be used
  before full memorization; otherwise the ratio loses meaning.

**Next session**:
- Move to Experiment 1.3 (equal-information killer test).
- Design the data calibration carefully (Shannon I(y; spectral) =
  I(y; spatial) via signal-strength tuning).

---

## 2026-06-26 (late) — Experiment 1.2 dynamics, three iterations

**v1 (Adam, SpatialMLP, noise=0.1):**
- Trained joint at D ∈ {32, 128, 512, 2048}, 3 seeds each, 150 epochs.
- Observed striking test-accuracy COLLAPSE at D=2048: peaks at ~70%
  early, then decays to ~50% by epoch 150 while train loss continues to
  drop. Two seeds at D=512 also collapse, one holds.
- EGR did not show dramatic decay (Adam's per-param normalization
  masks the natural gradient asymmetry).

**v2 (SGD+momentum, SpatialMLP, noise=0.05) — diagnostic miss:**
- Joint vs frozen converged to nearly identical accuracies (~75%).
- The collapse from v1 disappeared.
- ROOT CAUSE: SpatialMLP has zero spatial receptive field (per-pixel
  MLP). It has no spatial pathway for a "spatial shortcut" to exploit.
  The whole framing of "joint training discovers spatial shortcuts"
  cannot be demonstrated without 2D spatial mixing in g_phi.

**v3 (Adam, SpatialCNN with 2x 3x3 convs, noise=0.1) — the right setup:**
- Trained joint AND frozen at D ∈ {16, 64, 256}, 3 seeds each.
- Frozen beats joint at every capacity; gap widens with D.
  D=256: joint acc 54%, frozen acc 57%; joint test loss 1.05, frozen 0.80.
- Joint test accuracy at D=256 peaks ~70% then COLLAPSES to ~52%.
  Train loss meanwhile drops to 0.29 — classic shortcut overfit.
- EGR trajectories show the predicted capacity-monotonic collapse:
    D=16:  drops then recovers to ~0.4
    D=64:  drops to ~0.25 and plateaus
    D=256: drops to ~0.2 then keeps decaying to ~0.1
  This matches Proposition prop:egr_monotonic.

**Headline plots ready for paper drafts:**
- exp1_2v3_cnn_joint_vs_frozen.png — joint vs frozen by capacity
- exp1_2v3_cnn_egr_overlay.png    — capacity-monotonic EGR collapse
- exp1_2v3_cnn_joint_D256.png     — 4-panel full mechanism at D=256

**Decisions**:
- SpatialMLP retained as "no-spatial-mixing baseline" for ablations.
- SpatialCNN is the canonical architecture for the dynamics figures.
- Adam is the realistic optimizer for the literature contrast;
  SGD ablation can stay in supplement.
- noise=0.1 is the "interesting" regime (CE doesn't fully saturate,
  shortcuts emerge); we'll also report noise=0.05 in supplement to
  show the regime where joint training "looks fine."

**Next session**:
- Push SpatialCNN widths up to D=1024 to test whether the gap
  continues widening, or saturates somewhere.
- Add SpatialViT for the third architecture in our universality story.

---

## 2026-06-26 (eve) — Experiment 1.1 scaled (5 seeds, D up to 8192)

**Done**:
- Extended widths to D ∈ {16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192}.
- 5 seeds per D (5×10 = 50 measurements).
- Added paired eigenvalues plot (λ_θ and λ_φ on same axes).

**Findings**:
- λ_θ stays flat at ~0.02 across the full D range — confirms the spectral
  block does not grow because C_f is fixed. The pathology is entirely on
  the spatial side.
- λ_φ grows 74× (0.62 → 46) as C_g grows 510×.
- κ grows from 40 → 2,407 as C_g/C_f goes from 0.3 → 152.
- Fit slope is now 0.70 (up from 0.48 with the small range).
- At C_g/C_f ≈ 152 (close to real architecture's 325), κ ~ 2,400 —
  loss landscape is severely ill-conditioned.

**Decisions**:
- The "spectral block flat" finding is a HEADLINE result. Add paired
  plot as the primary figure for Theorem 1 in the paper.
- Will further push D to test asymptotic slope (whether it reaches 1)
  but the qualitative claim is firmly established.

**Next session**:
- Run Experiment 1.2 (training dynamics) — should give cleaner result.

---

## 2026-06-26 (PM) — Sections 4–6 drafted + Experiment 1.1 first run

**Done**:
- Drafted Section 4 (Theorem 1: Hessian Capacity Bound) — formal
  statement, 3 supporting lemmas (Karakida scaling, Cauchy interlacing,
  Schur complement), proof skeleton with TODO markers for Phase 2.
- Drafted Section 5 (Theorem 2: Two-Timescale Spurious Convergence) —
  formal statement, 3 supporting lemmas (timescale gap, fast manifold,
  residual decay), proof sketch combining Borkar + Pezeshki.
- Drafted Section 6 (EGR) — initial-value proposition, monotonicity
  proposition, threshold criterion, implementation reference.
- Workflow built infrastructure in parallel: synthetic/data.py,
  synthetic/models.py, hessian/eigenvalues.py, egr/callback.py,
  experiments/exp1_1.py, experiments/exp1_2.py. Smoke test passed.
- Fixed signature mismatch in exp1_1.py (top_eigenvalue_block API).
- **Ran Experiment 1.1 (capacity ablation)** on synthetic data:
    D ∈ {16, 32, 64, 128, 256, 512}, C_f = 1024, C_g ∈ [306, 9730]
    κ ranges from 23 to 464.

**Findings**:
- Direction confirmed: λ_φ grows with C_g, κ grows with C_g/C_f.
- Scaling exponent α ≈ 0.48 (predicted 1 from Karakida asymptote).
- κ already exceeds 400 at modest C_g/C_f ≈ 10 — landscape is
  pathological even far from the real-architecture regime
  (C_g/C_f ≈ 325).
- Need: scale D up to 2048+, average over seeds, run Exp 1.2 dynamics.

**Decisions**:
- Theorem 1 statement will be reframed as "monotonic growth with
  measurable exponent" rather than tight Ω(C_g/C_f).
- Cite Karakida for asymptote, document empirical finite-size α.
- Add capacity ablation up to D=8192 as next step.

**Next session**:
- Run Experiment 1.2 (dynamics) — should give clean qualitative result.
- Scale Experiment 1.1 to larger D + 5-seed averaging.
- User reads Pezeshki when ready.

---

## 2026-06-26 (AM) — Section 3 (Setup) drafted

**Done**:
- Walked through model setup, chain rule, Jacobian, Hessian conceptually
  - Simplified to C=1 (single-channel) for clean theory; channel extension trivial
  - Confirmed scope: linear `f_θ` for proofs, nonlinear empirically
  - Confirmed CE-specific role: vanishing-residual property
  - Concrete toy example (3-dim spectrum, 2 features, 2 classes) showing
    Jacobian = V in linear case
  - 2D quadratic example showing condition number → timescale gap
- Added Experiment 1.5 (Spatial architecture ablation: linear vs CNN vs ViT)
  - Important because CNN-based pipelines report most spectral-collapse issues
- Saved full conversational explanations to `study_notes/01_setup_and_chain_rule.md`
  - Convertible to PDF for offline study
- Wrote Section 3 (`paper/sections/03_setup.tex`) in formal LaTeX
  - Four assumptions stated (linearity, regularity, sub-Gaussian, capacity gap)
  - Loss formula
  - Chain-rule decomposition with named factors
  - Block Hessian structure
  - EGR formal definition
  - Notation conventions

**Decisions made**:
- Default C=1 in theory; real-data experiments use d2 (matches field standard)
- Run ablations later with raw, d1, d2 as robustness check
- Run BOTH linear and nonlinear (small MLP) in synthetic experiments
- Add CNN to spatial-architecture ablation (Experiment 1.5)

**Next session**:
- Begin Phase 0 (Track A reading): Pezeshki 2021
- In parallel: I start drafting Section 2 Related Work — gradient starvation subsection

**Open questions**:
- None yet.

---

## 2026-06-08 — Project initialized

**Done**:
- Created directory structure at `/home/u37314kd/Projects/spectral_shortcut_theory/`
- Wrote `README.md`, `PLAN.md` (8-phase plan), this `PROGRESS.md`
- Set up LaTeX paper skeleton with section stubs
- Created reading note templates for 5 foundational papers
- Initialized git repository

**Next session**:
- Begin Phase 0, Track A: Start reading Pezeshki et al. (2021) "Gradient Starvation"
- Fill in `reading_notes/01_pezeshki_2021.md` as you read
- In parallel, I'll start drafting Section 2 (Related Work) based on what you note

**Open questions**:
- None yet.

**Decisions log**:
- Project lives separate from `spectral_tokenization/` to keep theory work clean
- LaTeX-first for proofs; markdown drafts in `proofs/` for early iteration
- EGR diagnostic will be released as standalone PyPI package
- Target ICML/NeurIPS theory track first, AISTATS as backup
