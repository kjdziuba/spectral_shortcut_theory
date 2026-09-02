# Round-3 review triage (2026-09-02)

Reviewers: Fable (21 findings, deep), ChatGPT (truncated ~F9), Gemini (4).
Verdicts below verified against artifacts before acting.

## Verified against our artifacts
- **Best-val inversion (Fable F2, ChatGPT F8, Gemini F2): CONFIRMED, decisive.**
  e3c_analysis_runs.csv, best-val selection: joint − frozen_random = +0.033
  (h48) / +0.041 (h192); joint − frozen_pca = +0.104 (h48) / −0.004 (h192).
  Final-5 gaps were a selection artifact. SURVIVES: post-peak degradation
  (joint 0.87→0.54 h48 vs frozen_pca 0.77→0.73) and the SGD equivalence,
  which is METRIC-ROBUST (best-val 0.655 vs 0.654). Section 8.4 rewrite:
  dual-metric tables, headline = trajectory/drift + regime dissociation,
  NOT "frozen beats joint".
- **K=64/128 confound (Fable F1): REFUTED.** All nine April 5fold arms have
  reduce_dim=128 (configs pulled from cluster); E3c local arms uniform
  K=64. Paper fix only: caption the capacity table so K=64 row = O'Leary
  default config, not the compared baseline.
- **FTIR padding (all three): PARTIALLY HANDLED.** CoreDataset pads labels
  with 255 = ignored in loss+metrics (verified). Residual: zero-tokens in
  attention/BN as position cue (Islam et al. 2020) — caveat + optional
  fold-0 crop control.

## Actions taken (2026-09-02)
- Cluster arms added (agent, gated on smoke 19676080): joint_speclr0.1,
  joint_speclr10 (mechanism discriminator, prediction logged),
  ft_real_speclr0.1, lpft (frozen->unfreeze, Kumar et al. 2022 protocol),
  seeds {1,2} fold0 x 4 pivotal arms, LR sensitivity {3e-6,3e-5} fold0 x 2
  arms. ~27 jobs. Drift logging extended to all joint arms.
- Local laneSGD2 running: frozen_pca:48:{0,1,2}:sgd (positive control),
  joint_linear/frozen_random:192:{0,1,2}:sgd (width robustness).
- Pre-registered ordering for LR arms: harmful-drift predicts
  speclr0.1 >= joint >= speclr10; starvation predicts reverse.

## Adopted refutation commitments (Fable's, verbatim intent)
- P1 (finetune_real < frozen-MLP): refuted if >= 3/5 folds >= frozen-MLP
  and NB-corrected CI excludes −0.02 harm margin → report as prediction
  not supported. NOTE: E3c best-val reanalysis (done BEFORE campaign
  results) raises genuine uncertainty here — that makes it a real test.
- P2 (frozen >= joint, 3 datasets): evaluate on K-matched arms (all are);
  refuted if joint wins sign-consistently on any dataset.
- P3 (historical table replicates): R1 supersedes archived numbers on
  disagreement; commit in advance.
- LR-arm ordering: if x10 >= joint >= x0.1, harmful-drift is refuted in
  favor of starvation — report as such.

## Paper-edit queue added to P3 (from round 3)
1. 8.4 rewrite dual-metric (BIG — claims change; do before any reviewer
   sees Section 8 again).
2. 8.3: "regime violation" → "regime where the bound is true but
   uninformative" (Fable F18 + ChatGPT agree); soften "behave exactly as
   proved" (ChatGPT F6); µ̂ two-point "trend" → "directionally consistent".
3. 9.1 contradiction with 8.4 (Fable F3): regime-resolved rewrite; add
   two-timescale remedy to operational list; demote untested additive
   baseline to future work.
4. Companion citation: demote to "in preparation (SPEC 2026)"; R0/R1
   first-party numbers become primary (Fable F4). "BlockViT" = internal
   name for O'Leary modified ViT — say so explicitly.
5. 7.4 killer-test reframing (report train acc; drop "killer" name);
   7.5 live TODO removal; equivalence claims → TOST/CI with pre-declared
   margin ±0.02; Nadeau-Bengio corrected paired tests for fold comparisons.
6. Positioning paragraphs: Kumar et al. 2022 (LP-FT; our delta =
   IN-distribution degradation of RANDOM-init features), Huang et al. 2022
   modality competition, Frankle et al. 2021 frozen-random precedent,
   TTUR (Heusel 2017), OGM-GE (Peng 2022), Fujimori 2020 modality-specific
   early stopping.

## Open items
- Breast folds patient-level? (Fable F8.3) — ASK AUTHOR: are there
  multiple cores per patient in the breast TMA? If yes, historical folds
  may leak patients across folds.
- A1 mechanism check (Fable F11a): measure ||h_bar||^2/M_h and
  lambda_max(S_h^p)/M_h per arm/width — localize why default init kills
  the SLOPE not just level. + residual+LN arm and trunc-normal ViT-init
  arm. Local, cheap, queued behind laneSGD2.
- Direction-resolved gradient energy during training (ChatGPT F5,
  Fable F12): E3c+ instrumented joint run projecting grad/update energy
  onto mean-spectrum / class-contrast / LDA subspace / remainder. Local.
- Synthetic Thm-2 bound instantiation (Fable F19i): verify "bound bites
  in synthetic" numerically before keeping the sentence.
- Reviewer calibration note: Gemini 70%→85% is an outlier vs Fable
  20%→45-50%; Fable's is evidence-based (it verified citations/claims),
  trust its number for planning.
