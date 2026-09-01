# P2 — Section 8 rebuild skeleton (2026-09-01)

Decisions locked with author (2026-09-01):
- Section 8 uses BOTH evidence sources, two roles: companion 5-fold numbers as
  production-scale motivating observation (artifact-backed rows ONLY — provenance
  trace in progress; audit A7 already killed 4/7 rows), E3c matrix as the
  controlled, instrumented mechanism study.
- Section 7 stays in main text as compact synthetic verification (E1v3/E2 reruns
  launched 2026-09-01) — its job is to show the Thm-2 bound is non-vacuous where
  containment holds, so Section 8 can honestly report it vacuous where it doesn't.

Budget: ~2.1 pp main text + appendix. Every number below maps to an artifact
file (see table at bottom). NOTHING enters the tex without a row in that table.

---

## 8.1 Pipeline and capacity (~0.3 pp — mostly keep, one deletion)

- Keep: BlockViT description, capacity table (C_g/C_f = 177–299; arithmetic
  re-verified 2026-09-01: 18,271,540/61,108 = 299.0 ✓, 21,472,564/121,588 = 176.6 ✓).
- Keep: channel-convention paragraph (3-channel is the conservative choice).
- **DELETE**: the "theory predicts D_curv ≳ c1'·√(C_g/C_f) ~ 13–17×" paragraph.
  DEAD per E3a (tracker P2 note). Replaced by 8.3's measured story.
- Companion-citation fixes (from audit, verified 2026-09-01):
  - A52: the transfer experiment is breast **QCL only** — say "QCL infrared
    hyperspectral images" there; mention FTIR only for the broader companion
    benchmark context, clearly separated.
  - A29: closing paragraph may credit the companion ONLY with what it contains
    (pixel-level benchmark across three datasets; breast-QCL spatial transfer
    experiment with training-dynamics figure). Drop "capacity ratio sweeps",
    drop "all conform to the predicted theory".
  - A51: fine-tuning degradation (0.90→0.79, if row survives provenance) is
    "consistent with the shortcut dynamics underlying Thm 2", NOT a "predicted
    dynamical consequence" — the theorem bounds gradient magnitude; the
    degradation direction is empirical.

## 8.2 Production-scale observation (~0.3 pp — companion rows, backed only)

- Rows retained = whatever the provenance trace backs to result files.
  Audit A7 verdict: joint 0.675, Peak 0.84, MLP 0.90 plausibly backed;
  frozen-random 0.70, PCA 0.78, SlidingWin 0.95 (contradicts companion's own
  best 0.896), fine-tuned 0.79 UNBACKED → cut unless the trace finds artifacts.
- Framing: "observed in production: joint training loses to pretrain-and-freeze"
  — an *observation that motivates the controlled study*, not a theorem test.
- FREEZING-CLAIM SPLIT (round-2b, ChatGPT #3): pretrain-and-freeze is an
  empirically-supported escape route, NEVER "theorem-guaranteed". The theorem
  test is joint ≈ frozen-at-init, which 8.4 runs properly.
- Fix "five variants" vs row-count mismatch (A7).

## 8.3 Geometry at initialization (~0.6 pp — E3a/E3b/E3d/E3e)

1. Measured at-init D_curv on the real pipeline: **0.46** at production width
   M=192 — the naive block-scalar reading of Thm 1 does NOT manifest here.
   Say this first, plainly. The cap itself is validated (λ_θ cap slope −0.002
   breast, −0.012 prostate vs M — the theorem's inequality holds).
2. Diagnosis: **rank-1 input structure**. Σ_X effective rank 1.07/942 (breast),
   1.02 (prostate) — all tissue spectra share the mean shape, so the cap
   C_θ = L̃²λ_max(Σ_X) is enormous and the bound, while true, is uninformative
   at buildable widths. (This is a scoping discovery, not a failure — the
   theorem's regime assumption is measurably violated by real spectra.)
3. **Direction-wise starvation** (the rescue, E3b): spectral block's top
   eigenvector aligns 0.954 with the mean-spectrum direction; top-40 of 61,108
   dims hold 83% of trace; the clinically decisive CancerEpi–CAS contrast
   direction is **12–28× starved**; spatial top eigenvalue exceeds the θ
   spectrum from rank 2–5. Starvation is direction-wise, not block-scalar.
   → MAIN FIGURE 1 (E3b spectrum + alignment panel).
4. **Normalization is the lever** (E3e): wn_norm inflates the cap 2.4×; without
   it D_curv crosses parity (1.18 at M=192). Internal BNs absorb input
   centering exactly (linear proj + post-BN cancels constant shifts) — the
   lever acts through normalization choice, not preprocessing. (Headline in
   main text, mechanism detail → appendix.)
5. Prostate replication one-liner (E3d): same geometry, second dataset/modality.

## 8.4 Controlled retraining dynamics (~0.6 pp — E3c, 30 runs)

- Design: arms {joint_linear, frozen_random, frozen_pca, joint_mlp} ×
  widths {48, 192} × 3 seeds, AdamW; + SGD pair {joint_linear, frozen_random}
  ×3 seeds at h=48 (theorem-regime check, θ in zero-weight-decay group).
- Headline (pre-registered): **frozen ≥ joint everywhere under AdamW**.
  h48: joint 0.538±0.082 vs frozen-random 0.648±0.063 (−0.110);
  h192: joint 0.646±0.050 vs frozen-random 0.731±0.023 (−0.085).
  Falsifier (joint − frozen_pca strongly positive would refute): −0.193 / −0.081.
- **Harmful drift, not lazy pinning**: θ moves 55–60% (relative) under AdamW
  and this movement *hurts* — outside the theorem's regime the spectral module
  is not starved into stillness, it is actively degraded. joint_mlp drifts
  86–95% and persists the gap (nonlinear tokenizer, same phenomenology).
- Honest nuance: at h=192 frozen-PCA (0.727) ≈ frozen-random (0.731);
  informativeness advantage only at h=48 (0.731 vs 0.648). Stability, not
  informativeness, is what the theorem speaks to — say so.
- **SGD dissociation (theorem's regime)**: joint ≈ frozen-random
  (means 0.538 vs 0.510, +0.028 within seed noise) — Thm 2's functional-
  equivalence read validated where its assumptions apply. Dissipation share
  drops from ~0.95 (AdamW) to ~0.27 (SGD joint) — optimizer preconditioning
  is what breaks the envelope, consistent with 04's adaptive-optimizer scoping.
- **μ̂ grows with width**: 6.0e-4 (h48) → 1.9e-3 (h192), ×3.2 — empirical
  bridge datapoint for prop:ntk_classprior (Ω(M) class-prior curvature).
  (4-point curve pending optional h∈{96,384} runs — decide before P5 figure.)
- EGR honesty paragraph: scalar EGR is **blind** here — it grows (0.8 → 4–12)
  while the model degrades; old draft's "early EGR collapse" prediction is
  retired. Motivates direction-wise diagnostics (ties back to 8.3.3 and
  Section 6 caveat rem:egr_ratio_caveat). → appendix figure (trajectories).
- → MAIN FIGURE 2: arm × width F1 (bars ± sd) or training curves.

## 8.5 Scoping: what the theorems do and do not predict here (~0.3 pp)

- Containment violated on real data: jac_op grows 0.16 → ~10⁴ over training.
- Realized Thm-2 bound is vacuous: bound/measured ‖Δθ‖ ≈ 2×10⁴–2×10⁵ (report
  a representative row, table → appendix). The theorem is a *scoping* result
  on real data; Section 7 (synthetic, containment holds) shows the bound bites
  in its own regime. This contrast is the paper's honesty backbone — state it
  as a feature, not a concession.
- AdamW is outside the flow/GD theorems; SGD pair is the in-regime check.
- Retire "all conform to the predicted theory" sentence (overclaim).

## Figures (feeds P5)

- Fig-8A (main): E3b direction-wise starvation (spectrum decay + contrast ratio).
- Fig-8B (main): E3c arm × width results.
- Appendix: EGR trajectories; jac_op growth; envelope fit example (μ̂, Ĉ);
  E3e normalization-lever bars; bound-vs-measured table.

## Claim → artifact map (provenance discipline; extend as tex is written)

| Claim / number | Artifact |
|---|---|
| D_curv 0.46 @ M=192; cap slopes −0.002/−0.012 | results/exp1_8_real_dcurv.csv, exp1_8_real_dcurv_prostate_qcl.csv |
| Σ_X eff rank 1.07 / 1.02 | exp1_8 per-batch sigma stats (same CSVs) |
| Alignment 0.954; top-40 = 83% trace; contrast 12–28×; spatial ≥ θ from rank 2–5 | results/exp1_8b_{spectra,summary}.csv |
| wn_norm 2.4× cap; parity 1.18 without | results/exp1_8_real_dcurv_{centered,nobn_uncentered,nobn_centered}.csv |
| All E3c F1/drift/μ̂/B̂/dissipation/EGR/jac_op numbers | results/e3c_analysis_runs.csv (regenerated + verified 2026-09-01, 30 runs) |
| Bound/measured 2×10⁴–2×10⁵ | results/e3c_analysis_runs.csv (thm2_bound / theta_disp_abs) |
| Companion table rows | PENDING provenance trace (agent, 2026-09-01) — only traced rows enter |
| Capacity table C_f/C_g | instantiated-model counts (08 tex, re-verified arithmetic 2026-09-01) |
| Synthetic bound-bites contrast | PENDING E1v3/E2 rerun (agent, 2026-09-01) → Section 7 |
