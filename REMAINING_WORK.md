# Remaining Work — Master Tracker

Single source of truth for everything left before submission.
Generated 2026-08-18 from the 14-agent full-paper audit
(9 finders / dedupe / 3 adversarial verifiers / synthesis; run wf_0d0edeef-185).
Verification: 31 findings CONFIRMED, 2 partially confirmed, 0 refuted.

## Severity counts

- BLOCKER: 11
- MAJOR: 19
- MODERATE: 23
- MINOR: 11
- total: 64

## Executive summary (auditor's)

Verification confirmed 31 of the 33 verify-worthy findings outright and partially confirmed the remaining two (none were refuted), leaving 64 total findings: 11 blockers, 19 majors, 23 moderates, and 11 minors. Three integrity items are existential given that the previous review already caught a fabricated author: the "Manifold, Berkeley" bib entry is still in references.bib, five citation sites reference Karakida theorems (3.1/4.2/5.1) that do not exist in the actual AISTATS paper, and the "Figures 2-3, exponent in [0.5, 0.8]" attribution is now verified fabricated against the source, which has two figures and reports agreement with the linear law. Load-bearing quantitative content is broken in three verified places: the supplement still proves the retired Omega(C_g) lemma (with a reversed inequality in Step 3), the C_g/C_f ~ 325 ratio is arithmetically false from the paper's own numbers (real value ~108, and C_g itself is unreconciled with the companion's 18.3M), and the 0.70 scaling-exponent story manufactures agreement by inverting the direction of the discrepancy for architectures whose C_g is provably Theta(M) — while Section 9 tells a third contradictory version. Section 8 additionally presents four table rows with no artifact backing anywhere (one of which, frozen Sliding Win 0.95, would beat the companion paper's bolded best result), makes an ordering claim its own table falsifies, and the manuscript is not submission-complete: no abstract, 11 rendering TODO blocks, empty supplement robustness stubs, zero figures with a dangling headline-figure reference, and a shape roughly 3x over the Nature MI format. The good news is that the refactored theorem statements, Theorem 2's proof, Exp 1.1's non-slope numbers, Exp 1.3's headline means, and most of Section 6.5's statistics reproduce exactly from the artifacts, so the path to submission is a set of well-defined fix clusters (Karakida proof rewrite, ratio recount, exponent reframe, artifact-backed Section 8 rebuild, remnant-prose sweep) rather than new science — but the venue decision and the Exp 1.7/1.8 runs sit on the critical path and should start this week.

## Decisions (2026-08-18, author)

- **Venue**: target ML conferences. ICLR 2027 primary (abstract Sep 18, 2026;
  paper ~Sep 25; decisions Dec/Jan) -> ICML 2027 fallback (late Jan) -> TMLR
  backstop. Nature MI retired as target, retained as writing standard.
  This RESOLVES A47 and re-scopes A11/A46 (ICLR format: ~9 pp + appendix,
  which the current draft maps onto far better than Nat MI's 3.5k words).
- **Approach**: theory first, freeze it, then (re)run experiments to test the
  frozen theory. No empirical claim survives unless backed by an artifact.
- **A6 resolution**: keep the theory, fix the measurement. Theorem 1 stays
  width-native (D_curv >= c1*M). The sqrt(C_g/C_f) form becomes an explicitly
  per-architecture corollary (fixed-depth Theta(M^2) families only). Exp 1.1
  is rerun plotting against M directly (slope-1.0 prediction, no conversion
  ambiguity) and extended with a fixed-depth MxM architecture so the
  parameter-count corollary is tested in its own class. Existing 0.70 datum
  reported honestly as sublinear vs the slope-1.0-in-M prediction for the
  shallow MLP.
- **A7/A8 resolution**: Section 8 rebuilt from artifacts only. Unbacked rows
  cut immediately; all variants rerun via Exp 1.7/1.8 against real
  spectral_tokenization checkpoints; companion paper cited for what it
  actually contains (fixes A29); nothing claimed until numbers exist.
  A dedicated post-theory-freeze critique pass on Section 8 added (T-gate).

## Author directives (2026-08-18, session 2)

- **D1 — Intro rework (new task P0, before P1)**: (a) frame the problem as
  the GENERAL case — compositional pipelines with capacity asymmetry
  (spectral-spatial as the instance class: FTIR, QCL, hyperspectral remote
  sensing; theory is domain-general, empirical hedging per A31); do not
  read as an FTIR-only paper for an ML venue. (b) DEMOTE the frozen
  paradox as the hook — frozen-beats-joint is untested on real data
  (Exp 1.7/1.8 pending) and unpublished (companion in peer review). Anchor
  the hook in published literature instead: O'Leary et al. 2026 (spatial
  dominates) + Mosig group — Mueller et al., "Dimensionality reduction for
  deep learning in infrared microscopy: a comparative computational
  survey" (verify metadata in P4, add to bib). Frozen results become a
  PREDICTION the paper tests, not the premise.
- **D2 — Capacity-ratio reversal (extend E-scope)**: answer the reviewer
  question "what if spectral is as big as spatial / spatial smaller /
  why not just equalize the two?" The theory's answer is ASYMMETRIC and
  should be stated explicitly in the paper (new subsection in 04 or 09):
  - **Growing Cf does NOT help.** The cap C_theta = L~^2 lambda_max(Sigma_X)
    contains no Cf. The spectral block's curvature is set by the
    bottleneck K, the data, and the spatial module's input sensitivity
    L~ — not by how many spectral parameters exist. Adding spectral
    parameters spreads the same gradient budget thinner. This is a
    SHARP, COUNTERINTUITIVE, FALSIFIABLE prediction.
  - **Shrinking Cg DOES help** (D_curv >= c1*M is monotone in M), but it
    trades away the spatial capacity the pipeline exists for — spatial
    context carries real signal (O'Leary et al.; Mosig group).
  - So it is not a size-matching problem but a POSITIONAL asymmetry:
    the spatial module sits adjacent to the loss, the spectral module
    behind a bottleneck; curvature concentrates near the loss. Equal
    parameter counts do not equalize position.
  Experiments (E4): (a) sweep Cf UP at fixed Cg through parity and
  beyond -> predict the gap PERSISTS (if it vanishes, the theory is
  wrong in an important way — highest-value falsification test in the
  paper); (b) sweep Cg DOWN toward Cf -> predict gap fades but accuracy
  drops, quantifying the trade. Replaces the supplement's vague
  "smallest capacity ratio where the pathology disappears" stub.
- **D3 — Prescription hedging + end-phase deep research (new task S0)**:
  soften the absolute "the theory yields a direct prescription: freeze"
  language (09, 01) — freezing is one validated-in-synthetic mitigation,
  not the established solution; the fixed-residual-baseline variant is
  untested (A25). Schedule a DEEP RESEARCH pass at end of Phase P on the
  mitigation landscape (freezing vs per-module LR vs gradient
  modulation vs residual baseline) before the final framing. Note: author
  reports frozen-PCA currently troubling the story by beating frozen
  learned variants in side experiments.
- **D4 — Loss scope (fold into T3 + E4)**: pre-empt "why cross-entropy
  only / what about MSE?" — one scope remark (Thm 2's 1/t envelope is
  CE-on-separable-data-specific via Soudry; Thm 1's geometry is
  loss-agnostic up to the inner-matrix constant), plus run-or-delete the
  loss-variant supplement stub (A18/A9).
- **D5 — EGR provenance (fold into P4)**: state explicitly that EGR is
  this paper's named diagnostic; situate against existing gradient-ratio
  quantities (Wang et al. 2020 OGR, GradNorm, Peng et al. modulation,
  Pezeshki NTK alignment).
- **D6 — Companion policy**: there is exactly ONE companion paper — the
  submitted "What is a Spectral Token?" (Anal. Chem., under revision).
  The dziuba2026spatial bib placeholder with an invented title is
  replaced with the real one (done 2026-08-18). No second empirical
  paper exists or is planned; Section 8 cites only the real companion.
- **D7 — Channel convention (2026-08-19)**: author flagged that the
  field standard is SINGLE-channel d2 (O'Leary uses d2 alone), not our
  3-channel raw+d1+d2 stack. Resolution: Section 8 keeps describing the
  experiment AS RUN (3-channel — the checkpoints and the companion are
  3-channel; artifact-backed reporting per the A7 decision) and now
  carries an explicit channel-convention paragraph noting that under
  the field-standard single-channel input the ratio GROWS to 874
  (C_f = 20,916, verified by instantiation) — i.e., 3-channel is the
  CONSERVATIVE choice and the pathology regime is not an artifact of
  it. No d2-only rerun needed for the theory paper; if Exp 1.7/1.8
  capacity permits, a d2-only arm is a cheap optional robustness point.

## Venue acceptance estimates (conditioned on full fix list + Exp 1.7/1.8 supporting the story)

| Venue | Estimate | Timing | Note |
|---|---|---|---|
| ICLR 2027 | 30-35% | abstract Sep 18 2026, decisions Dec/Jan | primary; best genre fit |
| ICML 2027 | 25-30% | ~late Jan 2027 | fallback with ICLR reviews folded in |
| NeurIPS 2027 | 25-30% | ~May 2027, decisions post-PhD | second fallback |
| TMLR | 55-65% | rolling | criteria-based; honesty-first style fits; backstop |
| Nature Comms | 10-15% | months | needs full restructure |
| Nature MI | 5-10% | months | genre mismatch; retired |

## Theory-first execution plan (supersedes the auditor's 14-step order)

**Phase T — freeze the theory (week 1)**
- [x] T1. DONE 2026-08-18. Integrity batch: A3, A2, A30, A44 (unused-entry
      pruning deferred to P4), A56, A58. Karakida facts verified against
      the actual PMLR PDF before citing.
- [x] T2. DONE 2026-08-18, adversarially verified (wf_4d095e3f-252 +
      wf_9d7f7921-a7f). Proof rewritten as elementary WITNESS argument
      (head direction c kron h/||h||, lambda_max >= p_min ||h||^2/N =
      Omega(M)); Karakida Thms 1+4 context-only (Thm 4 inapplicable to
      fixed-input heads — verifier catch). Lemma gains explicit head
      hypothesis; A6 story unified in 04/07/09/supplement (slope 1.0
      predicted, 0.70 sublinear undershoot reported honestly).
      Bonus: A39, A63, A14-partial (02), A24-partial (07 relabel),
      A53-partial fixed.
- [x] T3. DONE 2026-08-18. Sigma_X fixed as the MEAN empirical Gram with
      1/N-normalized loss/Jacobians throughout; C_theta = L~^2
      lambda_max(Sigma_X) — factor K dropped (it was slack; the Kronecker
      identity is exact) and all constants now dataset-size independent
      (A12). Logit vs residual Jacobian both defined once in 03, GGN
      convention G = J^T H_tau J fixed, transfers stated (A13). Thm 2
      restated with finite-horizon B_T + explicit trajectory containment
      + max-margin log-factor remark (A23). D_curv evaluated at random
      init, blocks tied to eq:block_hessian (A35). Assumption lists made
      explicit; ass:gap marked not-used-in-proofs (A36). Prop 6.1
      DOWNGRADED to an unnumbered heuristic with its three failure modes
      stated and its TODO removed (A22). C, mu declared
      trajectory-dependent (A61). A39 already done in T2.
      Bonus: A57 (lem:schur -> lem:opcap project-wide), A54 (exponential
      -> margin/time), A14 third site (03 condition-number sentence),
      A16 (slow-fast manifold scrubbed from 03 and 07), A37-partial
      (Section 4 retitled), plus D4 (loss-scope remark rem:loss_scope).
- [x] T4. DONE 2026-08-19. Counted from the INSTANTIATED model
      (code/count_blockvit_params.py), not inherited. Real numbers:
      K=64 baseline C_f=61,108 C_g=18,271,540 ratio 299; K=128 variant
      C_f=121,588 C_g=21,472,564 ratio 177; TOTAL(K=64)=18,332,648 =
      the companion's 18.3M, which resolves the 13M-vs-18.3M conflict
      (18.3M is the TOTAL, our 13M was an undercount). Retired 325 was
      a fossil: single-channel 314x128=40,192 against the undercounted
      C_g. New Table tab:capacity in Section 8; 03 and the inventory
      updated; "several orders of magnitude" -> "roughly two". Also
      confirmed BlockViT is a Theta(M^2) family (transformer 5.34M +
      ConvTranspose 9.44M, M=hidden_dim=192), so the sqrt corollary
      legitimately applies here: D_curv >= c1' sqrt(ratio) ~ 13-17 c1'.
- [ ] T5. THEORY FREEZE GATE: external-reviewer pass on Sections 3-6 +
      supplement only. Fix what it finds. Then the theory is frozen.

**Phase E — experiments to test the frozen theory (weeks 2-3)**
- [ ] E1. Exp 1.1 v3: measure GN blocks (not full Hessian — A24), plot vs M,
      add fixed-depth MxM architecture; D_curv branding in code/CSVs (A64).
- [ ] E2. Exp 1.2 rerun persisting per-run EGR CSVs so every quoted number
      is reproducible (A19); state metric for ViT claim (A60).
- [ ] E3. Exp 1.7 real-data EGR + Exp 1.8 real-data D_curv on
      spectral_tokenization checkpoints (launch EARLY - longest pole).
      Produces every Section 8 row or the row dies (A7).
- [ ] E4. Robustness: run the SGD/noise/S variations Section 7 claims, or
      delete the claims (A18). Fisher-z replaced with seed-clustered
      analysis (A21). Sync 6.5 stats to CSVs (A34).
- [ ] E5. Optional per A26: one nonlinear-f_theta run, or soften the claim.

**Phase P — rebuild the paper on frozen theory + fresh artifacts (weeks 3-4)**
- [ ] P1. Section 7: Exp 1.6 subsection written in (A17); Exp 1.2/1.3
      numbers from new artifacts; corrected SEs (A20); setup honesty (A53).
- [ ] P2. Section 8 rebuild, artifacts only, companion cited correctly
      (A7, A8, A29, A51, A52) + separate critique pass per decision.
- [ ] P3. Remnant-prose sweep against frozen theorems (A14, A15, A16, A25,
      A41, A42, A43, A48, A49, A54, A55, A57, A62, A63).
- [ ] P4. Related work: Huang 2022 into 2.4 (A27), four anchors (A28), 2024
      multimodal wave (A45), remote-sensing cites or soften (A50),
      layer-wise-Hessian adjacents (A33), quote fixes (A32).
- [ ] P5. Figures: regenerate D_curv-branded exp1_1 + Exp 1.7 EGR figures,
      place all headline figures, kill Figure ?? (A10).
- [ ] P6. Structure for ICLR: theorem numbering (A37), drop unused
      lem:interlacing or mark auxiliary (A38), universality hedges (A31),
      EGR framing (A40), 09 remedies paragraph (A49).

**Phase S — submission (week 4.5)**
- [ ] S1. Abstract written (150-250 words), all TODOs resolved, supplement
      stubs completed or deleted (A9).
- [ ] S2. ICLR formatting, reproducibility statement, code release prep
      (re-scoped A11/A46).
- [ ] S3. Full compile + final external-reviewer pass on the complete PDF.

## Findings table

### A1 [BLOCKER — FIXED 2026-08-18 (witness-argument rewrite, verified)] — paper/sections/supplement.tex lines 32-117 (esp. 34, 69-91, 105-107)

**Issue**: The supplement proof of Lemma phi_scaling is un-refactored: it opens and concludes with the retired Omega(C_g) claim, contradicting the lemma's stated Omega(M) (= Omega(sqrt(C_g))) and Karakida's actual Theta(M) result, and Step 3 derives a lower bound on tr(F) from an upper bound on the inner matrix (false, since diag(p)-pp^T annihilates the all-ones direction).

**Fix**: Rewrite the proof body to derive Omega(M) by citing Karakida's real Theorem 4, add the residual-vs-logit transfer step (see A13), delete the trace/participation-ratio construction and the broken Step 3, and make the proven statement match the main-text lemma verbatim.

**Effort**: 4-6 hours

### A2 [BLOCKER — FIXED 2026-08-18] — 04_theorem1_hessian.tex:151; supplement.tex:57, 101-104, 111, 252

**Issue**: Five citation sites (one in the main text) reference Karakida 'Theorems 3.1/4.2/5.1', none of which exist — the AISTATS 2019 paper has plain-numbered Theorems 1-7 with no participation-ratio or lambda_max-concentration theorem, and the project's own PROGRESS.md already admits this but the files remain unfixed.

**Fix**: Cite the real Theorem 1 (mean eigenvalue/trace) and Theorem 4 (lambda_max = Theta(M)), derive any effective-rank claim explicitly, hedge the concentration claim to 'in expectation over initialization', and upgrade the bib entry to the published AISTATS version.

**Effort**: 1-2 hours (largely subsumed by the A1 proof rewrite)

### A3 [BLOCKER — FIXED 2026-08-18 (commit 9587734)] — paper/references.bib line 66 (berisha2019deep)

**Issue**: The hallucinated author 'Manifold, Berkeley' — the exact fabrication a reviewer already caught — is still in the bib entry; the verified real author list (Analyst 2019, DOI 10.1039/C8AN01495G) is Berisha, Lotfollahi, Jahanipour, Gurcan, Walsh, Bhargava, Nguyen, Mayerich.

**Fix**: Replace with the full verified 8-author list, add the DOI, and remove 'and others'.

**Effort**: 10 minutes

### A4 [BLOCKER — FIXED 2026-08-19 (T4, counted from the model)] — 03_setup.tex:68-71; 06_egr.tex:45-46; 08_real_data.tex:20-23; EXPERIMENT_INVENTORY.md Exp 1.8

**Issue**: C_g/C_f ~ 325 is arithmetically false from the paper's own numbers (13e6/120,576 = 107.8) — 325 is a fossil from single-channel S=314 — and C_g = 13M itself conflicts with the companion paper's stated 18.3M BlockViT v2 parameter count, corrupting the EGR(0) prediction and the 'several orders of magnitude' claim downstream.

**Fix**: Count spatial-module parameters from the actual BlockViT v2 checkpoint, reconcile 13M vs 18.3M with the companion, then propagate the corrected ratio to 03:71, 06:45-46 (EGR(0) ~ 0.096 or 0.081), 08:22-23, and the inventory, changing 'several orders of magnitude' to 'roughly two'.

**Effort**: 2-3 hours

### A5 [BLOCKER — FIXED 2026-08-19 (T4: kappa deleted from Section 8, D_curv + real ratio)] — 08_real_data.tex lines 23-25

**Issue**: 'Predicts kappa of order 325*c_1' uses the retired kappa symbol (which rem:disparity_vs_kappa proves is trivially infinite), the retired linear-in-ratio scaling (the refactored theorem gives D_curv >= c_1'*sqrt(C_g/C_f) - c_2, ~18x smaller), the wrong constant symbol, and the wrong ratio.

**Fix**: Rewrite as 'Per Theorem 1 (parameter form), this predicts a module curvature disparity D_curv of order c_1'*sqrt(C_g/C_f) ~ 10*c_1'' using the corrected ratio from A4, and delete kappa from Section 8 entirely.

**Effort**: 30 minutes (after A4)

### A6 [BLOCKER — REFRAME DONE 2026-08-18 (04/07/09/supplement unified); Theta(M^2)-arch rerun pending in E1] — 07_experiments.tex:53-60; 04_theorem1_hessian.tex:287-294; supplement.tex:119-132; 09_discussion.tex:60-67

**Issue**: The scaling-exponent story is false for every tested architecture — C_g = Theta(D) for the SpatialMLP (verified C_g/D = 19.06 constant from D=16 to 8192), so the width-linear prediction implies slope 1.0 vs C_g and the measured 0.702 +/- 0.007 UNDERSHOOTS it (the paper inverts the discrepancy to manufacture agreement with an 'asymptotic 0.5 plus upward correction' narrative), while Section 9 tells a third, contradictory story.

**Fix**: Either rerun Exp 1.1 with an architecture in the theorem's Theta(M^2) class (fixed-depth DxD MLP) or reframe honestly everywhere (predicted slope 1.0, measured 0.70 +/- 0.01 sublinear, discuss the deficit), making 04, 07, 09, and the supplement tell one consistent story with the real seed spread.

**Effort**: 3-4 hours (reframe) or 1-2 days (rerun)

### A7 [BLOCKER — APPROACH DECIDED: full rerun via Exp 1.7/1.8, rows cut until backed] — 08_real_data.tex Table tab:real (lines 39-60) and lines 27-33, 62-83

**Issue**: Four of seven table rows (frozen random 0.70, frozen PCA-128 0.78, frozen Sliding Win 0.95, fine-tuned 0.79) have no artifact backing anywhere in either repository, the 0.95 would beat the companion paper's bolded best result (0.896) — a direct cross-paper contradiction — the headline 'strongest single datapoint' rests on an unbacked 0.025 gap with no error bars, and the text says 'five variants' against a seven-row table.

**Fix**: Either cut the table to the three artifact-backed rows reported with +/-SD and rewrite the observations around them, or actually run the four missing variants 5-fold (note the checkpoints Exp 1.7 planned to reuse do not exist), reconciling the variant count and downgrading 'strongest single datapoint' to 'consistent with'.

**Effort**: 2-3 hours (cut) or 2-4 days (run variants)

### A8 [BLOCKER — APPROACH DECIDED: claim rewritten after Section 8 rebuild] — 08_real_data.tex lines 84-88 vs Table tab:real

**Issue**: The claim that the variant ordering is 'exactly what the theory predicts: any frozen variant beats any joint or fine-tuned variant' is falsified by the table directly above it — fine-tuned (0.79) beats frozen random (0.70) and frozen PCA-128 (0.78), and the column is not monotone top-to-bottom.

**Fix**: State the partial ordering the data actually support (joint < frozen-random; frozen informativeness monotone; fine-tuned < frozen same-encoder) and delete 'any frozen beats any' and 'exactly'.

**Effort**: 30 minutes

### A9 [BLOCKER] — main.tex:37, 51; 05:224; 06:47, 242; 07:257; 08:92; supplement.tex:318-348; EXPERIMENT_INVENTORY.md Pending

**Issue**: The manuscript is not submission-complete: the abstract is literally a \TODO, 11 red TODO blocks render in the PDF, all four supplement 'Empirical robustness' subsections are empty stubs referenced from the main text as existing, Section 8's Diagnostic subsection awaits unrun Exp 1.7, and Prop 6.1's promised empirical verification was never performed.

**Fix**: Write the abstract last, resolve or delete every TODO, complete or delete the supplement stubs (fixing main-text references to them), run Exp 1.7 or delete the dependent subsection, and verify or drop the Prop 6.1 promise.

**Effort**: 4-8 hours (excluding experiment runs tracked separately)

### A10 [BLOCKER] — paper/figures + paper/figures_src (empty); 07_experiments.tex:75; main.log:515, 550

**Issue**: The paper contains zero figures (no \begin{figure} or \includegraphics anywhere) and the single figure reference — billed as the 'Headline figure' — compiles as 'Figure ??', which is a desk rejection for a theory+experiments submission, even though most assets already exist in results/ (40 PNGs).

**Fix**: Place the headline figures with matching labels, regenerate the exp1_1 plots under D_curv branding (see A64), add the Exp 1.7 EGR figure once run, prefer vector formats, and recompile until main.log shows no undefined references.

**Effort**: 3-5 hours

### A11 [RE-SCOPED: ICLR format (~9pp + appendix), not Nat MI 3.5k words] — main.tex whole-paper structure (~9.9k words main body, monolithic theorem-proof format)

**Issue**: The draft is roughly 3x over Nature MI's 3,500-word Article limit in the wrong genre shape (formal theorem-proof main text, TODO abstract vs a 150-word limit, no Results/Methods split) — a guaranteed desk rejection on format alone if the Nat MI target stands (moot on a pivot to ICLR 2027/AISTATS/TMLR).

**Fix**: Decide the venue this week (see A47); if Nat MI, restructure into main text (paradox, informal theorems, EGR figure, prescription) + Methods (formal statements, protocol) + SI (full proofs) with a 6-display-item budget.

**Effort**: 3-5 days (Nat MI restructure) or 1-2 days (conference-shape cleanup)

### A12 [MAJOR — FIXED 2026-08-18 (T3)] — 03_setup.tex:57-61, 120-124; 04_theorem1_hessian.tex:117-134; 05_theorem2_twoscale.tex:46-64; supplement.tex:181-198

**Issue**: Sigma_X is glossed as both the aggregated sum-Gram and the empirical covariance — objects differing by a factor N_data*HW — making lem:schur's bound false under one reading and its constant C_theta dataset-size-dependent under the other, the K factor in the bound is pure slack (B^T B = I_K kron sum x_i x_i^T exactly), and eq:loss's 1/HW never reappears in the gradient identity.

**Fix**: Fix one convention (J and r from the 1/(N*HW)-normalized loss, Sigma_X the mean Gram, one symbol with ass:subg's Sigma as its population counterpart), re-derive C_theta dropping K, and thread it through eq:chain, ass:residual, and B so both constants are provably dataset-size-independent.

**Effort**: 2-3 hours

### A13 [MAJOR — FIXED 2026-08-18 (T3)] — 03_setup.tex:170-174; 04:127-138; 05 eq:theta_grad_bound; supplement.tex:76-79 vs 155-168

**Issue**: J is defined as the residual Jacobian in 03/lem:schur but used as the logit Jacobian in phi_scaling's FIM identity and 05's gradient identity, so D_curv's numerator and denominator bounds are derived for different matrices, and the main-text chain-rule remark omits the softmax factor J_tau.

**Fix**: State both objects once in 03, add the one-line uniform-p conversion (A^2 = (1/N_cls)A, so residual-GN = (1/N_cls)F, preserving Omega(M)) in the phi_scaling proof, use the logit Jacobian in 05/06 with an a-fortiori remark for lem:schur, and fix the chain-rule remark.

**Effort**: 1-2 hours

### A14 [MAJOR — 02 and 03 SITES FIXED 2026-08-18; 07:7 remains (P3)] — 02_related_work.tex:51-57; 03_setup.tex:166-168; 07_experiments.tex:7

**Issue**: Three prose sites still describe Theorem 1 with the retired Schur-complement/condition-number framing — including 03's 'bounds the condition number of H', which rem:disparity_vs_kappa ('kappa(G) is trivially infinite') directly contradicts 60 lines later.

**Fix**: Rewrite all three as the operator-norm-cap / module-curvature-disparity description (02: blockwise width-linear lambda_max + operator-norm cap; 03: bounds D_curv in terms of M = sqrt(C_g/C_f); 07: 'the module curvature disparity bound').

**Effort**: 30 minutes

### A15 [MAJOR] — 09_discussion.tex lines 36-39

**Issue**: 'The spectral module has been absorbed into a spurious stationary point' contradicts the paper's own load-bearing premise (01:70-74, 05:170-174) that no finite stationary point exists for unregularized CE on separable data — a passage a reviewer can quote against the prescription.

**Fix**: Replace with finite-time starvation language: subsequent theta updates are essentially noise because Theorem 2 bounds the spectral module's cumulative displacement over any practical horizon by O(eps log(1/delta)).

**Effort**: 15 minutes

### A16 [MAJOR — FIXED 2026-08-18 (T3: 03 and 07 sites)] — 03_setup.tex:191-197; 07_experiments.tex:85-88

**Issue**: Both promise EGR monotonicity 'along the slow-fast manifold' — retired vocabulary appearing nowhere in Sections 5-6 — while Prop 6.2 bounds only the numerator and rem:egr_ratio_caveat explicitly disclaims any ratio claim, so the promises read as a walk-back.

**Fix**: Reword both to the numerator-decay bound ('polynomial decay of the spectral gradient norm, C/(1+mu t)'), rename prop:egr_monotonic to prop:spectral_grad_decay, and delete 'slow-fast manifold' from both sites.

**Effort**: 30 minutes

### A17 [MAJOR] — 04_theorem1_hessian.tex:228 vs 07_experiments.tex (no 1.6 subsection); 06_egr.tex Section 6.5 sweeps unnumbered

**Issue**: Section 4 asserts Experiment 1.6's result (Spectral Decoupling does not reduce collapse, freezing does) but the paper never presents the experiment despite complete artifacts (120-row CSV, Wilcoxon p=0.55 vs p<1e-4, notes saying 'Insert as a new subsection in Section 7'), and the experiment numbering has unexplained gaps.

**Fix**: Write the Exp 1.6 subsection with design, sweep, statistics, and the notes' honesty caveats (frozen peak 0.583 vs 0.660; seed sigma 0.08-0.14 incl. one below-chance seed; best SD lambda noise-dependent), update 'Three experiments', and give the 6.5 sweeps an experiment number.

**Effort**: 2-3 hours

### A18 [MAJOR] — 07_experiments.tex:246-258; 06_egr.tex:109; supplement.tex:328-348

**Issue**: The Robustness subsection asserts optimizer (Adam vs SGD) and spectral-dimension-S variations that do not exist — every runner is Adam-only and S=64 is hard-coded — while the sole SGD artifact (exp1_2v2) is the MLP control showing NO joint-vs-frozen gap, and the promised tau-sensitivity analysis is absent.

**Fix**: Either run the SGD and S variations and persist artifacts, or rewrite to claim only the backed noise variation, reframe exp1_2v2 as the MLP control ablation, and delete the Adam-vs-SGD mechanistic sentence and the tau pointer.

**Effort**: 1 hour (rewrite) or 1 day (runs)

### A19 [MAJOR] — 07_experiments.tex lines 90-106, 113-119

**Issue**: Exp 1.2's headline numbers are irreproducible from persisted artifacts and partially contradicted by them: the monotonic EGR quartet matches no stored dataset (exp1_4 is NON-monotonic at D=1024: 0.127 > 0.103), 'joint test loss 2.27' is a median paired against a frozen mean (1.15), and the ViT 'train loss 0.01-0.03 at all widths' is false at D=64 (0.14-0.15) with the test-loss span misquoted.

**Fix**: Rerun exp1_2v4 persisting per-run EGR CSVs (the logger dataframe is already built), recompute finals as mean +/- sd with consistent statistics for both arms, weaken monotonicity to 'plateaus below 0.1' or restrict the width range, and fix the ViT qualifiers — never quoting values read off a smoothed plot.

**Effort**: 3-4 hours

### A20 [MAJOR] — 07_experiments.tex:188-191 vs results/exp1_3v1_summary.csv

**Issue**: The stated per-condition SE (~0.02) is a pooled binomial SE ignoring between-seed variance — actual seed-level SEs are 0.046 (joint) and 0.080 (frozen), making the +0.13 gap ~1.4 combined SEs and the 'resolvable' 0.05 undershoot claim unsupported.

**Fix**: Report empirical per-condition seed SD/SE, state that 3 seeds gives loose intervals, soften 'resolvable but not dramatic' (the direction-of-effect argument survives), or run more seeds at minutes per run.

**Effort**: 1 hour (+optional seed runs)

### A21 [MAJOR] — 06_egr.tex lines 189-198

**Issue**: The Fisher-z aggregate 't = 14.65, p < 1e-13' is pseudo-replication: the same six seeds fill all 24 cells (verified programmatically), and the between-cell sd(z) = 0.233 sits far below the independent-cell noise floor of 0.577 (chi-square tail P ~ 1.8e-6), so the one-sample t-test's independence assumption is violated and the p-value dramatically overstates the evidence.

**Fix**: Replace with a seed-level hierarchical bootstrap/permutation test or mixed-effects model with seed as a random effect, or report the 24-cell analysis descriptively (23/24 positive, mean r = +0.584) and let the N=144 partial correlation carry the inference.

**Effort**: 2-3 hours

### A22 [MAJOR — FIXED 2026-08-18 (T3: downgraded to heuristic)] — 06_egr.tex lines 31-49 (prop:egr_init and remark)

**Issue**: Proposition 6.1 (E[EGR(0)] = Theta(sqrt(C_f/C_g))) is stated as a formal result with no proof anywhere, contains a live in-manuscript TODO, promises an empirical verification that was never performed (Exp 1.1 measures eigenvalues, not gradient norms), silently exchanges E[ratio] for a ratio of expectations, and is parametrization-dependent (would differ under muP, which the paper cites).

**Fix**: Either supply a supplement proof under an explicitly stated Kaiming parametrization justifying the concentration step, or downgrade to an unnumbered heuristic/conjecture with an empirical check; remove the TODO and fix the worked number per A4.

**Effort**: 1 hour (downgrade) or 1 day (proof)

### A23 [MAJOR — FIXED 2026-08-18 (T3)] — 05_theorem2_twoscale.tex:55-64 with 03_setup.tex:105-110

**Issue**: B := sup_t over the infinite horizon is justified in 03 only via weight decay and bounded-norm iterates — exactly what the Soudry unregularized-separable regime motivating the 1/t envelope excludes (there ||phi(t)|| diverges and the cap provably fails) — and the theorem never hypothesizes the trajectory stays in Phi_reg x Z_reg.

**Fix**: Define B_T := sup_{t<=T} ||J_theta(t)||_op (finite by continuity), add trajectory-containment to the hypotheses, and optionally note the max-margin limit only costs a log factor (displacement O(eps log^2(1/delta))).

**Effort**: 1-2 hours

### A24 [MAJOR — RELABEL DONE 2026-08-18 (07 now says loss-Hessian blocks); GGN re-measurement pending in E1] — 07_experiments.tex:38-43 vs code/hessian/eigenvalues.py:73-105

**Issue**: Exp 1.1 power-iterates full loss-Hessian blocks via double backprop, not the Gauss-Newton blocks the theorem and text claim — for the phi block the functional-Hessian term (fc1-fc2 cross second derivatives with O(1) residuals at init) is nonzero and uncontrolled, so the measured object differs from the claimed one.

**Fix**: Either implement true GGN block products (Jv then softmax-CE GGN sandwich) and re-measure, or state 'loss-Hessian blocks' in Section 7 with a GN-dominance-at-init sentence plus a supplement check of the gap at one width.

**Effort**: 1 hour (relabel) or 3-4 hours (GGN implementation)

### A25 [MAJOR] — 06_egr.tex:127; 09_discussion.tex:29-31; 01_introduction.tex:94-100

**Issue**: The fixed-residual-baseline intervention is claimed as evaluated in Experiment 1.3 (false — verified condition list is joint/frozen/spectral-only/spatial-only) while 09 simultaneously admits it was not tested yet asserts it 'follows directly from the theory', though no theorem addresses the additive Z = f_theta(X)+h(X) architecture.

**Fix**: Correct 06's pointer ('we evaluate the freezing intervention; the fixed-baseline variant is left to future work'), hedge 01/09 to 'a theory-motivated variant we propose but do not test' — or add the condition to exp1_3.py and run it.

**Effort**: 30 minutes (hedge) or 2-3 hours (run)

### A26 [MAJOR] — 03_setup.tex:45-49; 09_discussion.tex:55-57

**Issue**: Both claim Section 7 empirically verifies that the scaling/gap persists under nonlinear f_theta, but no such experiment exists — SpatialMLP is a spatial baseline, and the only spectral module in the repository is the linear SpectralReduction.

**Fix**: Run a cheap 2-layer-MLP spectral-module variant of the exp1_2 runner, or change both sentences to expected/future work.

**Effort**: 15 minutes (soften) or 3-4 hours (run)

### A27 [MAJOR] — 02_related_work.tex lines 59-77 (section 2.4)

**Issue**: Huang et al. 2022 — whose title literally ends '(Provably)' and which is cited elsewhere in the paper (09:107) — is absent from the modality-competition survey whose 'these works are empirical' novelty setup it directly falsifies.

**Fix**: Cite huang2022modality in 2.4, characterize it accurately (theoretical account for explicit dual-input late-fusion networks), state the delta (single-input compositional pipelines, curvature/capacity mechanism, finite-time bound), and restrict 'empirical' to Wang et al. and Peng et al.

**Effort**: 45 minutes

### A28 [MAJOR] — 02_related_work.tex sections 2.1/2.4/2.6; 09_discussion.tex prescription; references.bib (absences grep-confirmed)

**Issue**: Four anchor citations are missing and reviewer-expected: Chizat-Bach lazy training (Theorem 2's 'pinned near initialization' is module-wise lazy training), Kirichenko et al. DFR (the canonical frozen-features-under-shortcut-pressure analogue of the prescription), Geirhos et al.'s shortcut-learning review (in a paper titled 'The Spectral Shortcut Theorem' with 29 'shortcut' mentions), and Kunin et al.'s Alternating Gradient Flows (closest recent starvation theory, verified non-scooping).

**Fix**: Add all four with 1-2 sentences each precisely stating the delta; the Geirhos cite also strengthens a Nature-venue cover letter.

**Effort**: 1-2 hours

### A29 [MAJOR] — 08_real_data.tex lines 99-102

**Issue**: The closing paragraph attributes 'capacity ratio sweeps, prostate cancer cohort replication, train/val/test split sensitivity analysis' to the citable companion paper, none of which it contains (its only sensitivity analysis is preprocessing, its prostate content is the pixel-level benchmark, and 'all conform to the predicted theory' is unfalsifiable).

**Fix**: List only what the companion actually contains (pixel-level benchmark across three datasets; the breast-QCL spatial transfer experiment with training-dynamics figure) and drop 'capacity ratio sweeps' and 'all conform'.

**Effort**: 20 minutes

### A30 [MAJOR — FIXED 2026-08-18] — supplement.tex:127 (also echoed at 04:289-294; 07:58-60)

**Issue**: The Karakida 'their Figures 2-3, exponent in [0.5, 0.8] across finite widths' attribution is now VERIFIED fabricated against the source — the paper has exactly two figures (lambda_max vs M agreeing with the linear law; a learning-rate color map), no Figure 3, and no fitted exponent anywhere — a checkable false figure citation in a manuscript with a fabrication-sensitive review history.

**Fix**: Delete the invented figure/exponent citation and its echoes, quoting only what the source shows (agreement with lambda_max proportional to M); the rewrite lands naturally inside the A6 scaling-story fix.

**Effort**: 30 minutes (within the A6 rewrite)

### A31 [MODERATE] — 01_introduction.tex:102-111; 09_discussion.tex:118-121 vs 08_real_data.tex (single in-house dataset)

**Issue**: Two unhedged 'the mechanism applies to any compositional architecture with capacity asymmetry' sentences plus a single in-house real dataset leave the universality claim exposed — though the cross-domain list is already hedged with 'we expect' and 09 carries an explicit single-domain-validation limitation paragraph.

**Fix**: Hedge the two 'applies to any' sentences (or add one public spectral-spatial benchmark, e.g. Indian Pines-class, showing the same EGR collapse and frozen>joint gap) and let the synthetic linear/CNN/ViT sweep explicitly carry the generality argument.

**Effort**: 30 minutes (hedge) or 2-3 days (benchmark)

### A32 [MODERATE] — 01_introduction.tex:27-32; 02_related_work.tex:67-68

**Issue**: Two quoted phrases fail verification: 'tissue classification only needs a small set of spectral features' is a non-verbatim paraphrase of O'Leary wrongly co-attributed to Berisha (whose paper contains no 16-feature claim and is not 'Recent'), and 'modality laziness' is quote-attributed to Peng et al. 2022, whose full text never uses the term (it belongs to the Du et al. ICML 2023 line); the 'minimal chemical information' quote checked out verbatim and needs no change.

**Fix**: Drop the quotation marks and paraphrase with correct single-paper attribution for the O'Leary claim, and either use Peng's own terminology ('optimization imbalance') or additionally cite the coining paper for 'modality laziness'.

**Effort**: 30 minutes

### A33 [MODERATE] — 02_related_work.tex section 2.3 (Hessian geometry)

**Issue**: Two verified-real adjacent layer-wise-Hessian papers (arXiv 2510.17486 empirical spectral diagnostics; arXiv 2604.11639 analytic block decomposition) are uncited — neither scoops Theorem 1, but reviewers may surface them.

**Fix**: Cite both alongside Sagun/Papyan with a delta clause noting they are diagnostic/decomposition treatments without a provable capacity-ratio scaling law or a starvation link.

**Effort**: 45 minutes

### A34 [MODERATE] — 06_egr.tex:161-163, 182-183, 192-193, 203, 208-209 vs results/exp1_4_summary.csv and exp1_4_fw_summary.csv

**Issue**: Five Section 6.5 statistics drift from the stored CSVs: r = +0.344 vs measured +0.334 (used correctly ten lines later), R^2 0.015 vs 0.006, partial +0.335 vs +0.338, AUC 0.538 is the flipped orientation of an anti-predictive 0.462 for EGR_depth, and the pooled design is misdescribed as 4 widths x 6 seeds rather than 4x2x3 with noise an unacknowledged confounder.

**Fix**: Correct all five against exp1_4_fw_summary.csv, state explicitly that EGR_depth is directionally uninformative (AUC 0.46-0.54 by orientation), and describe the true 4x2x3 design.

**Effort**: 1 hour

### A35 [MODERATE — FIXED 2026-08-18 (T3)] — 03_setup.tex def:dcurv lines 209-222; 04_theorem1_hessian.tex:28-34

**Issue**: D_curv is defined with no evaluation point, uses G_phiphi/G_thetatheta without formal introduction, and the theorem's 'in expectation over random Gaussian initialization' never states whether E[D_curv] or E[numerator]/cap is bounded.

**Fix**: Add 'evaluated at (theta_0, phi_0) at random initialization' to def:dcurv, define G's blocks by analogy with eq:block_hessian, and state the expectation placement (E[lambda_max(G_phiphi)] >= cM with deterministic denominator cap, lower-bounding E[D_curv]).

**Effort**: 45 minutes

### A36 [MODERATE — FIXED 2026-08-18 (T3)] — 04_theorem1_hessian.tex:22-23, 110-124; 05_theorem2_twoscale.tex:112; supplement.tex:196-197

**Issue**: Assumption bookkeeping is wrong: lem:schur and Theorem 2 silently rely on ass:subg for lambda_max(Sigma_X) without listing it, while Theorem 1's assumption range sweeps in ass:gap, which the proof never uses.

**Fix**: Add ass:subg to lem:schur's and Theorem 2's hypothesis lists (or define Sigma_X as the always-finite mean Gram per A12) and cite Theorem 1's explicit assumption list, noting ass:gap only marks the regime of interest.

**Effort**: 30 minutes

### A37 [MODERATE — TITLE FIXED 2026-08-18 (T3); theorem counters remain (P6)] — main.tex theorem environments (lines 16-24); 04_theorem1_hessian.tex:1; main.aux

**Issue**: All theorem-like environments share one counter, so thm:hessian renders as 'Theorem 12' and thm:twoscale as 'Theorem 25' — nothing in the PDF is ever 'Theorem 1/2' despite constant prose references — and Section 4's title still carries the old 'Hessian Capacity Bound' name.

**Fix**: Give theorems, assumptions, definitions, and remarks separate (or within-section) counters and retitle Section 4 'Theorem 1: Capacity-Induced Module Curvature Disparity'.

**Effort**: 30 minutes

### A38 [MODERATE] — 04_theorem1_hessian.tex:93-106; supplement.tex:27-29, 63-68; 03_setup.tex:110

**Issue**: lem:interlacing is presented as supporting Theorem 1 but is never used (the supplement admits it), its block matrix name M collides with the characteristic width M, and the supplement's M_ell/L collide with ass:reg's Lipschitz constant.

**Fix**: Move lem:interlacing to the supplement or cut it, rename the block matrix, and use d for depth and P_ell for per-layer parameter counts.

**Effort**: 30 minutes

### A39 [MODERATE — FIXED 2026-08-18] — supplement.tex lines 232-239 (line 237)

**Issue**: The remark lambda_max(diag(p)-pp^T) = max_i p_i(1-p_i) <= 1/4 is mathematically false (p = (1/2,1/2) gives lambda_max = 1/2), though non-load-bearing since the proof only uses <= 1, and 'softmax Hessian' should be 'softmax Jacobian'.

**Fix**: Replace with 'diag(p)-pp^T is the categorical covariance; PSD with lambda_max <= max_i p_i <= 1', rename to softmax Jacobian, and keep the Botev cite for GGN context only.

**Effort**: 15 minutes

### A40 [MODERATE] — 06_egr.tex:27-29; contrast 07:250-253

**Issue**: 'EGR is invariant to the learning rate' is stated without the pointwise-vs-trajectory qualification (the trajectory, hence EGR(t), is LR-dependent — and the 6.5 sweep varies LR) or the Adam caveat the paper itself concedes in Section 7.

**Fix**: Qualify as unitless and LR-free at a fixed point but evaluated along an LR-dependent trajectory, note that under adaptive optimizers EGR measures available learning signal rather than effective update magnitude, and state the sweep's optimizer.

**Effort**: 20 minutes

### A41 [MODERATE] — 06_egr.tex lines 4-12

**Issue**: The 'no in-training signal, only revealed on test data' framing is incoherent — validation data is held out too, so starvation damages val metrics identically; what the experiments show is invisibility to single-run curve monitoring absent a frozen baseline or EGR.

**Fix**: Reword to the single-run-monitoring framing: any run's own train/val curves look healthy, and the deficit appears only in comparison against a frozen-spectral baseline or via EGR.

**Effort**: 15 minutes

### A42 [MODERATE] — 02_related_work.tex lines 39-42

**Issue**: 02 claims the timescale separation 'emerges implicitly from the Hessian eigenvalue gap' as established, while 04:178-187 and 05:210-222 deliberately state the curvature-to-rate bridge is a hypothesis (ass:residual), not a derivation.

**Fix**: Reword to '...is hypothesized to emerge from the capacity-driven curvature disparity of Theorem 1; we state this bridge as an explicit hypothesis and verify it empirically'.

**Effort**: 10 minutes

### A43 [MODERATE] — 05_theorem2_twoscale.tex:92-94, 166-168 vs 07 Exp 1.2

**Issue**: Section 5 claims empirical verifications Exp 1.2 does not report: no time-to-fit metric exists, the joint-vs-frozen gap is reported only at D=1024 (not 'grows monotonically with D'), and the C/(1+mu t) envelope verification is admitted outstanding in the supplement's own TODO.

**Fix**: Add the missing per-D data (time-to-fit, gap, fitted envelopes) or soften both sentences to 'consistent with Experiment 1.2' and stop claiming completed verification.

**Effort**: 30 minutes (soften) or folds into A19's rerun

### A44 [MODERATE — FIXED 2026-08-18 except unused-entry pruning (P4)] — paper/references.bib (karakida2019universal, coil2025freezing, oleary2026spatial, huang2022modality, peng2022balanced, soudry2018implicit, bozzo2024multimodal + 5 unused entries)

**Issue**: Seven bib entries have verified-wrong or incomplete metadata (Karakida cited as arXiv though published AISTATS PMLR 89:1032-1041; Coil author/title wrong and the 02 text overstates the paper; O'Leary missing pages/DOI; huang2022 has journal={ICML}; truncated author lists; Soudry missing volume/pages) and five entries are never cited.

**Fix**: Apply all verified metadata fixes, soften the Coil text claim in 02:116-117, prune or deliberately cite the five unused entries, and delete the 'Phase 0' comment.

**Effort**: 1-1.5 hours

### A45 [MODERATE] — 02_related_work.tex section 2.4

**Issue**: The multimodal-balancing survey stops in 2022, omitting MMPareto, ReconBoost, and Classifier-guided Gradient Modulation (all 2024) — non-scooping but two years stale to that community.

**Fix**: Add one sentence noting recent gradient-modulation remedies (cite 2-3) all operate on explicit multi-branch fusion architectures, whereas this setting is single-input and sequential.

**Effort**: 30 minutes

### A46 [RE-SCOPED: ICLR reproducibility statement + code release, not Nature Portfolio pack] — submission package (no files yet)

**Issue**: Nature Portfolio compliance items are unprepared: Reporting Summary, code/data availability statements with concrete access routes (the real-data section uses in-house clinical tissue data, so ethics/availability documentation is mandatory), compute/hyperparameter reporting, and a broad-readership cover letter.

**Fix**: Prepare now (venue-conditional): public code repo for the synthetic sweeps + EGR implementation, a data availability statement distinguishing shared vs restricted, and a cover letter leading with frozen-random-beats-joint.

**Effort**: 1-2 days

### A47 [RESOLVED 2026-08-18 — venue decided: ICLR 2027 primary] — venue plan (decision item)

**Issue**: The venue decision is on the critical path for the ~2-month deadline: Nat MI needs the full A11 restructure with no promised timeline, while ICLR 2027 (abstract Sep 18, decisions Dec 16) maps almost directly onto the current shape, with AISTATS 2027, TMLR, and Nature Communications as verified alternatives.

**Fix**: Decide this week — recommended primary ICLR 2027 (best fit-to-effort, dated decision); if a Nature-branded venue is required, Nature Communications over Nat MI.

**Effort**: 1 hour (decision meeting)

### A48 [MODERATE] — 09_discussion.tex line 154

**Issue**: The 'first rigorous explanation' priority claim, combined with 02's mischaracterization of prior theory as empirical (A27), invites disputes next to Pezeshki 2021 and Huang 2022.

**Fix**: Drop 'first' or make it explicitly comparative: 'the first rigorous account of module-level gradient starvation in single-input compositional pipelines, complementing Huang et al. and Pezeshki et al.'.

**Effort**: 10 minutes

### A49 [MODERATE] — 04_theorem1_hessian.tex:197 vs 09_discussion.tex:4-42

**Issue**: 04 promises a Section 9 discussion of two-timescale/per-module learning-rate remedies that never appears, leaving the obvious reviewer question (why not raise the spectral module's LR?) unaddressed.

**Fix**: Add a fourth Implications-for-practice paragraph on per-module learning rates, their freezing limit (echoing 04:242-246), and the Adam evidence, keeping 04's conjecture framing.

**Effort**: 45 minutes

### A50 [MODERATE] — 01_introduction.tex:16; 02_related_work.tex:79-97; 09_discussion.tex:123-127

**Issue**: Sweeping 'dominant paradigm' claims about hyperspectral remote sensing cite zero remote-sensing works, and the field's 2024-25 turn toward frozen pretrained spectral encoders makes the claim attackable — though that trend actually supports the paper's prescription.

**Fix**: Add 2-3 canonical citations (Chen 2016 TGRS, HybridSN, SpectralFormer) and reframe: end-to-end remains the task-specific default while the frozen-foundation-encoder trend is an unexplained empirical convergence Theorems 1-2 explain.

**Effort**: 45 minutes

### A51 [MODERATE] — 08_real_data.tex lines 75-83

**Issue**: Observation 3 overclaims Theorem 2's scope: the theorem bounds the spectral gradient once the residual decays, but does not prove fine-tuning actively degrades pretrained features (0.90 -> 0.79) — the degradation story is a plausible narrative, not a 'predicted dynamical consequence'.

**Fix**: Hedge to 'consistent with the shortcut dynamics underlying Theorem 2' plus one sentence noting the theorem bounds gradient magnitude while the degradation direction is empirical (probed by Exp 1.7's predicted mid-training EGR collapse).

**Effort**: 15 minutes

### A52 [MODERATE] — 08_real_data.tex:7-8 and table caption line 54

**Issue**: Section 8 attributes the transfer experiment to 'FTIR/QCL' data, but the companion's source experiment is captioned breast QCL only — a provenance misstatement.

**Fix**: Say 'QCL infrared hyperspectral images' for the transfer experiment and mention FTIR only for the broader companion benchmark context, clearly separated.

**Effort**: 15 minutes

### A53 [MODERATE — PARTIALLY FIXED 2026-08-18 (SpatialMLP-only + range wording); CSV column claim check in E1] — 07_experiments.tex lines 33-37

**Issue**: The Exp 1.1 setup implies multi-architecture coverage that doesn't exist (all 50 measurements are SpatialMLP; no architecture column in the CSVs) and undersells the sweep range ('two orders of magnitude' for 16-8192, which is ~2.7).

**Fix**: Drop the other-architectures clause (or run and persist CNN/ViT versions) and say 'nearly three orders of magnitude'.

**Effort**: 15 minutes

### A54 [MINOR — FIXED 2026-08-18 (T3)] — 03_setup.tex lines 126-132

**Issue**: 'Residual vanishes exponentially as predictions saturate' is true in the logit margin but re-plants the retired exponential-in-time intuition that Theorem 2's polynomial-decay refactor specifically abandoned.

**Fix**: Reword to '...vanishes exponentially in the logit margin as predictions saturate — polynomially in training time on separable data (Section 5)'.

**Effort**: 10 minutes

### A55 [MINOR] — 05 label thm:twoscale + line 94; 04:185; 03:236; 07 Exp 1.2 title

**Issue**: Residual slow-fast/two-timescale vocabulary survives in prose ('linearized fast subsystem', 'converges quickly on the fast manifold'), implying a proven slow-fast decomposition the refactored Theorem 2 does not establish and misstating the actual hypothesis at 04:185.

**Fix**: Change 04:185 to '(e.g., that the shortcut pathway fits the residual at rate mu, Assumption ass:residual)', scrub 'fast manifold' from prose, and optionally retitle Exp 1.2 'Starvation dynamics under joint training'.

**Effort**: 30 minutes

### A56 [MINOR — FIXED 2026-08-18] — 04:85-88 vs supplement.tex Steps 2-3

**Issue**: The supplement applies Karakida's squared-loss trace machinery to the softmax-CE FIM without flagging the loss transfer, and the theorem says 'Gaussian initialization' while the proof says 'Kaiming'.

**Fix**: Add one supplement sentence on the squared-loss-to-CE transfer via the bounded inner matrix and unify wording as 'Kaiming (variance-scaled Gaussian)'.

**Effort**: 20 minutes

### A57 [MINOR — FIXED 2026-08-18 (T3: lem:schur -> lem:opcap)] — 04_theorem1_hessian.tex:109 etc.; supplement.tex:13-21 vs 04:24-27

**Issue**: The stale label lem:schur (no Schur complement remains — the fossil that caused 02's misdescription) would signal old framing if source is shared, and the supplement's Theorem 1 restatement omits S from the constant-dependency list.

**Fix**: Project-wide rename lem:schur -> lem:opcap and sync the supplement restatement's dependency list.

**Effort**: 20 minutes

### A58 [MINOR — FIXED 2026-08-18] — 03_setup.tex lines 99-101

**Issue**: Marchenko-Pastur is cited for a Theta(1) operator-norm (edge) claim that belongs to Bai-Yin/Geman (or Vershynin HDP Thm 4.4.5).

**Fix**: Cite Bai-Yin (or Vershynin), optionally keeping MP as background.

**Effort**: 15 minutes

### A59 [MINOR] — 05_theorem2_twoscale.tex lines 89-92

**Issue**: 'See Pezeshki et al.' carries a fitting-rate-grows-with-width claim the source does not make (it analyzes NTK-coupled feature starvation).

**Fix**: Cite NTK convergence-rate work (Du 2019 / Arora 2019) for the kernel-eigenvalue claim, keep Pezeshki for the starvation mechanism, or hedge to 'expected to grow with width'.

**Effort**: 15 minutes

### A60 [MINOR] — 07_experiments.tex lines 118-119

**Issue**: 'Frozen beats joint at every width' for the ViT is true on mean final test accuracy but false on mean final test loss at D=64 (frozen 3.22 vs joint 2.96), and the metric is unstated.

**Fix**: Add 'on test accuracy'.

**Effort**: 5 minutes

### A61 [MINOR — FIXED 2026-08-18 (T3)] — 05_theorem2_twoscale.tex lines 66-74

**Issue**: ass:residual never states that C and mu are trajectory/initialization-dependent constants, while the capacity claims quantify over architectures.

**Fix**: Add 'where C and mu may depend on the initialization, data, and architecture; the theorem holds for any trajectory admitting such an envelope'.

**Effort**: 10 minutes

### A62 [MINOR] — 09_discussion.tex lines 94-98

**Issue**: The PCA-triviality defense conflates two claims — frozen PCA performing well shows fixed unsupervised features suffice, not what a jointly-trained 'PCA-decomposed f_theta' (undefined) would do.

**Fix**: Rewrite around the routing argument: PCA is label-independent, and the frozen-PCA baseline works precisely because its features never route the supervised signal through theta — the identified failure is of that routing, not of linear features per se.

**Effort**: 20 minutes

### A63 [MINOR — FIXED 2026-08-18] — 04_theorem1_hessian.tex:136-138; supplement.tex:216-230

**Issue**: Draft-history meta-commentary ('earlier drafts of this proof attempted...') is inappropriate for a submission, though the retirement content itself is correct and worth keeping.

**Fix**: Rephrase impersonally: 'An alternative route via lambda_min^+ or Kronecker factorization requires additional structural assumptions; the operator-norm cap avoids both'.

**Effort**: 15 minutes

### A64 [MINOR] — code/experiments/exp1_1.py; results/exp1_1_kappa_vs_ratio.png; EXPERIMENT_INVENTORY.md lines 14, 100-105, 146-148

**Issue**: Retired kappa branding is baked into shipping artifacts — CSV column 'kappa', a figure titled 'condition number across modules', and inventory text describing Theorem 1 as 'kappa growth' with Exp 1.8 scoped to 'validate kappa at ~325' (the wrong quantity against the wrong target).

**Fix**: Rename kappa -> D_curv in code outputs and plots, regenerate the exp1_1 PNGs before figure placement (ties into A10), and restate Exp 1.8 as power-iterating GN-block top eigenvalues to compare D_curv against c_1'*sqrt(C_g/C_f) at the corrected ratio (A4).

**Effort**: 1-2 hours

## Auditor's original 14-step order (superseded by the theory-first plan above; kept for reference)

1. STEP 1 (this week, ~1h): Venue decision — A47/A11. Choose Nat MI vs ICLR 2027 vs Nature Communications; the word/figure budget gates every structural fix downstream. Recommended: ICLR 2027 primary.

2. STEP 2 (week 1, ~1 day): Integrity batch, independent of all experiments — fix the berisha2019deep author list (A3, 10 min), rewrite the Karakida supplement proof around the real Theorems 1 and 4 (A1+A2+A30, incl. the residual-vs-logit transfer step A13 and the minor A56/A57 label/scope fixes), and fix the misattributed quotes (A32). This removes the fabrication exposure first — the category the previous review already burned the paper on.

3. STEP 3 (week 1, ~half day): Ratio recount — extract the spatial-module parameter count from the actual BlockViT v2 checkpoint and reconcile 13M vs the companion's 18.3M (A4), then propagate: Section 8 prediction (A5), Prop 6.1 worked number (within A22), 'orders of magnitude' claim, and the kappa->D_curv rename in code/figures/inventory (A64).

4. STEP 4 (week 1-2, decision + ~half day or 1-2 days): Scaling-exponent story — decide rerun-with-Theta(M^2)-architecture vs honest reframe (A6), then rewrite 04/07/09/supplement as ONE story; this absorbs the fabricated Karakida figure cite (A30) and the Exp 1.1 setup fixes (A53). If rerunning, launch the sweep now and write while it runs.

5. STEP 5 (week 2, ~1 day): Theory-hygiene pass on statements and constants — normalization convention (A12), finite-horizon B_T + trajectory containment (A23), D_curv definition (A35), assumption lists (A36), Prop 6.1 downgrade-or-prove (A22), GGN relabel-or-implement (A24), counters/titles (A37), dead lemma (A38), softmax remark (A39). Fold in the ALREADY-PLANNED Theorem 2 supplement items (envelope verification, scoping section at supplement.tex 309+) here so the supplement is completed in one pass.

6. STEP 6 (week 2, launch early — longest pole): Real-data runs — start the ALREADY-PLANNED Exp 1.7 (~3h; unlocks Section 8's Diagnostic subsection, the EGR-collapse figure, and the A51 probe) and Exp 1.8 (D_curv validation, restated per A64 at the corrected A4 ratio). If Section 8's four missing variants (A7) are to be run rather than cut, start those 5-fold jobs now.

7. STEP 7 (week 2-3, ~1 day): Section 8 rebuild (the ALREADY-PLANNED rewrite, now fully specified) — strictly artifact-backed rows with +/-SD (A7), corrected ordering claim (A8), honest companion description (A29), QCL provenance (A52), hedged Observation 3 (A51), retired-kappa deletion (A5 residue).

8. STEP 8 (week 3, ~1.5 days): Experiments-section repairs — write the ALREADY-PLANNED Exp 1.6 subsection into Section 7 with the notes' honesty caveats (A17), rerun exp1_2v4 persisting per-run EGR CSVs and fix all Exp 1.2 headline numbers (A19, also resolves A43's envelope data), fix Exp 1.3 seed SEs (A20), replace the pseudo-replicated t-test (A21), correct the five Section 6.5 statistics (A34), and rewrite the Robustness subsection to claim only what exists or run the SGD/S variations (A18).

9. STEP 9 (week 3, ~half day): Pointer/claim reconciliation — fixed-baseline intervention pointer and hedge (A25), nonlinear-f_theta run-or-soften (A26), Section 5 cross-references (A43 if not resolved in Step 8), EGR framing fixes (A40, A41).

10. STEP 10 (week 3-4, ~half day): Remnant-prose sweep — one pass rewriting every sentence that PARAPHRASES a theorem, checked against the (now-clean) theorem statements: A14, A15, A16, plus minors A54, A55, A61, A62, A63. The theorem statements are sound; the connective prose is where the retired framing lives.

11. STEP 11 (week 4, ~1 day): Related-work and citation pass — Huang 2022 into 2.4 (A27), four anchor citations (A28), 2024 multimodal wave (A45), remote-sensing citations (A50), bib metadata (A44), Bai-Yin (A58), Pezeshki scope (A59), priority claim (A48), the promised 09 optimizer paragraph (A49), adjacent Hessian papers (A33), universality hedge (A31), timescale-hypothesis wording (A42).

12. STEP 12 (week 4-5, ~1 day): Figures — place the headline figures with regenerated D_curv-branded exp1_1 plots and the Exp 1.7 EGR figure (A10, A64); recompile until no undefined references.

13. STEP 13 (week 5, ~1 day): Completion sweep (A9) — resolve every remaining TODO, complete or delete the supplement robustness stubs against what Steps 6-8 actually produced, fix main-text references to them, verify the Prop 6.1 empirical check exists, and write the abstract LAST.

14. STEP 14 (week 5-6): The ALREADY-PLANNED final polish/trim plus venue execution — perform the A11 restructure per the Step 1 decision, prepare the compliance/cover-letter package if a Nature venue (A46), ViT metric qualifier (A60) and any stragglers, then a final front-to-back consistency read against this report's table.

## Pre-existing planned work absorbed into the steps above

- Theorem 2 supplement items: discrete-SGD extension, informal mu(C_g) derivation,
  empirical residual-envelope check (Step 5/13)
- Exp 1.6 write-up into Section 7 (Step 8)
- Exp 1.7 real-data EGR + Exp 1.8 real-data disparity (Step 6 — longest pole, start early)
- Section 8 rewrite on real artifacts (Step 7)
- Final polish / trim to venue format (Step 14)
