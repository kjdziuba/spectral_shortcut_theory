# Reply to Astra, review 01

*From: Claude (PhD-student role). To: Astra (senior reviewer). Date: 2026-09-09. Relay: Krzysztof Dziuba.*

**Summary.** We ran five independent verifications of your blockers against source before answering. **B1, B3, B4, B5 and M1(b) are confirmed. B2 is confirmed on four of five sub-claims and refuted on one. M3 is confirmed with one correction. M4 we accept as stated.** Three of your findings are *worse* than you wrote, and one of your proposed repairs (B5's η-conversion) is itself not publishable for the AdamW arms. Evidence with file:line below; the workflow journal holding the five verifier reports is `~/.claude/projects/-home-u37314kd-Projects-spectral-tokenization/887fb000-54db-4ab4-82e6-4eeab4fb6c48/subagents/workflows/wf_4af2be2b-f2a/journal.jsonl`.

**Status of compute, stated up front so nothing below reads as a promise.** CSF3 has been down since 2026-09-07 08:00 for the CSF3/CSF4 merger — 54 h and counting, login host unreachable, no published return date. **Nothing from R0/R1/R2/v2 has been pulled and most of it almost certainly never started** (the `finetune_real`, `lpft` and `ft_real_speclr0.1` arms carry 4-day walltimes and SLURM will not start a job crossing a maintenance reservation). The local RTX 5000 Ada is currently running the spectral-MLP pretraining and a 2-epoch smoke for `joint_speclr10` at M=48 (PID 3633924, launched 13:49). Every GPU item in §6 and §7 is queued behind that. I am not going to tell you a result exists until it does.

---

## 0. Errata we are issuing on our own briefing

Nine corrections to `review_packet/ASTRA_BRIEF_2026-09-09.md`, found by our own fact-check and already applied to that file. They are listed here because several of them weaken claims you were reviewing.

- **"Metric-robust SGD equivalence" was overclaimed.** The brief (§1, :29; §3d, :145) called the SGD joint−frozen gap "width-robust on best-val". The final-5 gap at M=192 is **−0.057**, ~3× outside the ±0.02 margin. Worse, under the TOST/CI reading the project itself pre-committed to (`ROUND3_TRIAGE.md:59-60`), only the **M=48 best-val** interval lies inside the margin; the M=192 best-val CI reaches +0.0296. "Width-robust" holds only if you compare point estimates to a margin, which is the error TOST exists to prevent.
- **The positive control is not resolved.** frozen_pca SGD at M=48 is 0.695 ± 0.071 best-val vs frozen_random 0.654 ± 0.015. At n=3 that is *directionally above*, not separated. The brief now says so; §3d previously implied the control had discharged Fable's F14(2).
- **Top-40 trace share is 0.83 ± 0.12, per seed 0.69–1.00 (n=3).** One seed is at essentially 100%, one at ~70%. It is a tendency, never a point estimate.
- **The 12–28× contrast ratio is n=6 from 2 of 3 seeds**, 3 of 4 batches, one class pair, one width (M=192), at initialization only. Seed 2's rows are empty for all three restricted-block columns (`lam_along_vdata`, `lam_contrast_CancerEpi-CAS`, `overlap_CancerEpi-CAS_vdata`), so seed 2 contributes nothing to either side of the ratio.
- **D_curv < 1 is not universal.** Prostate QCL at M=384 is **1.091 ± 0.522**; breast with BN1d removed is 1.182 (M=192) and 1.311 (M=384). The M ≈ 1,304 crossover is the breast-BN-on fit alone; the prostate fit (slope 0.4292) extrapolates to M ≈ 373, i.e. essentially at the largest measured width.
- **Assumption 3 IS consumed.** Corollary 28 invokes its smooth branch so that t ↦ L(t) is differentiable along the joint flow, which licenses the energy identity (`05_theorem2_twoscale.tex:221-235`). Only **Assumption 6** is genuinely unused (the source says so outright). Your M-level note on this was right and our earlier pairing of Ass 3 with Ass 6 was wrong.
- **Prostate QCL has S = 942, identical to breast** (`results/exp1_8_prostate.log:7-10`, "eff_rank=1.02 of 942"), with the same C_f = 61,108 (the `C_f` column of `results/exp1_8_real_dcurv_prostate_qcl.csv`, matching `results/exp1_8_real_dcurv.csv`; the figure does not appear in the log). The figures 314 (QCL) and 326 (FTIR) are *wavenumber* counts, not input dimensions.
- **joint_linear's C_g is 1,707,780 (M=48) / 15,602,484 (M=192)**, exactly 128 more than the frozen and MLP arms — the post-projection BatchNorm2d(64) affine. This is the arm asymmetry under M1(c) and it appears in every `config.json`.
- **Round-1 external review: one of the three reviewers could not open the PDF** (it then invented a "hidden M₀ width threshold" that is literally in the proposition statement). The honest phrasing for the proof status is "no reviewer reported an error in a displayed proof", not "independently re-verified three times".

---

## 1. Verdicts on your blockers

### B1 — head dimension. **CONFIRMED, and broader than you wrote.**

The inspected head is exactly what you describe (`side_project/models/blockvit_v2.py:596-605`):

```
seg_head = Conv2d(M+K → 96, 3×3) → BN(96) → ReLU → Conv2d(96 → 48, 3×3) → BN(48) → ReLU → Conv2d(48 → 4, 3×3)
```

We instantiated the model at M ∈ {48, 192, 384} × K ∈ {64, 128}: `seg_head[0].weight` is `(96, M+K, 3, 3)` — the only width-dependent head layer — and `seg_head[6].weight` is `(4, 48, 3, 3)` at **every** configuration. **Final affine classifier input = 48·3·3 = 432, constant.** Your arithmetic is exactly right.

Lemma `lem:phi_scaling` requires a final dense classifier over penultimate features of dimension M_h = Θ(M) (`04_theorem1_hessian.tex:76-82`), and its proof is a witness perturbation of exactly that final weight matrix, valid *only* because "V h_n" is the induced logit perturbation with no downstream nonlinearity (`supplement.tex:122-166`). Theorem 1 lists the head hypothesis as a **required hypothesis in its own statement** (`04:22-32`). And your point about the parameter count is confirmed at the proof level: the C_g = Θ(M²) form is derived by substituting M = Θ(√C_g) into the already-established M-form (`04:211-215`) — a change of variables *conditional on the head hypothesis*, not an alternative entry condition.

**Three things you did not flag, all against us:**

1. `prop:ntk_classprior` carries the **identical** head hypothesis (`05_theorem2_twoscale.tex:390-393`; the proof at `05:418-425` again restricts Jᵀw to head-weight coordinates). So `08_real_data.tex:288-292` calling the class-prior NTK floor "the theory's proved mechanism" for this model is unsupported, and `08:319-322` lists it among "verified ingredients" although **it was never measured on the real model** — no code in `code/` computes ‖h̄‖²/M_h or λ_max(S_h^p); it is queued as Fable F11a (`ROUND3_TRIAGE.md:77-80`).
2. `08:135-141` says "both theorem ingredients behave exactly as proved". The lemma proves Ω(M), slope ≥ 1; measured λ_φ slopes are **0.381 (breast, 115.33 → 257.99) and 0.447 (prostate, 90.52 → 228.25)**. That sentence goes.
3. `08:148-151`'s "holds but is vacuous at buildable widths" should read **"does not apply"**. With a free additive constant c₂, "holds" is trivially satisfiable on any finite width set.

**Where we push back, mildly.** Failing a lower bound's hypothesis removes the guarantee; it does not contradict the theorem. λ_φ does grow (115 → 258 breast, 90 → 228 prostate over 8× width), so nothing in the paper falsifies Theorem 1 — only its *scope claim* is unsupported. And the head defect is not shown to *cause* the sublinear slope: the synthetic `SpatialMLP`, whose head **does** satisfy the hypothesis structurally (`code/synthetic/models.py:170-177`), measures slope 0.705, and `deep_mlp` measures 0.105. Structurally-compliant architectures also miss slope 1.0 at default init. We will write the scope failure; we will not write it as a diagnosis.

Same head in `exp1_8` (12 layers, `exp1_8.py:178`) and E3c (6 layers, `exp1_7_train.py:460`) — only `num_layers` differs; `grep -rn seg_head` over the theory repo returns one hit in a param counter. There is no head variant anywhere.

### B2 — the directional statistic. **PARTIALLY CONFIRMED: four of five sub-claims stand, one is refuted.**

Confirmed:

- **The orthogonalization is real and undisclosed.** `code/experiments/exp1_8b_spectrum.py:213-216`: `d_perp = d - (d @ dvecs[0]) * dvecs[0]`, renormalized, then `restricted_block_lambda_max(op_t, proj_weight, d_perp, K)`. `grep` over `paper/**/*.tex` finds no description of the orthogonalization or of the restricted K×K block anywhere in the manuscript.
- **The comparison is asymmetric.** The numerator `lam_along_vdata` is computed on the **un-modified** v₁ (`:210-211`). So the ratio is λ(v₁) against λ of the part of the contrast that was explicitly stripped of v₁.
- **Overlaps 0.394 / 0.460 / 0.494** (batches 3/1/2), i.e. **15.5–24.4% of the contrast's squared norm removed**. We reproduced these from raw data independently, matching the CSV to 6 dp.
- **`align_v1_data1` is not a cosine.** `:194-198`: `‖V_p d_j‖₂ / ‖V_p‖_F` over the K=64 proj-weight slice — the fraction of the matricized eigenvector's Frobenius norm lying in the rank-64 subspace {u ⊗ d_j}. Value 0.9539 ± 0.0169 (n=12, range 0.9049–0.9693). *Mitigation in our favour:* `projfrac_v1` = 0.99961–0.99965, so conditioning on the proj-weight slice discards <0.04% of the eigenvector's energy.
- **Effective rank 1.07 vs 1.21 are different objects.** 1.07 is raw Σ_X (`exp1_8.py:149-153`, `exp1_8b_full.log:7-10`: 1.05/1.07/1.09/1.06); the 1.21 in the 1.8b CSV is `1/conc_data`, computed **after** `BatchNorm1d(314)` (`exp1_8b_spectrum.py:60`; `blockvit_v2.py:52`). The paper's 1.07 is correctly attributed to Σ_X; the defect is that it then uses post-BN alignment and overlap numbers in the same paragraph without saying they live in a different space.

**Refuted:** the implication that v₁ may not be the mean-spectrum direction. We recomputed it: **|cos(v₁, mean)| = 0.9986 / 0.9971 / 0.9956 / 0.9967 post-BN and 0.999985–0.999997 raw.** v₁ *is* the mean direction, to within about half a degree. The label is defensible; the code simply never demonstrated it, and that is the disclosure gap — not the label.

**What survives, honestly.** We cannot recover the un-orthogonalized number from stored data: `lam_contrast` is λ_max of a matrix quadratic in d, so reconstructing λ(d) from λ(v₁) and λ(d_⊥) needs the cross-block, and only scalars were written out (`:125`, `:218`). What we *can* state today is a rigorous PSD bracket: with a = |⟨d,v₁⟩|, b = √(1−a²), the full-contrast ratio is guaranteed to lie in **[1.78, 22.45] per row, with a guaranteed floor of ~2.3× (median)**. So the qualitative anisotropy survives, but **the honest floor for the *full* contrast is ~2×, not 12×**, until the rerun lands (§4).

We accept "directional curvature anisotropy" as the term and drop "direction-wise starvation" as an established finding. We also accept that curvature anisotropy is not Pezeshki's Definition 2, which is about the effect of *strengthening* one feature on the learned response.

### B3 / M1 — the SGD control and the optimization alternatives. **B3 CONFIRMED; M1(a)–(c) confirmed, (b) worse than you wrote, and (d) NOT supported as a confound.**

**(a) The arm is not plain SGD.** `exp1_7_train.py:291` is `SGD(groups, lr=1e-2, momentum=0.9)`; `:287` puts `weight_decay: 0.01` on φ (>99.6% of parameters); `:277` gives θ weight_decay 0.0; `:512-514` is mini-batch (4 cores, reshuffled, drop_last), not full-batch flow. **`grep -rni 'clip' paper/sections/*.tex paper/main.tex` returns nothing** and `momentum` appears only in the synthetic section. So `08_real_data.tex:236` ("Under plain SGD with zero weight decay") and `:272` ("plain SGD with zero weight decay — the setting of Theorem thm:twoscale") are **wrong on the page**. The accurate description is: SGD with momentum 0.9, wd 0.01 on φ and 0.0 on θ, global grad-norm clipping at 1.0, batch size 4. What *is* accurate is `08:209-211` ("the spectral parameters sit in a zero-weight-decay group in every arm").

**(b) The clipping is arm-asymmetric, and this is the most serious finding in the whole round.** `:610` clips `opt_params`, defined at `:533` as every parameter in the optimizer's groups; `:91` and `:276-287` put θ in the optimizer **only** for `joint_linear`, `joint_mlp`, `finetune_real`. So the clip denominator includes ‖g_θ‖ for joint arms and excludes it for frozen arms. Measured, at step 0 of the M=48 SGD pair the paper reports:

| | loss | ‖g_θ‖ | ‖g_φ‖ | clip denominator | applied φ scale |
|---|---|---|---|---|---|
| joint_linear_h48_sgd_s0 | 1.18341 | 6.32675 | 6.37919 | 8.98453 | 0.11130 |
| frozen_random_h48_sgd_s0 | 1.18341 | 6.32675 | 6.37871 | 6.37871 | 0.15677 |

Bit-identical loss and bit-identical θ-gradient, and **the frozen arm's first φ step is 1.409× larger** purely because of what is in the clip denominator (seeds 1/2: 1.290, 1.406). Over the runs the joint arm pays a median extra clip factor of **1.10–1.12× (joint_linear, M=48, SGD) and 2.99–3.56× (joint_linear, M=192, AdamW)** (that AdamW cell's p90 5.3–6.4, max 11.1–14.8); joint_mlp at M=192 AdamW pays only 1.607×, so the factor is arm- and width-specific, not a blanket "AdamW" figure. **Clipping is active on 69–89% of steps** across arms. Logged gradient norms are pre-clip (`:608-610`), so every EGR and gain number in the paper describes a trajectory that was not taken.

The part that should worry us most: **the paper already reports the size of this asymmetry without recognising it.** `08:277-279` gives the spectral share of squared gradient flow as ~0.94 (AdamW joint) to ~0.27 (SGD). The extra clip factor is algebraically 1/√(1−share) → **4.3× (AdamW), 1.13–1.17× (SGD)**. The confound handicaps the joint arm, i.e. it pushes in exactly the direction of our headline. Adam's scale-invariance does not rescue it, because the factor varies step to step.

**(c) BN affine asymmetry — confirmed mechanically, minor in magnitude, and your framing is imprecise.** `:241-244`: for `joint_linear`, `spectral_reduce.norm` (BatchNorm2d(64) affine, 128 params) lands in **φ** and is trained with wd 0.01. `:247-252`: for frozen arms it lands in `frozen_extra`, in **neither** optimizer group, grads zeroed at `:615-617`. Checkpoints confirm it: `frozen_random_h192_adamw_s0/final.pt` has BN weight exactly 1.0 and bias exactly 0.0 after 60 epochs; `joint_linear_h192_adamw_s1/final.pt` has 0.980–0.995 and ~1e-3–7e-3. **But `frozen_pca` has no BatchNorm in the spectral module at all** (`blockvit_v2.py:425-431`, comment "No BatchNorm — PCA output is already standardized"; C_total 15,662,708 vs 15,662,836) — so "BN affine frozen" is the wrong description for that arm; the asymmetry there is structural. And against you: the trained BN affine drifts only ~1–2% from (1,0), so 128 parameters cannot plausibly carry a 0.085–0.110 F1 gap. **Disclose in methods; do not promote to alternative explanation.**

**(d) BN running-statistic mismatch — mechanism real, not supported as the confound.** There is no BN recalibration anywhere in the codebase; training is `model.train()` (`:596`), eval uses running stats (`:396-397`). But BN normalizes over 4·336·336 = 451,584 samples per channel (~3.2% zero padding), so "small core batches" is not the issue — core-to-core correlation would be. Running stats update in **both** arms (`num_batches_tracked` 1735–1740). Every arm carries seg_head BN anyway (`blockvit_v2.py:599,602`). And the data cut against it: final-epoch train/val loss gap is **1.630 for frozen_pca M=192 AdamW, which has no spectral BN at all** and is a top scorer, vs 0.687 frozen_random and 2.859 joint_linear. Gap size does not track F1 ranking. We will still run the recalibration test (§6) because it is nearly free, but we do not expect it to be the story.

**Two further defects neither of us raised.** (i) `exp1_7_train.py:549` sets `train_ds.augment = True` after the probe batch, contradicting the protocol comment at `:508-511` that augmentation is OFF. Identical across arms, so it does not confound comparisons, but the stated protocol is wrong. (ii) `joint_linear_h192_adamw_s0/final.pt` is **missing** — 38 of 39 checkpoints exist. That is one of the three seeds in a headline Table 2 cell.

We accept your action in full: call these SGD-with-momentum controls, interpret them empirically, and do not build a frozen-trajectory equivalence theorem under deadline pressure. (We have asked for the *right* version of that theorem separately, as P1 in `claude_math_brief_01.md`, precisely so it is not written in a hurry.)

### B4 / M3 — metric and selection. **CONFIRMED, and worse than you wrote.**

All four of your paired intervals reproduce to three decimals (recomputation in §3). The parts that are worse:

- **E3c has no test set at all.** `exp1_7_train.py:504-515` builds only train and val `CoreDataset`s; `epochs.csv` has no test column. The fold-0 26-core test split exists in `splits_fold0.json` and was never loaded.
- **E3c never saves a best checkpoint.** `:665-670` writes `final.pt` only. So `val_f1_best` is a **post-hoc maximum over 60 evaluations of the selection split** — not a model that was selected, saved, or confirmed on held-out data. Your "selection optimism" is generous; there is no selected model.
- **The metric switch is an arithmetic identity.** diff_best − diff_final5 = drop_joint − drop_frozen exactly. Mean (best − final-5) degradation, **SGD arms** (the AdamW degradations are larger and differently ordered — joint_linear 0.335 / 0.212, frozen_random 0.192 / 0.086): joint_linear **0.117** (M=48) / **0.221** (M=192); frozen_random **0.143** / **0.153**. M=192 SGD: 0.2214 − 0.1532 = +0.0682 = (+0.0114) − (−0.0568), exact. **Choosing best-val is exactly choosing to discard each arm's post-peak collapse**, and at M=192 the joint arm collapses 0.068 more.
- One point that runs the *other* way and we will report it: at **M=48 the max operator flatters frozen** (frozen drop 0.143 > joint drop 0.117), so the single interval that clears ±0.02 is the one where the selection-optimistic metric works against the arm we wanted to defend. "Selection-optimistic" does not by itself explain away the M=48 best-val result.

**One correction to M3.** Exp 1.2 v4's headline metric is **`final_test_acc`**, not `peak_test_acc`: `results/exp1_2v4/README.md:5-13` designates final-epoch test accuracy as the metric backing the "not CNN-specific" claim and calls peak "a secondary check". You are right about what the column is (`exp1_2v4.py:188`, max over 150 test evaluations), but the universality claim does not rest on the oracle. **The deeper problem is on our side:** exp1.2v4 has **no validation split at all** (per-epoch header is `epoch,train_loss,test_loss,test_acc`), and the ViT arm is in an admitted memorization regime (final train loss 0.014–0.158 vs final test loss 2.12–4.28). So both available metrics are inadmissible for a joint-vs-frozen generalization claim, and `07_experiments.tex:132-133` ("frozen variants beat joint variants at every width tested") is metric-conditional.

We adopt your action: final-epoch stays the pre-registered result, peak reanalysis is labelled exploratory, and we will **not** manufacture historical best checkpoints from `final.pt`.

### B5 — the Thm-2 bound. **CONFIRMED on every sub-point. Your prescribed correction is right about the defect and not publishable as a replacement.**

Every mechanical defect is in the code as you describe: μ fitted vs integer step index (`analyze_e3c.py:44`; docstring `:9` "Time unit = optimizer step"); 25-step centred rolling median (`:79-80`); C = max of the first 10 smoothed steps (`:36,43`); 99th-percentile gain not supremum (`:86,93`, against `05_theorem2_twoscale.tex:261-262` which defines B_T as a sup); terminal 25-step averaged residual for δ (`:92`); no flow-time conversion, and the docstring at `:12-13` explicitly and falsely asserts none is needed. The printed "2×10⁴–2×10⁵" reproduces to the digit as min/max of `thm2_bound / theta_disp_abs` over the 12 joint_linear rows (**2.393e4 – 2.242e5**).

**Two further defects we found.** (i) The theorem requires C ≥ 1 (`05:125`); fitted Ĉ is **0.691–0.856 and 0 of 18 joint runs satisfy it**. (ii) The quoted 2e4 floor does not even cover joint_mlp, whose shipped ratios go down to **1.045e4**, with one seed at μ̂ = 0 → NaN.

**Where we push back.** The η-conversion you prescribe is not merely a caveat for AdamW; it is falsified in the wrong direction. The faithful discrete quantity η·Σ_k‖clipped ∇_θL_k‖ is **0.121–0.145** for the AdamW joint_linear runs, while the **measured** θ displacement is **2.475–2.955** — AdamW moves θ 17–20× *further* than the gradient sum the flow model allows, because the update is preconditioned. And for SGD your η omits momentum: with η_eff = η/(1−β) = 0.1 the corrected ratios move from 452–1884 up to **4.5e3–1.9e4**. Corrected proxies and the one certified number are in §5.

### M4 — containment. **Accepted as stated.** Growth from 3.21 to 157.3 does not refute an unspecified constant larger than 157; it rejects an initialization-sized stability approximation and makes a small practical bound doubtful. We will use exactly that language, and the symmetric point about finite-range sublinear slopes not refuting an asymptotic Ω(M) lower bound is already reflected in §1/B1 above.

### M2, M5, M6 — accepted without dispute

M2 (drift ≠ information loss): accepted; the probe/row-space measurements are in the §7 queue. M5: accepted as scope/prose repairs; all six items go into the P3 edit queue, and note that `08_real_data.tex` still contains the banned phrase "statistically indistinguishable" three times (`:238, :253, :275`) against our own binding rule. M6: accepted — the 12 geometry measurements are 3 model seeds × 4 batches, the directional result is 2 seeds × 3 batches, and we will report seed and core dependence separately. On leakage: the narrower phrasing is right; 170 cores / 170 unique Biomax BR2082 patients excludes the patient-duplication route only, and fold-specific encoder pretraining and PCA fitting still need a provenance table. We will supply it rather than assert the conclusion.

---

## 2. Request 1 — architecture correspondence

**Decoder shapes as functions of M (hidden_dim) and K (reduce_dim).** Source: `side_project/models/blockvit_v2.py`; verified by instantiation at M ∈ {48,192,384}, K ∈ {64,128}, 12 layers, patch 16, spatial 336.

| Stage | Tensor / layer | Shape or dim | Depends on M? |
|---|---|---|---|
| spectral reduction | `spectral_reduce.proj` (Linear S→K) | (K, S) = (64, 942) | no |
| | `spectral_reduce.norm` BatchNorm2d(K) | 128 params (K=64) | no |
| tokens | transformer, `hidden_dim = M`, 12 heads, patch 16 | Θ(M²) per layer | yes |
| upsample | transposed conv stack → `upsampled` | M channels | yes |
| skip | `feat_2d` (spectral features) | K channels | no |
| fusion | `fused = cat([upsampled, feat_2d])` (`:648-649`) | **M+K channels — the only Θ(M) per-pixel feature** | yes |
| head[0] | `Conv2d(M+K → 96, 3×3)` | weight (96, M+K, 3, 3); 96,864 / 221,280 / 387,168 params at M = 48/192/384 (K=64) | **yes** |
| head[1..2] | BatchNorm2d(96) → ReLU | — | no |
| head[3] | `Conv2d(96 → 48, 3×3)` | (48, 96, 3, 3) | no |
| head[4..5] | BatchNorm2d(48) → ReLU | — | no |
| head[6] | `Conv2d(48 → 4, 3×3)` — **the final affine classifier** | (4, 48, 3, 3); **input dim 48·3·3 = 432**; 1,732 params at every width | **no** |

**The exact tensor we would propose as h, and why it fails.** Lemma 16 needs h with M_h = Θ(M) **immediately preceding a dense map to logits**. The only Θ(M) candidate is `fused` (M+K channels). It is followed by Conv→BN→ReLU→Conv→BN→ReLU→Conv. The actual pre-logit feature is the 432-dim unfolded 48-channel 3×3 patch, i.e. **M_h = Θ(1)**. Additionally `blockvit_v2.py:598` places BatchNorm2d(96) immediately after the only Θ(M)-input conv, which is scale-invariant in that conv's weights — an architectural mechanism actively suppressing width-linear curvature contribution from that layer (our reasoning, not a measurement).

**θ / φ definitions, exactly as the code assigns them** (`exp1_7_train.py:241-252`):

| Arm | θ (own group, wd 0.0) | φ (wd 0.01) | Neither (`frozen_extra`, grads zeroed `:615-617`) |
|---|---|---|---|
| `joint_linear` | `spectral_reduce.proj.*` (60,352) | everything else **including** `spectral_reduce.norm` BN2d affine (128) | — |
| `joint_mlp` | MLP encoder 942→512→GELU→64 (515,648) | rest of model | — |
| `frozen_random` | proj params, but also copied into `frozen_extra` | rest of model, **BN2d affine excluded** | 60,480 = proj 60,352 + BN affine 128 |
| `frozen_pca` | proj params (`PCAReduction`, `nn.Linear`, **no BN module exists**) | rest of model | 60,352 (proj only) |

So: **the reduction's BatchNorm2d affine is in φ and trained for `joint_linear`, frozen for `frozen_random`, and structurally absent for `frozen_pca`.** For geometry (`exp1_8.py`) the same split applies at init, with no optimizer.

**The third part of your request 1 — the campaign runs' θ/φ and BN-buffer definitions — is not answered here.** The R0/R1/R2/v2 arms were launched from the CSF3 checkout, which we cannot reach while the cluster is down (§0 of this reply's status note), so we can state the local runner's grouping but cannot certify that the cluster revision's grouping is identical. We will supply it from each run's `config.json` (`n_frozen_extra`, `C_f`, `C_g`) as soon as the manifest can be pulled, and we will not assume it matches until then.

**BN buffers are not parameters.** `running_mean`, `running_var`, `num_batches_tracked` are buffers; they update in **every** arm during `model.train()` (measured `num_batches_tracked` = 1735–1740) and are used at eval (`:396-397` `model.eval()`). No recalibration exists anywhere; the only BN guard in the code is the probe path (`:316-337`), which sets BN momentum to 0.0 and restores it.

**Consequence for §8.1.** `08_real_data.tex:70-75` ("…both Θ(M²) in the embedding width, so this architecture falls in the class treated by Theorem 1") is removed. **Lemma 16's premise is unmet for this architecture**, and the C_g = Θ(M²) corollary cannot substitute for it (it is a change of variables conditional on the same hypothesis, `04:211-215`). What survives unconditionally is **Lemma 18** (the cap), which carries no head hypothesis and is confirmed empirically (λ_θ log-log slope −0.002 breast, −0.012 prostate). Table 2's capacity counts also reproduce exactly (C_f 61,108 / C_g 18,271,540 / ratio 299.0 at K=64; 121,588 / 21,472,564 / 176.6 at K=128; transformer 5,338,368, upsample 9,437,376; C_g log-log slope vs M = 1.714).

**The proposed fix is a new lemma, not a relabeling.** We have asked Astra for an **interior-layer witness**: λ_max(G_φφ) ≥ σ_min(J_Φ)²·λ_max(S_h^p) with h = `fused` and Φ the downstream Conv-BN-ReLU-Conv-BN-ReLU-Conv stack — problem **P3 in `review_packet/astra/claude_math_brief_01.md:31-38`**. If that is provable at init w.h.p. for a fixed-depth conv/ReLU stack, Theorem 1 applies to the production model with an honest new lemma. If it is not, the real-architecture connection is cut for this submission. We are not computing a transformer-token Gram and calling it the Lemma-16 head Gram.

---

## 3. Request 2 — campaign decision table

**Honestly: there is no campaign table.** CSF3 is down; nothing has been pulled; the R0 re-evaluation and the 1-day arms are "possibly partially complete" and the 4-day arms (`rr_finetune_real`, job 19676527, `--array=0-4%2`; `lpft`; `ft_real_speclr0.1`) almost certainly never started. I will not present a pending design as a result table. What follows instead is the **complete local E3c evidence base, all 39 rows, under both metrics**, with the caveats that make it not a substitute.

### 3.1 Arm × width, both metrics, mean ± sd over 3 seeds

**AdamW (lr 1e-4):**

| arm | M=48 final-5 | M=48 best-of-60 | M=192 final-5 | M=192 best-of-60 |
|---|---|---|---|---|
| joint_linear | 0.5382 ± 0.0821 | 0.8728 ± 0.0295 | 0.6458 ± 0.0501 | 0.8579 ± 0.0564 |
| frozen_random | 0.6477 ± 0.0630 | 0.8396 ± 0.0254 | 0.7310 ± 0.0226 | 0.8167 ± 0.0171 |
| frozen_pca | 0.7307 ± 0.0030 | 0.7687 ± 0.0210 | 0.7265 ± 0.0116 | 0.8619 ± 0.0680 |
| joint_mlp | 0.5658 ± 0.0705 | 0.8998 ± 0.0004 | 0.5750 ± 0.1837 | 0.8972 ± 0.0010 |

**SGD (lr 1e-2, momentum 0.9):**

| arm | M=48 final-5 | M=48 best-of-60 | M=192 final-5 | M=192 best-of-60 |
|---|---|---|---|---|
| joint_linear | 0.5384 ± 0.0748 | 0.6554 ± 0.0092 | 0.4467 ± 0.0718 | 0.6680 ± 0.0134 |
| frozen_random | 0.5105 ± 0.0506 | 0.6536 ± 0.0145 | 0.5034 ± 0.0752 | 0.6567 ± 0.0038 |
| frozen_pca | 0.4412 ± 0.1116 | 0.6952 ± 0.0706 | not run | not run |

**The 9 laneSGD2 rows** are `frozen_pca_h48_sgd` × 3 (CSV data rows 7–9), `joint_linear_h192_sgd` × 3 (rows 25–27), `frozen_random_h192_sgd` × 3 (rows 13–15). They are the replication that gives the M=192 SGD pair, and **`08_real_data.tex:203-205` reports the SGD comparison "at M=48" only** while the M=192 triplet, pointing the other way on final-5 (−0.057), sits unreported in the same CSV. That is selective and it is being fixed.

### 3.2 Paired differences, joint_linear − frozen_random (t₀.₉₅,₂ = 2.920, n = 3)

| Optimizer | Width | Metric | Per-seed | Mean | 90% CI | Inside ±0.02? |
|---|---|---|---|---|---|---|
| AdamW | 48 | final-5 | −0.230, −0.131, +0.032 | **−0.1095** | [−0.3326, +0.1136] | no |
| AdamW | 48 | best-val | −0.014, +0.039, +0.075 | **+0.0332** | [−0.0427, +0.1090] | no |
| AdamW | 192 | final-5 | −0.123, −0.017, −0.115 | **−0.0852** | [−0.1851, +0.0148] | no |
| AdamW | 192 | best-val | +0.084, −0.043, +0.082 | **+0.0411** | [−0.0813, +0.1635] | no |
| SGD | 48 | final-5 | +0.132, −0.101, +0.053 | **+0.0279** | [−0.1725, +0.2283] | no |
| SGD | 48 | best-val | +0.008, −0.003, +0.001 | **+0.0019** | [−0.0076, +0.0114] | **yes** |
| SGD | 192 | final-5 | −0.013, −0.091, −0.066 | **−0.0568** | [−0.1236, +0.0100] | no |
| SGD | 192 | best-val | +0.016, −0.001, +0.019 | **+0.0114** | [−0.0069, +0.0296] | no |

**One interval out of eight clears the margin.** Its width (0.019) barely fits inside a 0.04 window.

### 3.3 Per-run appendix (all 39 rows)

`arm | M | seed | final-5 | best | θ drift (rel) | μ̂ | r_final | thm2_bound (as shipped)`

**AdamW.** frozen_pca 48: s0 0.7342/0.7694, s1 0.7289/0.7473, s2 0.7289/0.7893 · frozen_pca 192: 0.7380/0.7964, 0.7268/**0.9322**, 0.7147/0.8572 · frozen_random 48: 0.7192/0.8536, 0.6231/0.8550, 0.6007/0.8103 · frozen_random 192: 0.7213/0.8120, 0.7148/0.8358, 0.7568/0.8025 · joint_linear 48 (drift 0.552/0.640/0.598): 0.4893/0.8391, 0.4922/0.8939, 0.6329/0.8853 · joint_linear 192 (drift 0.537/0.551/0.555): 0.5980/0.8957, 0.6979/0.7930, 0.6415/0.8848 · joint_mlp 48 (drift 0.862/0.867/0.852): 0.5467/0.8994, 0.6438/0.8999, 0.5068/0.9001 · joint_mlp 192 (drift 0.964/0.931/0.942): 0.6560/0.8971, 0.7042/0.8963, **0.3647**/0.8983.

**SGD.** frozen_pca 48: 0.4043/0.7765, 0.5666/0.6592, 0.3527/0.6499 · frozen_random 48: 0.4548/0.6369, 0.5537/0.6624, 0.5229/0.6614 · frozen_random 192: 0.4485/0.6545, 0.4726/0.6545, 0.5892/0.6610 · joint_linear 48 (drift 0.369/0.373/0.355): 0.5870/0.6449, 0.4523/0.6594, 0.5760/0.6620 · joint_linear 192 (drift 0.346/0.336/0.335): 0.4350/0.6707, 0.3814/0.6534, 0.5236/0.6799.

### 3.4 Caveats that make this not a campaign substitute

1. **One fold, one dataset** (breast QCL fold 0), **28 val cores**, **115 train cores**, n = 3 seeds.
2. **No test set** — E3c never built one (`exp1_7_train.py:504-515`). Every number above is on the selection split.
3. **No best checkpoint** — `final.pt` only (`:665-670`), and **38 of 39 exist** (missing `joint_linear_h192_adamw_s0`).
4. **6 layers, no BN1d, K=64** — not the 12-layer / K=128 production config. We will not pool these with cluster results.
5. **The clipping confound of §1/M1(b) is present in every row.**
6. SGD is under-trained: r_final 0.385–0.556 across all SGD runs (0.385–0.531 for the matched joint_linear/frozen_random pair) vs 0.033–0.106 for AdamW joint_linear (0.033–0.164 including joint_mlp); arm × width **mean** best-val F1 0.654–0.695 under SGD vs 0.769–0.900 under AdamW. **This is not an equivalence result.** Only 1 of the 8 intervals in §3.2 clears ±0.02, the M=48 final-5 interval is 0.40 wide (absence of power, not evidence of similarity), and the M=192 final-5 point estimate is −0.057 in the opposite direction. The defensible sentence is your own: *similar best-validation scores in under-fitted SGD controls at M=48; broader equivalence is not established* — and whatever similarity is there is similarity at low fit.
7. Two runs have μ̂ = 0 → NaN bound (`joint_mlp_h48_adamw_s1`, `frozen_pca_h48_sgd_s0`).

### 3.5 The local pivot plan, and the hygiene runs you asked for

Launched today (fold 0, 6-layer E3c runner): `frozen_pretrained`, `finetune_real`, `joint_speclr0.1`, `joint_speclr10`, `finetune_speclr0.1` — priority M=192 `finetune_real` + `frozen_pretrained` × 3 seeds, then the speclr arms, then M=48. Currently at the pretraining/smoke stage.

**Being added to the runner now, per your Priority 2** — these are protocol changes, not new arms:

- **identical BN-affine treatment** across the matched comparison (put `spectral_reduce.norm` in φ for every arm, or in none);
- **per-group gradient clipping** (φ clipped against φ only) so the clip denominator is arm-independent, plus a logged **clip coefficient per step**;
- **applied per-block update norms** logged (post-clip, post-preconditioner) alongside the existing pre-clip gradients;
- **best-val checkpoint saved** at the selection epoch, plus milestone checkpoints;
- **the fold-0 26-core test set loaded** and evaluated once at the selected checkpoint (it exists in `splits_fold0.json` and costs nothing).

We will report a local single-fold n=3 result as exactly that, and mark P1/P2/P3/LP-FT as **pre-registered and pending** in the paper rather than dressing fold 0 up as a 5-fold replication. LP-FT has no local substitute at all right now.

---

## 4. Request 3 — directional correction table

All rows M=192, breast fold-0 train split, **at initialization**, post-BatchNorm1d input space. Source `results/exp1_8b_summary.csv`; definitions `code/experiments/exp1_8b_spectrum.py`.

| seed | batch | λ along v₁ (`lam_along_vdata`, `:210-211`) | λ along d_⊥ (`lam_contrast_CancerEpi-CAS`, `:213-216`) | ratio | \|⟨d,v₁⟩\| | 1−overlap² (fraction of ‖d‖² kept) |
|---|---|---|---|---|---|---|
| 0 | 1 | 430.22 | 22.04 | **19.52** | 0.4599 | 0.7885 |
| 0 | 2 | 416.37 | 35.66 | **11.68** | 0.4942 | 0.7557 |
| 0 | 3 | 375.03 | 14.82 | **25.30** | 0.3938 | 0.8449 |
| 1 | 1 | 718.98 | 33.60 | **21.40** | 0.4599 | 0.7885 |
| 1 | 2 | 720.95 | 40.20 | **17.93** | 0.4942 | 0.7557 |
| 1 | 3 | 671.93 | 23.99 | **28.01** | 0.3938 | 0.8449 |

min 11.68, max 28.01, median 20.46 → the paper's "12–28×". (The script's own printed aggregate, `:274`, is an *unpaired* mean-over-mean of 19.18×, logged as "19×" — a third number, and neither is in the paper.) Batch 0 is single-class — its own `class_counts` field in `results/exp1_8b_summary.csv` is `{"CancerEpi": 29664}`, with no CAS pixels at all — so no contrast can be formed there. (The `[101066, 25577, 0, 0]` at `exp1_8b_full.log:4` is the *whole 24-core split's* class total, not batch 0's; it shows that only CancerEpi and CAS are present anywhere in the split, which is why the ≥100-pixel guard at `:86` admits one pair out of the four declared at `:163`.) Seed 2's rows are empty.

**Additions you asked for:**

- **cos(v₁, mean spectrum)**, our recomputation: **0.998584 / 0.997095 / 0.995598 / 0.996696** post-BN (batches 0–3) and **0.999985–0.999997** raw. v₁ *is* the mean direction. Overlap of d with the mean direction itself: 0.4413 (b1), 0.4628 (b2), 0.3702 (b3) — i.e. orthogonalizing against v₁ is essentially orthogonalizing against the mean spectrum.
- **`align_v1_data1` definition**: `‖V_p d_j‖₂ / ‖V_p‖_F` where `V_p` is the (K=64) × (C·S=942) matricized proj-weight slice of the θ-block top Lanczos eigenvector (`:194-198`). It is the fraction of that matricized eigenvector's Frobenius norm lying in the rank-64 subspace {u ⊗ d_j}, **not a vector cosine**. Value 0.9539 ± 0.0169 (n=12, range 0.9049–0.9693). `projfrac_v1` = 0.99961–0.99965, so slice-conditioning is numerically immaterial.
- **Normalization state**: alignment, overlaps and contrast curvature are **post-BatchNorm1d(314)** (`:60`), where effective rank is 1.2136 (per batch 1.1452/1.2379/1.2628/1.2086, = 1/`conc_data`). The paper's **1.07 is raw Σ_X** (`exp1_8.py:149-153`). Centered post-BN effective rank recomputes to 1.32–1.46. These must be labelled separately in §8.3.
- **Rigorous bracket on the FULL contrast, available today.** With a = |⟨d,v₁⟩|, b = √(1−a²), the PSD GGN seminorm triangle inequality gives (a√λ_v − b√λ_⊥)² ≤ λ_full ≤ (a√λ_v + b√λ_⊥)². Per row the full-contrast ratio is bounded in: **≥ 2.29, 1.78, 3.01, 2.35, 2.04, 3.11** and **≤ 14.92, 17.39, 22.45, 13.93, 11.98, 20.64.** Guaranteed floor ~1.78, median floor ~2.3×.
- **Numerical residual**: Lanczos residual up to 0.14 on these blocks; GGN operators validated to 6.9e-16–1.22e-15 relative error against dense `eigvalsh` on tiny configs (`code/hessian/`).

**Queued GPU work (not done).** The un-orthogonalized full-contrast curvature **cannot be recovered from stored data** — `lam_contrast` is λ_max of a matrix quadratic in d, so reconstructing λ(d) needs the cross-block, and only scalars were written (`:125`, `:218`). Rerun recipe: add `restricted_block_lambda_max(op_t, proj_weight, d, K)` alongside the existing `d_perp` call at `:212-219`, keep the orthogonalized column for continuity, and run at existing defaults with `--seeds 0 1 --restricted_seeds 0 1`. **~1–2 h on an idle GPU.** A stronger variant assembles the 2K×2K block on the basis {e_i ⊗ v₁, e_i ⊗ d_⊥} in 128 matvecs, giving λ_max along **any** direction in span{v₁, d} — including the exact full contrast — for roughly two blocks' cost. **The four-direction overlap table** (full contrast, orthogonalized contrast, empirical mean, top Gram vector, with pairwise overlaps and a training-only probe) is the same rerun plus a probe fit, and needs a batch with all four classes populated — batch selection is the real cost there, not the eigensolve. Queued behind the pretraining/smoke job.

**Wording we propose for §8.3** (replacing `08_real_data.tex:159-166` and the caption at `:178-181`): "the θ-block's top GGN eigenvector holds 99.96% of its energy in the projection weight and, viewed as a 64×942 matrix, places 0.954 ± 0.017 of its Frobenius norm in the rank-64 subspace {u ⊗ v₁} spanned by the leading direction v₁ of the *post-BatchNorm* input Gram; v₁ is itself the mean direction of the post-BN inputs (|cos| ≥ 0.996). Along the component of the CancerEpi–CAS contrast orthogonal to v₁ (which carries 76–84% of the contrast's squared norm), the restricted spectral curvature is 12–28× smaller than along v₁ (n = 6: 2 seeds × 3 batches, one class pair, M = 192, at initialization)." We also drop "clinically decisive", and disambiguate the *other* 0.954 in the paper (`08:96`, a Sliding-Window test F1 in an unrelated table).

**Fairness note.** The docstring at `:17-20` and the surrounding code make clear the intended question was "how much curvature remains for the contrast after the mean-spectrum direction is accounted for", which is legitimate. The defect is reporting, not construction.

---

## 5. Request 4 — bound audit (unit-annotated)

`thm2_bound := (B_p99 · Ĉ / μ̂) · log(Ĉ / r_T)`, compared against measured `theta_disp_abs`.

| Theorem quantity | What `thm:twoscale` needs | What the code computed | Units | Defect |
|---|---|---|---|---|
| **C** | envelope constant on the full-data residual along the flow; theorem requires **C ≥ 1** (`05:125`) | `max` of the first 10 points of a **25-step centred rolling median** of the per-mini-batch RMS residual (`analyze_e3c.py:36,43,79-80`) | dimensionless | smoothed mini-batch ≠ full-data; **Ĉ = 0.691–0.856, 0/18 joint runs satisfy C ≥ 1**; docstring `:7` says "smoothed r(0)", code says max-of-first-10 |
| **μ** | decay rate in **flow time** | fit against the **integer step index** (`:44`; docstring `:9` "Time unit = optimizer step") | step⁻¹ | **needs μ_flow = μ_step/η**; docstring `:12-13` explicitly denies the conversion is needed — false, because the LHS ‖θ(T)−θ₀‖ carries no clock, so the step clock does not cancel |
| **B_T** | **sup**_{t≤T} ‖∇_θL‖/‖r‖ (`05:261-262`) | **99th percentile** of realized gain (`:86,93`) | dimensionless | percentile ≠ supremum, so it is not even an upper bound in the theorem's own sense (B_sup/B_p99 = 1.6–3.1 over the 12 joint_linear rows; 1.6–5.7 over all 18 joint rows, and up to 9.3 across all 39); and gains are logged **pre-clip** (`exp1_7_train.py:609-610`) while clipping binds on 69–89% of steps; docstring `:11` claims B̂ is used |
| **δ** | residual level defining the hitting time T_δ = inf{t : ‖r‖ ≤ δ} | terminal **25-step mean** of the smoothed residual (`:92`) | dimensionless | terminal time ≠ first hitting time |
| **envelope validity** | ‖r(t)‖ ≤ C/(1+μt) for the logged residual | holds on the smoothed curve (violated 0.00–0.06% of steps) | — | **the raw per-step residual violates it on 22.4–27.5% of steps** |
| **LHS** | ‖θ(T)−θ₀‖ along the flow | `group_l2_disp` (`exp1_7_train.py:635,652`) | parameter norm | correct |
| **the flow itself** | a.e. gradient flow, containment, θ zero-wd | AdamW(1e-4) or SGD(1e-2, momentum 0.9), φ wd 0.01, clip 1.0, batch 4 | — | outside the theorem's hypotheses regardless of the constants |

**Corrected proxy numbers** (bound_corrected = B̂_sup · Ĉ / (μ̂/η) · log(Ĉ/r_final), η per run's `config.json`):

| Arm | corrected / measured |
|---|---|
| AdamW joint_linear | **17.4 – 46.5** (mean 35.6; M=48: 42.9/27.8/34.1, M=192: 46.5/45.0/17.5) |
| AdamW joint_mlp | 2.10 – 10.44 (5 runs; 1 NaN) |
| SGD joint_linear, η = 1e-2 | **452 – 1884** (mean ~1033; M=48: 934/720/452, M=192: 1884/716/1491) |
| SGD joint_linear, η_eff = η/(1−β) = 0.1 | **4.5e3 – 1.9e4** |

**The only certified number in the set.** For SGD the update rule gives Σ‖Δθ‖ ≤ (η/(1−β))·Σ_k‖clipped ∇_θL_k‖ with θ's weight decay at 0 (`exp1_7_train.py:277`). Computed from `steps.csv`: sums **63.4–73.5** vs measured displacements **1.544–1.721**, i.e. **bound/measured = 40.3–44.9** for the six SGD joint_linear runs. This is provable from the update rule and computed from logs, not from the theorem's constants.

**Why we cannot do the same for AdamW.** η·Σ_k‖clipped ∇_θL_k‖ = **0.121–0.145** while measured displacement is **2.475–2.955**. The preconditioned update moves θ 17–20× *further* than the flow model allows. No η-rescaled flow bound is certified for the AdamW arms in either direction.

**Concession.** The column is a **proxy**, not an instantiated bound, and the "2×10⁴–2×10⁵" figure must be **withdrawn**, not corrected: it is 1/η too large by construction, uses a percentile where the theorem says supremum, and its stated range does not cover joint_mlp (1.045e4 floor).

**Recommended wording for `08_real_data.tex:313-316`:**

> The theorem's displacement bound cannot be instantiated quantitatively on this pipeline. It is a gradient-flow statement, whereas these runs use AdamW or SGD with momentum 0.9, a constant learning rate, and global gradient clipping that binds on 69–89% of steps; and its residual-envelope constants are estimable here only from a smoothed mini-batch residual whose raw counterpart violates the fitted envelope on roughly a quarter of steps. For the SGD arm, where the flow model is closest to the dynamics, the elementary triangle-inequality bound on the realized update exceeds the measured spectral displacement by a factor of 40–45; for the AdamW arms the preconditioned update moves θ more than seventeen times further than the gradient sum the flow model predicts, so no comparable factor is certified. We therefore report the bound as qualitatively loose on real data rather than attaching a number to it.

**Code fixes that go with it.** (i) `analyze_e3c.py:7` vs `:43` (Ĉ definition) and `:11` vs `:93` (B̂ vs B_p99) are **docstring/code contradictions**; `:12-13`'s no-conversion claim is false. (ii) Switch `thm2_bound` to B̂_sup and flag it as uncertified, or remove the column. (iii) The μ̂ values at `08:286-288` are step-clock rates — the 3.2× width ratio survives the units change, the absolute 6.0e-4 / 1.9e-3 do not. (iv) `ASTRA_BRIEF:157` and `:272` propagate the withdrawn number and are being updated.

We accept your Priority-4 prescription for the synthetic illustration (deterministic full-batch, step-size refinement, unsmoothed envelope over a stated horizon, explicit numerical accounting), and we accept that fitting every constant to the same trajectory and observing an inequality is an audit, not a validation. The clean discrete lemma set is asked of Astra as **P4 in `claude_math_brief_01.md:40-45`**, prioritized first precisely because the audit needs it.

---

## 6. Request 5 — alternative-mechanism check

**What we can run now, in order:**

**(1) Clip-binding statistics — already computed, no GPU needed.** Fraction of steps with clipping active: 88.5% (AdamW M=48 s0), 72.8% (AdamW M=192 s0), 83.9% (SGD M=48 s0), 88.2% (SGD M=192 s0); range across arms 69–89%. Median extra clip factor paid by the joint arm: **2.99–3.56× (AdamW M=192, p90 5.3–6.4, max 14.8)**, 1.10–1.12× (SGD M=48), 1.607× (joint_mlp M=192). Step-0 evidence table in §1/M1(b). Recomputed spectral share (joint_linear, median over epochs ≥ 56): 0.946 / 0.928 (AdamW M=192 / M=48), 0.202/0.253/0.268 (SGD M=48) — matching `08:277-279`; those shares *imply* an extra φ clip factor of 1/√(1−share) = 4.3× (AdamW) and 1.13–1.17× (SGD).

**(2) BN recalibration on the 38 existing `final.pt` — feasible without retraining, ~3 GPU-h.** Each checkpoint (`experiments_shortcut/e3c/breast_f0/`, 62.9 MB, keys `['epoch','model_state_dict','theta0','config']`) carries all 9 BN buffers including `spectral_reduce.norm.running_mean/var` and, for `frozen_pca`, the already-fitted projection — no PCA refit needed. Protocol: deterministic re-evaluation first (must reproduce `epochs.csv`'s last row), then **training-data-only** recalibration on disposable copies with weights fixed, retaining both numbers. **Caveat: `joint_linear_h192_adamw_s0/final.pt` is missing**, so a recalibration check at that headline cell runs on 2 of 3 seeds unless it is retrained.

**(3) Per-block applied update norms — needs the runner patch (§3.5), then rerun.** The logged quantities today are pre-clip gradients (`:608-610`); the applied θ and φ updates, the clip coefficient, and the Adam preconditioner effect are not logged. This is the instrumentation that lets us report raw gradient, momentum update, and applied update as three separate objects, per your Priority 3.

**(4) The decisive control — per-group clipping counterfactual, ~5 GPU-h.** Rerun the M=192 AdamW `joint_linear`/`frozen_random` pair (3 seeds each) with φ clipped against φ only, everything else identical, and report whether the −0.085 final-5 gap survives. **This is the single cheapest experiment that can invalidate our headline**, and it should run before any new geometry.

**(5) Feature/probe measurements (your M2), ~2 GPU-h.** Fixed-panel post-normalization feature change, scale/basis-aware row-space comparison for the linear encoder, and a standardized frozen-feature probe at init and at final. `exp1_7_train.py:316-337` already has a BN-guarded probe path to build on.

**Order: (1) done → (4) → (2) → (5) → (3).** (4) before (2) because a confound that changes the sign of the headline outranks a confound the data already argue against (§1/M1(d)).

---

## 7. Request 6 — proposed thesis, outline, and the 72-hour list

### 7.1 Two theses, and which one is the default

**(a) The floor — accepted as the default if no new theorem lands.** Your framing, adopted essentially verbatim:

> Module capacity and initialization curvature do not by themselves determine whether a spectral encoder learns useful features or should be frozen. We establish conditional curvature and finite-time displacement results for a linear bottleneck, identify precisely where their hypotheses fail in a realistic spectral–spatial network, and show that comparisons of freezing and joint training depend on optimization details and checkpoint selection. Real-data curvature is strongly anisotropic; its causal relationship to representation learning is not established here.

Title: **"The Spectral Shortcut? Limits of Curvature-Based Explanations in Spectral–Spatial Learning."** "Theorem" leaves the title.

**(b) The conditional ceiling — only if the mathematics lands.** If Astra proves **P3** (interior-layer witness, `claude_math_brief_01.md:31-38`), **P4** (discrete-time Theorem 2 with correct units and momentum/clipping, `:40-45`) and **P1** (the joint-vs-frozen trajectory-comparison bridge under a PL/NTK hypothesis, `:11-20`), the paper becomes a **regime-resolved theorem with a real-architecture witness**: Theorem 1 applies to the production decoder through a new lemma rather than a false premise; Theorem 2 has a certified discrete instance that covers our actual SGD arm; and the joint≈frozen comparison stops being an analogy. That is a materially stronger paper, and it is the only route by which "the theory describes this model" can be written honestly. **We are not planning around it.** The abstract we draft on Sept 10 is thesis (a); if a proof lands by Sept 12 we swap one paragraph, not the paper.

### 7.2 Nine-page outline, mapped to your allocation

| Pages | Section | Content | Reader takeaway |
|---|---|---|---|
| 1.00 | 1. Problem | Spectral→spatial pipelines; freezing is reported to help; three competing explanations (capacity/curvature, optimization asymmetry, checkpoint policy); related positioning against Pezeshki, Chizat, Kumar, Zhang'24, Karakida | Freezing benefits do not identify their cause |
| 0.75 | 2. Setup and observables | F = g_φ∘f_θ; the three distinct observables — curvature at init, representation change, selected-checkpoint performance — and why they are not interchangeable | These are three questions, not three metrics for one |
| 1.50 | 3. Theory | Lemma 18 (cap, unconditional); Lemma 16 (floor, conditional on the head hypothesis); Theorem 1 as a conditional statement; Prop 38 as the one verified instance; Theorem 2 statement + proof sketch; **an explicit hypothesis ledger** | Exactly what is proved, and which implications are absent |
| 1.00 | 4. Synthetic | Exp 1.1 v3 init-arm width sweep (λ_φ slopes 0.105–0.989 across init schemes; λ_θ slope ≈ 0 everywhere); the bounded Thm-2 illustration if it succeeds | Conditions and parameterization matter; toy scope is limited |
| 1.25 | 5. Real geometry | Architecture-to-assumption map (the 432 result); D_curv inverted on breast-BN-on, 1.09 on prostate at M=384, 1.18–1.31 with BN1d removed; rank-1 Σ_X and v₁ = mean direction; directional anisotropy with the full-contrast measurement and its bracket | Anisotropy is measurable; a scalar certifies nothing about useful learning |
| 2.50 | 6. Dynamics and policy | Matched hygiene runs (identical BN treatment, per-group clipping, saved best-val checkpoint, held-out test evaluation); AdamW vs SGD; spectral-LR ×0.1/×1/×10; final-window vs selected-checkpoint results side by side; **the clipping asymmetry reported as a measured confound** | Which practical conclusions survive a fair training policy |
| 1.00 | 7. Limitations and consequences | What can be diagnosed; what intervention (if any) helps; what remains unproved; the pre-registered-and-pending campaign | Honest scope |

Appendix (unlimited): full proofs, per-seed tables, all 39 E3c rows, the bound audit, EGR (demoted), Exp 1.3, Cor 28, Prop 31, historical corrections, the AI-use disclosure.

Main text carries **one geometry figure, one trajectory/intervention figure, one performance table.** We accept all your cuts (§6 EGR out of main text; Exp 1.3 to appendix; Cor 28 and Prop 31 to appendix; §8.2 historical table removed until provenance resolves; Ass 6 de-numbered) with one amendment: **Ass 3 stays numbered** because Cor 28 consumes it (§0).

### 7.3 The 72-hour experiment list, in priority order

One local GPU, ~50 usable GPU-hours in 72 h wall-clock, currently occupied by pretraining/smoke.

| # | Experiment | Hours | Can it change a sentence in the abstract? |
|---|---|---|---|
| 1 | **Per-group clipping counterfactual**, M=192 AdamW joint_linear/frozen_random × 3 seeds | 5 | **Yes — it can delete the empirical headline.** Highest priority. |
| 2 | **BN recalibration** on 38 `final.pt`, original + recalibrated retained | 3 | Yes, if it restores the late collapse |
| 3 | **Matched-hygiene M=48 grid**: frozen_random, joint ×1, joint spectral-LR ×0.1, joint ×10, joint ×1 + LR schedule; 3 seeds; identical BN treatment, per-group clip, saved best-val checkpoint, **test evaluation on the 26 held-out fold-0 cores** | 12–13 | Yes — this is the prospective result the abstract needs |
| 4 | **Full-contrast + four-direction geometry rerun** (`exp1_8b_spectrum.py` variant, 2K×2K span block) | 1–2 | Yes — converts "12–28× along d_⊥" into a defensible statement about d |
| 5 | Finish the launched pivot arms: `frozen_pretrained`, `finetune_real` at M=192 × 3 seeds | 5 | Yes — the only local evidence on P1 |
| 6 | **Head-witness measurement** on the real 432-dim classifier: λ_max(S_h^p) and ‖h̄‖²/M_h per arm and width (completes Fable F11a's first two parts) | 2 | Diagnostic, not headline — but it is the measurement §8 currently claims and never made |
| 7 | Bounded synthetic Thm-2 instantiation (full-batch, no momentum/clipping, step-size refinement) | 4 | Only if it succeeds; otherwise the honest report is "no useful numerical instance" |
| 8 | Feature/probe measurements (M2) | 2 | Supporting |
| — | **Deferred**: shallow 2×2 σ_b isolation; F11a's residual+LayerNorm and trunc-normal ViT-init arms; joint_mlp probes; a general Adam theorem | — | Decoration for this submission |

Total for 1–6: ~28–30 GPU-h, which fits. Items 7–8 fit only if nothing fails.

### 7.4 Claims we are removing

| Claim | Location | Why |
|---|---|---|
| "this architecture **falls in the class** treated by Theorem 1" | `08_real_data.tex:70-75` | Head hypothesis unmet (M_h = 432, constant); the C_g = Θ(M²) form is a change of variables, not an entry condition |
| "both theorem ingredients behave **exactly as proved**" | `08:135-141` | Proved rate is Ω(M); measured λ_φ slope 0.381 / 0.447 |
| "the class-prior NTK floor" as a **verified ingredient / proved mechanism** | `08:288-292`, `08:319-322` | Same head hypothesis (`05:390-393`); never measured on the real model |
| "**direction-wise starvation**" as established | §1 thesis, `08:162-165` | Curvature anisotropy ≠ starvation; measured on d_⊥, not d; honest full-contrast floor ~2× |
| "**plain SGD with zero weight decay — the setting of Theorem 2**" | `08:236`, `08:272` | SGD momentum 0.9, φ wd 0.01, clip 1.0, batch 4; discrete SGD is OPEN |
| joint≈frozen is "**precisely** the theorem's prediction" | §8.2/§8.4 | The theorem bounds θ displacement along one trajectory; the frozen run has its own φ path |
| "**vacuous by 2×10⁴–2×10⁵**" | `08:313-316` | 1/η mis-scaled, percentile-not-supremum, range does not cover joint_mlp |
| "**frozen beats joint everywhere**" | §8.4 | Inverts on best-val; and the comparison carries an arm-asymmetric clipping confound |
| "**metric-robust** SGD equivalence" | `ASTRA_BRIEF:29,145` | final-5 M=192 gap −0.057; only 1 of 8 intervals clears ±0.02 |
| "**statistically indistinguishable**" (×3 in §8, ×1 in §7) | `08:238,253,275`; `07:206` | Banned by our own laneSGD2 phrasing rule |
| "holds but is **vacuous** at buildable widths" | `08:148-151` | Should be "does not apply" |

### 7.5 Claims newly supported by this round

- **Arm-asymmetric gradient clipping is present by construction in every E3c comparison** and is measured on the runs we checked: identical step-0 gradients with a 1.29–1.41× larger frozen φ update; median extra joint clip factor 2.99–3.56× for joint_linear at M=192 AdamW (1.10–1.12× for joint_linear at M=48 SGD, 1.607× for joint_mlp at M=192 AdamW); binding on 69–89% of steps; and the paper's own "spectral share ≈ 0.94" is algebraically its magnitude. It is a *measured asymmetry*, not yet a demonstrated cause of the gap — that is what experiment 1 in §7.3 tests. Reportable either way, independent of whether our headline survives.
- **The head-dimension gap is a general diagnostic point**, not a bug in one model: an encoder–decoder segmentation head with a fixed-width final classifier removes the hypothesis that width-based curvature-floor arguments require, regardless of total parameter count — it does not show the floor is false. That is a transferable lesson for anyone applying Karakida-style width results to real architectures.
- **Corrected proxy magnitudes**, with one certified number: the SGD triangle-inequality bound exceeds measured displacement by 40–45×, while the AdamW trajectory moves θ 17–20× further than the flow model permits — i.e. the failure is directional, and the theory's scope boundary at adaptive optimizers is now quantified rather than asserted.
- **v₁ is the mean-spectrum direction** (|cos| ≥ 0.9956 post-BN, > 0.99998 raw) — the rank-1 Σ_X mechanism is confirmed, and only its reporting was loose.

---

## 8. Questions back

1. **Do we hold a slot for P3?** Given B1, our default is to cut the real-architecture theorem connection entirely and write Lemma 18 as the only unconditional ingredient. If you think the interior-layer witness (`claude_math_brief_01.md:31-38`) is provable at init for a fixed-depth conv/ReLU stack, we will reserve ~0.4 page in §3 and run the λ_max(S_h^p) measurement on `fused` as well as on the 432-dim feature. If you think it is out of reach in three days, say so and we cut now rather than on Sept 13.

2. **Which clipping control do you want, given we can afford one at n=3?** (a) per-group clipping (φ clipped against φ only), which isolates the confound but changes the optimizer for both arms; or (b) clipping off with a short stability check, which is closer to the theory but risks divergence in the joint arm and would then prove nothing. We lean (a).

3. **Should the matched-hygiene runs use the existing fold-0 26-core test split?** It exists in `splits_fold0.json`, E3c never loaded it, and using it costs nothing — giving us selected-checkpoint test F1 on a genuinely untouched set for the first time in this project. The counter-argument is that a single 26-core test set from one fold invites over-reading. Your call on whether one honest test number beats none.

4. **On the geometry rerun, is the span{v₁, d} block worth 2× the cost?** The cheap version adds one call and gives λ along the full contrast. The 2K×2K version (128 matvecs) gives λ_max along *any* direction in span{v₁, d}, so the paper could report a curvature profile rather than two endpoints. We would rather spend the extra hour there than on item 6, but you may disagree about which is decoration.

5. **What is your bar if the clipping counterfactual kills the effect?** If the joint−frozen gap disappears once the clip denominator is arm-independent, our empirical core reduces to "an unreported optimization asymmetry explained our headline". Is that still an ICLR paper as the diagnostic-limits contribution plus the geometry, or does it become a workshop/TMLR paper? We would rather know your threshold before Sept 12 than argue about it on Sept 14.

---

*Artifacts referenced: `results/e3c_analysis_runs.csv` (39 rows), `results/exp1_8b_summary.csv`, `results/exp1_8b_spectra.csv`, `results/exp1_2v4_summary.csv`, `code/experiments/{exp1_7_train,exp1_8,exp1_8b_spectrum,analyze_e3c}.py`, `side_project/models/blockvit_v2.py`, `experiments_shortcut/e3c/breast_f0/` (38 of 39 `final.pt`, gitignored at `.gitignore:45`). Math thread: `review_packet/astra/claude_math_brief_01.md`.*

---

*Audit note (2026-09-09, post-draft): this file was re-checked line by line against `results/e3c_analysis_runs.csv` (arm × width × metric tables and all eight paired differences recomputed independently), the five verifier reports in `wf_4af2be2b-f2a/journal.jsonl`, and your reply. Twelve corrections were applied: three per-seed rounding errors in §3.2 (−0.015→−0.014, −0.102→−0.101, −0.014→−0.013); wrong CSV row citations for the laneSGD2 rows in §3.1 (now 7–9 / 25–27 / 13–15); §1/B4's best-minus-final-5 degradations labelled SGD-only with the AdamW figures added; §3.4's "equivalence there is equivalence at low fit" withdrawn as an overclaim and replaced with your own non-inferiority wording, with the r_final and F1 ranges scoped to the arms they came from; the clip-factor ranges in §1/M1(b), §6 and §7.5 scoped to the specific arm and width (joint_mlp pays only 1.607×); §7.5's clipping bullet demoted from "measured confound in every comparison" to a measured asymmetry that is not yet a demonstrated cause; §7.5's "defeats width-based curvature-floor arguments" softened to "removes the hypothesis they require", matching §1/B1's own hedge; the §1 B3/M1 heading amended so it no longer reads as blanket confirmation when M1(d) is not supported; §4's batch-0 class counts corrected (batch 0's own counts are `{"CancerEpi": 29664}`; `[101066, 25577, 0, 0]` is the split total); §0's C_f = 61,108 citation moved off `exp1_8_prostate.log`, which does not contain it; §0's "NaN for both contrast columns" corrected to all three restricted-block columns; and §2 now states plainly that the campaign runs' θ/φ definitions are not supplied. No other number in the file failed to reproduce.*
