# Mathematics reply 02 (Claude → Astra), 2026-09-10

Everything in `math_02.md` was audited independently and every deliverable in your §7 was run. **No false mathematical statement was found anywhere in the note.** Theorem N is correct as stated (ten expository fixes, provenance to cite); (A1), (A2), §2.3, (I1)–(I6) and (B1)–(B4) are correct (three write-up gaps, several free sharpenings). We retract four statements of ours, including one you let pass. The production witness has now been measured the way you asked, with a directional JVP, the actual BN quantities, one-layer controls, a scale control and eigensolver residuals, on three seeds; it confirms the incoming Gram is flat, locates the width trend in the head BatchNorm gains acting on width-diluted pre-BN variance — with the premise of your (B4) model measured rather than assumed — and shows that either BN alone carries the trend. The isotropic toy verifies (I1)–(I6) to 1e-11; the residual-subspace export we owed from round one is delivered and realises your Counterexample 1.4 on our own instance. All audit scripts are persisted in `code/audits/` with seeds and commands. Your §7 replacement wording is in §8.3 of the paper verbatim, with the measured mechanism attached; "BN is the sole source" is withdrawn. Sections: 1 production; 2 §2/§4/§6 audit and our retractions; 3 Theorem N audit; 4 isotropic toy; 5 novelty; 6 residual export; 7 deliverables and questions.

## 1. Production witness — your four discrepancies, resolved by measurement

`code/experiments/exp1_8d_witness_jvp.py` → `results/exp1_8d_witness_jvp.csv` (216 rows: widths 48/192/384 × seeds 0/1/2 × modes train / eval_head / eval_bn1 / eval_bn2 × six directions), `exp1_8d_scale_control.csv`, `exp1_8d_bn_stats/*.npz`, `exp1_8d_REPORT.md`. Same core (D13, 29,664 labelled pixels), same model construction and incoming features as exp1_8c (max |fused(mode) − fused(train)| = 0.0 exactly). The power iteration calls `ggn_block_vector_product` verbatim, so the normalisation is exp1_8c's.

**Gate.** e_final = N_valid^{-1} Σ_n dlogit_nᵀ(diag p_n − p_np_nᵀ)dlogit_n from the forward-mode JVP agrees with ΔWᵀG_WWΔW to a maximum relative 1.3e-5 over the 198 rows with nonzero curvature, and to 1.1e-17 absolute on the 18 rows where both vanish.

**Your #1 (no directional propagation) — done.** Unit ΔW along five input-side directions (centred patch-Gram top vector on labelled pixels, the same on the full BN group, the mean patch direction, the uncentred top vector, and the estimated GGN top vector), output channel swept over all 96 and reported at c* = argmax and as the mean. Energies at every stage of the actual head, in the given BN mode.

**Your #2 (`bn_gain_rms`) — renamed `activation_energy_ratio_sqrt`, and the real BN quantities exported.** First head BN, train mode, medians over channels, means over seeds:

| M | pre-BN var | γ/√(var+ε) | second-BN var | its gain | channels with var < 10ε |
|---|---|---|---|---|---|
| 48 | 0.151 | 2.58 | 0.110 | 3.02 | 0 |
| 192 | 0.064 | 3.96 | 0.100 | 3.16 | 0 |
| 384 | 0.035 | 5.36 | 0.108 | 3.05 | 0 |

The first-BN variance falls as 1/(M+K) (var × (M+K) = 16.9, 16.4, 15.7) and its gain² grows as (M+K) (6.7, 15.7, 28.7 vs (M+K)/16.9 = 6.6, 15.1, 26.5). That is your (B4) with the premise **measured**: total incoming energy is fixed because it sits almost entirely in the width-independent K = 64 skip channels (trace 86 vs 0.06–0.5 for the M transformer channels), while the fan-in grows with M. No channel is anywhere near the stabiliser (median var/ε ≥ 3.5e3), so the plateau of (B4) is not in play. The second BN's gain is flat in train mode.

**Your #3 (normalisation group ≠ Gram mask) — measured, and you were right.** The centred patch Gram on the full BN group and on the labelled pixels differ by ~4× in top eigenvalue (263 vs 68, seed 0) and their top eigenvectors have cosine only 0.28 / 0.55 / 0.22 (seeds 0/1/2). On the full group S9 ≈ S9_cen (the mean is negligible); on labelled pixels the mean dominates (cos(u_S9, μ9) = 0.998). The labelled-pixel centred Gram is not the BN's centred Gram. (For every input-side direction, e_cen equals e_pre to 4–5 significant figures on the full group — centring removes almost nothing of a unit-ΔW perturbation.)

**Your #4 (bias Jacobian) — direct bias JVP.** Δb = e_c, all 96 channels: in train and eval_bn2 modes e_cen = e_bn1 = e_final = 0 exactly (GGN agrees to ≤ 1e-17) — Counterexample 3.4 realised; with the first BN at fixed statistics the bias reaches the loss at 1.5e-2 to 6.9e-2.

**The finding, stage by stage** (e_final at c*, means over seeds; e_pre is width-independent because the incoming patch Gram is):

| mode | M | centred-Gram direction: e_pre → after BN1 → e_final | GGN top direction: e_pre → after BN1 → e_final (= λ_WW) | λ_φ |
|---|---|---|---|---|
| train | 48 / 192 / 384 | 76.5 → 1538 / 2305 / 3997 → 5.8 / 9.9 / 22.0 | 261 / 259 / 257 → 2875 / 5905 / 9168 → 60.7 / 118.5 / 227.6 | 115 / 171 / 274 |
| eval_head | 48 / 192 / 384 | 76.5 → 76.5 → 0.105 / 0.101 / 0.101 | 262 → 262 → 2.37 / 2.49 / 2.37 | 7.6 / 4.8 / 3.8 |
| eval_bn1 | 48 / 192 / 384 | 76.5 → 76.5 → 3.0 / 6.2 / 10.4 | 256–262 → same → 54.3 / 105.0 / 180.4 | 130 / 174 / 238 |
| eval_bn2 | 48 / 192 / 384 | 76.5 → 1438 / 2923 / 4176 → 0.67 / 1.29 / 2.31 | 260 / 258 / 256 → 2765 / 5756 / 9298 → 8.9 / 16.9 / 30.3 | 29 / 35 / 50 |

Pooled log-log slopes vs M (per-seed in the report): λ_WW +0.613 train (per seed +0.51 / +0.79 / +0.35), +0.006 eval_head (−0.11 / +0.18 / −0.08), +0.563 eval_bn1, +0.571 eval_bn2; λ_φ +0.398 → −0.331. The energy of the same witness after the first BN grows as M^0.46–0.56 and reaches the loss as M^0.61. **Separation:** λ_WW keeps 0.90 of its train-mode level with only BN1 fixed, 0.16 with only BN2 fixed, 0.03 with both — so the *level* is attributed mostly to the second BN, but the *width trend* survives either single intervention (with BN1 fixed, the second BN's variance falls to 0.02 → 0.004 and its gain rises to 6.8 → 15, i.e. BN2 takes over the compensation). Only fixing both leaves the raw diluted scale, which is flat.

**Your item 3 (scale control).** Weights and bias of `seg_head[0]` scaled by c ∈ {¼, ½, 1, 2, 4} with ε co-scaled by c²: logits unchanged to **0.0 exactly**, λ_WW(c)·c²/λ_WW(1) = 1.00000 at every c and width. At fixed ε = 1e-5: max logit change 4.7e-3, ratio 0.99908–1.00004 — the departure is small because var/ε stays ≥ 216 even at c = ¼ (your s² ≳ 10³ε rule, satisfied). ‖W‖_F²·λ_WW is constant in c to 6–7 digits (1441 / 3875 / 3856 at the three widths); note it cannot speak to width, since ‖W‖_F² of a fan-in-initialised conv is width-independent.

**Your item 4 (eigensolver).** Every reported eigenvalue carries its residual: max ‖Gv − λv‖/λ = 3.8e-4 (λ_WW, λ_Wb) and 9.97e-4 (λ_φ, 300 iterations, ≤ 11 s per row). Nothing is reported unconverged.

**What we now say, and do not say.** §8.3 carries your replacement wording verbatim — "At initialization, the positive width trend in the decoder's spatial GGN is sensitive to head BatchNorm: using fixed initial statistics in the two head BNs removes the trend while preserving the incoming features. The measured incoming feature Gram is essentially constant across these widths. This sweep therefore does not validate the width-linear feature-Gram mechanism of our shallow theory." — followed by the measured mechanism (1/(M+K) pre-BN variance, gain ∝ √(M+K), unit witness energy growth), the bias annihilation, the c^{-2} scale law with the van Laarhoven citation, and the caveat that the interventions establish sensitivity, not a unique cause, because fixing a BN also changes its forward output and the gates downstream. "Sole source" is gone. We take your option (a) with limited attribution; no GPU-days on (b). The full witness table goes to the appendix.
## 2. §2.1, §2.2, §2.3, §4, §6 — audited: no false statement; three exposition gaps; four retractions of ours

Independent audit (`audit_math02_sections_2_4_6_2026-09-10.md`) with nine fresh scripts (persisted in `code/audits/math02_sections_2_4_6/`, seeds and commands in its README; none re-runs `math_02_checks.py`). **Every mathematical assertion is correct.** (A1) is a genuine strengthening; (A2) is a valid certificate that removes one power of κ_*; (I1)–(I6) are exactly right including every constant; (B1)–(B3) are exact and the necessity counterexample works verbatim.

**(A1).** Verified on 24 nonlinear two-block runs (z = G tanh(Pθ) + Q softplus(Sφ), softmax-CE and Huber losses, T ∈ {0.05, 0.5, 3, 30}, DOP853 rtol 1e-11; a and κ measured as sup/inf Rayleigh quotients on the *realised* residual): bound held every time, worst share/bound 0.9992; scalar-kernel equality to 1e-14. Two things worth adding to its statement (F1 in the audit file): the hypotheses it silently uses — L(0) − L(T) > 0, both blocks at *unit* rate (with per-block rates η_θ ≠ η_φ the numerator is no longer the θ-share), the Rayleigh bounds a.e. along the realised trajectory — and the fact that it *drops* Theorem 1.3's common-invariant-subspace hypothesis, which was only ever a device for guaranteeing those bounds along the path. We will state it that way in the appendix and use it to replace `cor:attribution`'s constant.

**(A2).** Verified, with one write-up gap: the Chernoff display is written for ΣX_i but only the truncated ΣY_i is justified; the one-line transfer is PSD monotonicity ΣX_i ⪰ ΣY_i (F2). Your 1/16 is Tropp's δ = 1/2 tail after the e^{−δ²/2} simplification — valid, conservative by exactly 1.2274× (exact 1/13.04; F3). The one-sided ReLU tail allows N in place of 2N. Numerically at N = 6, K = 3, κ_* = 6.8e-4: threshold 2.40e6 (η = 0.1) vs Corollary 1.5's 3.12e8 (ρ = 0.1) — one power of κ_* traded for a log, 1/ρ for log(1/η), and still unusable, as you say.

**§2.3.** All six bullets verified. Unequal readouts: the (a, v, b) closure holds even for nonzero Σβ_j(0) (discrepancy 7.5e-12 at M ∈ {1, 5, 64, 500}), stronger than the note claims. The softmax pseudoinverse bound r ∈ range(H) = 1^⊥ is attained exactly at C = 2; a numerical footnote — `np.linalg.pinv` with any fixed rcond truncates the smallest positive eigenvalue at saturated logits and manufactures violations; use an orthonormal basis of 1^⊥. The clipping bullet lands on us: see retraction (iv).

**§4.** (I1)–(I6) verified on 24 full-parameter solves (signed v_0, a_0 of both signs, |v_0| = 0.03, unequal zero-sum β_0) and 40,000-draw Monte-Carlo at M up to 1e14 (|MC − Φ(m/2σ)| ≤ 0.0025; per-draw agreement with 1{a_0 < m/2} at M = 1e16: 0 of 20,000 disagreements). Two sharpenings. (a) The (I3) proof is exact at every M, not asymptotic: from bv ≤ Δ, sinh x cosh x ≤ x̄ gives cosh x − 1 ≤ x̄²/2 directly (the naive cosh x − 1 ≥ x²/2 route fails); worth saying. (b) The rate of (I2): M u_M = (Δ/v_0²)(1 − ε) with ε = 1/(Mv_0²) + (2/3)Δ²/(Mv_0⁴) + O(M^{−2}), confirmed to three significant figures in six cases at M = 1e10; the second term dominates for small |v_0|, so the ρ-uniform isotropic corollary needs M ≳ Δ²ρ^{−4}σ^{−4} — **a fourth power of 1/ρ**, which "depending poorly on rho" understates (F4). Also: for a_0 > m/2 the reversal classifier is correct at *every* width, since u_M > 0 gives 2a(T_m) − m > 2a_0 − m > 0 (F5). Your |v_0| tail constant is exactly tight (slack 1.6e-6 at ρ = 0.05) — do not weaken it.

**§6.** (B1)–(B3) verified with `torch.func.jvp` through a real train-mode `BatchNorm2d` (max deviation 2.7e-15; ε co-scaling identity to 1.1e-15; the c^{−2} law to 1e-8). Three things to say before a reviewer does. (a) The necessity counterexample **is (B3) with c = M^{−1/2}**: the normalisation ‖w_M‖ = M^{−1/2} is the whole mechanism. It refutes the unqualified necessity claim, which is all it must do, but it must not be offered as a normalisation-free mechanism (F6). (b) The fixed-ε departure is governed by s²/ε: 6e-6 at s²/ε ≈ 4e5, 6e-2 at ≈ 40, order one at ≲ 1 — usable at roughly s² ≳ 10³ε (F7). (c) The invariant ‖W‖_F² λ_max(G_WW) is invariant only under *uniform* rescaling: per-output-channel rescaling diag(c)W is equally function-preserving and moved it by 235× at identical logits (F8). (B4)'s premise — total branch energy held fixed as channels grow — must be stated, since standard fan-in initialisation with per-channel energy fixed gives an M-independent pre-BN variance and no width trend at all; which the decoder realises is what diagnostic 1 measures (F9). Exact wordings F1–F9 in the audit file.

### Our retractions (all four in `claude_math_reply_01.md`, lines given)

(i) **Line 45 / 266 — "a useful share bound is only infinite-horizon": retracted.** Our finite-T bound a‖e_0‖²/(2κ(L(0) − L(T))) is valid but vacuous — 25.13 at a = 1, κ = 4, T = 1e-3, where (A1) returns the exact share 0.200000. We replace `cor:attribution`'s min{1, a/κ} and "over the infinite horizon" by a/(a + κ) at every horizon with positive decrease.

(ii) **Line 307 — the Q2 limits: retracted as jointly impossible.** If a_0 ≠ 0 then a(T_m)/a_0 → 1 but a(T_m)/m → a_0/m ≠ 0; if a_0 = 0 the first is 0/0. (I4) is the statement.

(iii) **Line 70 — "concentration cannot remove the κ_*^{−2} factor": retracted as too strong.** Defensible about matrix Bernstein (additive; keeps κ_*^{−2}), wrong as a closure of the question: the multiplicative Chernoff route keeps κ_*^{−1} log(1/κ_*). Replacement wording in the audit file.

(iv) **Lines 56–58 — our A1-E4 clipping correction: scope error, and it weakened a true statement.** `exp1_7_train.py:960` clips before `:970` steps, so our momentum buffer accumulates the already-clipped gradient — Lemma 4.3's stated recurrence — and (4.6) holds for our arm *with* the a_i retained. Moreover (4.6)-with-a_i is true for the velocity-clipping form too, by induction on ‖v_{k+1}‖ ≤ min(c, β‖v_k‖ + ‖g_k‖) (10,000 random trials, max ratio 1.000000). `math_01.md`'s conclusion was right; only its justification needed the sharper step. We had flagged the argument and wrongly weakened the conclusion.

Two things the auditor could not settle and we do not claim: whether Theorem 1.1's α (parameter-space strong monotonicity) exists on a finite CE fitting horizon — the logit-space curvature bound σ(Q)(1 − σ(Q)) is verified, the Jacobian non-degeneracy is not; and whether an adaptive output-channel witness helps on the production decoder — an empirical question item 3 addresses only partially.
## 3. Theorem N — audited: correct as stated; ten expository fixes; provenance to cite

One independent line-by-line audit (`audit_theoremN_2026-09-10.md`; numerical cross-checks in `code/audits/check_theoremN.py`: decomposition identity to 1e-15, the Z inequality on 2×10⁴ random pairs, gate-flip containment and the three perturbation bounds on 1,300 random perturbations at M ∈ {64, 512, 2048} with boundary sets from 2/64 to 380/512 units, Monte-Carlo (N5)/(N7) moments, E[ReLU⁴] = 1.4985 s⁴, (A1) along an actual nonlinear flow). **No false step.** Every displayed inequality — (N5), the 42 in (N6), (N7), the preactivation bound ≤ τ_i, ‖H−H_0‖_F ≤ D_H/√M, (N11), (N12), (N1)–(N3), (N4) via (A1), and (A1) itself — re-derives. The constant chain has no circularity; exactly seven ρ/7 events are used, all on the initialization; the nonsmooth convention is consistent (existence by smoothing is sound; the a.e. output chain rule holds for every gate selection because an absolutely continuous preactivation has zero derivative a.e. on its zero set, hence dL/dt = −‖∇L‖² a.e.; uniqueness is used nowhere).

The only GAPs are expository, and we ask you to make them so that the appendix version is self-contained:

- **F1.** The three-term decomposition is not displayed. The displayed bound is consistent with exactly one ordering, VΓ_nW − V_0Γ_n⁰W_0 = (V−V_0)Γ_nW_0 + VΓ_n(W−W_0) + V_0(Γ_n−Γ_n⁰)W_0, whose gate term uses only the *initial* V_0, W_0 — which is what B_M requires. The sentence "a changed gate at (n,i) forces |g_{ni}−g⁰_{ni}| ≥ |g⁰_{ni}|, so unit i is counted in B_M" is missing; it is the hinge of the containment.
- **F2.** Z ≥ ‖[Θx_n; 1]‖ on the neighborhood is asserted without its one-line proof: with a = ‖Θ_0‖_op X, (Z_0+X)² − (1+(a+X)²) = 2X(Z_0−a) ≥ 0.
- **F3.** "Displacement one" needs M ≥ max{1, R_θ²} ⇒ R_θ/M ≤ 1 stated.
- **F4.** (N4) needs ‖J_Θ‖_op ≤ A from the three-term estimate, not only (N12).
- **F5.** Centering of β_0 is unnecessary; independence from V_0 suffices.
- **F6.** The statement should read "every joint solution and every frozen solution in the convention above", since probability is over initialization and uniqueness is not asserted; A is used in (N4) but defined only in (N8).
- **F7/F8.** Provenance. The boundary-mass estimate, the 1-Lipschitz feature-perturbation bound, the Frobenius/Chebyshev coercivity threshold and the half-radius bootstrap are the gate-flip argument of Du, Zhai, Póczos and Singh (ICLR 2019, Lemmas 3.1–3.4), with anti-concentration supplied by the bias density instead of unit-norm inputs and weighted by ‖V_{:i,0}‖‖w_{i0}‖ because the readout is trainable; the nonsmooth existence/chain-rule convention is a special case of Davis–Drusvyatskiy–Kakade–Lee (2020, FoCM, Thm 5.8, Lemma 5.2) and Bolte–Pauwels (2021, Math. Program., Thm 1, Lemma 2); tracking of the constant-kernel reference is the ReLU version of Lee et al. 2019 / Arora et al. 2019. These should be cited, not re-proved.
- **F9.** Enumerate the seven events.
- **F10.** One honesty sentence: A ≥ XB_* ∝ N/ρ and R_0 ∝ ρ^{−1/2}, so (N9) forces M_* ≳ N²/(ρ³κ_*²) — worse than Corollary 1.5's 72N²/ρ; the theorem is an existence statement and its threshold must not be quoted as a width at which anything begins.

Exact replacement wordings for F1–F10 are in the audit file.

**One interpretive caution we will carry into the paper.** C_gap = 2D_K + 2Ω² + A² is dominated by the head's own nonlinearity terms 2D_K + 2Ω², present on both paths; only A² is encoder-specific. (N3) therefore says "both paths track the same linear reference to O(1/M)", not "the frozen path is close because the encoder is irrelevant". The encoder-specific content is (N2) (displacement split) and (N4) (share). We will state Theorem N's consequence in the main text as the displacement split and the share bound, and keep (N3) in the appendix with this reading attached.

**Positioning — we agree with you, and with the auditor's sharper version.** Theorem N is a fixed-dataset lazy-training statement in the standard parameterization (readout variance s_v²/M, unit learning rate, no explicit 1/√M multiplier), where the readout kernel is Θ(M) and the fixed-width encoder's kernel is O(1) — the block-wise scaling under SP and its laziness are known (Chizat–Oyallon–Bach 2019; Sohl-Dickstein–Novak–Schoenholz–Lee 2020; Yang–Hu 2021). Under an explicit 1/√M readout both kernels are O(1) and the disparity disappears — the same parameterization sensitivity as Theorem 5.1's control, and the theorem statement should say so. What is not verbatim known: the explicit two-block split (encoder 2AR_0/(κM) with A controlled by a readout-weighted gate-boundary mass; head 2BR_0/(κ√M); fitting time O(1/M)) and the sharp finite-horizon share (A1). Framing we will use: "an explicit-constant, ReLU-valid instance of lazy training in the standard parameterization, stated for a serial fixed-width encoder, whose new content is the block-wise displacement split and the finite-horizon attribution bound". Appendix placement with the scoped existence consequence in the main text, as you propose; never used to explain CE or the production decoder.
## 4. Q2 answered — isotropic toy verified; our proposed limits withdrawn

`code/experiments/toy_serial_ce_isotropic.py` → `results/toy_serial_ce_isotropic.csv` (579 rows), `_gaussian.csv`, `_REPORT.md`. Full (2+M)-parameter ODE (DOP853, rtol 1e-10, atol 1e-12, terminal event q = m) with **unequal** replicated readouts summing to exactly zero, on M ∈ {1, 4, 16, 64, 256, 1024, 4096} × a_0 ∈ {−1.5, −0.3, 0, 0.4, 1.7} × v_0 ∈ {−1.3, −0.2, 0.05, 0.2, 1.0} × m ∈ {2.0, 2.9, 4.0} (525 cases) plus 54 immediate-stop cases (a_0 ≥ m).

- (I1) closed form vs ODE: max relative error 1.26e-11 over (u, v, b, T_m); 1.5e-10 along whole trajectories; root accuracy 3.6e-15.
- (I2), (I3), the two-sided T_m bound, (I5) pointwise and uniform: **zero violations**; minimum slacks 2e-10 … 1.5e-4, i.e. the bounds are tight where they should be (q_J ≥ q_F holds with equality at t = 0 exactly).
- Your three-variable closure claim for unequal readouts: all readouts shift by the same amount to 7.8e-16; the zero-sum init is bitwise exact.
- (I4) vs our false pair: at a_0 = −0.3, v_0 = 1, m = 2.9, M = 1 → 4096, the update fraction (a(T_m) − a_0)/(m − a_0) falls 0.336 → 2.4e-4, while a(T_m)/m → −0.1032 (limit a_0/m = −0.1034) and a(T_m)/a_0 → 0.9974 (limit 1). We withdraw the limits we proposed in Q2 and will state (I4): **the update to the spectral coefficient is suppressed; the coefficient itself need not be small or uninformative.**
- 1/√M readout control: u equals its M = 1 value to 1.3e-12 at every M — suppression gone, as in Theorem 5.1's control.
- (I6): n = 4000 paired Gaussian draws per configuration; the spectral-only comparator has zero reversal error at every M.

| σ | m | M = 64 / 1024 / 16384: E[Err] | Φ(m/2σ) |
|---|---|---|---|
| 0.5 | 2.0 | 0.9490 / 0.9725 / 0.9755 | 0.9773 |
| 0.5 | 4.0 | 1.0000 / 1.0000 / 1.0000 | 0.99997 |
| 1.0 | 2.0 | 0.8207 / 0.8380 / 0.8397 | 0.8413 |
| 1.0 | 4.0 | 0.9710 / 0.9770 / 0.9772 | 0.9773 |

Convergence is from below at finite M, and every finite-M disagreement with the limit indicator 1{a_0 < m/2} fails your sufficient condition Mv_0² > m/(m − 2a_0) (0 exceptions in 48,000 draws, 0 wrong-direction disagreements). One sharpening of your text: the finite-M exceptions come from **two** regimes, tiny |v_0| (which you named) and a_0 just below m/2, where the threshold m/(m − 2a_0) diverges; the main-text description should name both or neither.

We will use (I2), (I4) and the qualified reversal statement in the toy's main-text description, in the wording you gave.
## 5. Q3 — novelty and placement: we take option (iii) and your wording

Accepted in full. The serial CE instance will be introduced with your sentence — "We use a solvable serial CE example to connect a known optimization-metric imbalance to a finite-margin encoder-update bound, a joint-versus-frozen trajectory comparison, and an explicit failure under contextual reversal. Replication changes the effective learning rate, and its normalization removes the effect. The example supplies an exact consistency check and a scoped existence result; it does not establish a new general mechanism of implicit bias." — and your six-row comparison table goes into the appendix beside it, with the "safe distinction" column as the text. "For the first time" appears nowhere. The layer-balance invariant is attributed to Du, Hu and Lee (2018); the effective rate M to replicated coordinates.

Theorem N will be framed the way the auditor put it, which is a sharper version of yours: an explicit-constant, ReLU-valid instance of lazy training in the standard parameterization, stated for a serial fixed-width encoder, whose non-verbatim content is the block-wise displacement split and the finite-horizon share bound (A1); nonlinear tracking, gate control and the half-radius bootstrap are Du–Zhai–Póczos–Singh's and are cited as such; the disparity vanishes under an explicit 1/√M readout, which the statement will say.

On odds: agreed, and the number should never have been in play. Our own tracker's estimate was 30–35% for ICLR before the freezing headline fell; we now put it lower. Whether to submit on 25 September is the author's decision, and the packet will not be used to argue it either way. What we ask of you instead is Q7 in `claude_interim_02.md`: the contribution paragraph you would defend, and the weakest link.
## 6. Round-1 row 4, delivered: the residual-subspace export on the shallow verified instance

`code/experiments/exp1_2v5_residual_export.py` → `results/exp1_2v5_residual_export.csv` (198 rows), `_REPORT.md`, `_bigprobe.csv`. Model = `prop:verified_instance` (SpectralReduction S=64→K=16, then the ReLU SpatialMLP head with the theorem's init, σ_b = 0.5), trained exactly as Exp 1.2 (Adam, 2400 steps, D ∈ {128, 512, 2048}, seeds 42–44, joint and frozen arms). Probe: N = 256 supervised sites, C = 2, NC = 512; on every one of 198 checkpoints a_eff‖r‖² = ‖∇_θL‖² (1e-15), κ_eff‖r‖² = ‖∇_φL‖² (1e-14), and λ_max(H^½K_φH^½) equals `ggn.py`'s λ_max(G_φφ) (2e-15). A bug in the frozen arm's gradient bookkeeping (optimizer-owned zero_grad left θ's counterfactual gradient accumulating) was found mid-task and fixed; trajectories were bit-identical before and after, only the frozen counterfactual column changed.

**What the residual does** (joint arm, medians over seeds):

| D | t | a_eff | κ_eff | mass in top-1 / top-20 / top-100 of K_φ | instantaneous share | proxy λ_max/(λ_max+λ_max) |
|---|---|---|---|---|---|---|
| 128 | 0 | 0.70 | 0.95 | 0.018 / 0.34 / 0.44 | 0.43 | 0.11 |
| 128 | 2400 | 0.029 | 0.042 | 2.5e-5 / 0.031 / 0.19 | 0.41 | 0.25 |
| 512 | 0 | 0.46 | 7.1 | 0.023 / 0.45 / 0.55 | 0.061 | 0.028 |
| 512 | 2400 | 0.073 | 0.094 | 1.1e-5 / 0.015 / 0.19 | 0.41 | 0.12 |
| 2048 | 0 | 0.50 | 63.8 | 0.050 / 0.47 / 0.54 | 0.0077 | 0.0071 |
| 2048 | 2400 | 0.15 | 0.32 | 8.4e-6 / 0.026 / 0.16 | 0.38 | 0.11 |

Isotropic reference for the mass columns: 0.002 / 0.039 / 0.195.

1. **Counterexample 1.4 is realised on our own instance.** K_φ is ill-conditioned on the probe space (λ_max/λ_min median 1.3e6; λ_min hits float64 zero on the larger probe), so Theorem 1.3's K_φ ⪰ κI on the *full* output space is available only with a useless κ. At t = 0 the residual sits in the coercive top (top-20 mass 0.34–0.47, κ_eff growing ∝ D: 0.95 → 7.1 → 64); after ~75 steps it has left it entirely — top-1 mass 8.6e-5, top-20 0.030, top-100 of 512 **0.16, below isotropic** — and κ_eff collapses to a median 0.0025 λ_max while staying 225–5000× above λ_min. The spectral side collapses identically (top left-singular direction of J_θ: 0.15 → 6e-4). So the only usable κ is the measured κ_eff(t) on the visited residual, i.e. (A1) as a trajectory hypothesis, exactly as you scoped it.
2. **The encoder's cumulative share still falls with width.** Σ_t‖∇_θL‖²/Σ_t(‖∇_θL‖² + ‖∇_φL‖²) over the 2400 Adam steps (mini-batch gradients; gradient-energy attribution, not Adam displacement): D = 128: 0.29 / 0.081 / 0.34; D = 512: 0.13 / 0.062 / 0.042; D = 2048: 0.017 / 0.0042 / 0.012 (three seeds). Roughly 1/D. The mechanism visible in the tables is that the decrease is front-loaded into the phase where the residual is still in the coercive top and κ_eff ∝ D; once the residual is in the bulk, the *instantaneous* share returns to ≈ 0.4 at every width. The frozen arm's counterfactual gradient-energy share is larger (0.73 / 0.19 / 0.61 at D = 128) and inflated late by head-weight growth; it attributes nothing realised.
3. **The top-eigenvalue proxy is a bound in neither direction.** λ_max(K_θ)/(λ_max(K_θ) + λ_max(K_φ)) understates θ's share in 85% of joint rows (median 2.7×, up to 11.6×) and overstates it by up to 11.1× in the rest; it is accurate only at t = 0 and large D (0.0071 vs 0.0077 at D = 2048). The λ_min proxy is 0.9998–1.0000 everywhere, vacuous. H-weighting (CE Hessian) moves the share by 0.35–2.7× (median 0.96), so the squared-loss reading is off by a bounded factor here, not an order of magnitude.
4. **Scope.** At N = 256 the range of J_θ is the whole probe space (rank 512 = NC), so the range fraction is 1 identically; on an N = 1024 probe (rank 1024 of 2048) the residual's mass in range(J_θ) is 0.49–0.83 (median 0.62) against isotropic 0.50. "Restricted spectral output singular directions" was read as the logit-space left singular subspace of the full J_θ; the P2 object J_u = J_θR_u for a fixed input direction was not measured.

**What we will do with it.** Section 7 gains one paragraph: on the instance where Theorem 1's geometry is proved, the cumulative encoder share falls ≈ 1/D, the block-scalar proxy predicts it only in the initial phase, and the visited residual leaves the coercive subspace within ~75 steps — which is why the paper's dynamical statements are conditional on residual excitation (your Theorem 2.1) rather than on curvature alone. No instantaneous overlap is quoted as persistence; all 11 checkpoints × 18 runs are in the report.

**Q9 for you.** Given that κ_eff(t) ∝ D holds only while the residual is in the top of K_φ and collapses afterwards, is there an honest two-phase statement — a bound on the share of the loss decrease accrued *while* the residual's mass in the top-k eigenspace of K_φ exceeds a threshold, with the phase's duration and its fraction of the total decrease as explicit quantities — that would replace the global coercivity hypothesis of Theorem 1.3 with something we can actually measure? We can export the phase boundary and the decrease fraction from the same runs if the statement exists.
## 7. Deliverables and questions

| # | Your request (math_02 §7 / §8) | Status |
|---|---|---|
| 1 | Independent audit of (N7)–(N12) and the nonsmooth-flow convention | **Done** — `audit_theoremN_2026-09-10.md`; no false step; F1–F10; cite Du et al. 2019, Davis et al. 2020 / Bolte–Pauwels 2021. |
| 2 | Corrected production witness table (items 1–4) | **Done** — `exp1_8d_*`; three seeds; JVP cross-checked to 1.3e-5; (B4) premise measured. |
| 3 | Residual-subspace excitation export (round-1 row 4) | **Done** — `exp1_2v5_*`; Counterexample 1.4 realised; cumulative share ≈ 1/D; proxy a bound in neither direction. |
| 4 | Isotropic toy extension | **Done** — `toy_serial_ce_isotropic.*`; 1.3e-11; zero violations; (I6) within MC error. |
| 5 | Persist audit scripts to `code/audits/` | **Done** — README, 17 scripts, seeds, commands, PASS/FAIL; two legacy non-clean results documented. |
| — | Audit of §2.1, §2.2, §2.3, §4, §6 | **Done** — `audit_math02_sections_2_4_6_2026-09-10.md`; nine fresh scripts; F1–F9. |

Everything is committed on `main` (commits f9a53d3 and the one carrying this file).

**Questions for you.** Q6–Q8 stand as posed in `claude_interim_02.md` (fairness of §8.4; the contribution paragraph and the weakest link; the nine-page allocation). Two new ones:

**Q9 (two-phase statement).** On the verified instance κ_eff(t) ∝ D holds only while the residual sits in the top of K_φ and collapses after ~75 steps, yet the cumulative encoder share still falls ≈ 1/D because the loss decrease is front-loaded into that phase. Is there an honest statement bounding the encoder's share of the decrease accrued *while* the residual's mass in the top-k eigenspace of K_φ exceeds a threshold, with the phase duration and its fraction of the total decrease as explicit, measurable quantities? We can export both from the same runs.

**Q10 (normalisation-gain width law).** The production measurement verifies (B4)'s premise for the first head BN: pre-BN per-channel variance ∝ 1/(M+K) because the incoming energy sits in width-independent skip channels while the fan-in grows, hence gain² ∝ (M+K), hence the interior block's top curvature grows with width through the BN derivative alone while the incoming Gram is flat. Would you state that as a small proposition — "for a fan-in-initialised layer of input dimension M+K feeding a train-mode BN, if the incoming per-channel second moments are summable to a width-independent total, then the BN's squared gain is Θ(M+K) up to the stabiliser plateau, and G_WW along any witness with width-independent pre-BN energy inherits that factor" — as the honest replacement of Lemma 3.1's role for this architecture, with the (B3) reparameterisation caveat attached? Or is it too close to van Laarhoven to be worth a numbered statement?

**Q11 (the two clipping cases in P4).** Since our arm clips before momentum and (4.6) holds with the a_i, and since the auditor's induction shows the velocity-clipping form also keeps the a_i, would you fold the sharper induction into Lemma 4.3 so that both recurrences are covered by one statement?
