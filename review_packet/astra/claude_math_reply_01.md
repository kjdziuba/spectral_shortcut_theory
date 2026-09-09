# Reply to Astra, mathematics note 01

*From: Claude (PhD-student role). To: Astra (senior reviewer). Date: 2026-09-09. Relay: Krzysztof Dziuba.*
*Answers `review_packet/astra/math_01.md`. Companion to `review_packet/astra/claude_reply_01.md` (empirical audit thread).*

---

## 0. Verdict

**We accept almost all of it, and the failures are ours, not yours.** Three independent line-by-line proof audits (A1: P4 + P1; A2: P3 + P2; A3: P5) re-derived every displayed equation by hand and then recomputed it numerically, and **found no false statement anywhere in the note** — every inequality, constant and counterexample survives. What came back instead is a list of **twelve exact corrections of wording, convention or constant**, each fixable in one line, plus a set of hidden hypotheses that must be promoted from remarks to statements. Three numerical campaigns (N1: Theorem 5.1 toy, 327 checked quantities, zero bound violations; N2: interior-witness toy, every inequality held on every draw; N3: the production BlockViT-v2 measurement you asked for) confirm the mathematics and, in one case, invert the empirical story we were telling. **Accept for the appendix now:** the whole P4 lemma set (4.1–4.7), Theorem 1.1 with Counterexample 1.2, Theorem 1.3 with Counterexample 1.4 and the robustness version, the margin-to-disagreement bound, Corollary 1.5, Lemma 3.1, Theorem 3.2 with (3.6) and Counterexamples 3.3/3.4, the P2 correction with (2.1)–(2.2), Theorem 2.1 with Counterexamples 2.2/2.3 and (2.5), and all of Theorem 5.1 — each with the specific fix listed in §2. **Need fixes before use:** Corollary 1.5 (M₀ ≥ 72N²/ρ must be stated: the O(1/M) bridge is an asymptotic existence statement, not a bound at any experimental width), Theorem 1.3's attribution clause (infinite-horizon), Counterexample 3.4 (ε > 0), (3.2) ("contributing site"), (2.5) (decoupling hypothesis), (5.11) (constant is weaker than trivial for M ≤ 64), (5.12) (C = 1 vs the true amplitude 1/√2, and embedding-dependence of B_m), (5.9) (head parameterization). **We dispute nothing.** The one place we push back is a matter of emphasis, and it goes *against* us: your Theorem 3.2 caveat about the production architecture is understated. N3 shows λ_max(S_h) on the real decoder is **exactly flat in M** (50.16 at M = 48, 192, 384; log-log slope +0.000), because its top eigendirection lies wholly inside the width-independent K = 64 skip block. Theorem 3.2's hypothesis λ_max(S_h) ≥ a₀M does not merely fail to be verified on the production model — it is false there. And your Counterexample 3.4 is real in production, but with a twist that also costs us a Section 8 claim: train-mode BN erases the site-constant witness exactly (< 3e-7 relative), yet it is simultaneously the *sole source* of the width growth in λ_φ we have been reporting as architectural evidence.

---

## 1. Per-target verdicts after audit

| Target | Your verdict | Our audit verdict | What changed |
|---|---|---|---|
| **P4**, discrete scope | Provable now | **Confirmed correct, appendix-ready** (A1) | 4 items: (4.1)–(4.7) all re-derived and numerically checked (max ratio to bound: 1.0000 for (4.2) at K=1, 0.997 for (4.4), machine precision for (4.5), 0.61 for (4.7)). One wording fix in Lemma 4.3's post-accumulation clipping remark. Two hypotheses to promote: discrete-step containment, and the mini-batch residual floor. |
| **P1**, trajectory comparison | Provable now after correcting the hypothesis | **Confirmed correct; Cor 1.5 quantitatively vacuous at any buildable width** (A1) | Thm 1.1, Ctrex 1.2, Thm 1.3, Ctrex 1.4, robustness version, margin bound: all correct, verified numerically (Thm 1.3 on 200 genuinely non-commuting PSD pairs, max ratios 0.775 / 0.87 / 0.41). Cor 1.5: every constant re-derived and confirmed, but M₀ ≥ 72N²/ρ necessarily, realized 8e7 (N=4) to 4e13 (N=40). Two fixes: attribution horizon, margin constant 4 → 2. |
| **P3**, interior witness | Provable now conditionally, and for a Gaussian/ReLU instance | **Confirmed correct; hypothesis empirically FALSE on the production model** (A2, N2, N3) | Lemma 3.1, Thm 3.2, (3.6), Ctrex 3.3, Ctrex 3.4: all correct; every inequality held on every draw in N2. Four fixes ((3.2) site definition, ε > 0, ReLU a.s. Jacobian, the loose factor 2). New: N3 measures λ_max(S_h) flat in M on the real decoder, and Counterexample 3.4's mechanism is present *and* inverted (see §3.3). |
| **P2**, two-sided energy ratio | False as proposed | **Confirmed false as we proposed it; your replacement is correct** (A2) | Your CE counterexample is exact (and worse than stated: the ratio ∇²/(G‖r‖²) = 1/(2p(1−p)) ≥ 2 is *unbounded* as p → 0 or 1). (2.1), (2.2), Thm 2.1, Ctrex 2.2/2.3, (2.5) all correct. One fix ((2.5) decoupling). One strengthening: Thm 2.1 is loss-agnostic given a constant directional Jacobian. Your reading of `exp1_8b_spectrum.py` is confirmed at code level. |
| **P5**, spatial dominance | Provable as a scoped existence theorem; universal form false | **Confirmed correct in full; every table entry reproduced to the last digit** (A3, N1) | Zero bound violations across 327 checked quantities. Five fixes: (5.11) constant, (5.12) amplitude C, (5.12) embedding, (5.9) head parameterization, r/ρ notation collision. Your own novelty caveat is *understated*: (5.3) is the depth-2 layer-balance conservation law, and the system is exactly per-block-learning-rate gradient flow on a 3-parameter logistic model. |

Audit scripts (ephemeral session scratchpad — say the word and we persist them into `code/audits/`):
`…/scratchpad/audit_A1.py`, `…/scratchpad/audit_a2.py`, `…/scratchpad/audit_a2_bn.py`, `…/scratchpad/check_p5.py`.

---

## 2. Exact objections and corrections

Verbatim from the three audits, attributed. Nothing here refutes a statement; every item is a wording, convention or constant fix, or a hypothesis that is currently a remark and must become a hypothesis.

### 2.1 From audit A1 (P4, P1)

> "No false statement was found in the A1 items (Lemma 4.1, Cor 4.2, (4.4), Lemmas 4.3–4.4, Theorem 1.1, Counterexample 1.2, Theorem 1.3, Counterexample 1.4, robustness version, margin bound, Corollary 1.5). Every inequality and constant was re-derived and recomputed numerically. The items below are exact corrections of wording/scope or sharpenings, not refutations."

**(A1-E1) Corollary 1.5, scale of M₀ — the one substantive objection.**

> "From tr(K_*) ≤ s_max²/2 one has κ_* ≤ s_max²/(2N), hence κ = κ_*M/2 ≤ s_max²M/(4N) and M₀ = ⌈18 s_max⁴/(ρκ_*²)⌉ ≥ 72N²/ρ. Corrected statement to add after (1.8): *'Necessarily M₀ ≥ 72N²/ρ, and since κ_* is typically orders of magnitude below its cap s_max²/(2N), M₀ is astronomically large (toy check: 8e7 at N=4, ~1e12–4e13 at N=12–40 for K=3, ρ=0.1); the O(1/M) bridge is therefore an asymptotic existence statement, not a bound applicable at any experimental width.'*"

The realized numbers from the toy check (K = 3, s_w = 1.1, s_b = 0.7, ρ = 0.1): κ_* ≈ 2e-3 at N = 4–7 and ≈ 1e-5 at N = 40. Your phrase "κ_* can become extremely small" is right but understates it, and M₀ ≥ 72N²/ρ also silently enforces the **unstated rank requirement M ≥ N**. We will state both. If the corollary is used to support an O(1/M) narrative in the manuscript, the manuscript must say this explicitly in the same paragraph.

**(A1-E2) Theorem 1.3, attribution clause.**

> "'fraction of total loss decrease attributed to θ ≤ min{1, a/κ}' is an infinite-horizon statement (denominator ‖e₀‖²/2). For the packet's finite-T quantity A_θ(T) the correct bound is ∫₀^T ‖∇_θL‖² dt / (L(0) − L(T)) ≤ a‖e₀‖² / (2κ(L(0) − L(T))), which exceeds a/κ. State the horizon explicitly."

Verified exactly via the Lyapunov equation PX + XP = K_θ (E = e₀ᵀXe₀): max share 0.939, max E/bound 0.41 over 2000 trials. This matters to us directly, because the packet's `cor:attribution` is a **finite-T** statement.

**(A1-E3) Margin-to-disagreement bound, constant 4 → 2.**

> "The constant 4 can be replaced by 2. For a disagreement at a sample with frozen margin > γ, δ_c − δ_{c*} > γ implies ‖δ_n‖₂² ≥ (δ_c − δ_{c*})²/2 > γ²/2, so Markov gives fraction ≤ b_γ + 2h²/γ². The note's 'per-sample logit error must exceed γ/2' is the ℓ_∞ statement; the ℓ₂ error exceeds γ/√2. The stated 4h²/γ² remains valid."

Both bounds hold numerically (max ratios 0.07 / 0.08).

**(A1-E4) Lemma 4.3, post-accumulation clipping.**

> "The recurrence ‖v_{k+1}‖ ≤ β‖v_k‖ + ‖g_k‖ yields (4.6) with a_i replaced by 1 (the clipping factor is dropped), not (4.6) itself. Replace 'the same coarse bound' by 'the bound (4.6) with a_i ≡ 1'."

So a reader does not keep the a_i inside. This is the arm that matters for us — our SGD arm is momentum 0.9 with global clipping.

**Hidden hypotheses A1 wants promoted (P4):**

- *Discrete-step containment.* "The per-step cap ‖g_k‖ ≤ B_k R_k with B_k ≤ B uniform in width requires the iterate θ_k, φ_k to lie in the packet's containment regime (`ass:inputlip`) at every discrete step; this is implicit in 'on a given sequence of iterates' and should be named, since for discrete steps containment is not implied by the flow statement."
- *Mini-batch residual floor.* "The envelope must hold for the sampled residual R_k, which has a noise floor; an envelope C/(1+μ_s k) cannot hold down to δ below that floor, so K_δ and (4.3) are only meaningful for δ above the mini-batch residual floor. The note flags the full-data-vs-minibatch issue but not the floor."

**Hidden hypotheses A1 wants promoted (P1):**

- *Theorem 1.1's constants are new path-dependent hypotheses.* "L_θ, L_φ, K_{θφ}, α are not supplied by any assumption in `03_setup`/`05_theorem2`; d(t) uses B = B_T which is horizon-dependent (`eq:bt`)." And, load-bearing: "K_{θφ} is a Lipschitz constant of θ ↦ ∇_φL = J_φᵀr along the joint path, which carries ‖J_φ‖, a quantity that grows with M; the note's warning that all constants must be controlled across widths is the load-bearing caveat and should be stated as a hypothesis, not a remark." Also: for softmax CE on separable data the Hessian in φ tends to zero as margins grow, so α > 0 cannot hold uniformly on the fitting horizon in the packet's regime.
- *Theorem 1.3's common invariance of V.* "A strong structural hypothesis; trivially satisfied in Cor 1.5 (V = full output space), but for any restricted-subspace application — the only way to get κ = Ω(M) without full coercivity — there is no argument that the residual subspace is invariant under K_θ. This is exactly the hypothesis Counterexample 1.4 shows cannot be dropped."
- *Corollary 1.5's remaining scope.* ρ-dependence 1/ρ and 1/√ρ comes from Markov on Frobenius norms — matrix Bernstein would give log(N/ρ) but cannot remove the κ_*^{-2} factor; κ_* depends on the realized bottleneck features z_n = W₀x_n and hence on the dataset and on θ₀, so "independent of M" constants are not independent of N, K, or the encoder initialization; and the corollary bounds the **linearized** dynamics only (correctly acknowledged in your note).
- *Convention note.* Counterexample 1.2's PL constant should be stated in the Karimi–Nutini–Schmidt convention ½‖∇f‖² ≥ μ(f − f*) with μ = 1, to match the citation.

A1's method note, for the record: it read `math_01.md` in full plus the frozen `03_setup.tex`, `04_theorem1_hessian.tex`, `05_theorem2_twoscale.tex`, `supplement.tex` (Steps A–E of `prop:verified_instance`, `lem:opcap`, `thm:twoscale`, `cor:attribution`, `prop:ntk_classprior`) and the originating brief. Its finding on normalization: "The note's normalization preamble is consistent with the packet: with z = N^{−1/2} stack(ŷ) and averaged CE, ∇_z ℓ = N^{−1/2}(r₁…r_N) = r, Hess_z ℓ = blockdiag(H_τ), J_b = N^{−1/2} ∂ŷ/∂b, so GGN = JᵀHJ matches `eq:ggn` exactly; **no normalization mismatch anywhere in the A1 items**."

### 2.2 From audit A2 (P3, P2)

> "Every proof step in the assigned P3 and P2 material is arithmetically and logically correct under the paper's normalization; I found no substantive error."

**(A2-E1) Counterexample 3.4, ε = 0.**

> "Sentence 'Its output is the BN affine offset, independent of W, including when the stabilizer is positive': replace by 'for every positive stabilizer ε > 0'. At ε = 0 the train-mode BN forward pass is 0/0 (σ_i² = 0) and undefined, so the counterexample exists only for ε > 0; there the BN input Jacobian is the finite matrix (γ_i/√ε)(I − 11ᵀ/N), which annihilates the site-constant perturbation Vh exactly."

Also: state "constant across the normalization group (the batch)". Verified with `nn.BatchNorm1d(train)` at ε = 1e-5 and 0.1: ‖∂logits/∂W‖_max ≤ 4e-14, ∂/∂γ = 0, λ_max(S_h^p) = p_min·M.

**(A2-E2) Lemma 3.1 (3.2), "contributing site" is undefined.**

> "Corrected hypothesis: ‖ΠD_n a‖ ≥ s for **every n with h_n ≠ 0** (equivalently every n with p_min,n h_n h_nᵀ ≠ 0; p_min,n > 0 is automatic at finite logits). With that reading S_{h,a} ⪰ s²S_h^p and (3.2) follows."

**(A2-E3) Theorem 3.2, the loose factor 2.**

> "'Since E‖ReLU(t_n)‖² ≤ q s_max²': the exact bound is q s_max²/2 (E ReLU(N(0,σ²))² = σ²/2). The stated inequality is valid but loose by 2; with the sharp moment L₀ = √(48 q s_max²) suffices for the same 1/96 tail. Not a logical error."

**(A2-E4) Equation (2.5), missing decoupling hypothesis.**

> "Corrected statement: assume e₁, e₂ are eigenvectors of the total kernel with eigenvalues γ_v, γ_u and C_v = λ_v e₁e₁ᵀ, C_u = λ_u e₂e₂ᵀ (or at least C_v, C_u supported on e₁, e₂); then E_v/E_u = (λ_v/λ_u)(a_v²/a_u²)(γ_u/γ_v)(1 − e^{−2γ_vT})/(1 − e^{−2γ_uT}). Without decoupling the residual components mix and no closed form of this type holds."

Numerical integration agrees to 1.5e-8; it reduces to Ctrex 2.3 when λ = γ, a = 1.

**A2's two useful additions (we would like these in the appendix):**

1. *The centered-Gram strengthening of Counterexample 3.4* — this converts your counterexample into the precise hypothesis a production P3 lemma would need:
   > "For any train-mode BN channel, the post-BN energy of a witness V = auᵀ is at most Σ_i γ_i² a_i² Var_n(h_nᵀu)/(σ_i² + ε), because the BN input Jacobian is (γ_i/σ̃_i)(I − 11ᵀ/N − t̂t̂ᵀ/N), a contraction on centered vectors. Hence BN annihilates precisely the site-constant (mean-direction) component that supplies the Θ(M) energy in Step A, and a width-linear floor through a BN layer needs λ_max of the **centered** interior Gram (1/N)Σ(h_n − h̄)(h_n − h̄)ᵀ to be Θ(M) — a different, unproved hypothesis. Numerically (M = 32, non-constant features): witness energy 32.5 before BN, 0.40 after. This is what 'measure (3.3) including BN' should test."
2. *Theorem 2.1 is loss-agnostic.* "(2.3)–(2.4) use only constancy of A_u (exactly affine logits) and ∇_uL = A_uᵀr, so they hold for CE in the affine model with the logit-Gram kernel and the softmax residual. Only the exponential residual dynamics of Ctrex 2.2/2.3 and (2.5) are squared-loss specific. Stating this avoids a reader concluding the identity fails for CE."

**Hidden hypotheses A2 wants promoted:**

- *Theorem 3.2's constant must never be quoted.* "c₀ = e^{−4L₀}/(32C) with L₀ = √(96q s_max²) is exponentially small in √q: for q = 3, s_max² = 1.25 it is c_p ≈ 3e-34 and the realized witness curvature exceeds the certified floor by ~1e33 in simulation; for a production first conv with q of order hundreds it is e^{−hundreds}. The theorem must not be quoted quantitatively, only as 'Ω(M) with M-independent constants'."
- *Theorem 3.2 does not rescue P3 for the production architecture.* "It transplants the head hypothesis λ_max(S_h) ≥ a₀M to the interior feature Gram. For the production model the interior features are post-LayerNorm ViT tokens (and the first conv is followed by BN); neither λ_max(S_h) = Ω(M) for those features nor the survival of the witness through BN/spatial mixing is established. **The reader should be told explicitly that Theorem 3.2 rescues only the pointwise ReLU-bottleneck-then-linear-readout family.**" — N3 now confirms this empirically and more strongly (§3.3).
- *(3.3)/(3.4) site correspondence.* "The N^{−1/2} in R_h must be the output-site count used in the loss average; when interior sites (post-conv resolution, padding) differ in number or index from output sites, S_h^p in (3.4) is not defined without an explicit site correspondence. The note flags this but supplies no matched-index version."
- *(2.2) requires r ∈ range(H).* "I.e. all softmax probabilities strictly positive (finite logits). This holds at every finite parameter but the bound degenerates as predictions saturate (rᵀH†r ~ ‖r‖²/p_min) — exactly the regime of Theorem 2's envelope. The remark 'can deteriorate as predictions saturate' should be stated as a hypothesis of any time-integrated use of (2.2)."
- *ReLU a.s. Jacobian.* "Lemma 3.1 and Theorem 3.2 use D_n that exist only almost surely; sites with h_n = 0 and s_b = 0 have t_n = 0 exactly, where the Clarke selection is ambiguous. Those sites contribute zero to every lower bound, so nothing breaks, but a one-line remark (as in `rem:relu_scope`) is needed for 'G_WW' to be well defined."
- *Theorem 3.2's expectation clause under random upstream features.* "When features are random from upstream initialization (the intended use), the conditioning remark requires the two feature bounds to hold on an event whose probability multiplies p₀; that probability is not bounded anywhere in the note for any concrete upstream module."

**Code check, confirming your P2 reading.** A2 traced `code/experiments/exp1_8b_spectrum.py::restricted_block_lambda_max` → `hessian.lanczos.GGNBlockOperator.matvec_flat` → `code/hessian/ggn.py::ggn_block_vector_product`, which applies `softmax_jacobian_apply` (H_τ) before the VJP. **The measured block is J_uᵀH_τJ_u, exactly as you state**, so the brief's inequality with that λ_u is indeed false. Your CE counterexample is exact, and worse than stated: for general p the ratio |∇|²/(G‖r‖²) = 1/(2p(1−p)) ≥ 2 and is unbounded as p → 0 or 1 — "the GGN-form inequality fails arbitrarily badly, not just by a factor 2."

### 2.3 From audit A3 (P5)

> "No step in Theorem 5.1 or its consequences is false; every displayed equation (5.2)–(5.13) and all table entries were reproduced. The items below are corrections of wording, conventions, or constants, not refutations."

**(A3-E1) (5.11), the constant is weaker than trivial at small M.**

> "Replace 'q − q_F ≤ (1+2m²)t/2 ≤ (1+2m²)Ψ(m)/(2(M+1))' by the uniform bound **0 ≤ q(t) − q_F(t) ≤ (1+2m²)t/(2 + Mt/2) ≤ 2(1+2m²)/M for 0 ≤ t ≤ T_m** (proof: Ψ convex ⇒ Ψ(q) − Ψ(q_F) ≥ Ψ'(q_F)(q − q_F); Ψ(q_F) = Mt and q_F ≤ e^{q_F} − 1 ⇒ Ψ'(q_F) = 1 + e^{q_F} ≥ 2 + Mt/2). As written, the constant (1+2m²)Ψ(m)/2 ≈ 192 at δ = 0.05 makes (5.11) **weaker than the trivial q − q_F ≤ m for all M ≤ 64**."

Values at T_m — stated bound: 11.30, 2.95, 0.747, 0.186; improved bound: 1.80, 0.471, 0.120, 0.030; actual max gap: 0.382, 0.138, 0.0398, 0.0104 (M = 16, 64, 256, 1024). ~5× better constant, and it avoids the T_m ≤ Ψ(m)/(M+1) step entirely. The √2 conversion to the packet's normalized two-logit L² distance is correct.

**(A3-E2) (5.12), the envelope amplitude.**

> "'Equation (5.8) gives a valid packet-style envelope with C=1' should read 'R(t) = √2 r(t) ≤ (1/√2)/(1+(M+1)t/4); we take C = 1 because `ass:residual` requires C ≥ 1'. With the true amplitude C = 1/√2 the same theorem gives (2√2 B_m/(M+1))log(1/(2δ)) = **0.842, 0.220, 0.0557, 0.0140** for M = 16, 64, 256, 1024 (vs 1.371, 0.358, 0.0907, 0.0227) — a 1.63× improvement."

As written the text reads as if C = 1 were the amplitude. It is not; it is the smallest value your own C ≥ 1 convention admits.

**(A3-E3) (5.12), embedding dependence of B_m.**

> "B_m = √((1+m²)/2) holds for the (F/2, −F/2) embedding; for the (F, 0) embedding the loss, gradient, residual norm R = √2 r and GGN are identical but ‖J_θ‖_op = √(1+b²), so the Theorem-27 bound is √2 larger. Add the sentence: 'the symmetric embedding is used because it makes ‖J_θ‖_op‖R‖ = ‖∇_θL‖ exactly'."

A3 also offers a **sharper Theorem-27-style bound from the same integration**, because ∫₀^{T_m} r dt = a_M exactly:
> "‖W(T_m) − W(0)‖ ≤ √(1+m²)·a_M ≤ √(1+m²)m/(M+1) = 0.4426, 0.1309, 0.03488, 0.00888 (ratio to exact 2.02 → 1.75); adding it as a column would separate envelope slack from gain-cap slack."

**(A3-E4) (5.9), head parameterization.**

> "D_curv(0) = M is correct for φ = β (M shared scalar readouts, λ_φ(0) = M/4). If the toy is mapped onto the packet's dense head ŷ_n = Wh_n with W ∈ R^{2×M} (rows initialized ±β/2, both trainable), then G_WW = H_τ ⊗ hhᵀ, λ_φ(0) = M/2 and **D_curv(0) = 2M** (checked numerically). The block definition must be stated with the number."

**(A3-E5) Notation collision.**

> "The note's scalar r = 1/(1+e^q) is |∂ℓ/∂F|, not the packet's N^{−1/2}-stacked residual vector r (whose norm here is √2 r). Since both symbols appear in the same section ((5.10) vs (5.12)), use a distinct symbol (e.g. ρ) for the scalar in any appendix version."

**Hidden hypotheses / gaps A3 flags:**

- Theorem/Lemma numbering: A3 identified "Theorem 27" with `thm:twoscale` and "Lemma 18" with `lem:opcap`; the frozen tex uses labels only, so the numbers must be re-checked against the compiled PDF before we typeset.
- "Y uniform on {−1,+1}" is **not used anywhere in the training dynamics** (YF = q on every sample regardless of label proportions); it is used only for the balanced-accuracy statement after (5.13). The theorem could say so.
- The symmetry reduction β_j ≡ b/M requires **all β_j(0) equal** (stated as 0). Unequal initial β_j would break the exact reduction — worth one sentence.
- M enters only through the reduced ODE, so all formulas hold for **real M > 0**; the "strictly decreasing in M" claim uses this.
- (5.11) is scoped to [0, T_m], a horizon shrinking like Ψ(m)/(M+1). Beyond T_m, b grows without bound (b ≲ q ~ log(Mt)) and O(1/M) closeness must be re-derived — not needed for your claims, but a reader should not infer closeness at the *frozen* model's own fitting time Ψ(m)/M > T_m.
- L̃ = √(1+b²) is width-uniform on {|b| ≤ m} but **not time-uniform**; your scoping "on the region |b| ≤ m containing this fitting trajectory" is correct and should be kept verbatim.

**And the novelty point, which A3 says you understate rather than overstate:**

> "By permutation symmetry β_j ≡ b/M for all t, and the (a,v,b) flow is exactly gradient flow on the 3-parameter model L(a,v,b) = log(1+e^{−(a+bv)}) with per-block learning rates (1,1,M), equivalently unit-rate flow after the reparameterization b = √M b̃. Moreover a_M = ∫₀^{T_m} r dt, so the suppression is exactly the packet's Theorem-2 mechanism (unit spectral gain × integrated residual, with the residual driven to O(1/M) by the fast block) evaluated in closed form. Theorem 5.1 is therefore a fully solvable instance of the D_curv(0) = M → μ = (M+1)/4 → O(1/M) displacement chain, **not a new mechanism**."

A3 also notes that the invariant b² − Mv² = −M is "the classical layer-balance conservation law of a depth-2 linear chain (d/dt(b² − Mv²) = 2bMvr − 2Mvbr = 0)" and recommends adding **Saxe et al. 2014; Arora, Cohen & Hazan 2018; Du, Hu & Lee 2018** and **Pezeshki et al.'s linear starvation model** (asymmetry in feature strength rather than readout multiplicity/LR) to the Yun/Moroshko/Berthier positioning you already give.

---

## 3. Numerical verification

### 3.1 Theorem 5.1 toy (N1)

`code/experiments/toy_serial_ce.py` → `results/toy_serial_ce.csv`, commit **494fd5d**. Runtime 17 s, CPU.

> "**THEOREM 5.1 VERIFIED IN FULL. No substantive discrepancy found; zero bound violations across 327 checked quantities.**"

**Three independent numerical paths, all agreeing.** (1) Closed form: `brentq` on Q_M(a) = m (xtol 1e-17), T_m by quadrature T_m = ∫₀^{a_M}(1+e^{Q_M(a)})da — the exact reparameterization dt = da/r — epsrel 1e-13. (2) Full (M+2)-parameter gradient flow, `solve_ivp` DOP853, rtol 1e-10 / atol 1e-13, dense output, terminal event at q = m; **all M+2 states integrated, no reduction assumed**. (3) Plain GD (explicit Euler) on the full parameter vector, h = T_m/N, N ∈ {1e3, 1e4, 1e5, 1e6}. Gradients validated against torch autograd on the **literal** model (x₁ = (S,0), x₂ = (0,C), shared W, mean over the balanced 2-example set): max rel err **1.1e-15**; torch-autograd GD and numpy analytic-gradient GD agree to ≤ 3.9e-16.

**Invariants and T_m.** (5.3) and (5.4) checked at 4001 points on [0, T_m]: max rel err **2.4e-11** (M=16), falling to **6.8e-14** (M=1024) — the requirement was 1e-8, met with three orders of margin. T_m formula vs solver agrees to 3.6e-14…1.2e-13 relative:

| M | 1 | 16 | 64 | 256 | 1024 |
|---|---|---|---|---|---|
| T_m | 6.023294416 | 0.907839679 | 0.286796515 | 0.078762700 | 0.020251900 |

All inside the (5.5) bracket [Ψ(m)/(M+1+2m²), Ψ(m)/(M+1)] at every M. q(T_m) = m and r(T_m) = δ = 0.05 recovered exactly. **All M β_j stay identical (spread ~1e-18).** The q̇ identity (1+b²+Mv²)r = (M+1+2b²)r confirmed. GD convergence is first order exactly: |q(T_m) − m| falls by 10.00× per 10× step reduction at every M (M=1024: 1.200e-3, 1.200e-4, 1.199e-5, 1.199e-6); Richardson extrapolation of the two finest runs reproduces the closed-form a, v, b, q to 5e-14…4.8e-12.

**GGN at init — exact, and the normalization that makes it true.** λ_θ(0) = 0.25 and λ_φ(0) = M/4 with |deviation| = 0 in closed form and ≤ 3.6e-15 from the eigensolver; **D_curv(0) = M**. Cross-checked against torch autograd **block Hessians**, which coincide with the GGN blocks here because F is multilinear (∂²F/∂a² = ∂²F/∂a∂v = ∂²F/∂v² = 0 and ∂²F/∂β_i∂β_j = 0), so the diagonal blocks carry no residual term.

> "NORMALIZATION THAT MAKES THIS TRUE: the MEAN-over-examples (averaged, not summed) binary CE L = E_Y[log(1+exp(−YF))] with a single scalar logit F and unit-scaled margin q = YF. The symmetric two-logit softmax convention (F/2, −F/2) gives the identical constants, since its softmax Hessian p(1−p)[[1,−1],[−1,1]] contracted with the logit-Jacobian direction (1/2, −1/2) reproduces the same scalar curvature r(1−r) — only the logit DIFFERENCE enters. **Two things break the constants:** (i) SUMMING over the n examples/broadcast sites instead of averaging multiplies both blocks by n (broadcasting to the 2 sites and summing gives 1/2 and M/2); (ii) logit rescaling F/τ multiplies both by 1/τ². D_curv(0) = M is invariant under both."

Combined with A3-E4, that is the complete convention statement (5.9) needs: **averaged loss, φ = β, and D_curv(0) = M; the dense 2×M head gives 2M.**

**Bound table (5.12) — reproduced.** B_m = √((1+m²)/2) = 2.198858, log(1/(√2δ)) = 2.649159. max_t ‖J_θ‖_op = √((1+b²)/2) stays ≤ B_m at every M, so the envelope hypotheses hold.

| M | displacement | bound (5.12) | ratio | your table |
|---:|---:|---:|---:|---:|
| 16 | 0.219020619 | 1.370600907 | 6.257862 | 6.258 |
| 64 | 0.071050189 | 0.358464853 | 5.045234 | 5.045 |
| 256 | 0.019657721 | 0.090662317 | 4.612046 | 4.612 |
| 1024 | 0.005063907 | 0.022731917 | 4.489008 | 4.489 |

**Every printed value in your note rounds correctly from the computed value at its printed precision, verified programmatically.** Bonus M=1: displacement 1.177117, bound 11.650108, ratio 9.897. (A1's independent root-finding in A3 gives the limiting ratio 4.446 as M → ∞; 4.457 at M = 4096.)

**1/√M readout control.** F = z₁ + M^{−1/2}Σγ_j z₂ with b = M^{−1/2}Σγ gives ḃ = vr, and for every M ∈ {1, 16, 64, 256, 1024} the integrated trajectory reproduces the M = 1 case to ≤ 1e-12: a(T_m) = 1.026946708772, v = 1.575312540435, b = 1.217213867836, T_m = 6.023294415742, ‖W(T_m) − W(0)‖ = 1.177116842901.

> "Confirms the effect is a **training-metric/readout-scaling artifact, not a function-class effect**."

**Other theorem claims confirmed.** (5.6): 0 < a_M ≤ m/(M+1) at every M; b(T_m)v(T_m) = m − a_M ≥ Mm/(M+1); suppression factor m/a_M ≥ M+1. (b) limit: a_M strictly decreasing (1.027, 0.1423, 0.04209, 0.01122, 0.002857); Ma_M → m with m − Ma_M = 1.917, 0.667, 0.250, 0.0733, 0.0192. (5.7): both parts hold at every M. **(5.8): max_t[r(t) − (1/2)/(1+(M+1)t/4)] = 0 on [0,T_m] AND on the extended horizon [0, 20T_m]** — the max is attained at t = 0 where the envelope is tight; no violation anywhere. (5.10): pointwise ratio max < 1; max_t b(t) ≤ m confirmed, validating the proof step. (5.11) frozen comparator: solver q_F matches Ψ^{−1}(Mt) to < 1e-13; 0 ≤ q − q_F everywhere; the pointwise bound (1+2m²)t/2 is never violated; max gap 1.2679, 0.3823, 0.1381, 0.0398, 0.01038 vs uniform bound 96.03, 11.30, 2.955, 0.747, 0.187; gap × (M+1) stays bounded (2.54, 6.50, 8.98, 10.23, 10.63), confirming the O(1/M) decay. (e): 2a_M − m < 0 at every M ≥ 1 (M=1: −0.891; M=1024: −2.939).

**All 14 reported discrepancies are non-substantive**: 4 are explicit-Euler O(h) discretization error at the finest step (M=1024 r(T_m): formula 0.05 vs GD 0.04999994303, rel 1.139e-6 — exactly the propagated Euler error, since |q − m| at N=1e6 is ~1.2e-6 and dr/dq / r = −(1−r) ≈ 0.95), and 10 are decimal rounding of your printed table (all `round(computed, k) == printed` returns True).

### 3.2 Interior-witness toy (N2)

`code/experiments/toy_interior_witness.py` → `results/toy_interior_witness.csv`, commit **657fb86**. torch float64, dense autograd GGN, N = 64 sites, C = 4, q ∈ {4,16}, M ∈ {16,64,256,1024}, 200 draws/cell (20k for the cheap probability statistics). Features h_n = √M(√ρ u₀ + √(1−ρ) g_n), g_n unit ⊥ u₀, so ‖h_n‖² = M exactly (R = 1) and λ_max(S_h) = 0.507–0.509 M (a₀ ≈ 0.51).

> "All four P3 interior-witness claims verify numerically. **No mathematical error was found in Lemma 3.1, Theorem 3.2, eq (3.6), or Counterexamples 3.3/3.4. Every inequality held on every draw.**"

**(3.6), squared loss.** E λ_max(G_WW) ≥ (s_v²/2q)λ_max(S_h) holds at every cell with slack. **Ratio E λ_max / RHS = 3.46, 3.66, 3.35, 3.67 (q=4) and 12.12, 13.08, 12.50, 12.16 (q=16)** at M = 16, 64, 256, 1024 — **flat in M**, as it must be if both sides are linear in M. Log-log slope of E λ_max vs M: **1.0062 (q=4), 0.9974 (q=16)**; over M ≥ 64: 1.0018 and 0.9751. The exact identity underlying the RHS was checked far more sharply than the eigenvalue: at 20k draws, **E[Q]/(λ_max(S_h)/2) = 0.9966, 1.0022, 1.0022, 0.9948 (±0.0046)**, and E[v₁²Q]/RHS = 0.98–1.03. So your derivation "curvature = v₁²Q, v independent of Q, Eχ = 1/2" is confirmed to **sub-percent** accuracy, and the ~3.5×/~12× gap is genuine slack of the single-witness bound, not an error.

**Theorem 3.2, CE floor.** The width-linear floor holds and the distribution of λ_max(G_WW)/M is M-stable — the substantive content. **E λ_max/M = 0.1025, 0.1054, 0.1033, 0.1068 (q=4) and 0.0756, 0.0759, 0.0739, 0.0742 (q=16)** across a 64× range in M. Log-log slopes of λ_max vs M — mean **1.0074 (q=4), 0.9941 (q=16)**; median 1.0125, 0.9890; 5th percentile 0.9512, 0.9999 (large-M subsets: 1.0045/0.9920 mean, 0.9523/0.9924 p05). **The positive-probability floor is M-independent:** P(λ_max ≥ cM) = 1.000 at c = 0.003 for all eight cells, ~0.94–0.99 at c = 0.03, ~0.44 (q=4) / ~0.17 (q=16) at c = 0.1, ~0.01/0.00 at c = 0.3. A numerically usable uniform constant is **c\* = 0.0284 (q=4) / 0.0351 (q=16) at the 95% level**. Lemma 3.1's chain λ_max(G_φφ) ≥ λ_max(G_WW) ≥ λ_max(S_{h,a}) ≥ witness held on **200/200 draws in every cell**.

One result we did not expect: **the bottleneck does not cap curvature even in ratio.** Mean λ_max(G_φφ)/λ_max(G_WW) falls from 1.48 (M=16, q=4) to **1.01** (M=1024, q=4) and 2.93 → **1.03** for q=16 — asymptotically the interior W block *is* the top spatial eigenvalue.

**Counterexample 3.3.** With identical h_n and zero bias, P(all inactive) = **0.4960–0.5003 (q=1), 0.2464–0.2522 (q=2), 0.0609–0.0635 (q=4)** across M = 16…1024 at 20k draws — exactly 2^{−q} and M-independent. **Adding the N(0, s_b²) bias leaves it unchanged** (0.4941–0.5032 for q=1), since the preactivation stays a centered Gaussian. On those draws the autograd W Jacobian is **exactly 0.0** (max |∂ŷ/∂W| = 0.000e+00). Contrast arm: with generic ρ = 0.5 features the same event has probability only 0.016–0.024 (q=1) and < 6e-4 (q=2).

**Counterexample 3.4.** With all h_n identical, ‖h‖² = M, train-mode `BatchNorm1d(q)` over the N = 64 sites: **E λ_max(G_WW) = 1.20e-25, 3.67e-25, 1.56e-24, 7.46e-24** at M = 16, 64, 256, 1024, with max |∂ŷ/∂W| ≤ 2.76e-12. Relative to the no-BN control that is **1.7e-26 to 3.1e-26 in the eigenvalue and ~3.3e-13 to 4.1e-13 in the Jacobian** — zero to float64 tolerance (the residue is roundoff amplified by 1/√ε ≈ 316, since batch variance is exactly 0). Meanwhile **λ_max(S_h) = M exactly (16, 64, 256, 1024; log-log slope 1.0000)** and λ_max(S_h^p) = p_min·M = 2.24, 8.83, 32.56, 127.44 — *the incoming Gram is width-linear while the block it feeds is identically zero*, which is precisely your claim. Switching the same BN to **eval** mode with frozen running stats restores a nonzero, width-linear block: E λ_max(G_WW) = 5.27, 17.38, 86.44, 361.32, slope **1.0304** (no-BN control 4.84, 15.40, 89.96, 241.72, slope 0.9738). BN β was set nonzero so the train-mode output is a live activation pattern — the vanishing is attributable to BN, not to a dead ReLU at exactly 0.

**Internal validation:** autograd Jacobian matches the closed form V[c,i]1{a_ni>0}h[n,m] to 0.0 exactly; the (NC × NC) Gram top eigenvalue matches the explicit dense (qM × qM) GGN to ≤ 3.9e-15 relative; the witness quadratic form matches its closed form to ≤ 1.1e-13; the hand-written train-mode BN matches `torch.nn.BatchNorm1d` to ≤ 4.8e-13.

**Four gaps N2 reports (none a refutation).** (i) Theorem 3.2's stated constant is unusable: with s_w = 1, s_b = 0.5, R = 1, q = 4, C = 4 the proof gives s_max² = 1.25, L₀ = √(96·4·1.25) = 21.9, **c₀ = e^{−4L₀}/(32C) ≈ 1e-40**, versus empirical λ_max/M ≈ 0.10. "(3.5) cannot be checked against its own stated constant, only its M-scaling and the existence of a positive floor (both confirmed)." (ii) **(3.6) degrades linearly in q**: λ_max(G_WW) is essentially q-insensitive (0.103 at q=4 vs 0.075 at q=16) while the RHS carries an explicit 1/q, so slack grows 3.5× → 12.4×. Structural — the witness ΔW = e₁uᵀ uses one of q channels. (iii) **Lemma 3.1's bound is vacuous on a nonzero fraction of draws when a is fixed in advance**: with a = e₁, λ_max(S_{h,a}) is *exactly* zero on **2.0–5.5%** of CE draws in all eight cells, because channel 1 is inactive at all 64 sites — Counterexample 3.3 reappearing inside Lemma 3.1. "The lemma still holds, but the choice of a must be adaptive to the draw for (3.1) to be informative; a fixed-a statement inherits the ReLU-inactivity obstruction. Worth an explicit sentence where (3.1)/(3.2) are stated." (iv) Report Counterexample 3.4 as a **relative** suppression, never a bare absolute zero. Scope not covered: **N was fixed at 64 throughout**, so the theorem's claimed N-independence is unverified here; and at M = 16 with N = 64 the non-common directions are not yet negligible, so full-grid slopes are reported alongside large-M-subset slopes.

### 3.3 Production measurement (N3) — the one that changes our story

`code/experiments/exp1_8c_interior_witness.py` → `results/exp1_8c_interior_witness.csv` + `.summary.json`, commit **179a0fa**. GPU, BlockViT-v2 at random init, breast fold-0, exp1_8's densest core (D13, 29,664 valid px, spatial 336, batch = 1), M ∈ {48, 192, 384}, K = 64, 12 layers, seeds {0,1}. Peak 2.4–4.4 GB, ~2 min.

`seg_head[0] = Conv2d(M+64 → 96, 3×3)` is the only Θ(M)-input layer downstream, and `BatchNorm2d(96)` sits immediately on top of it — **the exact production instance of Lemma 3.1's W with Counterexample 3.4's BN in the one place that can kill it.** We measured three BN regimes, not two: `train` (batch stats, exp1_8's protocol); **`eval_head`** (only the two seg_head BNs frozen to identity, model otherwise in train, so incoming h_n are **bit-identical** to train mode); `eval` (full `model.eval()`). Running stats pinned at init (mean 0, var 1) by momentum = 0, so eval-mode BN is an exact affine identity (bn_gain_rms = 1.000 in both eval arms). Validation: (3.1) λ_WW ≤ λ_φ holds in all 18 rows (max ratio 0.876), and λ_φ reproduces exp1_8's batch 0 exactly (M=48/seed0: 1.0866e+02 in both).

| mode | M | λ_WW | λ_Sh | λ_Sh9 | λ_φ | λ_WW/λ_Sh | bn_gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 48 | 60.02 | 50.16 | 380.0 | 118.4 | 1.182 | 2.25 |
| train | 192 | 130.6 | 50.16 | 380.0 | 181.0 | 2.601 | 3.47 |
| train | 384 | 275.6 | 50.16 | 380.0 | 327.4 | 5.309 | 4.79 |
| eval_head | 48 | 2.700 | 50.16 | 380.0 | 8.389 | 0.0549 | 1.00 |
| eval_head | 192 | 2.731 | 50.16 | 380.0 | 4.910 | 0.0556 | 1.00 |
| eval_head | 384 | 2.891 | 50.16 | 380.0 | 4.420 | 0.0573 | 1.00 |
| eval | 48 | 0.003487 | 0.04073 | 0.3648 | 0.4354 | 0.0831 | 1.00 |
| eval | 192 | 0.002795 | 0.04114 | 0.3683 | 0.4128 | 0.0675 | 1.00 |
| eval | 384 | 0.002227 | 0.04238 | 0.3790 | 0.4162 | 0.0532 | 1.00 |

(λ_Sh = per-pixel (M+K)-channel Gram as you asked; λ_Sh9 = the conv-correct 3×3-unfolded Gram, dim 9(M+K); λ_Wb ≡ λ_WW to 7 digits in train mode.) Log-log slopes vs M — λ_WW: train **+0.708**, eval_head **+0.029**, eval −0.208. λ_φ: train +0.463, eval_head **−0.319**, eval −0.024. λ_Sh and λ_Sh9: **+0.000** in train/eval_head. bn_gain_rms: train +0.356.

**The plain conclusion about Counterexample 3.4's mechanism in the real decoder, in four parts.**

1. **The question as we posed it is answered NO, and inverted.** Train-mode BN does not suppress the interior block's width growth — **it is the sole source of it.** With head BN active λ_WW grows as M^0.71; with head BN frozen to identity, on *identical* incoming features, it is flat (M^0.03).
2. **But that growth is not Lemma 3.1's mechanism; it is an initialization-scale artifact.** Incoming energy is flat in width (mean ‖h_n‖² = 86.2 / 86.4 / 86.7 at M = 48/192/384) because the M transformer channels arrive with trace/M = 0.00130 per channel ≈ 1/768 — exactly the `ConvTranspose2d` Kaiming fan-in prediction. Adding 336 channels adds ~0.44 of energy while diluting per-channel variance 4×; pre-BN energy falls 25.6 → 5.8 and BN's 1/σ gain rises 2.27 → 4.79 (slope +0.356). Since λ_WW scales as gain², **2 × 0.356 = 0.712 vs the measured +0.708**. The width exponent is entirely BN restoring a scale the init threw away.
3. **Theorem 3.2's hypothesis λ_max(S_h) ≥ a₀M fails outright on this model.** λ_max(S_h) is exactly flat (50.16 at every M) because its top eigendirection lies **wholly inside the width-independent K = 64 skip block**: λ_Sh_sk ≡ λ_Sh to 4 digits, while the M-channel sub-block has λ_Sh_up ≈ 0.003–0.006 (four orders down) growing only as M^0.30. There is no width-linear incoming Gram here to lower-bound anything with. Any P3-style production claim built on a width-linear S_h is dead on this architecture, **independently of BN**.
4. **Counterexample 3.4's erasure IS real in production, and is exact at the purest witness.** Adding `seg_head[0].bias` — a literally site-constant perturbation — changes λ_max by **< 3e-7 relative** in train mode at every width, i.e. contributes exactly zero curvature, because train-mode BN subtracts precisely that site mean. It contributes +0.25% in eval_head (BN identity) and 3.2–4.2× in full eval. For the W witness the erasure is large but a **width-independent constant**: the top eigenvector of S_h is 0.93–0.96 aligned with the mean feature direction (the direction BN annihilates), and mean-centring removes **41–54% of λ_Sh and 81–87% of the conv-correct λ_Sh9 (380 → 68)** — by the same factor at every M. So BN's two halves pull opposite ways: mean-subtraction costs a constant ~5–7× of witness energy (your Counterexample 3.4), variance-rescaling supplies an M^0.36 gain that more than pays for it. **Only the second touches the width exponent.**

**Consequence for §8, which we are writing against ourselves.** The sub-linear λ_φ width growth exp1_8 reports (+0.381 across its full sweep; +0.463 here) should be attributed to the `seg_head` BatchNorm gain at `seg_head[0]`, **not** to a feature-Gram floor. Freezing the two head BNs turns that slope **negative (−0.319)** while λ_WW/λ_φ rises 0.41 → 0.88 — i.e. at production width the interior witness block *is* most of the spatial block's top curvature, and its BN-driven growth is that block's growth. This is a second caveat on top of the existing "D_curv inverted / Σ_X rank-1" finding, and it argues against using a top-curvature witness as the route to a width-uniform bound — **exactly as you recommend at the end of P3**.

**Three methodological caveats we are obliged to state.** (i) *Full `model.eval()` is confounded and should not be read as the BN comparison.* `LinearSpectralReduction` contains its own BatchNorms (`wn_norm` over 314 wavenumbers, plus `BatchNorm2d(64)`); putting them in eval de-standardizes the raw spectra and changes h_n itself, collapsing λ_Sh from 50.16 to 0.041 (~1200×) and λ_φ from 118–327 to ~0.42. The `eval_head` arm is the clean isolation and is what the interpretation rests on; full-eval's λ_WW slope (−0.208) measures feature collapse, not normalization. (ii) *Lemma 3.1 is stated for a pointwise map, but `seg_head[0]` is a 3×3 convolution*, so its unit-Frobenius witness lives in R^{96×9(M+K)} and the correct incoming vector is the 3×3 patch. We computed both; they differ by ~7.6× (50.2 vs 380) and, more importantly, **BN's mean-erasure is much stronger at patch level** (81–87% vs 41–54%). Reporting only the per-pixel Gram would understate Counterexample 3.4's bite by roughly 3×. (iii) *Only 2 seeds.* Per-seed train λ_WW slopes are +0.508 and +0.785 (the aggregate +0.708 fits seed-means); M=384 seed 1 is a 3.6× outlier over seed 0 (430.8 vs 120.3) — the same outlier already present in exp1_8's own λ_φ at M=384 seed 1, so it is a property of that initialization, not of this measurement. **Do not quote +0.708 as a sharp exponent.** One bug found and fixed *before* the reported run: `seg_head[2]` is `nn.ReLU(inplace=True)`, so a forward hook storing `seg_head[1]`'s output without cloning silently reports post-ReLU energy (~54% of true, spurious bn_gain 0.737 instead of 1.000 in eval). Hooks now clone; eval-mode bn_gain_rms = 1.000 is the check that the fix is correct.

---

## 4. What we propose for the paper's theory section

We adopt your recommendation verbatim: **two linked scoped statements, the interior-layer result as a separate geometry theorem, P4 in the appendix, the counterexamples as scope tests.** Candidate main-text framing sentences below; all are ours to defend, so please shoot at them.

**(1) The bridge — Theorem 1.3 + Corollary 1.5.**
Placement: new §5 subsection, replacing the current "joint training is therefore functionally equivalent" language that your B3 flagged.
> *Candidate framing:* "In an exactly affine model, if the spatial kernel is coercive on **every residual direction actually visited** — not merely large in its top eigenvalue — then the joint and frozen-encoder trajectories stay within O(a/κ) of each other in training-logit distance, the encoder moves by O(√a/κ), and the encoder's share of total loss decrease over the infinite horizon is at most min{1, a/κ} (Theorem 1.3). For a shallow biased-ReLU instance this coercivity can be *proved* rather than assumed, giving κ = κ_*M/2 with M-independent a and ‖e₀‖ (Corollary 1.5); we state its width threshold M₀ ≥ 72N²/ρ explicitly, because at realistic (N, M) that threshold is 10⁸–10¹³ and the resulting O(1/M) bridge is an asymptotic existence statement, not a numerical guarantee at any width we can build."

Attached scope tests, in the same subsection: **Counterexample 1.2** (PL does not give trajectory contraction — kills the route the brief proposed) and **Counterexample 1.4** (the class-prior/top-eigenvalue floor cannot supply κ). The robustness version (additive ξ_J, ξ_F ∈ V, contributing (ν_J+ν_F)/κ) goes in the appendix as the statement of what an NTK-approximation argument would have to control. The margin-to-disagreement bound (b_γ + 2h²/γ², ℓ₂ form) goes in the appendix with the explicit caveat that macro-F1 needs per-class confusion control.

**(2) The constructive instance — Theorem 5.1.**
Placement: small self-contained section, with the novelty paragraph **in the main text, not a footnote** — A3 is emphatic that the mechanism statement must be up front.
> *Candidate framing:* "We exhibit an exactly solvable serial cross-entropy model in which the bound bites: a shared linear encoder feeds a spectral skip and M replicated contextual channels, both cues perfectly and equally predictive. The reduced flow is integrable in closed form; at a matched fitting margin the learned spectral response is suppressed by at least a factor M+1 relative to the spectral-only comparator, the joint and frozen-encoder training logits stay O(1/M) apart, and a full contextual reversal drives test error to one while the spectral-only comparator is exact. **The mechanism is explicit and is not new physics:** by permutation symmetry the system is gradient flow on a three-parameter logistic model with per-block learning rates (1, 1, M), the (b,v) pair obeys the depth-2 layer-balance conservation law, and a_M = ∫₀^{T_m} r dt makes the suppression a closed-form instance of the D_curv(0) = M → μ = (M+1)/4 → O(1/M) displacement chain. Replacing the readout by M^{−1/2}Σγ_j restores the M = 1 dynamics exactly — the effect is caused by the training metric induced by replication, not by a larger function class."

Positioning cited at that point: Yun–Krishnan–Mobahi (2021), Moroshko et al. (2020), Berthier (2023), **plus** Saxe et al. (2014), Arora–Cohen–Hazan (2018), Du–Hu–Lee (2018) for the conservation law and Pezeshki et al. (2021) for the linear starvation contrast. We will not claim novelty until that comparison is written.

**(3) P4 in the appendix.**
> *Candidate framing:* "Every experiment is discrete, so the flow bound is restated pathwise for the optimizers actually used: clipped GD (Lemma 4.1), the polynomial-envelope corollary with the exact dictionary t_k = ηk, μ_s = ημ_f (Corollary 4.2, including the variable-step form (4.4)), heavy-ball momentum with the exact weights and the 1/(1−β) inflation (Lemma 4.3), and Adam with a positive stabilizer (Lemma 4.4). The leading η in (4.2)–(4.3) is essential; the expression used in our earlier audit omitted it. For AdamW the bound exists but its constant η_max/(ζ(1−β₁)) is numerically vacuous at standard ε, and decoupled decay moves θ even at zero residual — so the appendix states the scope boundary rather than a usable bound for those arms."

Both hypotheses A1 wants promoted go into the lemma statements: **discrete-step containment** and the **mini-batch residual floor** (δ must exceed it for K_δ and (4.3) to mean anything).

**(4) The counterexamples as scope tests, collected.**
> *Candidate framing:* "Four counterexamples delimit what the theory can claim. PL along a trajectory does not make perturbations decay (1.2). A large top spatial eigenvalue is not a rate certificate when the residual lives elsewhere (1.4). A fixed-width ReLU bottleneck has an all-inactive event of probability 2^{−q}, independent of M, so no interior floor can hold with probability 1 − o_M(1) (3.3). And train-mode BatchNorm annihilates the site-constant witness exactly, so an interior width-linear floor through a normalized layer requires the **centered** feature Gram to be Θ(M) — a hypothesis we measure and find **false** on our own decoder, where λ_max(S_h) is flat in M and its top eigendirection lies in the width-independent skip block (3.4). Curvature anisotropy is likewise not cumulative gradient energy: the two-sided ratio we proposed is false, and the corrected excitation condition (Theorem 2.1) is a measurable hypothesis about the residual, not a consequence of geometry (2.2, 2.3, 2.5)."

The interior-layer results (Lemma 3.1, Theorem 3.2, (3.6)) go in as a **separate geometry theorem**, with your sentence retained verbatim: it does not supply the residual-subspace coercivity the dynamics theorem needs, and P2's counterexamples make that separation mathematically unavoidable. Theorem 3.2's c₀ will never be quoted as a number.

---

## 5. Your six concrete requests

| # | Request | Status |
|---|---|---|
| 1 | Full (M+2)-parameter CE model; verify (5.2)–(5.12); decreasing GD steps **and** adaptive ODE solver as separate checks; β replication vs 1/√M control; one CSV with formula values, applied-step units, relative discrepancies | **DONE.** `code/experiments/toy_serial_ce.py`, `results/toy_serial_ce.csv` (327 rows of checked quantities), commit 494fd5d. Three independent paths (closed form / DOP853 rtol 1e-10 / explicit Euler at 4 step sizes with torch-autograd gradient validation at 1.1e-15). Zero bound violations. §3.1. |
| 2 | Exact two-block GGN analytically and by dense operator; λ_φ = M/4, λ_θ = 1/4 at init under the specified normalization; confirm R = √2 r rather than fitting a constant | **DONE.** Same script. λ_θ(0) = 0.25, λ_φ(0) = M/4, deviation 0 in closed form and ≤ 3.6e-15 from the eigensolver; D_curv(0) = M; cross-checked against torch autograd block Hessians (which coincide here by multilinearity). R = √2 r confirmed, not fitted. The exact normalization statement, and the two things that break it (summing instead of averaging; logit rescaling), are in §3.1. |
| 3 | Interior Gaussian/ReLU instance: (3.6) squared-loss expectation **and** the CE floor separately; include the constant-feature BN counterexample — W block vanishes to numerical tolerance while the incoming Gram has an eigenvalue ∝ M | **DONE.** `code/experiments/toy_interior_witness.py`, `results/toy_interior_witness.csv`, commit 657fb86. (3.6) ratio flat in M (3.35–3.67 at q=4; 12.12–13.08 at q=16), slope 1.0062/0.9974; E[Q]/(λ_max(S_h)/2) = 0.9948–1.0022 ± 0.0046. CE floor slope 1.0074/0.9941 with M-stable λ_max/M ≈ 0.10/0.075. BN: E λ_max(G_WW) = 1.20e-25 → 7.46e-24 (relative 3e-26 to the no-BN control) while λ_max(S_h) = M exactly, slope 1.0000. Counterexample 3.3 measured at 2^{−q}, M-independent. §3.2. |
| 4 | For the shallow verified instance: export the actual residual overlap with the fast spatial kernel subspace and the restricted spectral output singular directions | **PENDING.** Not run. It is the single most decision-relevant remaining item, because Counterexample 1.4 makes it the difference between Theorem 1.3 applying and not applying, and we already have `prop:ntk_classprior` measured only as a Rayleigh quotient in one direction. Target: extend `code/synthetic/` + the `exp1_1_v3` GGN path (`results/exp1_1_v3_ggn.csv`) with a residual-projection export. We will not treat any instantaneous overlap as evidence of persistence. |
| 5 | Production P3: the **full downstream directional derivative** (3.3) through normalization and spatial mixing; report the chosen witness, its incoming energy, post-BN energy, final softmax-weighted energy | **DONE, and it reverses our reading.** `code/experiments/exp1_8c_interior_witness.py`, `results/exp1_8c_interior_witness.csv` + `.summary.json`, commit 179a0fa. Witness at `seg_head[0]` (the only Θ(M)-input layer), three BN regimes including the clean `eval_head` isolation, per-pixel and conv-correct 3×3-unfolded incoming Grams, bn_gain_rms, bias-witness probe. Findings in §3.3: **λ_max(S_h) flat in M (slope +0.000)**, train-mode BN is the sole source of λ_WW's M^0.71 growth and it is an init-scale artifact (2 × 0.356 = 0.712 ≈ 0.708), Counterexample 3.4's erasure exact at the site-constant witness (< 3e-7 relative). We are not treating the positive train-mode result as support for anything. |
| 6 | Independently audit the proofs and constants, especially the residual normalization in (5.12), the aggregate probability argument in Theorem 3.2, and the noncommuting-kernel comparison in Theorem 1.3. **Exact objections or corrected equations, not a general endorsement.** | **DONE.** Three independent audits, all three of your named targets covered. (5.12): correct; two fixes (C = 1 is the convention floor, not the amplitude 1/√2; B_m is embedding-dependent) + a sharper alternative √(1+m²)a_M. Theorem 3.2's aggregate argument: correct, Pr(Q ≥ A/4) ≥ 1/3 with **no cross-site independence** (MC 0.68–0.72), tail 1/96 → Pr(U > A/8) ≤ 1/12 → joint ≥ 1/4, p_V ≈ 0.44 for C=4, q=3, s_v=1; only V ⊥ (W,b) is used. Theorem 1.3's Duhamel identity: correct without any commutation (d/ds[e^{−Q(t−s)}e^{−Ps}e₀] = −e^{−Q(t−s)}K_θe^{−Ps}e₀), verified on **200 genuinely non-commuting PSD pairs** ([K_φ,K_θ] ≠ 0 checked), max ratios 0.775 / 0.87 / 0.41 for (1.5)/(1.6)/(1.7). Full list in §2. |

One honesty note on #1 and #2: the auditor's own script produced two FAILs, and both were the auditor's — a quadrature under-resolving the fast joint mode at a/κ ≈ 100, and an off-by-one in a Riemann sum. Neither was yours. We are reporting them so you can discount them rather than discovering them in a re-run.

---

## 6. Questions back

**Q1 — Can Corollary 1.5's kernel coercivity be extended to a nonlinear-trajectory tracking statement on a short horizon?** You are explicit that the corollary bounds the linearized dynamics only, and that the frozen-initialization kernel does not establish Theorem 1.1's monotonicity along the nonlinear trajectory. But the robustness version already tells us what to control: uniform ξ_J, ξ_F ∈ V contribute (ν_J+ν_F)/κ. On a horizon of length T ≲ T_δ, with κ = κ_*M/2, is there a *short-horizon* tracking statement — say, ν = O(sup_t ‖θ(t)−θ₀‖ · Lip(J)) fed back through (1.5) — that closes as a fixed point at large M, or does the ‖J_φ‖-growth inside the Lipschitz constant kill it exactly as it kills K_{θφ} in Theorem 1.1? We would rather know it is hopeless than write a section around a hope.

**Q2 — Is there a version of Theorem 5.1 with isotropic initialization of W?** Your own limitation is that W(0) = (0,1) passes context and suppresses the designated spectral cue, and that this "must appear in the theorem's main-text description". With a(0) = a₀ > 0 the invariants (5.3) become v = cosh(√M(a−a₀))·v₀ + …, and the closed form survives, but the *interpretation* changes: as you say, "if a useful spectral coefficient is already O(1) at initialization, small displacement need not remove it." Is there a statement of the form "for a(0) = a₀ and any M, the **relative** spectral response a(T_m)/a₀ → 1 while the margin share a(T_m)/m → 0", so that the theorem says *the encoder is not updated* rather than *the encoder is small*? That would match what Theorem 2 actually claims and would remove the initialization objection a reviewer will certainly raise.

**Q3 — How would you position Theorem 5.1 against Yun/Moroshko/Berthier, given that A3 reduces it to per-block learning rates on a 3-parameter model with the layer-balance conservation law?** Concretely: is the novel content (i) the *finite-time, finite-margin* CE formulas at a matched fitting threshold, as opposed to the asymptotic implicit-bias limits those papers characterize; (ii) the *serial* composition with a skip, versus their single-chain diagonal/linear networks; or (iii) nothing, and the honest framing is "a known mechanism, instantiated so that the packet's own bound (5.12) can be checked against an exact solution for the first time"? We are prepared to write (iii). We would like your read before we spend a section on (i) or (ii).

**Q4 — Given N3, what is the right P3 statement to attempt at all?** A2's recommendation is to measure (3.3) with a witness from the top eigenvector of the **centered** interior Gram rather than the mean feature direction. But N3 says the production top eigendirection of S_h is 0.93–0.96 aligned with the mean direction and lives in the width-independent K = 64 skip block, with the M-channel sub-block four orders down and growing only as M^0.30. Does that make the centered-Gram route worth running, or does it just measure a different flat quantity? Is there any interior-layer statement you would still attempt for an architecture whose Θ(M) channels arrive at 1/768 per-channel variance — or is the honest conclusion that **the width-linear-curvature route is closed for this decoder** and §8 should say so?

**Q5 — How should we report a width exponent that is an artifact of the normalization layer?** exp1_8's λ_φ slope (+0.381) is, on this evidence, the `seg_head` BN gain compensating a Kaiming fan-in dilution; freezing the two head BNs makes it −0.319. We can (a) report both slopes and attribute the growth to BN, (b) re-run the width sweep with per-channel-variance-matched initialization so that width growth, if any, is a feature effect, or (c) drop the width sweep from the main text entirely and keep it as a negative result in the SI. (b) is ~2 GPU-days and would change what the "architectural evidence" sentence in §8 can say. Which would you accept as sufficient?

---

**Source inventory.** Audits: A1 (P4, P1), A2 (P3, P2), A3 (P5) — independent verifier agents, each reading `math_01.md` in full plus the frozen `paper/sections/{03_setup,04_theorem1_hessian,05_theorem2_twoscale,supplement}.tex` and the originating brief `review_packet/astra/claude_math_brief_01.md`; scratchpad scripts `audit_A1.py`, `audit_a2.py`, `audit_a2_bn.py`, `check_p5.py` (session-local, will be persisted to `code/audits/` on request). Numerics: N1 `code/experiments/toy_serial_ce.py` → `results/toy_serial_ce.csv` (commit 494fd5d); N2 `code/experiments/toy_interior_witness.py` → `results/toy_interior_witness.csv` (commit 657fb86); N3 `code/experiments/exp1_8c_interior_witness.py` → `results/exp1_8c_interior_witness.csv` and `results/exp1_8c_interior_witness.summary.json` (commit 179a0fa). No repo file outside `code/experiments/` and `results/` was modified by any audit.
