# Audit of Theorem N (math_02.md §3) — 2026-09-10

Scope: `review_packet/astra/math_02.md` §2.1, §2.2, §3 (Theorem N, constants (N5)–(N12), boundary-mass estimate, three-term Jacobian estimate, variation of constants, containment, (N4) via (A1), nonsmooth-ReLU convention); `math_01.md` Theorem 1.3, Counterexample 1.4, Corollary 1.5 (reused κ_* argument and Frobenius/Chebyshev threshold); `claude_math_reply_01.md` Q1 (l. 305). Every displayed inequality was re-derived by hand; the algebraic steps most prone to error were additionally cross-checked numerically (scratchpad script `check_theoremN.py`: decomposition identity to 1e-15, Z inequality on 2×10⁴ random pairs, gate-flip containment and the three perturbation bounds on 1,300 random perturbations at M ∈ {64, 512, 2048} with boundary sets ranging from 2/64 to 380/512 units, Monte-Carlo (N5)/(N7) moments, E[ReLU⁴] = 1.4985 s⁴, (A1) along an actual nonlinear flow). No violation.

Notation used below: g_{ni} = w_iᵀΘx_n + b_i, g⁰_{ni} its initial value, γ_{ni} ∈ [0,1] the selected gate, Γ_n = diag(γ_{n·}), ξ_n = [Θx_n; 1].

---

## Verdict

Theorem N is correct as stated. Every displayed inequality in §3 — (N5), the 42 in (N6), (N7), the preactivation bound ≤ τ_i, ‖H−H_0‖_F ≤ D_H/√M, (N11), (N12), (N1)–(N3), (N4) via (A1), and (A1) itself in §2.1 — is VERIFIED by independent derivation; the constant chain has no circularity; exactly seven ρ/7 events are used; the nonsmooth convention is consistent (existence by smoothing is sound, the a.e. output chain rule holds for every gate selection because an absolutely continuous preactivation has zero derivative a.e. on its zero set, hence dL/dt = −‖∇L‖² a.e., and no step uses uniqueness). I found no FALSE step. Items marked GAP below are expository, not mathematical: the three-term decomposition is not written out (the displayed bound is consistent with exactly one ordering — (V−V_0)Γ_nW_0 + VΓ_n(W−W_0) + V_0(Γ_n−Γ_n⁰)W_0 — and the gate term does use only initial V_0, W_0 as required); the inequality Z ≥ ‖ξ_n‖ on the neighborhood is asserted without the one-line proof; "displacement one" needs the explicit M ≥ max{1,R_θ²} ⇒ R_θ/M ≤ 1; (N4) needs ‖J_Θ‖ ≤ A (from the three-term estimate), not only (N12); "centered β" is unnecessary; and the theorem statement should say "every solution in the convention" rather than "both trajectories". UNCLEAR only in the sense of undeclared provenance: the boundary-mass estimate, the half-radius bootstrap, and the coercivity threshold are the Du–Zhai–Poczos–Singh (2019) gate-flip argument with bias-density anti-concentration, and the nonsmooth existence/chain-rule convention is a special case of Davis–Drusvyatskiy–Kakade–Lee (2020) / Bolte–Pauwels (2021); these should be cited, not re-proved. The threshold M_* is polynomial in N, 1/ρ, 1/κ_* (at least ∝ N²/(ρ³κ_*²)) and numerically vacuous, as Astra says.

---

## Line-by-line ledger

### 1. (N5) moment bounds — VERIFIED

Preactivation variance: Var(g⁰_{ni}) = s_w²‖Θ_0x_n‖²/K + s_b² ≤ s_max². For g ~ N(0,σ²): E[ReLU(g)²] = σ²/2, E[ReLU(g)⁴] = 3σ⁴/2.

- E‖H_0‖_F² = Σ_{n,i} N⁻¹ E ReLU(g⁰_{ni})² ≤ N⁻¹·N·M·s_max²/2 = M s_max²/2. Divided by M: s_max²/2. ✓
- E‖W_0‖_F² = Σ_i E‖w_{i0}‖² = M·K·(s_w²/K) = M s_w². Divided by M: s_w². ✓
- E‖V_0‖_F² = C·M·s_v²/M = C s_v². ✓
- E‖J_{Θ,0}‖²: for ΔΘ ∈ R^{K×D}, (J_ΘΔΘ)_{(n,c)} = N^{-1/2} Σ_i V_{ci}γ_{ni} w_iᵀΔΘx_n = N^{-1/2}[VΓ_nWΔΘx_n]_c, i.e. the Jacobian is over all K·D entries, stacked over N samples and C outputs, with the N^{-1/2} normalization. ‖J_{Θ,0}‖_F² = Σ_{n,c,k,d}(∂z_{(n,c)}/∂Θ_{kd})² = N⁻¹Σ_{n,c}‖x_n‖²‖Σ_iV_{ci}γ⁰_{ni}w_i‖² (outer-product structure in (k,d)). Conditioning on (W_0,b_0), V_0 independent and centered kills cross terms: E‖Σ_iV_{ci}γ⁰_{ni}w_i‖² = (s_v²/M)Σ_i(γ⁰_{ni})²‖w_i‖² ≤ (s_v²/M)Σ_i‖w_i‖². Taking E over W_0: ≤ (s_v²/M)·M s_w². Hence E‖J_{Θ,0}‖_F² ≤ C s_v² s_w²·N⁻¹Σ_n‖x_n‖² ≤ C s_v² s_w² X². Operator vs Frobenius: the event is on ‖·‖_op and Markov is applied to E‖·‖_F² ≥ E‖·‖_op². ✓ (Tighter: C s_v² s_w² tr Σ_X, as in Cor. 1.5; X² is a valid relaxation.)
- E‖e_0‖²: ‖e_0‖² ≤ 2‖z_0‖² + 2‖y‖², ‖z_0‖² = N⁻¹Σ_n‖V_0h_n + β_0‖² with h_n = ReLU(W_0Θ_0x_n + b_0). With V_0 centered and independent of (W_0,b_0,β_0), E[(V_0h_n)ᵀβ_0] = 0 whether β_0 is fixed or independent (centering of β_0 is not needed). E‖V_0h_n‖² = C(s_v²/M)Σ_iE h_{ni}² ≤ C s_v² s_max²/2. So E‖e_0‖² ≤ 2·C s_v² s_max²/2 + 2E‖β_0‖² + 2‖y‖², exactly the display. ✓
- Markov: with h_0² = (7/ρ)(s_max²/2), P(‖H_0‖_F² > h_0²M) ≤ (M s_max²/2)/(h_0²M) = ρ/7; likewise C_w² = 7s_w²/ρ, C_v² = 7Cs_v²/ρ, A_0² = 7Cs_v²s_w²X²/ρ, R_0² = (7/ρ)(Cs_v²s_max² + 2E‖β_0‖² + 2‖y‖²). Five events, each ≤ ρ/7. ✓

### 2. (N6) and the 42 — VERIFIED

K_* = E[H_{:i}H_{:i}ᵀ] = N⁻¹E[ψψᵀ] (H carries N^{-1/2}), identical to Cor. 1.5's K_*; H_0H_0ᵀ = Σ_iX_i with X_i = H_{:i}H_{:i}ᵀ i.i.d., EX_i = K_*. E‖M⁻¹ΣX_i − K_*‖_F² = M⁻¹E‖X_1 − K_*‖_F² ≤ M⁻¹E‖X_1‖_F² = M⁻¹E‖H_{:1}‖⁴ = M⁻¹N⁻²E(Σ_nReLU(g_n)²)² ≤ M⁻¹N⁻²·N·Σ_nE ReLU(g_n)⁴ ≤ 3s_max⁴/(2M). Markov at level κ_*/2 with ‖·‖_op ≤ ‖·‖_F: P(‖M⁻¹H_0H_0ᵀ − K_*‖_op > κ_*/2) ≤ (3s_max⁴/2M)(4/κ_*²) = 6s_max⁴/(Mκ_*²). Setting this ≤ ρ/7 gives M ≥ 42 s_max⁴/(ρκ_*²) (6·7 = 42). ✓ On the complement, λ_min(H_0H_0ᵀ) ≥ M(κ_* − κ_*/2) = κM with κ = κ_*/2. ✓ κ_* > 0: Cor. 1.5's hyperplane-jump argument re-derived — cᵀK_*c = 0 ⇒ F(w,b) = Σc_nReLU(wᵀz_n + b) ≡ 0 by continuity and full support; distinct z_n ⇒ distinct hyperplanes (normals (z_n,1) cannot be proportional); a point of H_n off the other finitely many hyperplanes sees a gradient jump c_n(z_n,1) ⇒ c_n = 0. ✓ The (A2) alternative changes κ to κ_*/4 and propagates consistently.

### 3. (N7) and the three-term encoder-Jacobian estimate — VERIFIED (GAP: decomposition not displayed)

Union bound: conditional on w_{i0}, a_n := w_{i0}ᵀΘ_0x_n and τ_i are fixed, b_{i0} ~ N(0,s_b²) has density ≤ 1/(s_b√(2π)); P(|a_n + b_{i0}| ≤ τ_i | w_{i0}) ≤ 2τ_i/(s_b√(2π)); over N sites ≤ 2Nτ_i/(s_b√(2π)). ✓ (No cross-site independence used.)

E B_M: V_{:i,0} ⊥ (w_{i0},b_{i0}) ⇒ E[‖V_{:i,0}‖‖w_{i0}‖1{·}] = E‖V_{:i,0}‖·E[‖w_{i0}‖P(·|w_{i0})] ≤ s_v√(C/M)·(2N/(s_b√(2π)))·E[‖w_{i0}‖τ_i], with ‖w‖τ_i = (X‖w‖² + ZR_φ‖w‖)/√M. Summing M identical terms: the M·M^{-1/2}·M^{-1/2} cancels, giving E B_M ≤ (2Ns_v√C/(s_b√(2π)))[X E‖w_{10}‖² + ZR_φ E‖w_{10}‖] = C_bd, M-independent. ✓ E‖w‖² = s_w², E‖w‖ ≤ s_w (Jensen; valid since both enter an upper bound with positive coefficients). Markov: P(B_M > 7C_bd/ρ) ≤ ρ/7. ✓ E‖V_{:i,0}‖ ≤ √(Cs_v²/M). ✓

Three-term decomposition (the key question). At any point of the neighborhood with any admissible gates,
  VΓ_nW − V_0Γ_n⁰W_0 = (V−V_0)Γ_nW_0 + VΓ_n(W−W_0) + V_0(Γ_n−Γ_n⁰)W_0    (identity; checked to 1e-15).
This is the unique ordering consistent with the displayed bound: term 1 carries current gates and initial W_0 (‖·‖ ≤ ‖V−V_0‖_F‖W_0‖_F since ‖Γ_n‖ ≤ 1); term 2 carries current V and current gates (‖·‖ ≤ ‖V‖_F‖W−W_0‖_F); term 3 — the gate-change term — carries INITIAL V_0 and INITIAL W_0, so it is bounded by initial norms as B_M requires: ‖V_0(Γ_n−Γ_n⁰)W_0ΔΘx_n‖ ≤ Σ_i‖V_{:i,0}‖|γ_{ni}−γ⁰_{ni}|‖w_{i0}‖‖ΔΘ‖_F‖x_n‖ ≤ X Σ_{i: gate changed at n}‖V_{:i,0}‖‖w_{i0}‖. Containment in the boundary set: if γ_{ni} ≠ γ⁰_{ni} then either sign(g_{ni}) ≠ sign(g⁰_{ni}) or g_{ni} = 0 (selected gate), and in both cases |g_{ni} − g⁰_{ni}| ≥ |g⁰_{ni}|; with |g_{ni} − g⁰_{ni}| ≤ τ_i (item 4) this gives min_n|g⁰_{ni}| ≤ τ_i, i.e. i is counted in B_M. (If g⁰_{ni} = 0 exactly, i is in the boundary set trivially, so no a.s. argument is needed.) Hence term 3 ≤ X B_M. Averaging: ‖(J_Θ−J_{Θ,0})ΔΘ‖² = N⁻¹Σ_n‖·‖² ≤ X²[·]²‖ΔΘ‖_F². ✓ Then with ‖V−V_0‖_F ≤ R_φ/√M, ‖W_0‖_F ≤ C_w√M, ‖V‖_F ≤ C_v + R_φ/√M ≤ C_v+1, ‖W−W_0‖_F ≤ R_φ/√M ≤ 1 (M ≥ R_φ²), B_M ≤ B_*: ‖J_Θ − J_{Θ,0}‖_op ≤ X[R_φC_w + (C_v+1) + B_*] = A − A_0, so ‖J_Θ‖ ≤ A. ✓ No gap in the mathematics; the gap is that the decomposition and the "changed gate ⇒ |g−g⁰| ≥ |g⁰|" sentence are not written (fix F1).

### 4. Preactivation change ≤ τ_i and the Z bookkeeping — VERIFIED (GAP: Z inequality unstated)

g_{ni} − g⁰_{ni} = w_{i0}ᵀ(Θ−Θ_0)x_n + (ω_i−ω_{i0})ᵀξ_n. First term ≤ ‖w_{i0}‖‖Θ−Θ_0‖_opX ≤ X‖w_{i0}‖R_θ/M. Second ≤ ‖ω_i−ω_{i0}‖‖ξ_n‖, ‖ξ_n‖ = √(‖Θx_n‖²+1) ≤ √(((‖Θ_0‖_op + R_θ/M)X)² + 1). R_θ/M ≤ 1: if R_θ ≥ 1 then R_θ/M ≤ R_θ/R_θ² ≤ 1; if R_θ < 1 then R_θ/M < 1 for M ≥ 1 — so M ≥ max{1,R_θ²} suffices (not stated; fix F3). Z inequality: with a = ‖Θ_0‖_opX, (Z_0+X)² − ((a+X)²+1) = 1 + a² + 2XZ_0 + X² − a² − 2aX − X² − 1 = 2X(Z_0 − a) ≥ 0 since Z_0 = √(1+a²) ≥ a. So √(((‖Θ_0‖_op+1)X)²+1) ≤ Z_0 + X = Z. ✓ (fix F2). Total ≤ X‖w_{i0}‖R_θ/M + Z‖ω_i−ω_{i0}‖ ≤ X‖w_{i0}‖/√M + ZR_φ/√M = τ_i, using R_θ/M ≤ 1/√M ⇔ M ≥ R_θ² and ‖ω_i−ω_{i0}‖ ≤ ‖φ−φ_0‖ ≤ R_φ/√M. ✓ Frozen path: Θ-term is zero, bound still ≤ τ_i. ✓

### 5. ‖H−H_0‖_F, (N11) — VERIFIED

|H_{ni}−H⁰_{ni}| ≤ N^{-1/2}|g_{ni}−g⁰_{ni}| (1-Lipschitz). Minkowski on u_{ni} = w_{i0}ᵀ(Θ−Θ_0)x_n and v_{ni} = (ω_i−ω_{i0})ᵀξ_n: (N⁻¹Σ_{n,i}u²)^{1/2} = (N⁻¹Σ_n‖W_0(Θ−Θ_0)x_n‖²)^{1/2} ≤ X‖W_0‖_F‖Θ−Θ_0‖_F; (N⁻¹Σ_{n,i}v²)^{1/2} ≤ Z‖ω−ω_0‖. So ‖H−H_0‖_F ≤ XC_w√M·R_θ/M + ZR_φ/√M = D_H/√M. ✓ ‖H‖_op ≤ h_0√M + D_H/√M ≤ (h_0+1)√M = h√M using M ≥ D_H. ✓ ‖HHᵀ−H_0H_0ᵀ‖ ≤ ‖H−H_0‖(‖H‖+‖H_0‖) ≤ (D_H/√M)(h+h_0)√M = (2h_0+1)D_H = D_K. ✓ HHᵀ ⪰ (κM − D_K)I ⪰ (κM/2)I ⇔ M ≥ 2D_K/κ. ✓

### 6. (N12) — VERIFIED

z (as N×C) = HVᵀ + N^{-1/2}1βᵀ. J_V: ΔV ↦ HΔVᵀ, ‖HΔVᵀ‖_F ≤ ‖H‖_op‖ΔV‖_F, so ‖J_V‖ ≤ h√M; J_VJ_Vᵀ = (HHᵀ)⊗I_C (entry (n,c),(n',c') = δ_{cc'}(HHᵀ)_{nn'}) ⪰ (κM/2)I, gate-free. ✓ J_ω: (J_ωΔω)_{(n,c)} = N^{-1/2}Σ_iV_{ci}γ_{ni}Δω_iᵀξ_n; Cauchy–Schwarz over i: Σ_c(·)² ≤ ‖V‖_F²Z²‖Δω‖², averaged over n ⇒ ‖J_ω‖ ≤ Z‖V‖_F ≤ Z(C_v+1) = Ω. ✓ J_β: (J_βΔβ)_{(n,c)} = N^{-1/2}Δβ_c ⇒ norm exactly 1; K_β = (11ᵀ/N)⊗I_C. ✓ Concatenation: ‖[J_ω J_V J_β]‖² ≤ Ω² + h²M + 1 ≤ (h²+Ω²+1)M = B²M for M ≥ 1. ✓ K_φ ⪰ J_VJ_Vᵀ ⪰ (κM/2)I from the V block alone, for every gate selection. ✓

### 7. (N1)–(N2), continuity, frozen system — VERIFIED

Energy identity (a.e., see item 10): dL/dt = eᵀė = eᵀJṗ = −‖Jᵀe‖² = −eᵀKe with K = JJᵀ ⪰ K_φ ⪰ (κM/2)I; d‖e‖²/dt ≤ −κM‖e‖² ⇒ ‖e(t)‖ ≤ R_0e^{−κMt/2}. Same for the frozen path (K_F = K_φ). ✓ Displacements: ‖Θ̇‖_F = ‖J_Θᵀe‖ ≤ A‖e‖ ⇒ ∫_0^∞ ≤ 2AR_0/(κM) = R_θ/(2M); ‖φ̇‖ ≤ B√M‖e‖ ⇒ ∫ ≤ 2BR_0/(κ√M) = R_φ/(2√M). Both are exactly half the (N10) radii (R_θ = 4AR_0/κ, R_φ = 4BR_0/κ). ✓ Continuity: all Jacobian/kernel bounds hold at every point of the closed neighborhood (they are statements about the neighborhood on the initialization event, not about the path); at a putative first exit time the path would be on the boundary while the integrated displacement is ≤ half the radius — contradiction. Velocity ≤ (A+B√M)R_0 and finite path length exclude finite-time escape. ✓ Frozen: Θ ≡ Θ_0 lies in the Θ-ball trivially; every bound is monotone in the constants, and Z could be replaced by Z_0 (smaller Ω, B, R_φ, D_H); using the same constants is valid. ✓ Fitting time: R_0e^{−κMt/2} ≤ δ ⇔ t ≥ 2log(R_0/δ)/(κM). ✓

### 8. (N3) — VERIFIED

K_0 = (H_0H_0ᵀ)⊗I_C + K_β ⪰ κMI on all of R^{NC} (event 6, K_β ⪰ 0). K_J − K_0 = J_ΘJ_Θᵀ + J_ωJ_ωᵀ + [(HHᵀ−H_0H_0ᵀ)⊗I_C]: norm ≤ A² + Ω² + D_K; K_F − K_0 omits the first: ≤ Ω² + D_K. ✓ Duhamel: d/ds[e^{K_0s}e_P(s)] = −e^{K_0s}(K_P(s)−K_0)e_P(s) a.e. (e_P absolutely continuous, K_P bounded measurable); integrating gives the display with no commutation. ‖e^{−K_0(t−s)}‖ ≤ e^{−κM(t−s)}, ‖e_P(s)‖ ≤ R_0, ∫_0^te^{−κM(t−s)}ds ≤ 1/(κM), so sup_t‖e_J−e_F‖ ≤ [(D_K+Ω²+A²) + (D_K+Ω²)]R_0/(κM) = C_gapR_0/(κM), C_gap = 2D_K + 2Ω² + A²; z_J − z_F = e_J − e_F. ✓ Remark (interpretation, not error): C_gap is dominated by the head's own nonlinearity terms 2D_K + 2Ω², present on both paths; only A² is encoder-specific. (N3) is therefore "both paths track the same linear reference to O(1/M)", not "the frozen path is close because the encoder is irrelevant". The encoder-specific statements are (N2) and (N4).

### 9. (A1) and (N4) — VERIFIED (GAP: attribution sentence)

(A1) in §2.1: ‖∇_θL‖² = eᵀK_θe ≤ a‖e‖², ‖∇_φL‖² = eᵀK_φe ≥ κ‖e‖² ⇒ pointwise κ·eᵀK_θe ≤ a·eᵀK_φe ⇒ eᵀK_φe ≥ (κ/a)eᵀK_θe. Energy identity: L(0)−L(T) = ∫_0^T(eᵀK_θe + eᵀK_φe) ≥ ((a+κ)/a)∫_0^TeᵀK_θe. Ratio ≤ a/(a+κ). ✓ Requires only both blocks at unit rate, the a.e. identity, and L(0) > L(T); no commutation, no affine model, no infinite horizon. Sharp: scalar kernels give equality. ✓ It strictly improves Theorem 1.3's (1.7) share a/κ. In Theorem N: a = A² from ‖J_Θ‖_op ≤ A (item 3, not (N12)); κ = κM/2 from (N12); both hold at every point of the neighborhood hence on the actual residual vectors along the joint path; loss decrease is positive for every T > 0 unless e_0 = 0. (N4) = A²/(A²+κM/2). ✓ Fix F4 for the attribution wording.

### 10. Nonsmooth convention — VERIFIED (UNCLEAR: provenance; cite)

Smoothing: σ_ε(u) = (u+√(u²+ε²))/2, σ_ε' = (1+u/√(u²+ε²))/2 ∈ (0,1), |σ_ε − ReLU| ≤ ε/2. On a compact neighborhood the smooth fields are bounded uniformly in ε ≤ 2 (|σ_ε(u)| ≤ |u|+1), so solutions exist on a common interval with a common Lipschitz constant; Arzelà–Ascoli gives p_ε → p uniformly; gates γ^ε_{ni} = σ_ε'(g^ε_{ni}) are [0,1]-valued, so a weak-* subsequence in L^∞ converges to γ_{ni} ∈ [0,1] a.e.; on the open set {g_{ni} ≠ 0} the convergence is pointwise to 1{g>0} (u/√(u²+ε²) → sign u uniformly on |u| ≥ δ), so the weak-* limit equals the ReLU derivative a.e. there. ✓ Passage to the limit: every parameter-gradient entry is linear in a single gate (Θ_{kd}: Σe_{(n,c)}V_{ci}γ_{ni}w_{ik}x_{nd}; W_{ik}: Σe V γ (Θx_n)_k; b_i: Σe V γ) with coefficients converging uniformly, or gate-free (V, β); ∫c_εγ^ε → ∫cγ by uniform convergence of c_ε and weak-* convergence of γ^ε. The same scalar γ^ε_{ni} appears in all three gated blocks, so the limit gates are consistent across blocks. ✓ Chain rule: e_{(n,c)} is a polynomial in AC parameters and ReLU∘g_{ni}; for AC g and Lipschitz ReLU, ReLU∘g is AC with (ReLU∘g)' = 1{g>0}ġ a.e. on {g ≠ 0}, and ġ = 0 a.e. on {g = 0} (an AC function has zero derivative a.e. on the preimage of any null set), so (ReLU∘g)' = γġ a.e. for EVERY selection γ ∈ [0,1] on the zero set. Hence ė = J(γ)ṗ a.e. with the same γ used in ṗ = −J(γ)ᵀe, and dL/dt = eᵀJṗ = −‖ṗ‖² a.e.; (N1) and (A1)'s denominator rest on exactly this. ✓ The convention's second requirement ("output chain rule a.e.") is therefore automatic given AC paths — consistent, not restrictive. Uniqueness: not used anywhere — (N1)–(N4) are proved for every solution, and (N3) for every pair; existence is the only thing needed and is supplied. ✓ Continuation: on the event, bounded velocity gives a limit at any finite maximal time inside the compact neighborhood; restart the local construction; the containment estimates apply to the concatenated solution. ✓ Provenance: this is the Clarke-subdifferential / conservative-field gradient flow of a definable loss; existence of AC trajectories and the a.e. chain rule for any selection are Davis–Drusvyatskiy–Kakade–Lee (2020, Found. Comput. Math., Thm 5.8, Lemma 5.2) and Bolte–Pauwels (2021, Math. Program., Thm 1, Lemma 2). Should be cited (fix F7).

### 11. Circularity — VERIFIED (none)

Order: (κ_*, κ) from data + variances → (h_0,C_w,C_v,A_0,R_0) from (N5) + ρ → (Z_0,Z,X) from data → h = h_0+1, Ω = Z(C_v+1), B = √(h²+Ω²+1), R_φ = 4BR_0/κ → C_bd (uses X, Z, R_φ, s_w, s_v, s_b, N, C), B_* = 7C_bd/ρ → A = A_0 + X{R_φC_w + (C_v+1) + B_*}, R_θ = 4AR_0/κ → D_H = XC_wR_θ + ZR_φ, D_K = (2h_0+1)D_H, C_gap → thresholds (N6), (N9). Each constant depends only on earlier ones; none on M. The non-circularity hinge is that τ_i allocates X‖w_{i0}‖/√M to the Θ-induced change without knowing R_θ, and M ≥ R_θ² is imposed afterwards — legitimate. τ_i and B_M are random and M-dependent but are not constants. ✓

### 12. Probability bookkeeping — VERIFIED (exactly seven)

1. ‖H_0‖_F ≤ h_0√M; 2. ‖W_0‖_F ≤ C_w√M; 3. ‖V_0‖_F ≤ C_v; 4. ‖J_{Θ,0}‖_op ≤ A_0; 5. ‖e_0‖ ≤ R_0; 6. λ_min(H_0H_0ᵀ) ≥ κM (needs M ≥ 42s_max⁴/(ρκ_*²)); 7. B_M ≤ B_*. Each fails with probability ≤ ρ/7; union bound ≥ 1−ρ. All seven are events on the initialization only; nothing on the path is probabilistic. No hidden a.s. event is needed (g⁰_{ni} = 0 is handled by the boundary set, item 3). ✓

### 13. Other

- "Pairwise distinct Θ_0x_n": used only for κ_* > 0 (Cor. 1.5's kink argument). Not used in (N7). Duplicates ⇒ κ_* = 0 ⇒ theorem void; near-duplicates ⇒ κ_* tiny ⇒ M_* astronomically large (already noted in math_01).
- Output bias β: harmless. K_β is parameter-independent and is placed in K_0 so it cancels from K_P − K_0; contributes "+1" to B² and 2E‖β_0‖² to R_0²; (N5)'s cross-term argument needs only independence from V_0.
- "Fully trained": every bound is uniform on t ∈ [0,∞) and e(t) → 0 on the event, so the trajectories are trained to interpolation; the O(1/M) fitting time is 2log(R_0/δ)/(κM). Justified.
- (N1) exponent uses only K ⪰ K_φ; fine for both paths.
- Size of M_*: A ≥ XB_* ∝ N/ρ, R_0 ∝ ρ^{-1/2}, R_θ ∝ N/(ρ^{3/2}κ_*) ⇒ (N9) forces M_* ≳ N²/(ρ³κ_*²), worse than Cor. 1.5's 72N²/ρ. Not an error; must be labeled vacuous numerically, as Astra does.
- Statement wording: "both nonlinear trajectories" should read "every joint solution and every frozen solution in the stated convention" (probability is over initialization; uniqueness is not asserted). The constant A appears in (N4) but is defined only in the proof.
- (N5) text "take beta independent and centered": centering unnecessary.

---

## Required fixes (exact replacement wording; none changes the mathematics)

**F1 — display the decomposition.** Replace "Decompose the encoder Jacobian change into changes in V, changes in W, and changes in gates. For each sample its matrix has the form `V diag(gates) W` followed by multiplication by `x_n`; averaging over N does not increase the bound. Cauchy–Schwarz and the definition of `B_M` give" with:

> For sample n let `Γ_n = diag(γ_{n1},…,γ_{nM})` be the selected gates and `Γ_n^0` the initial gates, so that `(J_Θ ΔΘ)_{(n,:)} = N^{-1/2} V Γ_n W ΔΘ x_n`. Then
> $$VΓ_nW − V_0Γ_n^0W_0 = (V−V_0)Γ_nW_0 + VΓ_n(W−W_0) + V_0(Γ_n−Γ_n^0)W_0 .$$
> Since `‖Γ_n‖_op ≤ 1`, the first two terms have operator norm at most `‖V−V_0‖_F‖W_0‖_F` and `‖V‖_F‖W−W_0‖_F`. For the third, a gate at `(n,i)` can differ from its initial value only if `sign g_{ni} ≠ sign g_{ni}^0` or `g_{ni}=0`; in either case `|g_{ni}−g_{ni}^0| ≥ |g_{ni}^0|`, so the preactivation bound forces `min_n |g_{ni}^0| ≤ τ_i` and unit i is counted in `B_M`. Hence `‖V_0(Γ_n−Γ_n^0)W_0 ΔΘ x_n‖ ≤ Σ_i ‖V_{:i,0}‖ |γ_{ni}−γ_{ni}^0| ‖w_{i0}‖ ‖ΔΘ‖_F ‖x_n‖ ≤ X B_M ‖ΔΘ‖_F`. Using `‖x_n‖ ≤ X` in the first two terms and averaging the squares over n gives

**F2 — state the Z inequality.** After "`Z=Z0+X`" insert:

> For `‖Θ−Θ_0‖_op ≤ 1`, `‖[Θx_n;1]‖ ≤ sqrt(1+((‖Θ_0‖_op+1)X)²) ≤ Z_0+X = Z`, because with `a=‖Θ_0‖_op X` one has `(Z_0+X)² − (1+(a+X)²) = 2X(Z_0−a) ≥ 0`.

**F3 — "displacement one".** Replace "It lies within displacement one in both blocks." with:

> Since `M ≥ max{1,R_θ²}` gives `R_θ/M ≤ 1` and `M ≥ R_φ²` gives `R_φ/√M ≤ 1`, it lies within displacement one in both blocks.

**F4 — (N4) attribution.** Replace "Finally, `(N4)` is (A1) applied to the uniform bounds in (N12)." with:

> Finally, `(N4)` is (A1) with `a = A²`, from `‖J_Θ‖_op ≤ A` on the neighborhood, and `κ = κM/2`, from (N12); both inequalities hold at every point of the neighborhood, hence on the actual residual vectors along the joint path, and the almost-everywhere energy identity supplies the loss-decrease denominator.

**F5 — β wording.** Replace "For the last bound, take beta independent and centered, or fixed: the random readout has zero mean, so its cross term with beta vanishes in expectation." with:

> For the last bound, take `β_0` fixed or independent of `V_0` (centering of `β_0` is not needed): `V_0` is centered and independent of `(W_0,b_0,β_0)`, so the cross term vanishes in expectation.

**F6 — theorem statement.** Replace "with probability at least `1-rho`, both nonlinear trajectories satisfy" with:

> with probability at least `1−ρ` over the initialization, every joint solution and every frozen solution in the convention above satisfy

and replace "There are finite constants `M_*, A, B, kappa, R0, Cgap`" with:

> there are finite constants `M_*, A, B, κ, R_0, C_gap` — `A` being the encoder-Jacobian cap on the neighborhood (N10), defined in (N8) —

**F7 — cite the nonsmooth machinery.** Append to the smoothing paragraph:

> This convention is the gradient flow of the Clarke subdifferential of a definable loss; existence of absolutely continuous trajectories and the almost-everywhere chain rule for every selection at kinks are established in Davis, Drusvyatskiy, Kakade and Lee (2020, Found. Comput. Math., Thm 5.8, Lemma 5.2) and Bolte and Pauwels (2021, Math. Program., Thm 1, Lemma 2); the level-set fact used is that an absolutely continuous function has zero derivative almost everywhere on the preimage of any null set.

**F8 — cite the gate-flip argument.** After "This estimate needs neither cross-site independence nor unchanged gates." insert:

> The boundary-mass estimate, the feature-perturbation bound, and the half-radius bootstrap below follow the gate-flip argument of Du, Zhai, Póczos and Singh (ICLR 2019, Lemmas 3.1–3.4), with anti-concentration supplied by the bias density instead of a unit-norm input, and weighted by `‖V_{:i,0}‖‖w_{i0}‖` because the readout is trainable here.

**F9 — enumerate the events.** Replace "The seven-event union bound gives probability at least `1-rho`." with:

> The seven events — the five bounds in (N5), the coercivity (N6), and `B_M ≤ B_*` — each fail with probability at most `ρ/7`; the union bound gives probability at least `1−ρ`.

**F10 — one honesty sentence on the threshold (for the paper, not the note).** After "Its finite-width threshold remains conservative." add:

> Explicitly, `A ≥ X B_* ∝ N/ρ` and `R_0 ∝ ρ^{-1/2}`, so (N9) forces `M_* ≳ N²/(ρ³κ_*²)`; the theorem is an existence statement and its threshold must not be quoted as a width at which the phenomenon begins.

---

## Positioning note

Agree with Astra: Theorem N is a fixed-dataset lazy-training statement for the standard parameterization (readout variance `s_v²/M`, unit learning rate, no explicit `1/√M` multiplier). In that parameterization the readout kernel is `Θ(M)` while the fixed-width encoder's kernel is `O(1)`; the block-wise scaling of NTK contributions under the standard parameterization, and the fact that this regime is lazy, are known (Chizat–Oyallon–Bach 2019, §3 remark on `1/M` initializations; Sohl-Dickstein, Novak, Schoenholz and Lee 2020, "On the infinite width limit of neural networks with a standard parameterization"; Yang and Hu 2021, abc-parameterizations, where SP with unit LR sits in the kernel regime). Note that with the explicit `1/√M` multiplier and `O(1)` readout entries both kernels are `O(1)` and the disparity disappears — the same parameterization sensitivity as Theorem 5.1's control, and the theorem should say so.

Steps that are known lemmas and should be cited rather than re-proved: (i) (N7) and the "changed gate ⇒ small initial preactivation" containment, the 1-Lipschitz feature-perturbation bound, the Frobenius/Chebyshev coercivity threshold, and the half-radius continuity bootstrap are Du–Zhai–Póczos–Singh 2019 (Lemmas 3.1–3.4, Thm 3.2) with the bias density replacing unit-norm anti-concentration; see also Arora–Du–Hu–Li–Wang 2019 (Lemma C.2) and Oymak–Soltanolkotabi 2020. (ii) κ_* > 0 for distinct inputs via the ReLU kink is the standard argument (Du et al. 2019 Thm 3.1 for non-parallel inputs without bias; with bias, distinctness suffices). (iii) The nonsmooth convention is Davis et al. 2020 / Bolte–Pauwels 2021. (iv) Tracking of the constant-kernel reference `e_ref` by the nonlinear residual is the ReLU version of Lee et al. 2019 Thm 2.1 / Chizat–Oyallon–Bach 2019 Thm 2.4 (smooth) and Arora et al. 2019 Thm 3.2 (ReLU).

What is not a verbatim known result: the explicit two-block split — encoder displacement `2AR_0/(κM)` with `A` controlled by a readout-weighted gate-boundary mass, head displacement `2BR_0/(κ√M)`, fitting time `O(1/M)` — and the joint-versus-frozen-nonlinear-head comparison (N3). The latter is, however, only a triangle inequality through the linear reference and is dominated by the head's own nonlinearity (item 8); it carries no content beyond "both paths are lazy". The encoder-specific content is (N2) and the sharp finite-horizon share (N4) via (A1), which is a genuine (if elementary) improvement over Theorem 1.3's `a/κ`. Recommended framing: "an explicit-constant, ReLU-valid instance of lazy training in the standard parameterization, stated for a serial fixed-width encoder, whose new content is the block-wise displacement split and the finite-horizon attribution bound"; do not describe nonlinear tracking, gate control, or the bootstrap as new. Placement in the appendix with the scoped existence consequence in the main text, as Astra proposes, is appropriate; it must not be used to explain CE or the production decoder.
