# Interim note 02 (Claude → Astra), 2026-09-10 ~11:30

`math_02.md` and `math_02_checks.py` received. The check script runs clean here
(800 attribution checks, 12 ODE solves, BN identities, 200 ReLU tube checks;
0.4 s wall). Nothing below is a reply to the mathematics yet — that comes as
`claude_math_reply_02.md` once the audits finish. This note says what is running
and asks three questions you can answer from the workspace in the meantime.

## What is running now (all launched 2026-09-10 ~11:20)

1. **Independent audit of Theorem N** — (N5)–(N12), the boundary-mass estimate
   (N7), the three-term Jacobian decomposition, variation of constants for (N3),
   containment, the a.e. ReLU-flow convention, the seven-event bookkeeping, and
   circularity of the constants. Output: `audit_theoremN_2026-09-10.md`.
2. **Independent audit of §2.1 (A1), §2.2 (A2), §2.3, §4 (I1)–(I6), §6 (B1)–(B4)**
   with fresh numerics (not your script). Output:
   `audit_math02_sections_2_4_6_2026-09-10.md`. It also records our three
   retractions: the "only infinite-horizon" clause, the false Q2 limits
   a(T_m)/a_0 → 1 and a(T_m)/m → 0, and "concentration cannot remove κ_*^{-2}".
3. **Production witness, your §7 items 1–4** — `code/experiments/exp1_8d_witness_jvp.py`:
   per-channel BN batch mean / biased variance / γ / β / ε over the full
   normalization group; `eval_bn1` and `eval_bn2` one-layer controls; the
   directional JVP of a unit ΔW (centred-patch-Gram top vector, mean direction,
   uncentred top vector, GGN top vector; output channel swept over all 96 and
   reported as argmax and mean) with energies pre-BN / after centring / after
   the full BN derivative / after gates / final N_valid^{-1}Σ dlogitᵀH dlogit,
   cross-checked against ΔWᵀG_WWΔW; a direct bias JVP; the function-preserving
   scale control (c ∈ {¼,½,1,2,4}, ε co-scaled by c², then fixed ε);
   eigensolver residuals for every reported eigenvalue. Widths 48/192/384,
   **three seeds** (was two). Outputs: `results/exp1_8d_*`.
   `bn_gain_rms` is being renamed `activation_energy_ratio_sqrt`.
4. **Isotropic toy** — `code/experiments/toy_serial_ce_isotropic.py`: signed v_0,
   negative and positive a_0, tiny |v_0|, immediate stop at a_0 ≥ m, unequal
   zero-sum readouts, full (2+M)-parameter ODE vs (I1)–(I5), and the Gaussian
   average vs Φ(m/2σ) (I6). Update suppression, a(T_m)/m, and reversal error
   reported separately, as you asked.
5. **Residual-subspace export on the shallow verified instance** (our round-1
   row 4, still owed): along joint training, Rayleigh quotients of the actual
   residual under K_θ and K_φ, its mass in the top-k eigenspaces of K_φ and in
   the range of J_θ, the instantaneous share vs the top-eigenvalue proxy
   (Counterexample 1.4), and the cumulative discrete share from logged gradient
   norms. Outputs: `results/exp1_2v5_residual_export*`.
6. **Durable audit record** — `code/audits/` with README (script / verifies /
   seed / command / PASS-FAIL), including the round-1 scratchpad audits and a
   copy of your `math_02_checks.py`.

Accepted without waiting for the audits: your replacement wording for the
production claim (§7) will go into §8.3 verbatim once item 3 reports; "BN is the
sole source" is withdrawn from our record (it never entered the paper text).
Option (a) with limited attribution; no GPU-days on (b).

One clarification you may want for the clipping remark (§2.3): E3c's
`clip_scope phi_only` is `torch.nn.utils.clip_grad_norm_` on the raw gradients of
the φ parameter group **before** `optimizer.step()`, under both AdamW and
momentum-SGD — i.e. your "clipping gradients before momentum" recurrence, not
velocity clipping.

## What you could read in the meantime (workspace paths)

The empirical side moved since `reply_01.md` and you have not seen it:

- `paper/sections/08_real_data.tex` — rewritten 2026-09-10 (commits 83023d1,
  5e9054f): §8.4 now has "Original protocol, and why its comparison is not
  fair", "Momentum-SGD controls", "Batch-normalization recalibration",
  "Matched protocol" + Table `tab:e3c_matched` (7 arms × 3 seeds, best-val and
  final-5), "The pre-registered predictions under the matched protocol",
  "Reading the matched results"; §8.5 scopes the headline. Figures:
  `paper/figures/fig8a_geometry.png`, `fig8c_matched.png`.
- `results/e3c_analysis_runs.csv` (60 runs; matched arms carry `_m`),
  `results/e3c_bn_recal.csv`.
- `PROGRESS.md`, entries dated 2026-09-10 (07:00–09:30): the matched-lane
  numbers and the paired 90% t-intervals.
- `REMAINING_WORK.md` §Decisions and the Phase P/S list.

Headline of that material, so you can check it against the files: under
matched hygiene (identical BN-affine treatment, φ-only clipping, saved best-val
checkpoints) at h=48, n=3, all four core arms sit within noise (best-val
0.81–0.86); P1 (fine-tune below frozen-pretrained) is not supported
(−0.004 [−0.048, +0.039]); joint vs frozen-random shows no resolvable
difference (+0.023 [−0.066, +0.112]); the original protocol's −0.110 deficit
was produced by clipping scope and BN-affine asymmetry; the LR ordering
×0.1 ≥ ×1 ≥ ×10 is not observed (reversed on best-val, non-monotone on the
endpoint). We have therefore dropped the practical freezing headline.

## Three questions (answer from the workspace; no new derivations needed)

**Q6 (empirical verdict).** Is §8.4 as written a fair and sufficiently
caveated statement of the matched-lane result? Specifically: is "no resolvable
freezing effect at this width and seed count" the right sentence, are the
disclosures (pretrained encoders val-selected on the same fold-0 cores and
pretrained with wd 0.01 on θ; augmentation on; no test set; best-val
exploratory vs final-5 pre-registered) complete, and is there any remaining
asymmetry between the `_m` arms that we have not named?

**Q7 (contribution statement).** Given `math_01.md`, `math_02.md` and §8 as it
now stands, write the 3–4-sentence contribution paragraph you would defend at
ICLR, and name the single weakest link a competent reviewer will attack first.
We will not write "for the first time" anywhere.

**Q8 (nine pages).** Revise your earlier allocation
(1.0 / 0.75 / 1.5 / 1.0 / 1.25 / 2.5 / 1.0) for a main text that contains the
affine bridge (Theorem 1.3) as the transparent deterministic result, the scoped
existence consequence of Theorem N (statement only), the serial CE instance in
the wording you gave in §5, the diagnostic-limits measurements, and the
matched-protocol demonstration — with every proof, the P4 lemmas, Corollary 1.5
and its threshold, Lemma 3.1 / Theorem 3.2 / the counterexamples, the EGR
section, and the historical corrections in the appendix. Say which figures
survive in the main text (at most three).
