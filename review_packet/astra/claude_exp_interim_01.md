# Experiments 2 and 3 — interim results and an interpretation question

Claude → Astra and Krzysztof, 2026-09-11 01:15. Grids still running: Exp 2
`sp`/`mup`/`frozen` arms (seeds 1–2 at the larger widths), Exp 3 `unready`
regime. Everything below is from `results/exp2_summary.csv`,
`results/exp2/PREDICTIONS.md` and `results/exp3_summary.csv` as of now;
numbers are at the pre-registered threshold L* = 0.30 unless stated.

## 1. What is settled

**Suppression by informative context is robust (Exp 2).** At L*, every
jointly trained model with an informative context leaves the encoder's
spectral accessibility near its initial value, while the same model trained
with an uninformative context aligns strongly:

| arm (Exp 2) | widths | probe at L* | reversal acc. at L* | context-random acc. |
|---|---|---|---|---|
| `sp` (seed 0) | 2, 4, 8, 2048 | 0.65–0.66 (init 0.647) | 0.12–0.13 | 0.51 |
| `ctxfree` (3 seeds) | 2, 8, 128, 512 | 0.89–0.94 | 0.875–0.898 | — |
| `frozen` (3 seeds) | 2, 8, 128, 512 | = init | 0.12–0.15 | — |

**Spectral learning at matched fit responds to the head's relative speed
(Exp 2, `headlr`, M = 32, seed 0).**

| κ | step at L* | probe gain | h_u | reversal | context-random | spec-only |
|---|---|---|---|---|---|---|
| 1 | 1,671 | +0.028 | 0.079 | 0.128 | 0.515 | 0.536 |
| 1/16 | 7,611 | +0.107 | 0.177 | 0.129 | 0.513 | 0.568 |
| 1/256 | 22,708 | +0.223 | 0.475 | 0.152 | 0.519 | 0.641 |
| 1/4096 | not reached (final loss 0.320 at 40k) | +0.266 | 0.693 | 0.184 | 0.529 | 0.697 |
| 1/32768 | not reached (final loss 0.347 at 40k) | +0.267 | 0.700 | 0.208 | 0.528 | 0.689 |

(h_u at init 0.055; the optimal probe is Φ(1.645·√h_u): 0.65 → 0.88 →
0.91 for h_u 0.055 → 0.475 → 0.70.) The readout-only multiplier `lrmult`
moves the same quantities in the same direction but weakly (probe 0.70 →
0.66 from ×1/16 to ×16; three seeds, monotone in every seed).

**Width (Exp 2).** At L* = 0.30 the `sp` arm is saturated at every width
(pilot + seed 0: probe 0.65–0.66 from M = 2 to 2048). At the secondary
threshold 0.15 (just above the context-oracle optimum 0.126) there is a
clear width trend in spectral learning: probe gain vs M has Spearman
−0.86 / −0.96 / −0.86 in the three seeds (e.g. seed 1: M = 2 → +0.262,
M = 2048 → +0.030), and the normalized readout `mup` is less suppressed at
large M (seed 0, M = 2048: +0.104 vs +0.037; seed 2, M = 512: +0.087 vs
+0.027). Reversal accuracy at 0.15 is flat at 0.06–0.08 for both.

**Real spectra, ready regime (Exp 3, complete, 30 runs).** With standardized
inputs a random encoder already reads the class (probe 0.91–0.94 at
initialization), so there is nothing for the head to starve, and there is
no probe gain at L* in any arm (−0.015 to +0.035). Yet every jointly
trained model predicts from context only: reversal 0.07–0.11, context-random
0.48–0.52 (chance), spectral-only 0.48–0.62, at every width 8–512 and for
κ = 1, 1/16, 1/256 alike (all fit within 70–230 steps). The
uninformative-context comparator reaches reversal 0.93 and context-random
0.92. Frozen encoder: reversal 0.07–0.11.

## 2. What the speed intervention does NOT do: move reliance

In both experiments, reliance on context at matched fit (reversal accuracy
≈ 0.1, context-random ≈ 0.5) is essentially unchanged by κ, by width, by
the normalized readout and by the readout multiplier. It is removed only
when the training context is uninformative. In Exp 2 the encoder can be
made to expose the spectral cue almost fully (h_u 0.70, probe 0.91) while
the head still predicts from context (reversal 0.21).

Reading in the theorem's terms: reversal flips only when the spectral
pathway carries more than half the margin, 2a(T_m) > m. In the CNN the
spectral pathway's contribution to the margin needs a head readout as well
as an aligned encoder; the head's gain for the context cue is set by the
cue's amplitude (β = 5 on eight directions in Exp 2; γ up to 47 in Exp 3),
and slowing the whole head scales both cue readouts together, so their
ratio at the head — hence the fitted model's reliance — is unchanged. The
whole-head rate κ moves the encoder-versus-head speed ratio (and therefore
spectral learning), not the context-versus-spectral gain ratio at the head.
The ready regime of Exp 3 isolates the head-level effect completely: no
encoder learning is needed, and the head still chooses the large-amplitude
cue at every speed.

So the evidence supports: "the contextual pathway's speed relative to the
encoder controls how much of the spectral cue the encoder learns before the
fit is reached"; it does not support "that speed controls whether the
fitted model relies on context". Reliance, in these constructions, follows
from the informative context itself and the head's amplitude-driven
preference.

## 3. Proposal: make the probe the failure-relevant quantity (recoverability)

The probe is a linear readout fitted on the frozen encoder. It therefore
measures exactly what a practitioner recovers by retraining the readout on
data without the shortcut (deep feature reweighting, Kirichenko et al.). At
matched fit, a fast wide head leaves an encoder from which only 0.65–0.75
spectral accuracy can be recovered; a slow head, a small width, a
normalized readout at large width, or an uninformative context leave
0.87–0.94. The reversal failure is common to all joint models at matched
fit; what the contextual pathway's speed determines is whether the failure
is *repairable at the readout* or *baked into the encoder*. That is a
consequence of encoder starvation that the theorem's suppression bound
speaks to directly (a(T_m) − a₀ is the recoverable spectral coefficient),
and it is operationally meaningful.

Questions for Astra:
1. Does this reframing hold up: claim part one (speed → encoder suppression)
   supported by interventions; claim part two restated as "reliance follows
   from the informative context; speed determines whether the spectral cue
   remains recoverable from the encoder"? How should §0's claim sentence be
   rewritten so the abstract does not overreach?
2. Is a direct recoverability arm worth adding (freeze the encoder at the
   L* checkpoint, retrain the head on context-random data, report the
   accuracy under reversal)? It is cheap, uses the saved encoders, and would
   turn the probe into an actual repaired classifier. Or is the probe (with
   its exact population companion h_u) sufficient?
3. In the theorem, is there a clean statement of the head-level gain ratio
   as a second competition parameter (amplitude of the contextual coordinate
   versus the spectral one, at fixed rates), so that the two parameters —
   encoder-versus-head speed and context-versus-spectral gain — are named
   separately in §3's interpretation?
4. Exp 3 ready regime: the correct reading is "no encoder suppression
   possible, head-level competition total"; is it right to keep it as the
   control that separates the two levels, with the unready regime as the
   encoder-level test?

## 4. Pending

Exp 2: `sp`/`mup` seeds 1–2 at M ≥ 128 and `frozen` M = 512 seed 2; then the
registered P1–P5 verdicts at L* (P1 will read "saturated: no trend", P3
holds, P5 holds; P2/P4 to be read as registered) and the labelled secondary
analysis. Exp 3 `unready` regime: 29 of 30 runs. All files: `results/exp2/`,
`results/exp3/`, `results/exp*_summary.csv`, `results/exp2/PREDICTIONS.md`.
