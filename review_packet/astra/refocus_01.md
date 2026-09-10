# Refocus response 01 — one theorem and the two transfer experiments

Astra → Krzysztof and Claude, 2026-09-10.

Read: `paper/REFOCUS_PLAN_2026-09-10.md`, including §4.4a, and `paper/ICLR_SKELETON_v3.md`; checked the serial proofs in `review_packet/astra/appendix_math.tex`, the generator and runner (`code/synthetic/data_v2.py`, `code/experiments/exp2_intervention.py`), and `results/exp2/calibration.json`. I have not evaluated the incoming full-grid results. The width-2 pilot finding is taken from your message; the plan itself records the width-8 pilot. No paper or experiment files were edited.

## 1. One main theorem for §3

**Use the following statement.** It combines the existing serial theorem, general-initialization corollary and readout-control proposition. The reference labels are `thm:astra_serial`, `cor:astra_isotropic`, and `prop:astra_serial_control`; use these semantic labels when transferring proofs, since the old subsection and result numbers are easy to confuse.

```latex
\begin{theorem}[Selective spectral learning at matched fit]\label{thm:serial}
Let $Y$ be uniform on $\{-1,1\}$, $x_1=(S,0)$, $x_2=(0,C)$,
and $S=C=Y$ in training. The shared encoder $W=(a,v)$ produces
$z_1=aS,z_2=vC$; the head is $F=z_1+bz_2$, $b=\sum_{j=1}^M\beta_j$.
For integer $M\ge1$, train $(W,\beta)$ by unit Euclidean gradient flow
on $\mathcal L=\mathbb E\log(1+e^{-YF})$, without other optimizer terms,
from fixed $a_0\in\mathbb R$, $v_0\ne0$, $\sum_j\beta_{j0}=0$.
Fix $m>0$ and $T_m=\inf\{t\ge0:q_J(t)\ge m\}$, $q_J=a+bv$;
thus $T_m=0$ if $a_0\ge m$.

\textbf{(a) Suppression.} If $a_0<m$, set $\Delta=m-a_0$,
$u_M=a(T_m)-a_0$. Then $T_m<\infty$ and
\[
 0<\frac{u_M}{\Delta}\le\frac1{1+Mv_0^2},
 \qquad Mu_M\longrightarrow\frac{\Delta}{v_0^2}\quad(M\to\infty).
\]
The spectral-only head $F=aS$ requires update $\Delta$.
If $q_F$ instead trains only $\beta$ from the same initialization with
$W$ frozen, then $q_J\ge q_F$ on $[0,T_m]$ and
$\sup_{0\le t\le T_m}(q_J-q_F)\le C_0/(p_0Mv_0^2)$,
where $C_0=1+2\Delta^2/v_0^2$ and $p_0=(1+e^{-a_0})^{-1}$.

\textbf{(b) Reversal.} On $S=Y,C=-Y$, the stopped margin is
$2a(T_m)-m$ when $a_0<m$, and $a_0$ otherwise.
Error is one if $a_0<m/2$ and $Mv_0^2>m/(m-2a_0)$;
it is zero for $a_0\ge m/2$. The spectral-only comparator has zero
error for every $a_0$.
For $(a_0,v_0)\sim\mathcal N(0,\sigma^2I_2)$, $\sigma>0$, with
zero-sum readouts and the same stopping rule,
$\lim_{M\to\infty}\mathbb E_{W_0}\mathrm{Err}_{\rm rev}
=\Phi(m/(2\sigma))$, where $\Phi$ is the standard normal CDF.

\textbf{(c) Normalization.} Replace $b$ by
$M^{-1/2}\sum_j\gamma_j$, with $\sum_j\gamma_{j0}=0$ and the same
$(a_0,v_0)$ and unit rates. The aggregate $(a,v,b)$ dynamics equal
those at $M=1$ for every width.
\end{theorem}
\begin{remark}[Special initialization]
For $(a_0,v_0)=(0,1)$, the scalar-logit GGN blocks
$G_{\thetap\thetap},G_{\phip\phip}$, $\thetap=(a,v)$,
$\phip=\beta$, give $\mathcal D_{\rm curv}(0)=M$
(generally $Mv_0^2$), and
$\Psi(m)/(M+1+2m^2)\le T_m\le\Psi(m)/(M+1)$,
where $\Psi(m)=m+e^m-1$.
\end{remark}
```

**Length and conventions.** The theorem plus remark occupies approximately **0.44 of a text page** (285.26/650.43 pt), below the 0.6-page limit, in a standalone harness using the repository's `iclr2027_conference` style, Times, ordinary theorem environments and unchanged margins/font size. This is a layout check, not a guarantee about surrounding floats. All deterministic width limits keep `(m,a_0,v_0)` fixed; the Gaussian statement keeps `(m,σ)` fixed. `Err_rev` is population classification error under the stated reversal, with any fixed tie rule. The conclusions do not depend on that rule in the strict finite-width regimes or in the Gaussian limit. No probability over readouts is needed: their sum must be zero, including in the Gaussian-initialization experiment.

**Proof reuse.** For part (a), put `u=a−a_0` and reuse

\[
 v=v_0\cosh(\sqrt M u),\qquad
 b=\sqrt M v_0\sinh(\sqrt M u),\qquad
 q=a_0+u+\tfrac{\sqrt M v_0^2}{2}\sinh(2\sqrt M u).
\]

The increasing fitting equation and `sinh(x)≥x` give the update bound and its limit; the existing transformed-margin comparison gives the finite-horizon gap. The sign of the reversed margin and dominated convergence give (b). For (c), differentiating the aggregate readout gives `ḃ=v/(1+e^q)`, independently of M. This proof also covers arbitrary zero-sum normalized readouts, even though the old control proposition first presented zero initialization. No new lengthy theorem is required.

**Three interpretation rules for the new draft:**

- The spectral-only comparator is trained to its own first hitting time of the same margin, from the same spectral initialization. The frozen comparator is a different comparison: its margin is evaluated at the same elapsed times as the joint path, only through the joint `T_m`.
- **Normalization removes the width dependence; it does not promise recovery.** In particular, with `(a_0,v_0)=(0,1)`, the normalized system is the unnormalized M=1 system, which still has reversal error one for every `m>0`. Do not write that the theorem proves normalization eliminates shortcut reliance or reversal failure. In a CNN, improvement after normalization is an empirical prediction, not part (c).
- Drop the phase-attribution implication entirely from this theorem. Retain the corrected finite horizon and initialization constants. Do not reintroduce the “factor of six” or “every quantity agrees to 10^-11” assertions. Numerical verification belongs in its own accurately scoped table. This handles review items 1–4, 11 and 12 without rebuilding the old framework.

## 2. Experiment 2: a fair analogy, with a cleaner speed intervention

### 2.1 Verdict on the eight-neighbour construction

**Yes, this is a fair controlled nonlinear analogue of unequal initial cue accessibility. It is not a numerical identification of the toy parameters.**

The routing does what you intend. At a pixel p, its u component carries its own label. Its v components carry other pixels' labels; the copies of its own label are retrieved at the eight appropriate neighbouring offsets. Independent pixel labels remove the usual benefit of denoising a label-smooth spectral field. A nonlinear head with learned spatial filters is a substantive extension beyond identical replicated features.

The calibration file supports the intended accessibility asymmetry: spectral-only random-encoder probe accuracies are **0.6471, 0.6529, 0.6082**, versus context-only **0.9100, 0.9139, 0.9045**. The recorded oracle accuracies are close to 0.95 for both cues. Use these recorded values, not the plan's approximate 0.93 context readiness or an inferred fixed `v_0`.

The key distinction is that a **fitted LDA can read context from a random encoder**. That does not mean the randomly initialized CNN already has the correct contextual predictor. In the toy too, zero-sum readouts initially give zero contextual margin despite nonzero context sensitivity. “Initially accessible to a trained readout” is the right description.

The CNN introduces two deliberate departures: noisy, imperfect cues and jointly learned routing/readout for both cues, whereas the toy has noiseless cues and a fixed spectral skip coefficient. State them briefly. They are exactly why this is an experiment testing transfer rather than another instance of the theorem.

**The pilot supports fast contextual fitting and little spectral adaptation at the selected threshold. It does not establish `Mv_0²≫1`.** There is no measured scalar `v_0` for this CNN, and the 2,633/10,714-step ratio compares different training conditions. Write “consistent with a regime in which context fits before substantial spectral adaptation” until a rate intervention tests that account. A flat width curve in this regime can coexist with competition; it is not automatically evidence that the mechanism fails outside replicated coordinates.

### 2.2 The single intervention I recommend

**Introduce one multiplier κ on the entire spatial head's learning rate, preserving the encoder rate, the data, and the initial function:**

\[
 \eta_W=\eta,\qquad
 \eta_{\mathrm{conv1}}=\eta_{\mathrm{conv2}}=\kappa\eta,
 \quad \text{including the first convolution's bias.}
\]

Use κ<1 to slow the head. This is my preferred answer to “encoder multiplier, smaller β, or something else?” It is equivalent in gradient flow to multiplying the encoder rate by `1/κ` and rescaling time, but is easier to interpret numerically because the encoder step is not increased. Neither choice is identical in discrete time unless the corresponding global rate rescaling is also made.

The connection to the toy is exact and elementary: keep the encoder's unit rate and give each contextual readout rate κ. Then

\[
 \dot a=\varrho,\quad \dot v=b\varrho,\quad
 \dot b=\kappa Mv\varrho,\qquad
 \frac{a(T_m)-a_0}{m-a_0}\le\frac1{1+\kappa Mv_0^2}.
\]

Thus κ moves the effective competition parameter through one while leaving initial predictions and cue information fixed. The unweighted initial GGN ratio remains `Mv_0²`; it is the **rate-weighted dynamics** that change. This short calculation is an experimental-design explanation, not a request for another main theorem.

Why not the alternatives?

- **Smaller data amplitude β changes more than speed.** Here the context oracle signal-to-noise ratio is `sqrt(8) β/sqrt(β²τ²+σ²)`. Lowering β at fixed τ and σ changes predictive information and random-encoder accessibility, as well as gradients. Recalibrating τ to preserve the oracle still need not preserve accessibility. That is a legitimate different experiment, but a less clean first intervention.
- **Slowing only the final readout leaves a learned route around the intervention.** The first convolution can still learn context and alter feature magnitudes. Keep the already declared `lrmult` arm as a readout-specific test, but its failure would not show that changing total head-versus-encoder speed cannot matter.
- **Increasing the encoder rate** changes the intended ratio too, but can add a step-size instability explanation. It is an acceptable equivalent implementation when appropriately rescaled; I favor lowering the head rate.

**Bounded amendment, not a new campaign.** Preserve the declared grid and pilot artifacts. Add a separately named whole-head-rate arm on pilot-only data at one ordinary width (M=32 is sufficient), using a coarse geometric bracket such as κ in `{1, 1/16, 1/256}`, extending downward only if it fails to reach a regime with appreciable spectral learning before matched fit. Check whether the slow head still reaches the fitting threshold within the compute budget. Use training fit and a separate calibration probe to choose the bracket; do not select κ by held-out reversal accuracy. Then freeze the intervention and confirm on fresh seeds. If you want a width sweep to straddle the transition, choose a single κ from those pilots and apply it unchanged across widths, checking the narrow and wide endpoints first. Do not promise in advance that a particular κ must generate a crossover.

**Two separate controls now have separate meanings.** Whole-head-rate scaling tests relative module speed. The existing `mup` arm tests the final readout's coordinate scaling. From the actual code, setting `γ=√M w` and using output `γ/√M` gives the effective update

\[
 w^+=w-(\eta/M)\nabla_w\mathcal L,
\]

while the encoder and first-convolution updates agree at corresponding states. With the paired initialization and plain SGD, the `mup` trajectory is therefore algebraically equivalent, up to floating-point arithmetic, to the **readout-only** rate-`1/M` arm at the same width. This does not make the whole CNN's dynamics equal across widths: its learned hidden features and their number still differ. Do not count these equivalent implementations as independent causal evidence, and do not equate either with whole-head rate scaling.

Call this arm **normalized readout** in the paper. The internal string `mup` need not be changed to preserve artifacts, but it does not establish that this network follows the full maximal-update parameterization of [Yang et al., Tensor Programs V](https://arxiv.org/abs/2203.03466).

### 2.3 Is the LDA probe the right measure of spectral learning?

**Keep it as the primary operational measure of newly exposed spectral information, measured relative to its initialization value. Do not identify it with the toy's signed coefficient update.** Fit the probe anew for each frozen encoder on the same independent probe-training set, with the same fitting rule, and evaluate on a disjoint fixed set. A probe can expose information that the trained CNN does not actually use; therefore report reversal/context interventions alongside it.

For this synthetic generator there is an exact, cheap companion that clarifies what the probe measures. On `spec_only`,

\[
 X=\alpha Yu+\sigma\epsilon,\quad \epsilon\sim N(0,I),\quad \|u\|=1,
\]

and, conditional on a fixed trained W, the optimal linear probe has accuracy

\[
 A_{\rm spec}^*(W)=\Phi\!\left(\frac\alpha\sigma\sqrt{h_u(W)}\right),\qquad
 h_u(W)=u^TW^T(WW^T)^\dagger Wu
       =\|P_{\mathrm{row}(W)}u\|^2.
\]

The two Gaussian classes have means `±αWu` and common covariance `σ²WWᵀ`; their likelihood-ratio test gives the formula, including singular W by restricting to its image. Thus `h_u(W)−h_u(W_0)` measures improved retention of the spectral direction and can be exported from a checkpoint without retraining the network. Empirical ridge-LDA is a finite-sample, regularized approximation to this population quantity, not an exact evaluation of it.

By comparison, `a_u=||Wu||²/||W||_F²` measures allocation of parameter energy. An invertible change of output coordinates `W→AW` preserves `h_u` and optimal probe accuracy but can change `a_u`. Consequently the proposed **`a_u>0.25` requirement is not a necessary condition for spectral learning**. Your own pilot already illustrates the problem: 0.127 energy fraction accompanies a 0.92 spectral probe. Preserve P3's literal pass/fail result, but do not treat failing its arbitrary 0.25 cutoff as evidence against learning.

Also, LDA cannot detect mere amplitude suppression when signal and noise are scaled together. In the noiseless toy, the single spectral coordinate with any nonzero coefficient already permits perfect decoding, even when its update is O(1/M). The experimental probe tests useful row-space adaptation in a noisy bottleneck; the toy tests comparative coefficient adaptation and actual classifier failure. These related statements should remain distinct.

Recommended minimal outcome set: **probe improvement from initialization; actual reversal/context-random performance; encoder alignment and gain as explanatory secondary quantities.** If checkpoints are available, add `h_u` as the population check above. No elaborate new diagnostic framework is needed.

### 2.4 Corrections to interpretation and reporting before the next grid

These are important scope/implementation fixes, not reasons to abandon the experiment.

1. **Keep the registered predictions, weaken the inference attached to failing them.** Flat P1 curves in an already suppressed regime do not falsify competition. P2's zero width trend is a hypothesis for this CNN, not a consequence of the normalized toy. A width trend under context-random training can reflect ordinary width-dependent optimization of the spectral task; it does not logically exclude an additional contextual competition effect. Compare informative and random-context conditions at matched width/rate and inspect the interaction, while reporting the original P1–P5 verdicts unchanged. Added widths and pilot-selected amendments must be identified separately from the original five-width prediction.
2. **Matched mean loss is an experimental analogue of matched margin.** The noisy CNN has heterogeneous margins; equal average loss is not equal per-example fit or an exact application of the theorem's stopping rule. Keep the threshold, report actual loss and training accuracy at crossing, and confirm on representative cases that halving the step size does not change the interpretation. Report non-reaching runs rather than treating their last checkpoint as matched fit.
3. **Do not call 0.15 a certified context-only loss floor.** For the specified Gaussian construction, direct one-dimensional quadrature gives context-oracle population CE about **0.12596 nats** (spectral-only about **0.12721**), with the calibrated β, τ, σ. This follows by putting `d=sqrt(8)β/sqrt(β²τ²+σ²)` and integrating `E_{Z~N(0,1)} log(1+exp(-2d(d+Z)))`. The pilot's reachable level and this population optimum are different quantities. Training loss below 0.10 can also reflect finite-sample fitting; it alone does not prove that spectral learning occurred. The independent probe and test outcomes answer that question.
4. **Use rate-weighted gradient terms if discussing loss-decrease shares.** With different rates, continuous-time descent is `−L̇=η_W||g_W||²+Σ_l η_l||g_l||²`, not an unweighted ratio of block norms. In normalized coordinates raw gradient norms also rescale. Additionally, the current runner evaluates threshold snapshots *before* the current backward pass, so `gnorm_theta/phi` there are from the previous iterate. Recompute them at that checkpoint if claiming a same-state identity; otherwise label them as previous-step diagnostics. This does not invalidate the snapshot's probe, alignment or accuracy.
5. **Pair contextual interventions when measuring reliance.** The current test conditions use separate seeds. Their aggregate accuracy differences are legitimate distributional comparisons. For a sharper reliance measurement, hold centre labels and spectra fixed and change only the assigned contextual field, with shared noise where appropriate. This distinguishes a response to the intervention from unrelated test-sample variation and requires no additional training.

Implementation details worth reconciling in the supplement: the code calibrates/readiness-probes a **3×3** window (`r=1`), while some plan text says 5×5; 3×3 already contains all eight target-label copies, so this is a documentation discrepancy rather than a flaw in the oracle. The current `LR_MULTS` ends at 16, whereas the plan originally includes 64; list the actually run arms and any amendment accurately. These observations are from the source snapshot I read, not judgments on the forthcoming result table.

## 3. Experiment 3: B primary, A secondary

**Choose B as the primary mechanism test.** Keep A as a bounded secondary transfer test if resources permit, and retain its result even if it does not support the hypothesis. B gives the cleaner way to vary contextual accessibility while retaining a genuinely measured centre spectrum and target. Its artificiality is acceptable when it is explicit; it limits the generalization claim, not the internal question the experiment asks.

**A's one-sentence description:**

> “We construct a contextual-shift benchmark from measured centre spectra and measured neighbour spectra, assigning neighbour classes to control their association with the centre label; the resulting neighbourhoods are assembled rather than naturally observed tissue neighbourhoods.”

**B's one-sentence description:**

> “We test contextual competition using measured centre spectra and their recorded labels, with a synthetic class-associated spatial intensity pattern along a training-estimated dominant spectral direction in the neighbourhood.”

Use “real-centre-spectrum experiment with constructed context” for B, not “validation on natural tissue context.” If the chosen classification problem is binary, say which preselected pair; if multiclass, define the assigned-class transition matrix rather than using “reversal” ambiguously.

### Why the reported 16–24% alignment does not decide between them

Taking your reported contrast fraction as given, its overlap with v1 is not a measurement of accessibility after random encoding. Discriminability depends on within-class covariance, preprocessing and the readout. A large-variance direction can carry substantial Euclidean mean contrast but poor standardized class separation. Conversely, several real neighbours can make context easy by averaging independent within-class variation, even when each neighbour requires the same spectral alignment as the centre. Correlated neighbours from one core reduce that averaging benefit. Therefore I do **not** accept that A's only possible asymmetry is the stated 16–24% projection.

Measure the same small readiness table for A and B: raw centre-only oracle/probe, random-encoder centre probe, and random-encoder neighbourhood probe, fitted on training-side data and evaluated on separate calibration data. This is not an expensive training campaign. If A already exhibits the needed accessibility difference, it becomes a more natural and valuable transfer test; if it does not, a null A result tests a different regime rather than refuting the designed mechanism. My choice of B as primary is based on experimental control, not a claim that A cannot work.

### Minimum construction requirements

- Estimate v1, centering, scaling and all preprocessing from **training patients only**, then freeze them. Confirm that the intensity pattern survives the actual preprocessing; per-spectrum normalization can alter or erase the intended amplitude cue.
- Preserve each real centre spectrum. In B, any baseline neighbour spectra must be sampled independently of the centre label before the constructed pattern is added; otherwise a second uncontrolled cue remains. Use the same assignment mechanism with label association 0.5 for the uninformative-context control, preserving marginal pattern amplitude/noise.
- Preserve the patient split for centres and real donor spectra. Do not insert training-patient donors into held-out evaluation patches. Treat patient/seed variation as replication, rather than treating every constructed pixel as independent clinical evidence.
- Calibrate amplitude/noise and assignment correlation on training-side calibration data. **Do not make B's contextual cue almost perfectly predictive if the centre-only task is intrinsically weaker, then attribute the preference solely to speed.** Approximately match independently measured cue predictive quality where feasible, as in Experiment 2. If it cannot be matched, report the imbalance and weaken the “equally predictive” transfer claim.
- Compare rate/normalization arms at paired initialization within each width, matched training loss and common context interventions. On noisy real spectra the spectral-only comparator will generally have nonzero error. The failure claim is excess shifted error relative to that reference, not the theorem's exact zero-versus-one dichotomy.

### Suggested §2 definition-of-failure paragraph

> “We compare models at their first attainment of a prespecified training-fit threshold: a common margin in the solvable model and mean training loss in the experiments. Spectral learning means adaptation of the designated spectral coefficient in the theorem and increased independently measured linear accessibility of the centre's spectral cue in the experiments. We measure contextual reliance by changing the assigned neighbourhood signal while preserving the centre and its label, and recording the resulting change in predictions or risk. We call that reliance a failure when the shifted risk exceeds that of a separately trained spectral-only comparator at the same fitting threshold. The theorem supplies an exact reversal failure; the noisy experiments measure the corresponding risk difference without assuming a zero-error reference.”

This definition separates a model's use of context from harm caused by that use, and separates information available in an encoder from information used by its trained head.

**Immediate recommendation:** integrate the compact theorem, preserve the running grid as declared, make the bounded whole-head-rate amendment, and use B for the primary real-spectrum mechanism test. The next decision should be based on the joint probe-and-shift response to that intervention, not on generating additional proofs.
