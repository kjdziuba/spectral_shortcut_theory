# ICLR reviewer pass 02 — refocused draft

Astra → Krzysztof and Claude.

Reviewed `paper/main_iclr_v2.pdf`, 29 pages, SHA-256 `fd28d95d635736dcf966fb7d67b560b5dd7f3da136cef4a480da2ff262eeadea`. Page references are PDF pages; line references are the printed review numbers. I read the nine main pages first, then Appendices A–F, the current sources, the dated plan/amendments and Claude's interim note. I independently recomputed summaries from copies of `results/exp2_summary.csv` (753 rows; SHA-256 `39156ca1e35061ad10c576c298c43a29d1bb6fbc5d84be5814fd268485635065`) and `results/exp3_summary.csv` (582 rows; SHA-256 `4f148cde5f73e5a3342a81a6892f66afae6bf9acab882f77d9613bd3e1c610a4`). These are review snapshots because the γ=10 experiment is still running. The latter snapshot contains eight γ=10 run records; the PDF explicitly reports seven. I do not treat that ordinary update as a discrepancy in the frozen PDF. I also inspected selected saved encoder checkpoints to check the proposed directional explanation. No training was rerun and no paper files were edited.

## I. Reviewer report

### Summary

The paper studies a shared linear spectral encoder followed by a larger contextual model. An exactly solvable serial logistic model establishes finite-fit suppression of a spectral coefficient's update, a conditional contextual-reversal failure, and a readout-normalization control. Synthetic interventions in a ReLU CNN show a substantial change in encoder probe accessibility when the head is slowed, accompanied by only a small improvement in the original classifier's poor reversal accuracy. Experiments with measured tissue spectra and constructed context distinguish high and low initial probe accessibility; at the larger contextual amplitude, slowing the head has little effect on the low-accessibility regime. Production diagnostics and a matched freezing study delimit the scope.

**This is a materially stronger and more coherent submission than v1. My present recommendation is 5/10, borderline with a lean toward rejection.** There is now a concrete empirical result worth discussing. The central language still converts that result into claims about recoverability, invariance of shortcut use, and transfer across regimes that the evidence does not establish. Several summary sentences also contradict the otherwise correct theorem or the result tables.

### Strengths

1. **The main theorem is now correctly scoped and readable.** General versus special initialization, the finite fitting horizon, immediate stopping and the expectation/limit convention are visible (pp. 3–4, lines 158–179). Appendix A supports the statement. The normalized system's continuing failure in the special case is explicitly explained (p. 4, lines 198–201). The problem is the prose surrounding the theorem, not a newly discovered false main theorem.
2. **The whole-head rate intervention produces a substantial, replicated encoder effect.** At M=32 and L*=0.30, changing κ from 1 to 1/256 increases mean probe accuracy from **0.6620 to 0.8505**; the paired gains are **0.1951, 0.1784, 0.1918**. Mean row-space retention increases from **0.0665 to 0.4062**. All nine checkpoints reach the threshold (p. 6, lines 294–295; Appendix C, Table 7; `exp2_summary.csv`). This is much stronger evidence than a curvature-versus-width plot.
3. **Encoder accessibility and fitted-classifier performance are usefully separated.** In that same comparison, reversal accuracy changes only from **0.1207 to 0.1426**, and context-random accuracy from **0.5083 to 0.5095**. The rate intervention improves the representation substantially without repairing the trained predictor. The ready real-spectrum regime supplies a complementary observation: high initial probe accuracy coexists with severe contextual-shift failure (p. 7, lines 372–377). These observations deserve the central position now given to them, with the qualifications below.
4. **The controlled data construction and scope disclosures are valuable.** Real centre spectra are distinguished from constructed neighbourhoods, preprocessing uses training-side data, validation/test results are separated, and the uninformative-context training controls succeed (pp. 7–8; Appendix D). The revised pairing and same-state-gradient implementation address earlier concerns, subject to the centre-preservation qualification below.
5. **The manuscript reports failures and the incomplete variant rather than hiding them.** Failed prediction labels, secondary thresholds, the pilot amendments, and non-reaching κ runs are present somewhere in the paper. That is good practice. Their aggregation and placement need correction; the underlying records make correction possible.

### Ranked weaknesses

**W1 — Major, highest priority: the headline exceeds the demonstrated dissociation.** Locations: abstract, p. 1, lines 22–27; contributions, p. 2, lines 61–74; p. 6, lines 300–310; conclusion, p. 9, lines 434–454.

The strongest supported result is that, in the synthetic experiment and tested rate range, slowing the head increases probe accessibility while the original predictor remains strongly context-reliant. This is not a binary decision about whether information is recoverable, not proof that speed has no effect on shortcut use, and not a uniform result across the real-spectrum regimes. Reversal accuracy changes modestly but consistently at the two κ endpoints. The ready real-data regime has no large measured probe deficit; the γ=30 unready regime has limited and heterogeneous gains. Its proposed amplitude-based explanation has supporting evidence but is not yet a uniquely identified mechanism.

The finite-width reversal condition is also omitted repeatedly outside the theorem. Moreover p. 9 identifies high random-probe accuracy with the theorem's `a_0≥m/2` case, although an arbitrary retrained encoder probe does not measure the fixed spectral contribution to the original classifier's margin. That identification is false. Sections II and III give exact replacements.

**W2 — Major: the recoverability contribution needs the operational consequence it currently claims.** Locations: p. 2, lines 66–68; p. 6, lines 303–310; Appendix A, Proposition A.2.

A linear probe on isolated spectral-only encoder outputs is a useful result. In the Gaussian synthetic experiment it has an exact population interpretation. It is not the same operation as retraining the original spatial head on encoded patches with context present but decorrelated. Nor does poor performance of one probe demonstrate that all practical repair is impossible. In the noiseless theorem, every nonzero special-case spectral coefficient already retains the label perfectly, even when its update is arbitrarily small; Appendix A says so explicitly.

The distinction between available core features and reliance on spurious features is already a central observation of [Kirichenko et al., *Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations*](https://arxiv.org/abs/2204.02937). The potential addition here is how relative module training rates change what subsequent retraining can achieve at matched original fit. Demonstrate that addition directly, or describe the present contribution as a probe-accessibility study without claiming practical repairability. I am not asking for a universal theorem or a new optimizer.

**W3 — Major reporting issue: some principal comparisons mix thresholds, seed counts or width sets.** Locations: p. 6, lines 278–315; Figure 3, p. 7, lines 347–351; Figure 4, p. 8, lines 403–407.

The κ=1/4096 and 1/32768 examples never reach L*. Their final probe/reversal outcomes nevertheless appear in text and a figure headed “at matched fit.” The claimed 0.87–0.94 recovery range pools different arms and stopping thresholds and is false at the stated primary threshold. The claimed “original five-width” verdicts use the seven-width amended grid. On the original five widths, the normalized-readout alignment-flatness clause actually passes its two-of-three-seed criterion, while its reversal-flatness clause fails. P3 fails both its magnitude and flatness requirements, not just the 0.25 cutoff. P6's probe result is described, but its registered reversal requirement should be reported explicitly as unmet in one seed. These are repairable accounting errors; they materially affect how strong and confirmatory the evidence looks.

**W4 — Moderate to major: the two real-data regimes support a descriptive separation, with a weaker causal interpretation than the prose gives.** Locations: pp. 7–8, lines 354–431; p. 9, lines 451–454.

High initial accessibility is a useful control for failure despite available spectral information. It does not imply that no further adaptation is possible. In the unready γ=30 setting, mean probe gains remain far below the context-random control, but one seed gains **0.1013**, contradicting “complete” suppression and the universal 0.05 maximum. The mean increase from κ=1 to 1/256 is approximately **0.013**; that is a small response rather than proved rate insensitivity.

My checkpoint check supports concentration of adaptation: for all nine M=32 γ=30 combinations of seed and κ, more than **99.99% of squared net encoder displacement** lies in the eight constructed-cue directions. This is useful evidence to report. It does not alone identify the instantaneous gradient decomposition or establish that cue amplitude uniquely causes the lack of recovery. Unlike the toy's zero aggregate readout, the CNN/patch head has a nonzero random readout, so contextual learning can proceed through encoder changes even with a nearly fixed head. Also, the real cue directions need not be orthogonal to the centre's class contrast, so learning those directions may itself improve the centre probe. Section V gives the defensible two-regime reading.

**W5 — Moderate: experimental definitions and quantitative scope need a final reconciliation.** Locations: pp. 2–3, lines 102–147; p. 5, lines 254–269; p. 9, lines 454–462.

There are different module learning rates in the principal interventions despite repeated “one rate for every parameter” wording. The empirical reference is a context-random-trained full model, not a trained spectral-only architecture. Experiment 2 preserves the target spectral component and noise under reversal, but not each complete centre spectrum: every pixel's nuisance v components also change. Equal fitted-LDA accuracy does not establish equal information in real data, and the neighbourhood-readiness calibration in Experiment 3 includes the centre. The production paragraph repeats the earlier rank-one causal attribution and mislabels orthogonalized-contrast curvature as full-contrast curvature. None of these requires enlarging the mathematical appendix; they require precise methods and narrower sentences.

**W6 — Moderate, mainly about significance and economy.** Locations: p. 2, lines 80–93; pp. 4–5, numerical-check paragraphs; p. 9 and Appendix E.

The exact model is intentionally a replicated-coordinate learning-rate construction. Its transparency is a strength, but its scientific value depends on the empirical consequence beyond that construction. The synthetic rate effect supplies part of that value; practical recovery and the real-data rate-by-amplitude interaction remain incomplete. Repeating the algebra check twice and retaining eight pages of old production material dilute the focused contribution. Keep the strongest direct result in the main text, retain the relevant counterevidence, and trim the historical material to the scope limitation it now serves.

### Questions for the authors

1. Will you replace the global “speed decides recoverability, not shortcut use” claim with the scoped rate/probe/predictor statement in Section III, and restore the finite-width reversal qualifier in every summary?
2. At the genuinely matched L* checkpoints for κ=1 and 1/256, does retraining the same spatial head with the encoder frozen on context-random data produce different shifted accuracy? What does the random-initial encoder permit under the same retraining protocol?
3. Can you regenerate the registered-verdict table on both the original and amended width sets, explicitly reporting every criterion and P6's two outcomes? Can the two non-reaching κ runs be moved out of the matched-fit panel?
4. Can you export the encoder-update projection check below, and either measure the centre/context contributions to the actual encoder gradient or describe the amplitude explanation as a hypothesis supported by concentrated updates?
5. Once γ=10 is complete, what is the paired rate effect relative to γ=30, with the same thresholds and complete seed counts? Does it change the claim after controlling the calibrated cue predictiveness and reporting the held-out centre benchmark?
6. What is the estimated variation across patients for the real-data contrasts? The current seed summaries are conditional on one split; please avoid implying a clinical population estimate from thousands of constructed pixels.

### Score and justification

**5/10 — borderline, leaning reject. Confidence 4/5 on the technical and artifact checks; lower on novelty judgments across reviewers.** This is the same internal ten-point convention as review 01, not a claim about the final ICLR 2027 review form.

The move from 4 to 5 reflects actual progress: a coherent scoped theorem, a large replicated synthetic rate/probe effect, and a useful real-spectrum control. The remaining objections concern the contribution claimed from those findings, material aggregation errors, and an untested operational interpretation. Correcting the prose and tables removes the reliability objection but does not by itself establish the claimed repairability result. A strong, small recovery experiment could move me toward a weak accept; additional general proofs would not be my priority.

### The single change most likely to move the score

**Turn probe accessibility into a controlled, actual recovery comparison using the already saved matched-fit encoders.**

At M=32, use the L*=0.30 encoders from κ=1 and 1/256, plus the random-initial encoder, for all three seeds. Freeze each encoder and train the same chosen spatial-head architecture on newly sampled context-random training patches, with a fixed shared protocol, paired head initializations and a stopping/model-selection rule fixed before inspecting shifted evaluation results. Evaluate on the same iid, reversed and context-random conditions, with context present. Report original shifted accuracy, probe accuracy, and shifted accuracy after retraining in one small table. This is approximately nine primary retraining runs, not a new architecture sweep. Add the real-spectrum version only if feasible after the mandatory γ=10 completion.

The question is whether the rate intervention changes how much robustness this specified retraining procedure can recover while the original classifiers have similarly poor shifted performance. Report optimization failures separately; failure of this retraining procedure would not prove that every repair is impossible. The clean Gaussian population probe is a useful reference. This experiment would substantiate the manuscript's practical interpretation and its distinction from existing “features are present but unused” work.

## II. Separate main-text claim ledger

These entries cover the main-text sentences I find false, materially overbroad or unsupported, grouping repeated versions of the same issue. Mathematical notation is transcribed; line-break hyphenation is removed. “Unsupported” is not a claim that the opposite is true. I exclude explicitly labelled predictions from being treated as established results and do not count the disclosed unfinished γ=10 status as misconduct or a hidden result.

### 1. Reversal failure is repeatedly missing its finite-width condition — blocking

**p. 1, lines 15–21:**

> “In a linear-encoder cross-entropy model with M replicated contextual readouts we prove that, at a matched fitting margin, the spectral coefficient moves by at most a fraction 1/(1+Mv_0²) of the update a spectral-only model needs, that the fitted model then fails under contextual reversal when its initial spectral coefficient is below half the margin, and that a M^{-1/2}-normalized readout removes the width dependence without removing the failure: the contextual pathway's training speed relative to the encoder is the competition parameter.”

**p. 2, lines 54–60:**

> “For the serial cross-entropy model with M replicated contextual readouts, gradient flow from general initialization moves the spectral coefficient at the matched fitting margin by at most a fraction 1/(1+Mv_0²) of the update a spectral-only model needs; the fitted model misclassifies every example under contextual reversal when the initial spectral coefficient is below half the margin, with expected error tending to Φ(m/2σ) over isotropic Gaussian initialization; and a M^{-1/2}-normalized readout removes the width dependence, though not the failure.”

**p. 4, lines 194–196:**

> “When the initial spectral coefficient is below half the margin, the fitted joint model relies on context strongly enough that contextual reversal misclassifies every example, while a spectral-only comparator trained to its own hitting time of the same margin is correct.”

**p. 9, lines 434–437:**

> “We prove, in a serial linear-encoder model, that a contextual pathway trained faster than the spectral encoder suppresses the encoder's update at matched fit, that the fitted model then fails under contextual reversal while a spectral-only comparator does not, and that readout normalization removes the width dependence of this effect without removing the failure.”

**Restricting file:** `paper/sections_iclr_v2/appendix_serial_proofs.tex`, Corollary A.1. Require sufficiently large M, or the sufficient finite-width condition `Mv_0²>m/(m−2a_0)`. Counterexample: m=2, a_0=0.4, v_0=0.05, M=1 gives fitted a≈1.98514 and positive reversed margin ≈1.97028, although a_0<m/2. The normalization clause is also special-case, not a universal persistence-of-error theorem: write “normalization removes width dependence, but need not remove failure.” The main theorem already says the correct thing.

### 2. The abstract combines distinct empirical regimes into one universal response — blocking

**p. 1, lines 22–26:**

> “Interventions on that speed in a ReLU convolutional model with two equally informative cues, and with measured tissue spectra in constructed neighbourhoods, show that an informative context suppresses the encoder's spectral learning at matched fit at every width, that slowing the head restores it monotonically, and that the fitted head nonetheless predicts from context whenever the context is informative.”

**Contradicting/restricting files:** `results/exp2_summary.csv`, `results/exp3_summary.csv`, Tables 7, 11 and 12. The substantial monotone restoration is an Experiment 2 result over κ∈{1,1/16,1/256}. The ready real regime starts with high accessibility; the unready γ=30 regime has small heterogeneous changes. “Whenever” ranges beyond the tested conditions. Split the synthetic finding from the real-data scope test and use “remained strongly context-reliant in the tested informative-context runs.”

### 3. A binary/global recoverability decision is not established — blocking

**p. 1, lines 26–27:**

> “The speed of the contextual pathway thus decides not whether the model fails under a context shift but whether the spectral cue remains recoverable from the encoder afterwards.”

**Restricting files:** `results/exp2_summary.csv`; `paper/sections_iclr_v2/appendix_serial_proofs.tex`, Proposition A.2. Accessibility changes continuously; it is already above chance for the fast heads. The theorem's noiseless spectral coordinate remains perfectly decodable whenever its coefficient is nonzero. Even the “not whether” clause is not a theorem: for the special initialization and m=2.9, effective readout speed κM=1 yields reversed margin ≈−0.864, whereas κM=0.1 yields ≈+2.151. Rate can change reversal failure outside the experimentally tested regime. Replace with the scoped wording in Section III.

### 4. The introductory claim has not been reconciled with the new empirical conclusion — important

**p. 1, lines 43–46:**

> “In a serial spectral–spatial model, increasing the effective training speed of the contextual pathway suppresses learning of an equally predictive spectral cue; at matched training fit this produces greater reliance on context and failure when that context changes.”

**Restricting files:** the theorem's Appendix A and the two experiment summaries. This works as a conditional existence claim for the specified toy; it is not the common established empirical chain. In Experiment 2 the probe effect is large and the shifted-accuracy effect small; in the ready real regime failure occurs without a large probe deficit. Make the introduction state the proved instance followed by the measured dissociation. The next sentence's “we predict ... any serial model” is explicitly a prediction, but it should now be retired or narrowed in light of these results.

### 5. Synthetic contribution: restoration and control scope are overstated — important

**p. 2, lines 61–68:**

> “In a ReLU convolutional head over a linear encoder, with two cues of equal oracle information but unequal initial accessibility, an informative context suppresses the encoder's spectral learning at matched fit at every width; slowing the whole head restores it monotonically in every seed, a normalized readout reduces it at large width, and an uninformative context removes it.”
>
> “The fitted head's reliance on context does not respond to any of these controls: the contextual pathway's speed sets how much spectral structure the encoder retains, hence what readout retraining can recover, not whether the shortcut is used.”

**Restricting file:** `results/exp2_summary.csv`. Say “increases probe accessibility” rather than unqualified restoration; it remains below the context-random control at κ=1/256. Resolve the ambiguous “reduces it” to “reduces suppression.” The last sentence includes the uninformative-context control, which plainly does change reliance, and ignores the +0.0219 reversal-accuracy change between κ endpoints. Restrict it to the informative-context rate/normalization interventions and say they do not remove the large shifted-performance deficit. Practical spatial-head retraining has not been tested.

### 6. Real-spectrum contribution: “every arm,” “complete,” and “nothing” exceed the data — important

**p. 2, lines 69–74:**

> “With measured centre spectra and their recorded labels in constructed neighbourhoods, the same reliance appears in every arm; encoder suppression is complete in the regime where the encoder must learn the contrast and, at the cue amplitude that makes the context accessible, insensitive to the head's rate, while in the regime where a random encoder already reads the contrast there is nothing to suppress.”

**Contradicting file:** `results/exp3_summary.csv`. Context-random-trained arms have high reversal accuracy. In the unready informative regime probe gain reaches +0.1013 in an individual run, while mean rate effects are small rather than exactly zero. Ready-regime probe gains also vary. Use “informative-context arms,” “limited adaptation relative to the control,” “small rate response over the tested range,” and “little measured room for improvement in the chosen probe.”

### 7. One global rate is incorrectly stated as the full experimental protocol — important

**p. 2, lines 102–104:**

> “Training minimizes the mean margin loss L=N^{-1}∑_p log(1+e^{-y_pF_p}) by gradient flow or plain gradient descent with one learning rate for every parameter, so that any difference in training speed between the two modules comes from the model, not from the optimizer.”

**p. 5, lines 253–256:**

> “The model is the linear encoder followed by a ReLU convolutional head (3×3 convolution to width M, ReLU, 3×3 convolution to one logit, circular padding), trained by full-batch gradient descent on 256 images (65,536 pixels) with one learning rate η=10^{-3} for every parameter, chosen as the largest rate with a monotone loss at the widest head.”

**p. 9, line 454:**

> “Plain gradient descent with one global rate is the only optimizer studied.”

**Contradicting files:** `code/experiments/exp2_intervention.py`, `code/experiments/exp3_train.py`, optimizer parameter groups; the rate arms in both summaries. Write “plain gradient descent with base rate η and the explicitly specified block multipliers.” The common-rate claim applies only to the baseline. Figure 1's embedded “one global rate” label needs the same repair; Appendix E also contains other optimizer families, so scope the sentence to Experiments 1–3.

### 8. Equal fitted-oracle accuracy is presented as equal information everywhere — important

**p. 3, lines 126–130:**

> “Both cues are calibrated to carry the same information about the label (equal oracle accuracy), so the model's preference cannot be attributed to one cue being more predictive.”
>
> “In the theorem this is the pair (a_0,v_0): the initial spectral coefficient is small relative to the margin, the contextual coefficient is of order one.”

**Restricting files:** `results/exp3/calibration3.json`; `code/experiments/exp3_train.py::calibrate`; Appendix A. Equal estimated ridge-LDA accuracy on calibration halves is not equality of label information or a Bayes-oracle guarantee on held-out patients. For example, the unready centre benchmark is 0.948 on the calibration split and 0.884 on validation; the contextual calibration target is 0.948. Do not infer that this establishes which cue is more informative on validation either—measure both there if that claim is needed. The synthetic Gaussian construction permits a stronger matched-signal-to-noise interpretation; the real-data statement should say “matched calibration discriminant accuracy.” The theorem allows general a_0 and v_0; the asymmetric case is a selected regime, not its entire hypothesis set.

### 9. The reference classifier in the definition differs from the implemented reference — important

**p. 3, lines 137–139:**

> “We call that reliance a failure when the shifted risk exceeds that of a separately trained spectral-only comparator at the same fitting threshold.”
>
> “The theorem supplies an exact reversal failure; the noisy experiments measure the corresponding risk difference without assuming a zero-error reference.”

**Restricting files:** both experiment runners' `ctxfree` arms. The implemented reference is a full spatial model trained with an independent context-label field; it still sees neighbours and nuisance context. `acc_spec_only` is a test condition, not a separately trained spectral-only model. Keep the valid comparison but define it by its actual training intervention, or add the stated centre-only training reference. This is a mismatch in my previously suggested paragraph as integrated with the actual experiments; it needs correction rather than preservation because I wrote it.

### 10. Full centre spectra are not held fixed in the synthetic reversal — important

**p. 3, lines 135–137:**

> “We measure contextual reliance by changing the assigned neighbourhood signal while preserving the centre and its label, and recording the resulting change in predictions or risk.”

**p. 3, lines 143–146:**

> “Every evaluation set is generated three ways from the same centre pixels, labels and noise: iid (context agrees with the label as in training), reversed (the neighbourhood carries the opposite label), and context-random (the neighbourhood carries an independent label, so context is present but uninformative).”

**Contradicting file:** `code/synthetic/data_v2.py::make_problem_v2`. Reversal changes `β∑_d y_{p-o_d}v_d` at **every** pixel, including the target centre. Its u component, label and noise are shared; its complete spectrum is not. Experiment 3 does hold the centre fixed. Describe these separately; Figure 1's “same centres” label needs the same qualification. The synthetic intervention remains meaningful, but it changes the global contextual feature field rather than only a target patch's neighbours.

### 11. Figure 1 gives a 3×3 receptive field for a two-convolution head — minor

**p. 3, lines 119–122:**

> “Each pixel's spectrum passes through the shared encoder; the spatial model reads the encoded 3×3 neighbourhood.”

**Restricting files:** `code/experiments/exp2_intervention.py::CNNHead`; `code/experiments/exp3_train.py::PatchHead`. The Experiment 2 head has a 5×5 effective receptive field from two 3×3 convolutions. The cue is placed in the inner 3×3; Experiment 3 reads a single nine-spectrum patch. Label the figure as cue placement or explicitly distinguish the heads. The following “requires encoder alignment” language also applies to the unready settings, not the ready real-spectrum regime.

### 12. Low spectral accessibility is called no accessibility — minor

**p. 5, lines 251–253:**

> “This is the nonlinear analogue of (a_0 small, v_0=O(1)): the contextual cue is initially accessible to a trained readout; the spectral cue is not.”

**Restricting file:** `results/exp2/calibration.json`. The spectral probe is 0.608–0.653, not chance. Say “the spectral cue is substantially less accessible.”

### 13. Initialization numbers are presented as common values — minor

**p. 5, lines 265–269:**

> “At every snapshot we record the encoder's linear accessibility of the spectral cue (a discriminant fitted on independent spectral-only data and evaluated on a disjoint set, reported as the gain over its initial value 0.647), the exact population counterpart h_u(W)=||P_row(W)u||² (the optimal probe has accuracy Φ(α√h_u); h_u=0.055 at initialization), the encoder's displacement, and accuracies on paired test sets that share labels and noise and differ only in the assigned context: iid, reversed, context-random, spectral-only, context-only.”

**Restricting file:** `results/exp2_summary.csv`, `threshold=init`. Actual training-run probe baselines are **0.647156, 0.657959, 0.614929**, and h_u **0.054746, 0.060406, 0.029236**. The printed constants are seed 0. Calibration uses different probe samples, so its 0.653/0.608 values must not be substituted as the run baselines. Qualify seed 0 or give the mean/range, including the repeated baseline in p. 6, line 272 and Table 4's caption. In the list of five tests, `ctx_only` also removes the spectral signal; it does not differ *only* in context.

### 14. “Every jointly trained model” contradicts the whole-head result — important

**p. 6, lines 270–273:**

> “At L* every jointly trained model with an informative context leaves the encoder's spectral accessibility near its initial value: the probe gain is 0.04 at M=2 and 0.002 at M=2048 (means over three seeds; h_u 0.09→0.05 against 0.055 at initialization), accuracy under reversal is 0.12–0.13, and accuracy with an uninformative context is at chance (0.50).”

**Contradicting file:** `results/exp2_summary.csv`, `headlr` at L*. These gains reach 0.203–0.223 at κ=1/256. The quoted width summaries are the `sp` arm. Begin “In the standard-parameterization baseline...” and use the actual mean initialization h_u≈0.0481.

### 15. The registered-width statements use a different width set — important

**p. 6, lines 278–282:**

> “We report P1–P5 as registered (Appendix C).”
>
> “P1 fails: the width trend in encoder alignment and reversal accuracy is monotone in two seeds but not the third, and its magnitude is negligible, because L* lies in a regime where the context fits before any spectral adaptation at every width, including M=2.”
>
> “P2 fails as stated (the normalized readout also shows a small trend in two seeds), although its second clause holds: at M=2048 the normalized readout leaves more encoder energy on the spectral direction than the standard one in every seed.”

**p. 6, line 315:**

> “The original five-width predictions are reported above as registered.”

**Contradicting/restricting files:** `results/exp2/PREDICTIONS.md` and Appendix C explicitly use seven widths; original §4.4 of `paper/REFOCUS_PLAN_2026-09-10.md` uses five. Original-width SP correlations are `a_u: −0.70,−1.00,−1.00`, reversal `−0.60,−0.90,−0.60`: only one seed meets both requirements. Original-width normalized-readout a_u correlations are `+0.10,−0.90,−0.30`, which **pass** the stated two-of-three alignment-flatness requirement. Reversal flatness fails (`−0.90,−0.70,−0.70`). Thus P2 overall still fails if “same quantities” includes reversal, but the reason must be stated correctly. Report original and amended grids separately. Low measured gains are consistent with early contextual fitting; “before any adaptation” and the causal “because” are stronger than needed.

### 16. P3 omits a failed requirement and explains away a real probe trend — important

**p. 6, lines 282–285:**

> “P3 fails its literal cutoff on the encoder's energy fraction while its substance holds: with the cause removed the encoder exposes the cue at every width (probe 0.89–0.94), the energy fraction merely spreading over more head directions as M grows.”

**Contradicting/restricting files:** the plan's P3 and `results/exp2_summary.csv`. P3 also required flatness; a_u and the probe have rank correlation −1 across the amended control widths in each seed. Mean probe accuracy falls **0.9365→0.8934**, and h_u **0.8628→0.5641**. This is not merely a coordinate-energy redistribution with unchanged accessibility. Also a_u concerns W, whose 12×256 dimensions do not grow with head width. Report both failures, then separately report the positive fact that the context-random controls learn substantially more than the informative-context baselines.

### 17. The strongest “matched-fit” reversal range includes non-reaching runs — blocking

**p. 6, lines 300–303:**

> “Accuracy under reversal at L* stays between 0.11 and 0.21 across every width, every κ including the extension, the normalized readout and the readout multiplier, and accuracy with an uninformative context stays at chance; only training without an informative context removes the reliance.”

**Contradicting file:** `results/exp2_summary.csv`; Appendix C, Table 7. Both extended κ runs are `max_steps` without an L* checkpoint, at final losses **0.3198 and 0.3471**. Among actually matched informative joint runs, per-run reversal accuracy is **0.0952–0.1520**; the larger ≈0.184/0.208 values are budget endpoints. The final “only” is valid only among the tested interventions. Separate the endpoints explicitly.

### 18. Near-full encoder recovery is an unmatched endpoint — important

**p. 6, lines 303–304:**

> “The encoder can be made to expose the cue almost fully (h_u 0.70, probe 0.91) while the head still predicts from context.”

**Restricting file:** the same two non-reaching `headlr` rows. This is a valid observation about budget endpoints, not the matched-fit comparison under which the paragraph is presented. At matched fit the strongest κ arm has mean h_u≈0.406 and probe≈0.850. Label the horizon and avoid “almost fully” if referring to h_u itself (0.70 is not near one).

### 19. The nonlinear predictor is assigned an unmeasured toy margin decomposition — important

**p. 6, lines 304–307:**

> “In the theorem's terms the fitted models sit in the regime 2a(T_m)<m: the contextual pathway's speed relative to the encoder controls how much of the spectral cue the encoder learns before the fit is reached; the head's gain for the large-amplitude contextual cue, which slowing the whole head leaves unchanged, controls which cue the fitted head uses.”

**Restricting files:** `code/experiments/exp2_intervention.py::CNNHead`, Appendix A and `results/exp2_summary.csv`. The CNN has no measured scalar a or uniform fitted margin m. Multiplying all head rates preserves their **initial relative rate scaling**, not a fixed learned functional gain ratio at matched fit: W, ReLU gates and head weights follow different paths. Treat this as an analogy and an amplitude hypothesis, not an invariant proved by the intervention. The actual rate/probe and shifted-accuracy outcomes suffice.

### 20. Probe retraining is equated with spatial-head repair, with incorrect accuracy ranges — important

**p. 6, lines 307–310:**

> “The probe is a linear readout on the frozen encoder, so it measures what readout retraining without the shortcut can recover (Kirichenko et al., 2023): a fast wide head leaves 0.65–0.75 recoverable spectral accuracy, a slow head, a small width, a normalized readout at large width or an uninformative context leave 0.87–0.94.”

**Contradicting/restricting files:** `results/exp2_summary.csv`; `code/experiments/exp2_intervention.py::spectral_probe`. At L*, mean probes are **0.6824** for SP M=2, **0.6678** for normalized M=2048, **0.8505** for whole-head κ=1/256, and **0.8934–0.9365** for context-random training. SP M=2048 is **0.6417**. The 0.87–0.94 grouping does not describe these arms at this threshold. The probe is fitted on spectral-only pixels and is not the original spatial head retrained with nuisance context present. Keep “linear-probe accuracy on isolated spectral inputs” unless a direct recovery arm is added.

### 21. Figure 3 hides differing horizons and seed counts — important

**p. 7, lines 347–351:**

> “Experiment 2 at matched fit L*=0.30 (three seeds; points are seeds, lines are means).”
>
> “(c) Whole-head rate multiplier κ at M=32: probe gain and reversal accuracy.”

**Contradicting file:** `results/exp2_summary.csv`, extended κ rows. The open markers are single-seed final-budget outcomes, not matched-fit points, and the caption never explains that distinction. Put them in a separate endpoint panel or label their loss, horizon and n=1 explicitly. The main matched-fit comparison can stand on its own three κ values.

### 22. The readiness regimes are identified with theorem cases without a mapping — blocking for the formal interpretation

**p. 7, lines 360–361:**

> “We therefore run two regimes on the same pixels and labels: ready (standardized inputs, nothing for the encoder to learn) and unready (whitened inputs, the theorem's case).”

**p. 9, lines 451–454:**

> “The initial accessibility asymmetry that the mechanism needs is set by the bottleneck and the input statistics: where a random encoder already exposes the spectral cue, as it does for one real class pair at bottleneck twelve (Section 5), the theorem's a_0≥m/2 case applies and no failure is predicted.”

**Contradicting/restricting files:** Appendix A; `code/experiments/exp3_train.py::PatchHead`; `results/exp3_summary.csv`. In the theorem a_0 is already the signed spectral contribution to the **actual classifier** through a fixed unit skip. An independently fitted probe's accuracy does not give that contribution for a random learned head. Ready models indeed fail here. The noisy whitened patch network is also not an instance of the theorem. Use “high/low initial probe accessibility” and “an empirical analogue of the adaptation question”; no theorem-case identification.

### 23. The reported neighbourhood readiness includes the centre — important

**p. 7, lines 366–368:**

> “τ is calibrated per regime so that the neighbourhood oracle equals the centre-only oracle (0.969 ready, 0.948 unready); through a random encoder the neighbourhood reads at 0.94–0.99 in both regimes.”

**Contradicting file:** `code/experiments/exp3_train.py::calibrate`, `Qa=enc(Pa).reshape(...)`; `results/exp3/calibration3.json`, `readiness_by_seed[*].patch`. The latter probe sees **all nine encoded spectra including the centre**. That is especially consequential in the ready regime, where combining two useful cues can exceed either cue's own accuracy. Call it patch readiness; export a donor-only encoded probe if a neighbourhood-only accessibility claim is needed. The calibration equality is an estimated training-side discriminant equality, as in item 8.

### 24. Ready-regime “context only” and “nothing can be starved” overstate observational evidence — important

**p. 7, lines 372–377:**

> “In the ready regime the probe is 0.91–0.94 before training and does not move at L* in any arm (gain −0.015 to +0.035), yet every jointly trained model predicts from context only: reversal accuracy 0.08–0.10 and context-random accuracy at chance for widths 8–512 and for κ=1,1/16,1/256 alike, all fitting within 70–230 steps; the uninformative-context comparator reaches reversal 0.93 and context-random 0.92, and the frozen encoder 0.09 (Table 12).”
>
> “Nothing can be starved here; the reliance is head-level competition between a large-amplitude cue and an O(1) one, which the theorem does not address, and it is complete.”

**Restricting file:** `results/exp3_summary.csv`. Gains are small, not identically zero; `ctxfree` is also jointly trained and does not fail. Poor aggregate shifted accuracy establishes strong reliance, not the absence of any spectral contribution to the function. The ready probe is high, not a guarantee that all useful adaptation is exhausted. Write “The initially accessible spectral information remains largely accessible, yet informative-context-trained heads are strongly context-reliant.” The comparison shows that a large deficit in this probe is unnecessary for failure; it does not identify a unique head-level gain mechanism.

### 25. The unready maximum and fitting-time range are means presented as universal bounds — important

**p. 8, lines 410–412:**

> “In the unready regime the probe starts at 0.56–0.60 and at L* has gained at most 0.05 in any arm, with reversal 0.07 and context-random at chance; the fit takes 69–185 steps even at κ=1/256 (Table 11).”

**Contradicting file:** `results/exp3_summary.csv`. In informative runs, seed 1 at κ=1/256 gains **0.10133**, and κ=1 gains **0.07867**; the context-random arms gain ≈0.34. The 69–185 range is of selected arm means; actual informative trainable runs span **59–226** steps. Say “mean gains of at most 0.046 among the γ=30 informative-context arms, with substantial seed variation.” Keep the distinction from the much larger control gains.

### 26. Gradient dominance is asserted as a uniquely established causal explanation — important, with supporting evidence available

**p. 8, lines 412–415:**

> “Slowing the head does not help the encoder here because, at amplitude 30, the contextual cue dominates the encoder's own gradient: the encoder learns the context directions before the whitened, high-dimensional class contrast, in the theorem's terms a large v_0 relative to a_0 at the encoder rather than a fast readout.”

**Restricting files:** `results/exp3_summary.csv` records whole-block norms, not this gradient decomposition; Appendix A assumes zero-sum readouts, whereas the real patch head is random and nonzero. Saved `results/exp3/enc_unready_*` checkpoints strongly support concentration of **net updates**, as detailed in Section V. Use that measured fact and say the amplitude account is consistent with it; “does not restore the control's probe performance” is more accurate than “does not help.” No exact identification with v_0 has been established.

### 27. γ=10 is neither the synthetic amplitude nor a demonstrated non-dominance regime — important/minor

**p. 8, lines 419–421:**

> “A variant with the cue amplitude reduced to γ=10, the scale of Section 4, tests whether the head's rate recovers its effect once the cue no longer dominates the encoder's gradient.”

**Contradicting/restricting files:** Experiment 2 uses β=5; `code/experiments/exp3_train.py` uses γ=10 in a different input dimension/preprocessing. Equal numerical amplitudes would not by themselves match effective scale. “Once the cue no longer dominates” presupposes the unmeasured result; selected γ=10 encoder updates remain highly concentrated in cue directions. Write “A lower-amplitude variant tests whether the rate response changes.” The bracketed run counts and per-seed interim examples are properly identified as interim; they should not be promoted to final evidence.

### 28. The real-data reading equates several unmeasured quantities with κMv_0² — important

**p. 8, lines 426–431:**

> “With measured spectra the outcome matches the synthetic experiment on reliance and sharpens the account of suppression: the fitted head uses the constructed context whenever it is informative, at every width and rate; whether the encoder learns the spectral contrast before the fit depends on the balance of gains at the encoder, set jointly by the head's relative speed (Section 4) and by the contextual cue's amplitude relative to the spectral one.”
>
> “Both are the theorem's κMv_0² seen from different sides.”

**Restricting files:** Appendix A; both experiment summaries and the patch-head code. The experiments vary rates and input amplitude; they have not measured a scalar effective v_0, nor shown the rate/amplitude interventions to be interchangeable in the trained nonlinear model. The hypothesis is plausible; present it as a possible explanation under test. Restrict “every width and rate” to those evaluated. No extra theorem is needed to make the observational statement useful.

### 29. Exact excess-risk numbers conflate means with common values — minor

**p. 8, lines 418–419:**

> “Excess shifted risk of the joint models over that comparator is 0.85.”

**p. 8, lines 430–431:**

> “Excess shifted risk relative to the uninformative-context comparator is 0.85 in both regimes.”

**Restricting file:** `results/exp3_summary.csv`. At matched widths 8 and 128, mean validation excess risks averaged over those widths/seeds are **0.8336 ready** and **0.8387 unready**; individual paired differences span **0.8152–0.8447** and **0.8027–0.8528**, respectively. Say “about 0.83–0.84 in the matched-width comparisons,” identify validation and the comparison set, or report each width. The “about 0.85” wording in the conclusion is a loose summary rather than a separate major error.

### 30. Figure 4 gives a uniform three-seed caption for incomplete cells — important until the grid is complete

**p. 8, lines 403–407:**

> “Experiment 3 at L*=0.30 on validation patients (three seeds; points are seeds).”

**Restricting files:** `results/exp3_summary.csv` and the explicitly interim γ=10 passage. Some orange cells have fewer seeds and missing κ values. The main text's bracket does not make the standalone caption correct. Mark per-cell n and interim status, or update the complete figure and caption together once all rows exist. This is a reporting repair, not evidence that the finished γ=30 or ready grids are incomplete.

### 31. The conclusion overstates transfer and absence of a speed effect — important

**p. 9, lines 438–444:**

> “In a ReLU convolutional head over a linear encoder on synthetic data, and with measured tissue spectra in constructed neighbourhoods, we find the corresponding suppression at every width and the corresponding failure under context shift whenever the context is informative, with excess shifted risk of about 0.75 (synthetic) and 0.85 (real spectra) over the uninformative-context comparator.”
>
> “What transfers with a qualification is the role of speed: the head's relative rate controls how much of the spectral cue the encoder learns before the fit, not whether the fitted head uses the shortcut, and where the contextual cue dominates the encoder's own gradient even that control disappears.”

**Restricting files:** both summaries, particularly the ready regime and the nonzero rate effects. The phrase “corresponding suppression” is not a common finding across both real regimes; the fixed-rate whole-head experiment shows a dissociation in effect size, not categorical independence. Use the conclusion proposed in Section III.

### 32. The production paragraph repeats two old scope errors and misnames the contrast — important

**p. 9, lines 456–462:**

> “On a production spectral–spatial segmentation model we measured the initialization curvature of the two parameter blocks and found the scalar ratio inverted (0.46±0.21 at width 192) because the inputs are effectively rank one (effective rank 1.07 of 942), and found its growth with width to be a normalization-gain effect (first-head-convolution slope +0.61 with batch statistics against +0.006 with both head BatchNorm layers at fixed statistics), while the class-contrast direction carries 12–28× less curvature than the dominant input direction (Appendix E).”

**Contradicting/restricting files:** `paper/sections_iclr_v2/appendix_production_v1.tex`, Table 14 and E.1.3; `results/exp1_8d_REPORT.md`. Near-rank-one inputs coexist with the inverted scalar ratio; effective rank alone does not identify its cause. The BN experiment establishes sensitivity and supports a normalization-gain interpretation, with the appendix explicitly declining a unique causal attribution. Most directly, **12–28× concerns the v1-orthogonalized contrast**, whereas the full contrast has ratio **2.6–4.8×, mean 3.7×**. Correct the direction or the number. The following statement that directional curvature does not measure impaired learning is sound.

### Appendix repairs worth doing while reconciling these claims

These are outside the requested main-text ledger but should not be left in the submission:

- Appendix C.2, p. 17, lines 877–879 says population CE≈0.126 implies training loss below≈0.13 cannot be reached by either cue alone. A population optimum is not a finite-sample training-loss lower bound; remove that inference.
- Appendix B, Table 1's caption says every relative discrepancy is at most 1.2e−13, but the same table prints a trajectory-invariant error 2.0e−11. Scope the caption to endpoint solver comparisons. Its `sup_t` gap labels still need the finite horizon, and its two-timescale displacement bound refers to material removed from the new theory; omit that block or supply the precise retained derivation.
- Appendix E, p. 23, lines 1230–1232 again infers “three orders of magnitude” from zero channels below 10ε. The latter does not imply all variances exceed 1000ε. Use the actual minimum ratio if that claim matters.
- Appendix A/B/E still contain old-review instructions and obsolete theorem references; curate them. Preserve limiting evidence, not the history of every earlier framing.

## III. Is the restated claim the right one?

**The distinction is right; “speed decides whether recoverable, not whether used” is too strong.** The evidence supports a difference in how strongly two measured outcomes respond within specified interventions.

I would defend this as the central empirical statement:

> **“At matched training loss in our synthetic spectral–spatial model, slowing the spatial head substantially increases linear accessibility of the spectral cue, while the trained classifier remains strongly context-reliant. Experiments with measured spectra show that this accessibility response depends on initialization readiness and cue construction.”**

A numerical version is stronger than an absolute slogan: “Reducing the head rate by 256 increases probe accuracy by 0.188 on average, but reversal accuracy by only 0.022.” These quantities are measured at the same L* checkpoints and are reproducible across the three seeds. They do not require a claim of no speed effect.

For the theoretical contribution:

> “An exactly solvable serial logistic model gives a finite-fit spectral-update bound, sufficient conditions for reversal failure, and a control showing that replicated-readout width acts through the training metric.”

For the ready real-spectrum result:

> “High spectral probe accuracy can coexist with severe contextual reliance, showing that a deficit in this probe is not necessary for failure.”

For γ=30:

> “With low initial spectral accessibility and the larger constructed context amplitude, slowing the head over the tested range yields only small probe gains relative to training with uninformative context.”

These statements form one paper. They do not require proving all cases or recovering the old general theory. They do require naming the outcomes accurately: adaptation, clean-input probe accessibility, use by the original predictor, and performance after an actual retraining procedure are different measurements.

**Suggested abstract core, leaving final γ=10 numbers out until complete:**

> “We study how relative module training rates affect spectral adaptation in serial spectral–spatial models. An exactly solvable logistic model bounds the spectral update at matched fit and establishes contextual-reversal failure under explicit conditions; readout normalization removes its width dependence but need not prevent failure. In a nonlinear synthetic model, slowing the spatial head increases spectral probe accuracy substantially while the trained classifier remains strongly context-reliant. Experiments using measured tissue spectra with constructed context distinguish high and low initial spectral accessibility and show a weaker rate response at high contextual amplitude. The results separate adaptation of the encoder from contextual reliance of the fitted classifier.”

If direct head retraining succeeds, add its measured consequence. Until then avoid “baked into the encoder,” “unrepairable,” and “what a practitioner recovers exactly.” The clean Gaussian optimal probe is already a precise positive result and should be described as such.

## IV. Registered verdicts: fairness, length and replacement

**Fair intent, incomplete execution.** Keeping failed labels is good. The paragraph is slightly too defensive, and the width-set mismatch makes part of it inaccurate. Keep approximately 100–140 words in the main text, with a compact criterion table in Appendix C. The original registration, the pilot amendment, the added κ prediction, and secondary analyses should be identifiable without reading the project log.

My recomputation at L*=0.30:

| Prediction/criterion | Original or relevant grid | Defensible verdict |
|---|---|---|
| P1, SP alignment and reversal each rank correlation ≤−0.8 in every seed | Original M={8,32,128,512,2048} | Not met; only seed 1 meets both |
| P2, normalized alignment flatness, abs(ρ)<0.5 in at least two seeds | Original five widths | Met for alignment: ρ=+0.10,−0.90,−0.30 |
| P2, analogous reversal flatness | Original five widths | Not met: ρ=−0.90,−0.70,−0.70 |
| P2, more alignment at M=2048 than SP | Paired seeds | Met |
| P3, a_u>0.25 and flat width trend | Amended control widths {2,8,128,512}; original grid not fully run | Both requirements fail on the available grid; original full-grid claim cannot be made |
| P4, alignment decreases with readout multiplier | Actually run multipliers {1/16,1/4,1,4,16} | Met; the original multiplier 64 was dropped |
| P4, reversal accuracy decreases | Same amended multipliers | Opposite sign; not met |
| P5, frozen reversal accuracy below 0.5 | Available amended widths | Met |
| P6, probe/h_u increase as κ decreases | κ={1,1/16,1/256}, three seeds | Met |
| P6, reversal rank criterion in every seed | Same κ grid | Not met: two seeds ρ=−1, third ρ=−0.5; mean reversal accuracy nevertheless improves slightly |

P2's “same quantities” wording is not completely explicit. Report alignment, probe and reversal separately rather than choosing the reading with the preferred verdict. The original five-width normalized probe correlations likewise pass the two-of-three flatness criterion; reversal is the failing component. Seven-width values may be reported as amended-grid results alongside them.

**Suggested main-text paragraph:**

> “The original width-based prediction was not met at L*=0.30. On the original five widths, normalized-readout alignment and probe flatness met the stated two-of-three-seed criterion, but reversal flatness did not; its high-width alignment improvement held. On the amended control grid, both the energy cutoff and flatness requirements failed, although context-random training produced substantially higher probe accuracy than informative-context training. The amended readout-rate sweep supported the alignment prediction but changed reversal accuracy in the opposite direction. Frozen controls remained below chance under reversal. The added whole-head-rate prediction held for probe accessibility and row-space retention; its reversal rank criterion failed in one seed. Low primary-threshold gains and the stronger labelled secondary trend are consistent with early contextual fitting.”

This reports failure without asserting that it falsifies every form of competition or that it somehow counts as the registered success. P3's “substance holds” should become the separate positive comparison, not a reinterpretation of a failed criterion. The κ-extension paragraph should say openly that the extension was motivated by the observed reversal outcome; being fixed before the *new* runs does not make that decision independent of the earlier outcome.

One further correction to the design log: §4.4c reverses a sufficient inequality as if it were an exact reversal threshold. From “κMv_0²>m/(m−2a_0) guarantees failure” one cannot infer that the opposite inequality guarantees recovery. The exact fitting equation determines the boundary. Do not carry that sentence into the paper.

## V. Experiment 3's two regimes and the γ=10 variant

### The two-regime interpretation

**Ready:** retain it as a control demonstrating failure despite high measured spectral accessibility. Replace “nothing to starve/head-level competition only” by “no large probe-accessibility deficit was observed, yet the original classifier remained strongly context-reliant.” This is a useful and supported separation. Encoder parameters do move, and the probe is not a complete description of all useful information. The theorem's a_0≥m/2 case does not apply to a fitted probe on this encoder.

**Unready at γ=30:** retain “limited spectral accessibility compared with the context-random training control, with only a small rate response over the tested κ range.” The mean κ endpoint difference is about +0.013 in probe accuracy, and individual gains are heterogeneous. Avoid “complete suppression” and “insensitive” without those qualifications.

The proposed contextual-amplitude explanation has real support. I computed, from the saved checkpoints,

\[
 R_C=\frac{\|(W_{L^*}-W_0)C^T\|_F^2}{\|W_{L^*}-W_0\|_F^2},
\]

where C has the eight orthonormal constructed-cue directions as rows. Across M=32, three seeds and κ∈{1,1/16,1/256}, R_C ranges **0.999950–0.999972** in the γ=30 unready regime. The corresponding ready range is **0.999898–0.999949**. Files are `results/exp3/enc_unready_sp_M32_s{0,1,2}.npz` and the matching `enc_unready_headlr_M32_h{0.0625,0.00390625}_s{0,1,2}.npz`, with ready counterparts; arrays `W0`, `W_0.3`, `C`. This export is a small, useful addition to the result record, not a new theoretical framework.

What it proves is that **net encoder adaptation is overwhelmingly concentrated in the constructed-cue span**. It is not a pathwise gradient-share measurement, and those cue directions can contain some centre-label information too. A gradient decomposition into centre and neighbour contributions, or projections onto the cue span and its complement at a few matched states, would substantiate the stronger gradient language. At minimum change the sentence to the measured update statement and call amplitude a supported explanation under investigation.

There is also a structural reason not to identify this with the toy's v_0 alone: the real model starts with nonzero random head weights. Freezing or nearly freezing that head does not eliminate the encoder's ability to amplify a contextual signal through those weights. In the toy, zero aggregate contextual readout initially gives `v̇(0)=0`. The empirical model intentionally goes beyond this restriction; it should not be described as a literal κMv_0² instance.

### Should γ=10 stay in the main text?

**Keep a compact, completed γ=10 comparison in the main figure, regardless of its sign. Move its full grid to the appendix.** The current paper explicitly claims an interaction between head rate and cue amplitude; γ=10 is the direct test of that claimed explanation. Its placement should follow that relevance, not whether it produces the preferred result. It does not need a third long regime narrative.

Replace the bracketed interim paragraph with a paired rate-effect comparison at γ=10 versus γ=30, with complete seed counts and controls. Use one small panel or table showing the κ=1 to 1/256 change in probe accuracy and reversal accuracy at L*. Report the within-seed difference of those rate effects across amplitudes. Keep calibration accuracy and initial accessibility visible so that a change in cue quality is not silently treated as a pure speed intervention. Do not use the incomplete γ=10 data to assert a restored mechanism now.

The current γ=10 snapshots suggest larger probe changes, but they do not yet supply the completed paired comparison. They also do not establish that cue dominance is gone: the five M=32 cue-span update projections corresponding to unique matched configurations in my CSV snapshot remain above **0.9996**. Thus a completed positive interaction would show amplitude-dependent accessibility response; it would not alone prove the categorical narrative “γ=30 gradient dominance / γ=10 no gradient dominance.”

If the authors instead remove the amplitude mechanism claim entirely and present γ=30 simply as a limitation of transfer, γ=10 can move wholly to the appendix. Under the current central interpretation, I would keep the compact main-text comparison and finish it before further polishing.

## VI. Acceptance estimate

**Approximately 35% for this draft's present scientific package and execution, with a subjective plausible range of 25–45%.** This is not a calibrated probability or a forecast based on a measured ICLR 2027 base rate. The disclosed incomplete variant does not receive credit as a successful result.

This is an increase from my v1 assessment because there is now a substantial controlled synthetic finding and a coherent paper structure. It is not yet the 45–60% scenario I described for a convincing synthetic intervention *and* a strong, consistently stated real-spectrum extension. The main opportunities are precise claim repair, correct matched-fit/registration reporting, completion of the amplitude interaction, and a small direct recovery experiment. More mathematical generality is not what is currently limiting the paper.
