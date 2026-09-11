# Astra — ICLR reviewer report 03

Reviewed version: **commit `3596a9c`, `paper/main_iclr_v2.pdf`, 40 pages, SHA-256 `70502959a528633b4c0ce6c404fb6b8dbdf32976865ec64704d5a08220844414`**. Page references below are PDF page numbers; line references are the printed review lines. I read the nine main pages before the appendices. I recovered the specified commit because the live workspace had advanced. Results and source references in the substantive review refer to that commit unless explicitly marked as later extensions. No paper files were edited.

**Recommendation: 6/10, weak accept, up from review 02's 5/10.** The recovery experiment addresses the principal gap I identified. The nonlinear-encoder experiment strengthens it, and the public data improve reproducibility. However, the new natural-context interpretation and parts of the replication summary overstate the evidence. These are required corrections to the scientific claims, not reasons to launch another mathematical or experimental programme.

The paper I would defend is: **a solvable example of contextual competition, followed by controlled experiments showing that similar contextual reliance can coexist with substantially different encoder accessibility and recovery after a specified retraining procedure.** The natural-context experiments delimit that explanation. They do not establish that same-class neighbours cannot compete, and they do not refute successful low-dimensional learned compression.

## 1. Reviewer report

### Summary

The paper studies a spectral encoder shared across centre and neighbouring pixels, followed by a contextual head. In an exactly solvable serial cross-entropy model, replicated contextual readouts change the optimization metric, suppress adaptation of a designated spectral coefficient at matched fit, and cause contextual-reversal failure under explicit initialization and width conditions. Readout normalization removes the width dependence without guaranteeing robustness.

Synthetic experiments intervene on the original head's learning rate and distinguish encoder accessibility, original contextual reliance, and recovery by retraining a fresh head from a frozen encoder. Slower original head training substantially improves the latter two representation measurements while leaving the original predictor strongly context-reliant. The ordering survives a two-layer ReLU encoder. Constructed-context experiments using measured breast spectra and a public hyperspectral scene reproduce suppression and reliance more broadly than they reproduce the learning-rate response. Natural-neighbourhood experiments find contextual reliance without the measured encoder deficit, and learned small bottlenecks remain competitive with no bottleneck under the declared comparisons.

### Strengths

1. **The central probe effect now has an operational consequence.** In the linear-encoder synthetic experiment, the three paired slow-minus-fast recovery gains under reversal are **+0.222, +0.180, +0.212**, mean **+0.205**. Randomized-context evaluation gives essentially the same mean gain. The original heads in both arms still fail badly. This closes the main gap in review 02. Locations: p.6, lines 317–323; Appendix C, recovery tables.

2. **The nonlinear encoder is a meaningful extension.** The two-layer ReLU encoder retains suppression, reliance, a positive rate response, and positive paired recovery improvements in all three seeds. Recovery gains are **+0.109, +0.134, +0.082**. The magnitudes differ from the linear case, as they should be allowed to. Locations: p.7, lines 353–359; Appendix C, nonlinear-encoder subsection and tables.

3. **The exact theorem is appropriately narrow and interpretable.** The fixed spectral skip, zero-sum initial readouts, initial spectral coefficient, finite reversal condition, stopping convention and normalization comparison are explicit. The substantive theorem and proof survive this review. The remaining coefficient-versus-whole-encoder error is in the summary prose, not the main theorem. Locations: p.4, lines 170–215; Appendices A/B.

4. **The real-spectrum and public experiments reveal useful limits.** High initial accessibility permits severe original contextual failure with successful subsequent recovery. Low accessibility produces suppression relative to context-random training, but slowing the head is not consistently effective on Pavia. This is informative evidence against treating one scalar rate or one probe as a complete account. Locations: pp.7–8, lines 369–426; Appendix F, especially p.30, lines 1570–1587.

5. **Unsuccessful predictions remain visible.** The original width prediction is not silently replaced by the later rate intervention. The revised registered-verdict paragraph in §4 is fair at its present length. Natural-context PCA insensitivity fails at the reported endpoints, and the public gate fails. Reporting these is a strength provided the interpretation does not turn failures into universal confirmations. Locations: p.6, lines 287–298; Appendices C/E/F.

### Ranked weaknesses and actions

**W1 — Major: the strongest result is still visually subordinate to the older width story.**

Locations: p.2, lines 54–99; Figure 3, p.7, lines 324–350; Figure 4, p.9, lines 432–461.

Gradient starvation, preference for accessible features, and classifier retraining are established ideas. The distinctive evidence here is the effect of the **original relative module learning rate on later recovery from a shared encoder, at matched original fit**, while original reliance changes little. Neither main experimental figure shows recovery. Meanwhile the introduction has acquired a dense compression dispute and the appendices now contain several additional experimental narratives.

**Action:** keep the one-theorem structure and make recovery part of the main visual argument. State the distinction from last-layer retraining precisely: these experiments initialize and train the entire nonlinear head. Do not add mathematics to compensate for positioning. The relevant last-layer precedent is [Kirichenko, Izmailov and Wilson, ICLR 2023](https://arxiv.org/abs/2204.02937).

**W2 — Major claim correction: the natural-context null is being turned into a causal exclusion.**

Locations: p.2, lines 75–76 and 90–95; p.8, lines 424–426; p.9, lines 476–485; Appendix E, pp.25–26.

Comparable probe scores and comparator-level recovery support **no detected accessibility deficit under these measurements and protocols**. They do not support “no competition arises,” “learns exactly,” or the explanation that a same-class neighbourhood cannot be a competing cue. The theorem's training cues already carry the same label. Label agreement does not establish identical statistical information, optimization accessibility, or gradients. Multiple noisy measurements of one class can provide distinct predictive routes.

The new final sentence also overreaches: it claims a general conditional rule about what can be recovered and says that “only retraining the encoder” works otherwise. The comparator was jointly trained with uninformative context; that is not a demonstrated necessity result for repairing an existing encoder.

**Action:** use the replacement in §2 below. Present natural context as a boundary of the observed suppression result. Retain its legitimate spatial benefit and its contextual reliance. Do not infer absence of competition from annotation homogeneity.

**W3 — Major positioning correction: the compression experiment largely agrees with the cited small-feature result.**

Locations: p.2, lines 88–95; Appendix E, P11 discussion, p.26, lines 1384–1395.

Learned-12 versus no-bottleneck performance is within the registered tolerances. That supports empirical compressibility in this model/task. PCA-16 performing worse shows that the chosen projection matters; it does not show that more than 12 learned features are necessary, that all input bands are necessary, or that the prior paper's learned bottleneck was hiding starvation. The contrast with PCA is itself discussed in O'Leary's full paper. See §3 for the precise positioning and citations.

**Action:** frame this as separating model reliance, learned compression, and the choice of spectral subspace. Do not present Appendix E as a rebuttal of O'Leary's small learned-feature finding. One short paragraph is enough; this should not become a second main contribution competing with recovery.

**W4 — Moderate: the public replication supports some components, not a general rate-to-recovery mechanism.**

Locations: p.2, lines 69–76; p.8, lines 419–426; Appendix F, p.30, lines 1570–1614.

Pavia reproduces suppression relative to the context-random comparator and severe original reliance. At gamma=30, slowing the head improves the probe in one seed and worsens it in two; recovery has the same mixed signs. At gamma=10, one substantial positive recovery effect coexists with one negative effect and one negligible positive effect. The high-accessibility condition recovers well from all sampled encoder histories, but a recovery rate effect there does not establish mediation through the probe.

**Action:** give a two-sentence component-wise replication summary in the main text, including the failed rate/recovery generalization. The public constructed-context result should not be hidden while the main text mentions only its natural-context outcome. Preserve every registered failure; describe partial directional patterns separately from passing a registered criterion.

**W5 — Moderate: recovery is procedure-dependent, and the natural-context comparator is limited.**

Locations: p.6, lines 276–278 and 317–323; p.8, lines 413–417; p.9, lines 483–485; Appendix F, p.30, lines 1598–1614.

The recovery protocol is a fair operational intervention, but its common budget does not equalize optimization difficulty across representations. All original E2/E3 recovery runs and the nonlinear synthetic recovery runs use the full 20,000 steps. A weak score is not an information-theoretic ceiling. On Pavia natural context, recovery to approximately 0.629 matches the within-family comparator, approximately 0.627, but that comparator is far below the per-pixel MLP's approximately 0.911. This limits how much “head-level” localization can establish.

**Action:** retain “specified procedure and budget”; distinguish the comparator's original classifier from its retrained head; use “near random” where justified, not literal equivalence. No new training programme is needed for the operational claim. A compact budget/learning-rate sensitivity check could strengthen it, but it is optional, not a new acceptance requirement.

**W6 — Moderate for reproducibility: some registration and integration claims do not match the actual records.**

Locations: p.6, lines 276–278; p.7, lines 356–359; p.9, lines 476–485; Appendix E, p.26, lines 1350–1382; Appendix F, p.30.

The written Exp4 fallback selects a common reached loss threshold and reports endpoints separately. It does not specify that endpoints become primary if no threshold is jointly reached. The endpoint comparison was planned, but calling its promotion the “registered fallback” is inaccurate. “Every verdict” is also false when the second P8 clause fails on the denoised copy and on Pavia. Finally, the main text still says every encoder is linear, and the nonlinear paragraph leaves a numerical range attached to an ambiguous metric.

**Action:** correct these bounded inconsistencies, using §6 and §7. Do not reopen the experiments or retrospectively rewrite the original registration.

### Questions for the authors

1. Will you adopt a core claim about the measured separation of reliance, accessibility and procedural recovery, with rate responsiveness explicitly conditional on the tested setting?
2. Will the main text say that the public constructed-context study reproduces suppression/reliance but not a consistent low-accessibility rate/recovery response?
3. Will you distinguish a planned endpoint analysis from the post-outcome decision to make it the primary Exp4 comparison, and keep the original threshold rule visible?
4. Can each recovery table identify original encoder checkpoint, evaluation side, fresh whole-head protocol, final-state stopping rule, and whether its comparator is original or retrained? Most of this is in code already.
5. Will the compression paragraph acknowledge that a useful low-dimensional learned representation is compatible with these results, rather than suggesting high spectral dimensionality has been established?

### Score and justification

**6/10 — weak accept. Confidence: 4/5 on the inspected mathematics and reported arithmetic; lower on exhaustive novelty and eventual reviewer preferences.** This is my stated review scale, not a claim about the eventual ICLR 2027 review form.

The operational recovery result and nonlinear extension justify the increase from review 02. Public replication removes an important reproducibility weakness. The natural results do not invalidate the existence/mechanism contribution, but they limit its explanation of natural spatial dominance. I am below a strong accept because the theoretical ingredients are familiar, the clearest empirical effects use designed contextual cues, the original width prediction fails, and the rate intervention generalizes incompletely.

I would defend acceptance of the scoped scientific contribution, with the claim corrections below required. I would not defend the literal “no competition,” universal repair prescription, or claimed rebuttal of low-dimensional learned sufficiency. A reviewer who centres those sentences could reasonably remain at 5.

### The single change most likely to move the score

**Replace Figure 3(d)'s single-seed trajectory with the paired recovery experiment, including random, fast-head and slow-head encoders.** Show individual seeds, original versus retrained reversal accuracy, and the mean paired gain; use separate small facets for linear and ReLU encoders. Move the trajectory to Appendix C.

The caption should specify: original encoders selected at L*=0.30; frozen thereafter; fresh entire head; paired initialization and common recovery data; 20,000 GD steps. This exposes the strongest new result without adding a page or an experiment. The remaining replication and natural-context corrections are necessary accuracy fixes, not invitations to expand the paper.

## 2. Check (a): what the natural-context experiment establishes

**Its measured result is useful; its present causal explanation is not established.**

Recomputed from `results/exp4_summary.csv`, `results/exp4_pca23_summary.csv`, `results/exp4_paviau_summary.csv` and the corresponding `recovery_summary.csv` files. Entries are three-seed validation macro-F1 means at the reported final states, not globally matched-loss comparisons.

| Dataset/copy | Natural vs shuffled encoder probe | Natural encoder + fresh head | Shuffled encoder + fresh head | Random encoder + fresh head | Original shuffled classifier, context-random |
|---|---:|---:|---:|---:|---:|
| Tissue, non-denoised | .672 / .670 | .681 | .688 | .494 | .680 |
| Tissue, PCA-23-denoised | .671 / .676 | .706 | .720 | .503 | .722 |
| Pavia, natural | .732 / .722 | .629 | .615 | .559 | .627 |

These support the registered **within-family head-level reading**: the frozen natural-context encoder supports comparator-level performance under the specified fresh-head procedure. The result is particularly convincing as a distinction from random initialization on tissue. It does not prove equal learned features, equal attainable performance, or that the encoder has retained all task-relevant information. The weak Pavia comparator makes that qualification material.

The annotation statistic is also narrower than the prose. In the sampled tissue training patches, no *labelled* neighbour disagrees with its centre; approximately 88% of neighbours share its label and 12% are unlabelled. Only approximately 61.5% of patches have all eight neighbours explicitly labelled as the centre class. Pavia has a similar missing-label issue. Source: `results/exp4/neighbourhood_stats.json` and `results/exp4_paviau/neighbourhood_stats.json`. Say **no observed cross-class label disagreement**; do not turn unknown labels into established homogeneity.

More fundamentally, even perfectly known label agreement would not establish noncompetition. The theorem's training spectral and contextual cues both predict Y. Its asymmetry is in their parameterization and accessibility. Natural neighbours may supply additional noisy measurements, different nuisance variation, or easier aggregation. The present experiment does not identify which of these explains its null accessibility comparison.

**Replacement for §5/§6:**

> In the sampled natural 3×3 neighbourhoods, informative-context and shuffled-context training produced similar encoder probe scores, and fresh-head retraining from the natural-context encoder reached the within-family comparator's level. Thus these measurements found contextual reliance without the encoder-accessibility deficit observed in our constructed low-accessibility settings. They do not identify why suppression was absent or establish that natural context cannot produce it.

Add in Appendix E:

> No labelled neighbour disagreed with its centre, although some neighbours were unlabelled. This annotation pattern describes the sampled setting; it does not by itself imply that neighbouring measurements contain no additional class information or cannot compete during optimization.

**Does this undercut the paper more than the draft admits?** Yes, if the claim is that the proposed competition explains natural histology's spatial dominance, or explains O'Leary's dimensionality findings. The directly tested natural setting does not show that proposed encoder deficit. No, if the claim is a controlled existence result with experimentally established manifestations and limits. Keep the negative result visible; do not rationalize it away as mathematically inevitable. Larger natural receptive fields and other tasks remain untested, not predicted confirmations.

P7 failing at endpoint also matters: natural spatial models do lose performance after PCA-16 here. Therefore the intended joint pattern of an insensitive spatial model and a sensitive per-pixel comparator was not established. P11 supports a different observation—small **learned** compression can work—which should remain distinct.

## 3. Check (b): compressibility, redundancy, and the prior papers

**The present sentence is not the strongest defensible positioning. It mixes distinct questions and implies a refutation that your evidence does not supply.**

I checked the full accepted O'Leary manuscript, beyond its abstract. Its bottleneck is learned for neural models and PCA-based for RF/SVM. Its spectrum-only MLP also retains performance with a small learned bottleneck; its results explicitly distinguish learned projections from leading PCs. Consequently spatial competition cannot explain away the spectrum-only finding, and “learned compression differs from PCA” is not a new correction to that paper. Your P11 broadly agrees with its empirical low-dimensional result. [O'Leary et al., accepted manuscript, methods and results pp.9–13](https://pure.manchester.ac.uk/ws/files/1773921257/final_manuscript.pdf).

Müller et al. compare compression methods for a particular colon-tissue CNN task and interpret the small performance differences together with spatial sensitivity. They discuss task/model limits and better exploitation of spectra. Do not attribute to that survey an information-theoretic claim that spectra are universally useless. [Müller et al., Analyst 2023](https://pubs.rsc.org/en/content/articlehtml/2023/an/d3an00166k).

**The strongest sentence against the inference from spatial reliance is:**

> Spatial reliance of a fitted predictor does not, by itself, establish spectral redundancy: at the same bottleneck dimension, our controlled rate interventions produce similarly context-reliant predictors whose frozen encoders support substantially different shifted accuracy after the same specified head-retraining procedure.

This challenges an inference from predictor behaviour. It does **not** challenge the existence of a sufficient small learned representation.

For the natural compression result, I would write:

> On the tested tissue split, a learned twelve-dimensional bottleneck performs within the registered tolerance of the no-bottleneck model, whereas replacing its spectral input by the leading sixteen PCs reduces performance; this demonstrates dependence on the chosen representation, not a requirement for high-dimensional encoder output. The corresponding per-pixel PCA gate does not pass on Pavia.

If the phrase “compressibility is not redundancy” is retained, define the latter as **dispensability of useful spectral information**, not redundancy of input coordinates relative to an adequate learned statistic. Otherwise the phrase is ambiguous: successful task compression is perfectly compatible with substantial dimensional redundancy.

Do not claim that all 942 channels are needed, that more than 12 learned features are needed, that choosing a few wavelengths is equivalent to selecting a few learned features, or that the intrinsic task dimension has been measured. The 942-vector also includes deterministic spectral derivatives; it is not 942 independent physical measurements. Your observations do not overturn the cited task-specific compression result or explain its nonspatial classifiers.

For the intro, replace the long dispute with two short sentences distinguishing these questions. Keep the dimensionality tables in Appendix E. The paper's strongest contribution is still the rate/recovery intervention.

## 4. Check (c): what “reproducible” can mean after Pavia

**Yes:** the controlled suppression and reliance pattern reproduces on a public scene using the provided construction and training procedure. **No:** the full rate-to-accessibility-to-recovery account does not reproduce uniformly. Prefer **“public-data replication of suppression and reliance, with mixed rate and recovery effects.”** Publicly rerunnable code and cross-dataset persistence of every prediction are different claims.

All 78 constructed-context Pavia runs reach L*=0.30. Standard-parameterization low-accessibility arms leave little probe gain, whereas the context-random comparator gains approximately .22–.24 and achieves much better reversal performance. That is substantial positive evidence. The following failed directions must remain equally explicit.

| Pavia regime | Slow − fast probe, seeds 0/1/2 | Slow − fast recovery reversal, seeds 0/1/2 |
|---|---|---|
| Low accessibility, gamma=30 | +.0234 / −.0087 / −.0060 | +.0124 / −.0214 / −.0187 |
| Low accessibility, gamma=10 | +.0726 / −.0017 / +.0053 | +.0608 / −.0318 / +.0007 |
| High accessibility | −.0013 / +.0057 / +.0077 | +.0144 / +.0201 / +.0274 |

Sources: `results/exp3_paviau_summary.csv`, `results/exp3_paviau/recovery_summary.csv`.

The gamma=10 recovery description should be “one substantial positive effect, one negative effect, one negligible positive effect.” There are technically two positive signs, so “one positive seed” needs a declared materiality threshold; do not invent one after seeing outcomes. Passing a subset of a required all-seed criterion is not passing that criterion.

High-accessibility recovery means are .885 from random initialization, .887 from fast-head training and .907 from slow-head training. The probe response does not have the same three-seed sign pattern. Thus a claim that accessibility, as measured here, **governs** the recovery effect is stronger than the intervention identifies. The rate can affect other properties of a representation or its subsequent optimization.

**Main-text replacement:**

> On a public Pavia University scene, the constructed-context experiment reproduces suppression relative to context-random training and severe contextual reliance. The low-accessibility learning-rate and recovery effects are inconsistent across seeds, delimiting their generalization beyond the synthetic and breast-tissue settings.

Keep gamma=10 in the main real-spectrum account because it informs the mechanism's scope. Its full seed table belongs in the appendix. Pavia does not require another main figure; a brief honest summary is sufficient.

Two qualifications belong in Appendix F. The split avoids overlapping 3×3 train/test patches but is a tile split within one scene, not an independent-scene replication. Also the binary Meadows/Trees evaluation is imbalanced: the validation majority baseline is approximately .784 and test approximately .838. Label accuracy and its baseline explicitly; “near chance” is not a universal .5 reference for this evaluation. Paired differences remain informative. A balanced/per-class metric from saved evaluations would help interpretation without demanding another training grid.

## 5. Recovery check retained from the preceding request

**The requested operational recovery experiment is now satisfied.** I am not moving the goalposts to require optimal decoding, a universal repair theorem, or a new natural-context mitigation.

Code inspection of `exp2_recovery.py` and `exp3_recovery.py` supports freezing encoder parameters, paired fresh-head initialization, common recovery samples within seed, original L* checkpoints and final-state evaluation. I checked equality of the original linear W0 checkpoints for the three synthetic and nine tissue fast/slow pairs. I recomputed the summaries; I did not rerun training.

Synthetic recovery uses newly generated context-random images. Tissue recovery redraws patches around the same training centres: fresh patch construction, not a new cohort. All 9 original synthetic and 27 tissue recovery runs use the 20,000-step budget. The nonlinear synthetic extension also uses the budget throughout. This establishes what this procedure obtains from each representation, not the best possible performance from it.

| Setting, three original seeds | Random encoder recovery | Fast encoder recovery | Slow encoder recovery | Mean slow − fast |
|---|---:|---:|---:|---:|
| Synthetic, linear encoder | .5740 | .5827 | .7872 | **+.2045** |
| Synthetic, ReLU encoder | .5798 | .5685 | .6767 | **+.1081** |
| Tissue, high accessibility | .9223 | .9417 | .9419 | +.0002 |
| Tissue, low accessibility, gamma=30 | .5048 | .5187 | .5257 | +.0070 |
| Tissue, low accessibility, gamma=10 | .5065 | .5611 | .6399 | **+.0789** |

All entries are reversal accuracy; real-data entries use validation patients. Sources: `results/exp2/recovery_summary.csv`, `results/exp2/recovery_summary_nl_init_nl_kappa1_nl_kappa256.csv`, `results/exp3/recovery_summary.csv`.

At tissue gamma=10, the paired test-patient gains are also positive in every original seed, mean +.0977. This supports keeping the finding. The gamma=30 null is procedural: the probes retain some predictive information even when the specified head training performs poorly.

The causal statement is a **total effect of the original rate intervention at a common attained training loss**, including its change in elapsed encoder-training steps. It is not a direct rate effect at equal training time, or a proof that the probe uniquely mediates recovery. The present scope sentence at p.6, lines 321–323 makes this largely clear and should be preserved.

## 6. Fresh main-text sentence ledger

This ledger distinguishes false numerical/general statements from unsupported causal scope and minor ambiguity. Quoted text is from the reviewed PDF, with mathematical typography standardized. Where only the offending clause is quoted, its sentence location is given. Files below contain the contradicting observation or the narrower proved/tested statement; absence of a stronger test is not presented as proof of the opposite.

### L1 — Required mathematical correction: coefficient versus whole encoder

**p.1, lines 15–20:**

> “An exactly solvable serial logistic model bounds the spectral encoder's update at matched fit by a fraction 1/(1+Mv₀²) of the update a spectral-only model needs, establishes contextual-reversal failure under explicit finite-width conditions, and shows through a M⁻¹ᐟ²-normalized readout that replicated-readout width acts through the training metric; normalization removes the width dependence but need not prevent the failure.”

`appendix_serial_proofs.tex` bounds the **spectral coefficient** by that fraction. The whole encoder has a different bound. In `results/toy_serial_ce.csv`, a0=0, v0=1, M=16, m=log(19) gives full displacement **.219021**, greater than m/17=**.173202**, while the spectral coefficient update **.142325** obeys the theorem. Replace with “update of the spectral coefficient.”

Apply the same precision to these summary sentences:

- **p.1, lines 46–49:** “We prove one instance exactly: in a linear-encoder model with a replicated contextual readout, a faster contextual pathway suppresses the encoder's update at matched fit, the fitted model fails under contextual reversal under explicit finite-width conditions, and readout normalization removes the width dependence (Section 3).”
- **pp.8–9, lines 431 and 465–466:** “We prove, in a serial linear-encoder model, that a contextual pathway trained faster than the spectral encoder suppresses the encoder's update at matched fit, that the fitted model fails under contextual reversal under explicit finite-width conditions, and that readout normalization removes the width dependence without necessarily removing the failure.”

Those two are ambiguous summaries rather than independently false whole-displacement theorems. Naming the coefficient resolves them.

### L2 — Required numerical qualification: fast is near random, not no better

Four occurrences:

- **p.1, lines 24–26:** “from the fast-head ones, no better than random encoders.”
- **p.2, lines 66–68:** “Retraining the head on decorrelated context recovers shifted accuracy of 0.75–0.84 after the slow head and 0.54–0.62 after the fast one, no more than from random encoders.”
- **p.6, lines 317–319:** “From the κ=1 encoders the retrained head reaches accuracy 0.62, 0.59, 0.54 under reversal (0.61, 0.59, 0.55 with context-random, 0.59 iid), no better than the same procedure on the random initial encoders (0.60, 0.59, 0.53)”.
- **p.9, lines 468–470:** “retraining a fresh head on decorrelated context recovers about 0.2 more shifted accuracy after the slow head than after the fast one, which yields no more than a random encoder.”

`results/exp2/recovery_summary.csv`: fast-minus-random reversal gains are +.0189, +.0013, +.0059, mean +.0087. These are small compared with +.2045, but not nonpositive, nor an equivalence test. Use **“near the random-encoder baseline”**, optionally giving its .53–.60 range.

### L3 — Required scope correction: natural context does not establish no competition

**p.2, lines 75–76:**

> “With natural neighbourhoods, on tissue and on a public hyperspectral scene, no competition arises (Appendices E and F).”

**p.8, lines 424–426:**

> “The construction matters: with the natural 3×3 neighbourhoods of the same cores, which never straddle two classes, the encoder learns the class signal equally with and without informative context and the model uses the full spectrum, on tissue and on a public hyperspectral scene (Appendices E and F).”

**p.9, lines 477–478:**

> “The cues that show the mechanism are constructed; natural 3×3 neighbourhoods, being the same class as their centre, did not produce it, and larger receptive fields are untested.”

Restricting files: `results/exp4*_summary.csv`, corresponding recovery and neighbourhood-statistics files; `appendix_serial_proofs.tex`. They support the measured null in §2 of this review, not causal impossibility or exactly equal learning. “Uses the full spectrum” is especially misleading on Pavia, where the per-pixel gate fails. Neither experiment tests necessity of every band. Say **“larger natural receptive fields”**: E2 already uses an effective 5×5 field.

### L4 — Required positioning/numerical qualification: the dense compression sentence

**p.2, lines 90–95:**

> “Appendix E tests that reading with natural neighbourhoods: a learned twelve-dimensional bottleneck loses nothing, sixteen principal components lose 0.07–0.11 macro-F1 per pixel and 0.04 for the spatial model, and the jointly trained encoder learns the class signal exactly as it does without informative context; compressibility is not redundancy, and a natural 3×3 neighbourhood, being the same class as its centre, is not the competing cue of the theorem.”

Several different restrictions apply:

- **“loses nothing”:** non-denoised validation .732 versus .744; denoised .727 versus .744. P11 is a declared tolerance comparison, not exact equality. Source: `exp4_summary.csv`, `exp4_pca23_summary.csv`, plan §8.5.
- **Unlabelled metric side:** the non-denoised spatial PCA gap is **.0747 validation**, **.0411 test**; the sentence's .04 uses the latter. The denoised gaps are **.0438 validation**, **.1994 test**. Label copy and split rather than mixing summary conventions. Sources: those CSVs and `results/exp4/pixel_sweep.csv`.
- **“exactly”:** similar scalar probes/recovery do not identify equal representations. Sources: the corresponding probe/recovery summaries.
- **Same-class exclusion and redundancy:** neither the theorem nor the empirical gate licenses these deductions. Pavia additionally fails the per-pixel PCA gate. See §§2–3 above.

Replace the whole sentence with the shorter scoped paragraph in §3. Its preceding sentence about O'Leary's small-feature interpretation should acknowledge that the evidence includes nonspatial models; do not imply their conclusion came solely from spatial reliance. External restricting source: the full O'Leary manuscript cited in §3.

### L5 — Required: the new universal repair statement is not an experimental result

**p.9, lines 483–485:**

> “We test no mitigation on natural context; the supported statement is conditional: where the spectral cue is accessible at the encoder output, retraining the readout on context-decorrelated data recovered it; where it is not, only retraining the encoder without informative context did.”

`exp2_recovery.py`, `exp3_recovery.py` and `exp4_train.py` test specified fresh whole-head training. The `ctxfree`/`shuf` arms in the summary files are joint training with uninformative context, not a necessary repair operation on an existing fitted encoder. No test establishes the word **“only”**, a binary accessibility threshold, or optimal recoverability. The low-accessibility probes are not zero-information representations.

Replace with:

> In the tested high-accessibility settings, fresh-head retraining recovered useful shifted performance despite severe original reliance. In the constructed low-accessibility settings, joint training with uninformative context produced stronger spectral accessibility; recovery from informative-context encoders depended on the setting and the specified retraining procedure. These comparisons do not identify a universally sufficient or necessary repair.

### L6 — Required integration fix: the encoder is no longer always linear

**p.2, lines 106–107:**

> “The encoder is linear, fθ(x)=Wx with W∈Rᴷˣˢ, in the theorem and in the experiments; the spatial model is a replicated readout in the theorem and a ReLU convolutional head in the experiments.”

**p.9, lines 476–477:**

> “The theorem has a linear encoder, noiseless cues and a fixed spectral skip coefficient; the experiments add noisy cues and jointly learned routing but keep a linear encoder and one head family each.”

`appendix_exp2_nlenc_tables.tex`, `results/exp2_summary.csv` and the nonlinear recovery CSV contradict “in the experiments”/“keep a linear encoder” as universal statements. Add **“in the primary experiments, with a two-layer ReLU encoder extension in §4.”** Describe the real patch head separately from the synthetic convolutional head when defining the model family.

### L7 — Required metric clarification: nonlinear probe gain is not reversal accuracy

**p.7, lines 356–359:**

> “A two-layer ReLU encoder in place of the linear one (registered before its runs) leaves the pattern unchanged: probe gain 0.00–0.02 against 0.26–0.29 without informative context, reversal at most 0.13, and 0.14–0.21 with the head slowed by 256×, in every seed (Appendix C).”

In `results/exp2_summary.csv`, the **.14–.21 range is probe gain** under the slow-head intervention. Grammatically it follows “reversal” without naming a new metric. Write **“while slowing the head by 256× raises probe gain to .14–.21.”** Scope “reversal at most .13” to the stated standard-parameterization nonlinear arms; it is not the maximum over frozen and slow-head variants. Say the qualitative pattern survives, with recovery values .65–.70 versus approximately .57, not the linear encoder's values.

### L8 — Required provenance clarification: “any shifted result” is too broad

**p.6, lines 276–278:**

> “A recovery procedure, fixed before any shifted result was inspected, freezes the encoder at its L* checkpoint, trains a fresh paired-initialization head on new context-random images for 20,000 steps, and evaluates it on the same paired test family.”

The plan's earlier experiment results and p.7, lines 355–356 explicitly place recovery design after original shifted results. Change to **“fixed before inspecting its recovery evaluation results.”** The supplied timestamps document the intended order; I have not independently established an external registration history.

### L9 — Required protocol distinction: “such retraining” has the wrong antecedent

**p.2, lines 84–88:**

> “That core features can be present in a representation while a spurious feature is used, and that last-layer retraining can then restore robustness, is the observation of Kirichenko et al. (2023); our addition is that the relative training rate of the modules at the original fit changes what such retraining can recover, measured at matched fit against a control.”

`exp2_recovery.py` and `exp3_recovery.py` optimize all parameters of a fresh nonlinear head, not only its final layer. Replace the final clause with **“we test how the original relative module rate changes recovery by retraining the entire nonlinear head from a frozen encoder, comparing original encoders at matched fit.”**

### L10 — Required qualification: successful recovery does not prove literal nonuse

**p.2, lines 72–74:**

> “Where the spectral cue is accessible, retraining the head recovers 0.90–0.96 shifted accuracy from every encoder, a random one included: the failure was in the use of the spectrum, not its availability.”

**p.8, lines 413–416:**

> “With high accessibility a fresh head retrained on decorrelated context reaches reversal accuracy 0.90–0.96 from every encoder, the random initialization included, against 0.08 for the original classifiers: the spectral information was available and unused.”

`results/exp3_summary.csv` and `results/exp3/recovery_summary.csv` support availability in the sampled encoders and strong original contextual reliance. They do not establish zero spectral contribution or preservation of all information. Prefer **“the original failure coexisted with spectral information that the specified retraining could use.”** Restrict the numerical range to the original three breast-tissue seeds.

### L11 — Minor but important to scope: the gamma=30 range is procedural

**p.8, lines 416–417:**

> “At γ=30 no encoder yields more than 0.45–0.56; at γ=10 the slow-head encoders yield 0.55–0.70 against 0.48–0.60 after the fast head.”

`results/exp3/recovery_summary.csv` samples nine encoder histories per regime under one recovery budget. Say **“Under this procedure, the nine sampled γ=30 encoders yield validation reversal accuracy .45–.56.”** For gamma=10, the paired gains are more informative than pooled ranges. Do not extend the sentence to the later seeds without updating its range.

### L12 — Minor scope correction: label-blind sampling is not exact label independence

**p.8, lines 378–383**, sentence beginning:

> “Each patch is a real centre with eight real donor spectra drawn independently of the centre label from the same split side”.

`code/experiments/exp3_train.py`, `sample_donors`, applies same-core resampling. The donor distribution can therefore depend on the centre core, which can correlate with label. Replace **“independently of”** with **“without conditioning on”**, and retain the sampler details in Appendix D. This is a precision correction, not evidence that this residual dependence explains the reported effect.

### L13 — Minor scope correction: spectra remain the inputs to context

**p.1, lines 40–42:**

> “Such comparisons do not settle whether joint training starves the encoder: the spatial stage can predict the label from the neighbourhood, and a model that learns to do so may never need the spectrum.”

`02_model.tex` and the data generators route every neighbour's spectrum through the encoder. Replace the ending with **“may have little incentive to improve access to the target's centre-spectrum cue.”** This is essential if “redundancy” is part of the motivation.

### L14 — Minor architectural scope: not all histology classifiers are serial

**p.1, lines 35–38:**

> “Infrared and quantum-cascade-laser histology classify tissue pixel by pixel from a spectrum of several hundred channels, and the models that do so are serial: a small encoder compresses each pixel's spectrum, and a much larger spatial network reads the encoded neighbourhood (Berisha et al., 2019; O'Leary et al., 2026).”

The cited O'Leary paper includes spectrum-only classifiers. Use **“one family of models used for this task is serial.”**

### L15 — Minor bridge correction: low accessibility is a selected regime, not every experiment

**p.3, lines 133–140:**

> “The spectral cue lies in the pixel's own spectrum along a direction the encoder must discover: a randomly initialized encoder attenuates it.”

> “The theorem allows any (a₀,v₀); the experiments select its regime of interest, a small initial spectral contribution and an initially accessible contextual one, and measure that accessibility asymmetry directly (Section 4).”

`results/exp3_summary.csv`, the readiness scans and `exp3_paviau_summary.csv` include high-accessibility cases. A probe is also not the initial contribution to a random head's margin, as §5 correctly says. Qualify **“in the constructed low-accessibility settings”** and **“an empirical accessibility analogue, rather than a measurement of the theorem's initial margin coefficients.”**

### L16 — Required causal qualification if retained: normalization sensitivity is not an identified explanation

**p.2, lines 97–98**, clause:

> “we measure learning outcomes rather than curvature, whose width dependence is normalization-driven”.

`appendix_production_v1.tex` limits the production normalization intervention and scalar-curvature diagnostic. Removing/changing BN changes the operating network and does not uniquely identify a causal decomposition of width dependence. Use **“whose observed width trend in the production model is sensitive to normalization (Appendix G).”** Do not make this a universal statement about curvature; the serial readout-normalization theorem is a separate, exact result.

### L17 — Scope clarification after adding Pavia: the amplitude response is breast-specific

**p.2, lines 69–72**, contribution clause:

> “with a rate response that grows as the cue amplitude falls”.

**p.9, lines 469–471:**

> “With measured spectra, high spectral accessibility coexists with severe contextual reliance, so a deficit in this probe is not necessary for failure; with low initial accessibility the head's rate matters more at the smaller constructed amplitude.”

The breast comparison at gamma=10 versus 30 supports its scoped average/paired reading. `results/exp3_paviau_summary.csv` and its recovery summary do not support a universal monotone amplitude rule. Say **“on the breast-tissue contrast, the measured rate response is larger at γ=10 than at γ=30”**, and report the public mixed response separately. This is not a demand to weaken the supported breast result.

I found no new error in Theorem 1's three formal conclusions, and no reason to retract the main synthetic recovery effect, the qualitative nonlinear extension, or the registered-verdict paragraph at lines 287–298. The ledger corrects their scope rather than replacing the scientific result.

## 7. Appendix and registration corrections needed for a consistent paper

These are separate from the main-sentence ledger because several new interpretation errors live only in the appendices.

1. **Exp4 fallback provenance — Appendix E, p.26, lines 1350–1354.** “The registered fallback applies and the primary comparison is at budget end” does not match plan §8.2, which chooses the highest jointly reached threshold in {.30,.40,.50} and reports endpoints separately. No such global threshold was reached. Write that the planned endpoint analysis is reported because the registered global matched-fit comparison was unavailable; if it was subsequently promoted to primary, label that promotion as an amendment. Leave the original registration intact. Endpoints also include the declared .10 loss stop, so not every arm has exactly 40,000 steps.

2. **Exp4 verdict concordance — Appendix E, p.26, lines 1374–1382.** “Agrees ... on every verdict” contradicts the failed second P8 clause on the denoised copy: the shuffled comparator is .722 versus the per-pixel MLP's .804, missing the .03 criterion. Report P8 clause-wise. P7 fails at endpoint on both copies; P10's combined insensitive-spatial/sensitive-comparator reading is not established. A numerical subcriterion passing is not the combined interpretation passing.

3. **Exp4 exact equality/causal exclusion — Appendix E, pp.25–26.** Replace “therefore a redundant copy,” “learns ... exactly,” “no competition for the head's rate to relieve,” and “answered ... in the negative” with the measured head-level result in §2. A model can use neighbours heavily while its encoder still preserves useful centre information. Do not identify these as opposing statements about all spectral use.

4. **Exp4 stale numbers and checkpoint mixing.** The primary slow-head endpoint probe is **.6276**, not **.619** (`exp4_summary.csv`; table already approximately .628). In the denoised recovery paragraph, **.706–.712** spans means for the final and L*=.30 checkpoints, not a seed range at one checkpoint. Label them, or give final .706 versus retrained-shuffled .720. The validation and test differences between these checkpoints are not interchangeable.

5. **P11 interpretation — Appendix E, p.26, lines 1384–1395.** “Equality is not a fitting artefact” is too strong merely because the no-bottleneck arm reaches its stopping loss faster. Say the registered finite-budget tolerance holds. “As any four-class discriminant is” must refer to a *linear discriminant's score-contrast space*, not a theorem that all four-class tasks have a low-dimensional linear sufficient statistic. Dense learned weights do not prove every input coordinate is necessary. The denoised test comparison differs by +.037, so do not imply the validation equivalence tolerance passed identically on test.

6. **Pavia all-arm ranges — Appendix F, p.30, lines 1570–1576.** The reported approximately zero gains and 15–420 fitting steps describe standard-parameterization arm means. They do not describe “every informative-context arm”: gamma=10 slow-head mean probe gain is **.0292**, with mean fitting time **1,901** steps. Restrict the range to the intended arms and say when numbers are seed means.

7. **Pavia recovery and mechanism summary — Appendix F, p.30, lines 1580–1587.** Replace “no encoder yields more than the random one” with the measured group means and mixed paired signs. Replace “rate response and the recoverability it governs replicate” with the component-wise result in §4. These data do not establish probe mediation or complete low-accessibility recovery replication.

8. **Pavia natural verdicts — Appendix F, p.30, lines 1590–1614.** “Matches ... every registered item” is false: G1 fails, the second P8 clause fails, and compression arms are not run under the gate rule. Do not label unrun P7/P10 as failures. “Sixteen PCs carry essentially all ... information” should be **“the tested classifiers did not meet the registered ≥.02 improvement criterion on both evaluation sides.”** Their best MLP gains are .0196 validation and .0085 test; this is a performance result, not an information bound.

9. **Pavia comparators and baselines.** The natural recovery .629 is compared in the text with the **original** shuffled model's .627; the freshly retrained shuffled encoder gives .615. Both comparisons are legitimate if labelled. “Nine-class chance .11” is not a general macro-F1 chance level under the actual class imbalance; remove it or define and compute the particular random predictor. The approximately .627 versus .911 gap establishes a weak within-family baseline, not uniquely whether architecture or GD is responsible.

10. **Pavia completion statement.** A statement that every run reaches L* must say **Experiment 3 analogue**. Several natural-context runs do not; the slow-head natural loss is .84. A blanket “every run” below the combined appendix is false.

11. **Projection audit trail remains incomplete in Appendix D.** The main text refers to Appendix D for the net-update percentages, but the reviewed `appendix_exp3.tex` still needs the actual definition/table. Add the normalized Frobenius projection of W*−W0 onto the orthonormal constructed-cue span, coordinate/preprocessing convention, checkpoint and arm ranges from `results/exp3/update_projections.csv`. Preserve the current main-text qualification that this is not a pathwise gradient decomposition. The concentration number alone is not a suppression diagnostic.

12. **Recovery protocol in the appendices.** Keep an explicit shared paragraph covering whole-head optimization, M=32, eta=.001, paired initialization/data, final-state evaluation and the full-budget outcomes, with nonlinear-encoder and natural-context differences stated. The registered nonlinear P14 tests the two endpoint rates; do not describe it as a three-rate monotonicity test. In plan §10.1, “fresh linear-readout head” should not imply only a linear layer was retrained.

## 8. Check (d): registration abstract

**Replace v3 with a corrected v4; do not use the candidate in plan §13 verbatim.**

It inherits the whole-encoder fraction and random-equivalence errors. It places the linear encoder's numerical values after “linear or a two-layer ReLU encoder,” incorrectly assigning those magnitudes to both. Its new natural-context sentence asserts no competition and implies the tissue PCA finding holds on Pavia, where the gate fails. These are material abstract errors.

I would register the following version. It keeps one central question, distinguishes proof from empirical scope, includes the nonlinear extension, and makes the public limitation visible. I would leave the compression details in the introduction/appendix rather than add a second abstract story.

> We study how relative module training rates affect spectral learning in serial spectral–spatial models. In an exactly solvable logistic model, we bound adaptation of the spectral coefficient at matched fit, establish contextual-reversal failure under explicit conditions, and show that readout normalization removes replicated-width dependence without necessarily preventing failure. Controlled synthetic experiments distinguish accessibility of the centre-spectrum cue, reliance of the fitted classifier, and recovery through a specified fresh-head retraining procedure. With a linear encoder, slowing the contextual head by 256× raises probe accuracy from 0.66 to 0.85 at matched training loss while original reversal accuracy remains 0.12–0.14. Retraining a fresh nonlinear head for 20,000 steps on decorrelated-context data, with the original encoder frozen, yields reversal accuracy 0.75–0.84 after slow-head training versus 0.54–0.62 after fast-head training. A two-layer ReLU encoder reproduces the qualitative ordering. Constructed-context experiments with breast-tissue and public hyperspectral spectra reproduce suppression and reliance, while the rate and recovery responses depend on the setting. In the tested natural 3×3 neighbourhoods, contextual reliance occurs without a detected encoder-accessibility deficit relative to shuffled-context training. These results show that similar contextual reliance can conceal different encoder quality and subsequent recoverability; they do not establish a general width law or optimal recoverability from the learned representations.

The numbers refer to the original three-seed linear experiment. If tables use five-seed summaries, label that distinction explicitly. The final sentence can be shortened for style, but retain the procedural scope. This is an abstract for the evidence you have, without promising an explanation of all natural spatial dominance.

## 9. Later extensions inspected separately

The live workspace advanced beyond the requested commit during this review. The following checks are an addendum, not a silent change to the reviewed PDF's population or line references.

- The added synthetic seeds 3/4 give slow-minus-fast recovery gains approximately **+.208 / +.207**. The main ordering now holds in 5/5 linear synthetic seeds.
- The added tissue gamma=10 seeds give gains approximately **+.119 / +.135**. That recovery ordering now holds in 5/5 seeds. These reinforce the central result; retain the original registered verdicts on seeds 0–2.
- Do not say the two added seeds agree on **“every direction”** as the newer `04_synthetic.tex` does. The live plan §11.1 records a positive width/reversal rank correlation for seed 3, unlike the proposed negative direction. Say they confirm the specified suppression and rate/recovery comparisons.
- The same live plan's displayed P1 correlations imply **3/5** alignment passes and **1/5** joint alignment/reversal passes at the stated −.8 cutoff, not its written 2/5 and 0/5. Correct the arithmetic; the overall registered P1 conclusion remains unsuccessful.
- The live plan says the gamma=30 added recovery advantage is at most .02, but seed 3's .593−.5705 is approximately **.0225**. This does not alter the qualitative amplitude comparison; it requires a numerical correction.
- Completed folds should be reported as their own extension when finalized. A fold-0 table with pending rows is not five-fold validation. I have not treated planned or partial fold runs as completed evidence for this score.

## 10. Submission decision and acceptance odds

The next work should be **consolidation**: fix the claim ledger, expose the recovery result in the main figure, state the public failures plainly, and finish/report the already registered extensions without growing the main story. Keep the one theorem. Keep natural context and compression as concise scope evidence. Do not start a fifth major experiment or restore the general mathematics programme.

My subjective estimate is **about 55% acceptance after these corrections**, with a broad plausible range of **40–65%**. This is a judgment about the paper and likely disagreement among reviewers, not a calibrated statistical forecast or a conference acceptance-rate calculation. Submitting the present overstatements unchanged would lower that assessment.

The improvement comes from the recovery consequence, nonlinear extension and public evidence. The remaining risk is contribution significance and honest scope, not a shortage of proofs. I cannot credibly promise 90%. You now have a defensible weak-accept paper; making its strongest result obvious and its limits accurate is the highest-value work left.

Audit limits: summary CSV arithmetic and relevant source/protocol inspection; no training rerun, no crawl of `experiments_shortcut/e3c`, and no `steps.csv` inspection. The specified PDF was recovered and hash-verified; the later-extension notes are explicitly separate.
