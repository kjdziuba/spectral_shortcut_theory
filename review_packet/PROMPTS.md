# External review prompts — theory freeze gate (2026-08-20)

**Artifact to attach:** `review_packet/theory_scope_2026-08-20.pdf`
(25 pp = Sections 3–6 + the supplement, extracted from the compiled paper.
Sections 1–2 and 7–9 are deliberately excluded: they are being rebuilt after
the theory freezes, and reviewing them now wastes the pass.)

**Use NEW chats, not the old ones.** The old chats hold the retired theory
(condition-number κ, slow-fast manifolds, Option 4A) and will anchor on it.
Exception: one short follow-up in each old chat, using Prompt D, to check that
their earlier objections actually landed.

Run A + B + C in parallel across the three services. Suggested pairing:
- **Prompt A (math referee)** → the strongest reasoning model available
- **Prompt B (deep research)** → ChatGPT/Gemini deep-research mode
- **Prompt C (area chair)** → the third service
- **Prompt D (follow-up)** → old chats only, cheap, optional

---

## PROMPT A — line-by-line math referee

> You are refereeing the theory of a machine-learning paper submitted to ICLR.
> The attached PDF contains only the theoretical core: the setup section, two
> theorems with their proofs, a diagnostic section, and the supplement with the
> full proofs. The experimental sections are excluded on purpose.
>
> Your job is to find mathematical defects. Verify every displayed equation and
> inequality by doing the computation yourself — do not accept a step because it
> looks standard. For each step ask: does the conclusion actually follow from
> what precedes it, and are all quantifiers, measurability, and normalization
> factors handled correctly?
>
> Pay particular attention to:
> 1. The proof of the spatial-block lemma in the supplement (the rank-one
>    witness argument and the softmax-weighted feature Gram). Is the central
>    inequality really deterministic and hypothesis-free? Does the lemma's
>    stated hypothesis match exactly what the proof consumes?
> 2. The proposition titled "A fully proved instance." This is the newest
>    material and has had the least scrutiny. Check every probabilistic step:
>    what is measurable with respect to what, whether conditional and
>    unconditional bounds are being mixed, whether the failure probabilities
>    sum correctly, and whether the constants are genuinely free of the width
>    and of the dataset size as claimed.
> 3. The operator-norm cap lemma and its exact Kronecker identity.
> 4. Theorem 2: the integration, the fitting-time step, the finite-horizon
>    constant, and whether the interpretation paragraphs claim more than the
>    theorem delivers.
>
> For every defect you report, state the specific failing step and either a
> counterexample or the precise reason the inference is invalid. Rank findings
> by severity. If a step is correct, do not report it; I want defects, not a
> summary. If you believe the theory is sound, say so explicitly and name the
> steps you checked hardest.

---

## PROMPT B — deep research: citations and prior art

> You are doing a citation-integrity and novelty audit for a machine-learning
> theory paper heading to ICLR. The attached PDF contains the paper's
> theoretical core only.
>
> Part 1 — citation fidelity. For every external work cited in the attached
> text, verify against the actual source: (a) that the work exists with the
> stated authors and venue; (b) that any theorem, lemma, equation or figure
> number referenced is real; (c) that the claim attributed to it is something
> the source actually establishes, within the source's own hypotheses. This
> paper has a history of citations being stretched past their model class —
> a real theorem invoked outside the setting it was proved for — so check
> hypotheses, not just statements. Flag anything you cannot verify rather than
> assuming it is fine.
>
> Part 2 — prior art. The paper's two claims are: (i) as a downstream module
> grows wider, curvature concentrates in it while an upstream module's
> curvature stays capped, so the ratio of their top Gauss-Newton eigenvalues
> grows linearly in width; (ii) consequently, under an assumed residual-decay
> envelope, the upstream module's total parameter displacement up to the time
> the loss is fitted is bounded logarithmically, so it stays near
> initialization for the whole practical training run.
>
> Search the literature for work that anticipates either claim: layer-wise or
> block-wise Hessian/Fisher spectral analysis, lazy training and NTK parameter
> movement bounds, gradient starvation, modality competition and multimodal
> imbalance, bottleneck architectures, and anything on freezing upstream
> encoders. For each relevant work, state precisely what it establishes and
> whether it subsumes, overlaps with, or is genuinely distinct from the claims
> above. I need to know what a reviewer will say has already been done.
>
> Give me a ranked list with citations and links.

---

## PROMPT C — hostile area chair

> You are a senior area chair at ICLR with a reputation for rejecting papers
> whose theory is technically correct but vacuous. The attached PDF is the
> theoretical core of a submission; the experiments are excluded.
>
> Read it as an adversary. I want the strongest objections a well-prepared
> reviewer could raise, whether or not they are formally errors. In particular:
>
> - Is the first theorem substantive, or is it arithmetic on two assumptions
>   dressed up as a result? Does the paper prove its hypotheses anywhere, and
>   is the proved instance a real answer or a toy that dodges the difficulty?
> - Is the second theorem an integration exercise whose entire content sits in
>   the assumed residual envelope? The paper contains a remark arguing this is
>   not circular — is that argument convincing, or does it dodge the objection?
> - Does the geometric claim actually connect to the dynamical one, or is the
>   paper trading on a suggestive resemblance between the two?
> - Are the hypotheses doing hidden work — assuming most of the phenomenon and
>   leaving the theorem to state the obvious?
> - Is the framing honest about what is proved versus assumed versus observed?
>
> Quote the specific sentences you would attack and write out what your review
> would say. Then give a verdict: accept, weak accept, weak reject, or reject
> on the theory alone, with your reasoning. Do not be polite about it.

---

## PROMPT D — follow-up in the OLD chats only

> Earlier you reviewed a draft of this paper and raised objections — among them
> the invalid eigenvalue-scaling claim, the invalid use of Cauchy interlacing
> on the smallest positive eigenvalue, the Kronecker factorization requiring
> per-pixel structure, and the conflict with Soudry et al. on separable data.
>
> The theory has since been substantially rewritten. The attached PDF is the
> current theoretical core. Please check specifically whether your earlier
> objections have been genuinely addressed rather than papered over, and
> whether the fixes introduced new problems. Do not re-review the whole paper;
> I have fresh reviewers doing that. Focus only on: did the repairs land?

---

## What to bring back

For each reviewer: the raw findings, unedited. I will triage against
`REMAINING_WORK.md` (known items are already tracked and should not consume a
fix round) and verify each surviving claim against the source files before
changing anything. The failure mode to avoid is fixing a "finding" that is
actually a misreading — every previous round has contained at least one.
