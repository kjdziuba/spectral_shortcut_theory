# External review — theory freeze gate (2026-08-20)

**Artifact:** `review_packet/theory_scope_2026-08-20.pdf`
(25 pp = Sections 3–6 + supplement, extracted from the compiled paper. Sections
1–2 and 7–9 are excluded on purpose: they are rebuilt after the theory freezes,
and reviewing them now would absorb the pass on already-tracked breakage.)

**Protocol: identical prompt, identical attachment, THREE NEW CHATS.**
One each in ChatGPT, Gemini, Claude. Do not continue the old review chats —
they carry the retired theory (condition-number κ, slow-fast manifolds, the
Option 4A argument) and will anchor on it. More importantly, for an agreement
comparison every reviewer must start from the same state; mixing one
context-carrying chat with two fresh ones contaminates the result.

Deep-research / extended-thinking mode ON wherever available.

The prompt is deliberately open-ended and does NOT name the objections our
in-house review already found. Naming them would produce agreement by priming
rather than by independent judgement.

---

## THE PROMPT (paste identically into all three)

> You are refereeing the theoretical core of a machine-learning paper being
> prepared for ICLR. The attached PDF contains the setup section, two theorems
> with their proofs, a diagnostic section, and the supplement containing the
> full proofs. The experimental sections are excluded deliberately — do not ask
> for them or speculate about them.
>
> I want an adversarial review. Assume the paper is wrong somewhere and try to
> find where. I would much rather hear a hard objection now than from a
> reviewer in December.
>
> Cover all four of the following.
>
> **1. Mathematical validity.** Verify every displayed equation and inequality
> by doing the computation yourself; do not accept a step because it looks
> standard or because the surrounding prose sounds confident. For each step,
> check that the conclusion actually follows from what precedes it, and that
> quantifiers, measurability, independence, and normalization factors are
> handled correctly. Give particular scrutiny to the proposition in the
> supplement labelled "A fully proved instance" — it is the most recently
> written material and has had the least review. Also check whether each
> theorem and lemma states exactly the hypotheses its proof consumes: neither
> assuming something it never uses, nor using something it never assumed.
>
> **2. Citation fidelity.** For every external work cited, verify against the
> actual source that the work exists as described, that any theorem, lemma,
> equation or figure number referenced is real, and that the claim attributed
> to it is something the source genuinely establishes *within the source's own
> hypotheses*. Pay attention to whether a real result is being invoked outside
> the setting it was proved for. Flag anything you cannot verify rather than
> assuming it is correct.
>
> **3. Substance.** Independently of formal correctness, assess whether the
> results are worth stating. For each theorem, ask: does it establish something
> non-obvious, or does it follow immediately from its assumptions? Are the
> hypotheses doing hidden work — assuming most of the phenomenon and leaving
> the theorem to state what remains? Does the paper's chain of reasoning
> actually connect its geometric claim to its dynamical one, or does it rely on
> the two merely resembling each other? Is the writing honest about what is
> proved, what is assumed, and what is only observed?
>
> **4. Prior art.** The paper's two claims are: (i) as a downstream module
> grows wider, curvature concentrates in it while an upstream module's
> curvature stays capped, so the ratio of their top Gauss-Newton eigenvalues
> grows linearly in width; (ii) under an assumed residual-decay envelope, the
> upstream module's total parameter displacement up to the time the loss is
> fitted is bounded logarithmically, so it remains near its initialization for
> the whole practical training run. Search for work that anticipates either
> claim and state precisely whether it subsumes, overlaps with, or is distinct
> from them. I need to know what a reviewer will say has already been done.
>
> **Output format — please follow exactly, so I can compare your review against
> two others:**
>
> First, a numbered list of findings, most severe first. For each finding give:
> - `SEVERITY:` blocker / major / minor
> - `LOCATION:` section, theorem/lemma name, or page
> - `CLAIM:` one sentence stating the defect
> - `EVIDENCE:` the specific failing step, a counterexample, or the source text
>   that contradicts the paper. A finding without this is not usable.
>
> Second, a section headed `CHECKED AND SOUND:` listing what you verified and
> found correct — I need to know where you looked and found nothing, not only
> where you found something.
>
> Third, a single line: `VERDICT:` accept / weak accept / weak reject / reject,
> on the theory alone, with one sentence of reasoning.
>
> Report only what you can defend. Do not include style, formatting, or writing
> quality suggestions.

---

## Optional, afterwards

Once the three fresh reviews are in, a short follow-up in each **old** chat is
cheap and worth doing — but keep it separate from the comparison above:

> Earlier you reviewed a draft of this paper and raised objections. The theory
> has since been substantially rewritten; the attached PDF is the current
> theoretical core. Check only whether your earlier objections were genuinely
> addressed rather than papered over, and whether the fixes introduced new
> problems. Do not re-review the whole paper.

---

## Triage on return

Bring the three reviews back raw and unedited. They get triaged against
`REMAINING_WORK.md` (already-tracked items must not consume a fix round) and
every surviving claim is verified against the source files before anything
changes. Every review round so far has contained at least one confident finding
that turned out to be a misreading — including from the in-house adversarial
agents. Agreement across all three reviewers is strong evidence a finding is
real; a finding from one reviewer alone still needs checking, not assuming.
