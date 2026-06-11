# Reading Notes — Phase 0

Five foundational papers. Read in this order. Each gets a markdown file using the same template.

## Order

1. `01_pezeshki_2021.md` — **START HERE.** Most critical paper. Builds the gradient starvation half of Theorem 2.
2. `02_sagun_2017.md` — Hessian eigenspectrum intuition.
3. `03_heusel_2017.md` — Two-timescale convergence, your template for Theorem 2.
4. `04_karakida_2019.md` — Fisher Information scaling, foundation for Theorem 1.
5. `05_pennington_bahri_2017.md` — Random matrix theory for Hessians.

## Template (every note follows this)

```markdown
# Title — Authors (Year)

**Link**: arXiv URL
**Read on**: YYYY-MM-DD
**Time spent**: X hours
**Effort**: 1 (skim) / 2 (working knowledge) / 3 (deep — can re-derive)

## 3-line summary
What did they do, in three sentences.

## Main theorem(s)
Restate in our notation.

## Proof technique
What tools did they use? (e.g., random matrix theory, NTK, contraction mapping)

## What we borrow
Specific lemmas, definitions, or techniques we will reuse.

## What we extend
How does our work go beyond theirs.

## Limitations / open questions
What did they not prove? What assumptions might break for us?

## My questions for next session
List anything confusing — we discuss next time.
```

## Why this format

- Forces synthesis (3-line summary is a writing exercise)
- Builds the related work section piece by piece
- Catches confusion early (questions at bottom go straight into next session)
- After 5 papers you have ~25 pages of organized notes — your personal reference for the rest of the project
