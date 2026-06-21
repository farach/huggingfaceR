# Companion Book — Working Drafts

This folder collects planning artifacts for a book that accompanies the
`huggingfaceR` package: a narrative curriculum teaching R programmers to access,
use, and **operationalize** Hugging Face models, culminating in an enterprise
production pipeline.

The book's through-line:

```
Hugging Face → find / fine-tune model → Microsoft Foundry
            → secure deployment + monitoring → employee/customer application
```

## Contents

- **`book-outline.md`** — the full structure: working title + alternates,
  audience/prerequisites/outcomes, an 8-part / 26-chapter learning arc, a
  recurring case study, appendices, the non-duplication boundary versus the
  pkgdown reference and vignettes, and a recommendation to build with Quarto Book.
- **`use-cases-and-audience.md`** — the positioning thesis: who genuinely
  appreciates reaching Hugging Face from R (and why they differ from Python +
  Transformers users), the audiences to target, and a ranked set of use cases
  (survey research, social science / economics, literature mining, zero-shot
  classification, semantic search, teaching, data journalism). Feeds the preface
  and Chapter 1.
- **`platform-comparison.md`** — where Hugging Face fits relative to Microsoft
  Foundry, GitHub Models, Kaggle, ModelScope, and OpenRouter: per-platform R
  access, pricing, strengths/weaknesses, a comparison table, and the synthesis
  argument for the HF → Foundry pipeline. Fully sourced.
- **`pipeline-narrative.md`** — the capstone chapter: a stage-by-stage walkthrough
  of the production pipeline, the role `huggingfaceR` plays at each stage
  (including how `endpoint_url` lets the same R code target a governed Foundry
  endpoint), reference architecture diagrams (ASCII + Mermaid), a serverless →
  dedicated → Foundry decision guide, and an openness-vs-governance tradeoff
  analysis.

## How these fit together (and avoid overlap)

Each file works at a different planning altitude, so they complement rather than
compete. To keep them from drifting, each topic has **one canonical home**:

- **`book-outline.md`** — the *structure* (chapters, arc, appendices). Canonical
  for "what goes where."
- **`use-cases-and-audience.md`** — the *positioning* (why / who / use cases).
  Canonical for audience and use cases; the outline points here instead of
  restating them.
- **`platform-comparison.md`** — *research input* feeding Part VIII / Ch. 23.
  Canonical for cross-platform facts.
- **`pipeline-narrative.md`** — a *drafted chapter* (the capstone pipeline).
  Canonical for the production-pipeline prose.

Rule of thumb: edit a topic in its canonical file and link from the others;
don't copy content between them.

## Status

These are drafts/outlines, not finished prose. Each comparison/pipeline document
carries a "could not verify" section — confirm time-sensitive platform facts
(pricing, rate limits, product names) against the cited URLs before publishing,
since several vendor pages blocked automated fetching during research.

These files are not part of the built R package or the pkgdown site.
