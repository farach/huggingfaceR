# CLAUDE.md — Project conventions for huggingfaceR

Guidance for working on this package. Keep it short and current.

## Design principles

### Defaults must be beginner-friendly and broadly known

Every user-facing default — model IDs, example datasets, parameters — should be
the **easiest, most broadly understood option that gets a newcomer running with
the least friction**. The mental model is base R's `mtcars`: a default everyone
recognizes, that just works, and that lets people start experimenting in one
line.

Concretely, when choosing a default model:

- **Prefer broadly known, widely used models** over the technically "best" or
  newest one. Recognition and reliability beat marginal quality for a default.
- **Prefer the option that needs no extra arguments.** A default should work from
  a bare call, e.g. `hf_translate("Hello")`, without the user first learning
  codes, flags, or schemas. (This is why `hf_translate()` defaults to the
  Helsinki-NLP `opus-mt-en-fr` language-pair model — direction is in the model ID
  — rather than NLLB, which requires FLORES-200 codes.)
- **Prefer small, fast, low-cost models** for defaults so first calls are quick
  and cheap. Power users can always pass a larger `model`.
- Document the default's identity and behavior in the roxygen `@param`, and point
  to the heavier/more-capable alternative in `@details`/`@examples`.

The goal is the quickest path to people *actively using* the functions and
experimenting — easy onboarding first, depth second.

## Working without an R instance

This environment has **no R interpreter**, so package code cannot be run, built,
or tested here. When making changes:

- Edit `R/*.R` **and** keep the corresponding generated `man/*.Rd` in sync by
  hand (roxygen2 cannot run here).
- Do as much as possible statically (code, docs, tests, NEWS), then **record what
  must be verified once an R instance is available** in
  `dev/R-VERIFICATION-TODO.md`.
- Append a new entry to that file for every change that needs live verification
  (e.g. confirming a new default model is actually served by the Inference
  Providers API, running `devtools::document()` / `R CMD check` / `testthat`).
