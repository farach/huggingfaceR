# R Verification TODO

This file tracks work that was done statically (without an R interpreter in the
development environment) and **must be verified once an R instance is
available**. See `CLAUDE.md` for why this exists.

When you have R: work through the relevant items, tick them off, and delete
items that are confirmed done.

## How to verify (general)

```r
# From the package root, with the dev toolchain installed:
devtools::document()      # regenerate man/*.Rd; confirm no diff vs. hand edits
devtools::test()          # run testthat
devtools::check()         # full R CMD check
```

A valid Hugging Face token (`HUGGINGFACE_API_TOKEN` / `hf_set_token()`) is needed
for any item that makes a live inference call.

## Open items

### Defaults — beginner-friendly model choices

- [ ] **`hf_translate()` default model `Helsinki-NLP/opus-mt-en-fr`.**
  Changed from `facebook/nllb-200-distilled-600M` for easier onboarding
  (no FLORES-200 codes needed). Confirm with a live call that this model is
  served by the Inference Providers API and returns `translation_text`:
  ```r
  hf_translate("Hello, how are you?")          # expect French, e.g. "Bonjour, comment allez-vous ?"
  hf_translate("Hello", model = "Helsinki-NLP/opus-mt-en-es")  # Spanish
  ```
  If `opus-mt-en-fr` is not reliably served serverless, reconsider the default
  (candidates that keep the no-extra-args property: another served `opus-mt-*`
  pair, or fall back to NLLB with sensible `source`/`target` defaults).

- [ ] **Audit other default models against the beginner-friendly principle**
  (see `CLAUDE.md`). Confirm each is currently served and is the most broadly
  known low-friction option for its task:
  - `hf_summarize()` → `facebook/bart-large-cnn`
  - `hf_classify()` / batch → `distilbert/distilbert-base-uncased-finetuned-sst-2-english`
  - zero-shot → `facebook/bart-large-mnli`
  - `hf_embed()` / steps → `BAAI/bge-small-en-v1.5`
  - `hf_ner()` → `dslim/bert-base-NER`
  - `hf_question_answer()` → `deepset/roberta-base-squad2`
  - `hf_table_question_answer()` → `google/tapas-base-finetuned-wtq`
  - `hf_chat()` / `hf_generate()` → `meta-llama/Llama-3.1-8B-Instruct`

### Defaults centralization (`R/defaults.R`)

- [ ] **`hf_default_model()` registry + wiring.** All `hf_*` functions now take
  `model = hf_default_model("<task>")` instead of an inline literal. This is a
  no-op refactor (resolved values are unchanged), but verify in R:
  ```r
  devtools::load_all()
  testthat::test_file("tests/testthat/test-defaults.R")   # registry + sync tests
  # spot-check a couple of signatures resolve correctly:
  eval(formals(hf_chat)$model)        # "meta-llama/Llama-3.1-8B-Instruct"
  eval(formals(hf_summarize)$model)   # "facebook/bart-large-cnn"
  ```

### Docs

- [ ] Run `devtools::document()` and confirm the **hand-edited `man/*.Rd` files
  match roxygen output exactly.** Several were updated by hand because roxygen2
  cannot run in the dev environment: `man/hf_translate.Rd`,
  `man/hf_default_model.Rd` (new), and the `\usage{}` blocks of every function
  whose `model` default was switched to `hf_default_model("<task>")`
  (~21 files). `document()` is the authoritative regenerator — confirm it
  produces no diff.
- [ ] Confirm `NAMESPACE` export of `hf_default_model` survives a
  `document()` regen (added by hand).
