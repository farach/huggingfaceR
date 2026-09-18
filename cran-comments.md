## Submission type

This is the first submission of huggingfaceR 2.3.0, an update to version 2.2.0
currently on CRAN.

This release adds optional local embeddings and text classification:

* Explicit setup, revision-pinned Hub snapshot downloads, and reusable local
  model handles use the suggested package 'reticulate'.
* Local predictions return the same tidy columns as the existing API functions.
  The API-first functions and their defaults are unchanged.
* Model loading uses standard safetensors files, refuses remote custom code,
  and supports cached offline use without falling back to hosted inference.
* Python is not initialized and dependencies are not installed when the package
  is loaded or its default API functions are used.

## Release validation

Final version-2.3.0 check results will be recorded here before submission.

## Network use

Examples that require API credentials, additional Python software, or model
downloads are wrapped in `\dontrun{}`. Tests requiring network access or an API
token are skipped unless explicitly enabled. Deterministic local-model tests
mock Python/download boundaries and do not initialize Python.

The separate opt-in GitHub Actions workflow downloads real models into a runner
cache, performs predictions, repeats them in fresh offline R processes, and
renders the local-model article. It also installs the exact GitHub revision
independently in a second R/Python environment. These model downloads are not
part of ordinary package checks or installation.

No model weights or third-party Python source are bundled in the package.

The package's long-form articles are published on its pkgdown site rather than
shipped as package vignettes, so `vignettes/` is excluded via `.Rbuildignore`
and the tarball contains no vignettes to build.

## Method references

There are no published method references for this package. It provides an
interface to public Hugging Face Hub and Inference API services and established
Python model libraries from R.
