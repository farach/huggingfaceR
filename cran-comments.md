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

`R CMD check --no-manual --as-cran` reported 0 errors, 0 warnings and 0 notes
on these GitHub Actions environments for the 2.3.0 release candidate:

* Windows Server 2022 x64, R 4.6.1
* macOS Tahoe 26.6.2 arm64, R 4.6.1
* Ubuntu 24.04.5 x64, R 4.6.1 and R 4.5.3

These cross-platform checks use the r-lib action's standard configuration,
with the CRAN incoming-feasibility check disabled. A separate Ubuntu R 4.6.1
check enables incoming feasibility; before publication, its only NOTE was the
new local-model article's URL returning 404. The page is generated and checked
in CI, and is published from main/docs on merge. The main-branch workflow
requires the live URL check to pass before a submission bundle is used.

Full logs, independent real-model results, and the exact checked source
package with its SHA-256 are retained as workflow artifacts. CRAN submission
is a separate, explicit step after those final main-branch checks.

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
