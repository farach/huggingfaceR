## Submission type

This is an update of huggingfaceR from 2.3.0, currently on CRAN, to 2.4.0.

This release makes existing functions faster and corrects the declared
dependency versions:

* Pairwise and nearest-neighbour similarity now use vectorized matrix
  operations, and batch, dataset, and Hub results are assembled column by
  column. Results are unchanged, except that a few edge cases that previously
  raised errors now return `NA`, as described in NEWS.md.
* 'RcppSimdJson' is a new suggested package for parsing large embedding
  responses. It is used only through `requireNamespace()`, with 'jsonlite' as
  the fallback, and the tests that use it are skipped when it is not installed.
* The minimum version of 'httr2' is now 1.1.1. The embedding, classification,
  and streaming chat functions use 'httr2' features that earlier releases do
  not have, so those functions failed for users with an older 'httr2'. The
  minimum versions of 'cli', 'dplyr', 'recipes', and 'testthat' were corrected
  in the same way.

## Release validation

`R CMD check --no-manual --as-cran` reported 0 errors, 0 warnings and 0 notes
on these GitHub Actions environments for the 2.4.0 release candidate:

* Windows Server 2025 x64, R 4.6.2
* macOS Tahoe 26.6.2 arm64, R 4.6.2
* Ubuntu 24.04.5 x64, R 4.6.2, R 4.5.3, and R-devel (2026-10-26 r90766)

These cross-platform checks use the r-lib action's standard configuration,
with the CRAN incoming-feasibility check disabled. A separate Ubuntu R 4.6.2
check with incoming feasibility enabled also reported 0 errors, 0 warnings and
0 notes.

The exact source package being submitted was also checked on win-builder,
R-devel (2026-10-30 r90805 ucrt), with Status: OK.

Full logs, independent real-model results, and the exact checked source
package with its SHA-256 are retained as workflow artifacts. CRAN submission
is a separate, explicit step after those final main-branch checks.

## Reverse dependencies

There are no reverse dependencies on CRAN.

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
