## R CMD check results

0 errors | 0 warnings | 0 notes

Checked with `R CMD check --as-cran` on the built tarball, with vignettes built.

All URLs were additionally verified with `urlchecker::url_check()`, which
reported no problems. An intermittent "possibly invalid URLs" NOTE can appear on
this machine when its TLS connection to huggingface.co is reset mid-check; the
URLs themselves resolve correctly.

## Submission type

This is a minor release (2.2.0) of huggingfaceR.

It realigns the package with the current Hugging Face Inference Providers
platform:

* Task requests now confirm that the first-party `hf-inference` provider serves
  a model before sending, and otherwise fail with an error naming the providers
  that do serve it, instead of returning an opaque "not found".
* The `HF_TOKEN` environment variable is honoured alongside the legacy
  `HUGGING_FACE_HUB_TOKEN`.
* Two default models that are no longer reachable were replaced.
* `hf_list_providers()` now covers non-chat models and gained a `task` column.
* Documentation links to the retired `huggingface.co/docs/api-inference` pages
  were updated to their current Inference Providers equivalents.

## Test environments

* Local Windows 11 x64, R 4.6.1

## Network use

All examples that contact the Hugging Face API are wrapped in `\dontrun{}`, and
tests that require network access or an API token are skipped unless
`NOT_CRAN=true`. Vignette chunks that call the API are conditional on a token
being present in the environment, so they are not evaluated during checks. The
check above was run with no Hugging Face token present, and the package made no
network calls during it.

## Method references

There are no published method references for this package. It provides an
interface to public Hugging Face Hub and Inference API services from R.
