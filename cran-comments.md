## R CMD check results

0 errors | 0 warnings | 0 notes

## Submission type

This is a minor release (2.2.0) of huggingfaceR.

It realigns the package with the current Hugging Face Inference Providers
platform: task requests now resolve which provider serves a model instead of
assuming the first-party `hf-inference` provider, the `HF_TOKEN` environment
variable is honoured alongside the legacy `HUGGING_FACE_HUB_TOKEN`, and a
default model that is no longer served by any provider was replaced.

## Test environments

* Local Windows 11 x64, R 4.6.1

## Network use

All examples that contact the Hugging Face API are wrapped in `\dontrun{}`, and
tests that require network access or an API token are skipped unless
`NOT_CRAN=true`. The package makes no network calls during `R CMD check`.

## Method references

There are no published method references for this package. It provides an
interface to public Hugging Face Hub and Inference API services from R.
