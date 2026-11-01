# Declare Optional Local Python Requirements

Declare a shared Python environment for local embeddings and
classification using
[`reticulate::py_require()`](https://rstudio.github.io/reticulate/reference/py_require.html).
This is opt-in: loading `huggingfaceR` and using its hosted API
functions do not require or initialize Python.

## Usage

``` r
hf_local_setup()
```

## Value

Invisibly, a list with `packages` and `python_version` requirements.

## Details

Requires optional `reticulate` version 1.41.0 or later. Declared
requirements are Python `>=3.10,<3.13`, transformers `>=4.41,<5`,
sentence-transformers `>=3,<6`, huggingface_hub `>=0.34,<1`, torch
`>=2.6,<3`, and numpy `>=1.26,<3`.

Call this before initializing Python. Requirements are resolved lazily
by reticulate when Python is first used. An explicitly selected Python
installation, including `RETICULATE_PYTHON`, is never overridden. In a
manually managed environment, install these requirements yourself and
restart R after changing the environment. Local download and load
operations check the selected environment's versions using Python
package metadata.

## See also

[`hf_download_model()`](https://farach.github.io/huggingfaceR/dev/reference/hf_download_model.md),
[`hf_load_local_model()`](https://farach.github.io/huggingfaceR/dev/reference/hf_load_local_model.md)

## Examples

``` r
if (FALSE) { # \dontrun{
requirements <- hf_local_setup()
requirements$packages
} # }
```
