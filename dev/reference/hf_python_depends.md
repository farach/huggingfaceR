# Install Python Dependencies

Installs python packages needed to run huggingfaceR functions

## Usage

``` r
hf_python_depends(
  packages = c("transformers", "sentencepiece", "huggingface_hub", "datasets",
    "sentence-transformers")
)
```

## Arguments

- packages:

  Python libraries needed for local model usage.  
  Defaults to transformers, sentencepiece, huggingface_hub, datasets,
  and sentence-transformers.

## Value

The value returned by \`reticulate::conda_install()\`, called for its
side effect.

## Examples

``` r
if (FALSE) { # \dontrun{
hf_python_depends()
} # }
```
