# Clear the Cached Inference Provider Mappings

\`huggingfaceR\` caches each model's Hugging Face provider mapping for
the duration of the R session so that repeated inference calls do not
re-query the Hub. Call this function to discard the cache after provider
availability changes, or in tests.

## Usage

``` r
hf_clear_provider_cache()
```

## Value

\`NULL\`, invisibly.

## Examples

``` r
hf_clear_provider_cache()
```
