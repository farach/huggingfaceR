# List Inference Providers for a Model

Query Hugging Face for provider availability, pricing, latency, and
capabilities. Coverage is assembled from two sources: the Hub's
\`inferenceProviderMapping\`, which covers every model and task, and the
router catalogue, which adds pricing and performance metrics for
chat-completion models. Models served by no provider return an empty
tibble.

## Usage

``` r
hf_list_providers(model_id, token = NULL)
```

## Arguments

- model_id:

  Character string. Model ID.

- token:

  Character string or NULL. API token for authentication.

## Value

A tibble with one row per provider, including a \`task\` column naming
the task each provider serves.

## Details

The Hub provider mapping is cached for the current R session (successful
lookups for 15 minutes). Call \[hf_clear_provider_cache()\] first if you
need a freshly fetched result.

## Examples

``` r
if (FALSE) { # \dontrun{
# Chat model: includes pricing and latency
hf_list_providers("meta-llama/Llama-3.1-8B-Instruct")

# Task model: covered via the Hub provider mapping
hf_list_providers("BAAI/bge-small-en-v1.5")
} # }
```
