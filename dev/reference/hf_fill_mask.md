# Fill Mask

Fill in a \[MASK\] token in text with predicted words. Commonly used
with BERT-style models.

## Usage

``` r
hf_fill_mask(
  text,
  model = hf_default_model("fill_mask"),
  mask_token = "[MASK]",
  top_k = 5,
  token = NULL,
  endpoint_url = NULL,
  max_active = getOption("huggingfaceR.max_active", 1L),
  ...
)
```

## Arguments

- text:

  Character vector of text(s) containing \[MASK\] token.

- model:

  Character string. Model ID from Hugging Face Hub. Default:
  "google-bert/bert-base-uncased".

- mask_token:

  Character string. The mask token to use. Default: "\[MASK\]". Some
  models use different tokens like "\<mask\>".

- top_k:

  Integer. Number of top predictions to return. Default: 5.

- token:

  Character string or NULL. API token for authentication.

- endpoint_url:

  Character string or NULL. A custom Inference Endpoint URL.

- max_active:

  Integer. Maximum number of inputs to send at once. Default:
  \`getOption("huggingfaceR.max_active", 1)\`, which sends one input at
  a time. Larger values send inputs concurrently. This is faster for
  long vectors but spends provider credits and rate limits faster.
  Results keep input order, transient failures (HTTP 429 and 5xx) are
  retried a bounded number of times, and the first failing input stops
  the call with its error, as in sequential mode. Set
  \`options(huggingfaceR.max_active = 4)\` to change the default for
  every per-input function.

- ...:

  Additional arguments (currently unused).

## Value

A tibble with columns: text, token, score, filled (the complete text)

## Examples

``` r
if (FALSE) { # \dontrun{
# Fill in the blank
hf_fill_mask("The capital of France is [MASK].")

# Get top predictions
hf_fill_mask("Paris is the [MASK] of France.", top_k = 3)

# Use with different mask token
hf_fill_mask("The capital of France is <mask>.", mask_token = "<mask>")
} # }
```
