# Summarize Text

Condense longer text into a shorter summary using a summarization model
via the Hugging Face Inference Providers API. Accepts a character vector
and returns one row per input, composing naturally with dplyr pipelines.

## Usage

``` r
hf_summarize(
  text,
  model = hf_default_model("summarize"),
  min_length = NULL,
  max_length = NULL,
  token = NULL,
  endpoint_url = NULL,
  max_active = getOption("huggingfaceR.max_active", 1L),
  ...
)
```

## Arguments

- text:

  Character vector of text(s) to summarize.

- model:

  Character string. Model ID from the Hugging Face Hub. Append
  \`":provider"\` to select an inference provider. Default:
  "facebook/bart-large-cnn".

- min_length:

  Integer or NULL. Minimum length of the summary in tokens. Default:
  NULL (model default).

- max_length:

  Integer or NULL. Maximum length of the summary in tokens. Default:
  NULL (model default).

- token:

  Character string or NULL. API token for authentication.

- endpoint_url:

  Character string or NULL. A custom Inference Endpoint URL. When
  provided, requests are sent to this URL instead of the public
  Inference API.

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

A tibble with columns: text, summary.

## See also

<https://huggingface.co/docs/inference-providers/tasks/summarization>

## Examples

``` r
if (FALSE) { # \dontrun{
hf_summarize("Long article text goes here ...", max_length = 60)

# Send up to four articles at a time
hf_summarize(articles$body, max_active = 4)

library(dplyr)
articles |>
  mutate(tldr = hf_summarize(body)$summary)
} # }
```
