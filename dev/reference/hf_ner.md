# Named Entity Recognition (Token Classification)

Extract named entities (people, organizations, locations, etc.) from
text using a token-classification model via the Hugging Face Inference
Providers API. Returns one row per detected entity, with character
offsets that let you highlight or join back to the source text. Inputs
that produce no entities (and \`NA\` inputs) yield a single row with
\`NA\` entity fields so every input is represented.

## Usage

``` r
hf_ner(
  text,
  model = hf_default_model("ner"),
  aggregation_strategy = "simple",
  token = NULL,
  endpoint_url = NULL,
  max_active = getOption("huggingfaceR.max_active", 1L),
  ...
)
```

## Arguments

- text:

  Character vector of text(s) to analyze.

- model:

  Character string. Model ID from the Hugging Face Hub. Append
  \`":provider"\` to select an inference provider. Default:
  "dslim/bert-base-NER".

- aggregation_strategy:

  Character string. How sub-word tokens are grouped into entities: one
  of "none", "simple", "first", "average", "max". Default: "simple".

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

A tibble with columns: text, word, entity_group, score, start, end.

## See also

<https://huggingface.co/docs/inference-providers/tasks/token-classification>

## Examples

``` r
if (FALSE) { # \dontrun{
hf_ner("Barack Obama was born in Hawaii.")

# One row per entity, ready to count or join
library(dplyr)
hf_ner(headlines) |>
  filter(!is.na(word)) |>
  count(entity_group, sort = TRUE)
} # }
```
