# Text Classification

Classify text using a Hugging Face model. Commonly used for sentiment
analysis, topic classification, etc. Vector inputs are sent in a single
batched Inference API request when possible, which is substantially
faster than one API request per text.

## Usage

``` r
hf_classify(
  text,
  model = hf_default_model("classify"),
  token = NULL,
  endpoint_url = NULL,
  ...
)
```

## Arguments

- text:

  Character vector of text(s) to classify.

- model:

  Character string. Model ID from Hugging Face Hub. Default:
  "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
  (sentiment analysis).

- token:

  Character string or NULL. API token for authentication.

- endpoint_url:

  Character string or NULL. A custom Inference Endpoint URL. When
  provided, requests are sent to this URL instead of the public
  Inference API. Use for models deployed on dedicated Inference
  Endpoints.

- ...:

  Additional arguments (currently unused).

## Value

A tibble with columns: text, label, score

## Examples

``` r
if (FALSE) { # \dontrun{
# Sentiment analysis
hf_classify("I love R programming!")

# Multiple texts
hf_classify(c("This is great!", "This is terrible."))

# Use in a pipeline
library(dplyr)
reviews |>
  mutate(sentiment = hf_classify(review_text)) |>
  unnest(sentiment)
} # }
```
