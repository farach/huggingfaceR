# Batch Text Classification (In-Memory)

Classify multiple texts in parallel. This function processes all inputs
in memory and returns results in a single tibble.

## Usage

``` r
hf_classify_batch(
  text,
  model = hf_default_model("classify"),
  token = NULL,
  batch_size = 100L,
  max_active = 10L,
  progress = TRUE,
  endpoint_url = NULL
)
```

## Arguments

- text:

  Character vector of text(s) to classify.

- model:

  Character string. Model ID from Hugging Face Hub. Default:
  "distilbert/distilbert-base-uncased-finetuned-sst-2-english".

- token:

  Character string or NULL. API token for authentication.

- batch_size:

  Integer. Number of texts per API request. Default: 100.

- max_active:

  Integer. Maximum concurrent requests. Default: 10.

- progress:

  Logical. Show progress bar. Default: TRUE.

- endpoint_url:

  Character string or NULL. A custom Inference Endpoint URL.

## Value

A tibble with columns: - \`text\`: Original input text - \`label\`:
Predicted label - \`score\`: Confidence score - \`.input_idx\`: Original
position in input vector - \`.error\`: TRUE if request failed -
\`.error_msg\`: Error message or NA

## Examples

``` r
if (FALSE) { # \dontrun{
# Classify many texts in parallel
texts <- c("I love this!", "This is terrible.", "Meh, it's okay.")
result <- hf_classify_batch(texts, max_active = 5)

# Check for errors
errors <- result[result$.error, ]
} # }
```
