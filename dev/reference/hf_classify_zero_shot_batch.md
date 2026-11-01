# Batch Zero-Shot Classification (In-Memory)

Classify multiple texts into custom categories in parallel without
training.

## Usage

``` r
hf_classify_zero_shot_batch(
  text,
  labels,
  model = hf_default_model("zero_shot"),
  multi_label = FALSE,
  token = NULL,
  batch_size = 50L,
  max_active = 10L,
  progress = TRUE,
  endpoint_url = NULL
)
```

## Arguments

- text:

  Character vector of text(s) to classify.

- labels:

  Character vector of candidate labels/categories.

- model:

  Character string. Model ID from Hugging Face Hub. Default:
  "facebook/bart-large-mnli".

- multi_label:

  Logical. If TRUE, allows multiple labels per text. Default: FALSE.

- token:

  Character string or NULL. API token for authentication.

- batch_size:

  Integer. Number of texts per API request. Default: 50.

- max_active:

  Integer. Maximum concurrent requests. Default: 10.

- progress:

  Logical. Show progress bar. Default: TRUE.

- endpoint_url:

  Character string or NULL. A custom Inference Endpoint URL.

## Value

A tibble with columns: - \`text\`: Original input text - \`label\`:
Predicted label (or labels if multi_label) - \`score\`: Confidence
score(s) - \`.input_idx\`: Original position in input vector -
\`.error\`: TRUE if request failed - \`.error_msg\`: Error message or NA

## Examples

``` r
if (FALSE) { # \dontrun{
texts <- c("I love my new laptop", "The game was exciting", "This recipe is delicious")
labels <- c("technology", "sports", "food")
result <- hf_classify_zero_shot_batch(texts, labels, max_active = 5)
} # }
```
