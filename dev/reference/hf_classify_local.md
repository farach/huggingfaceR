# Classify Text with a Loaded Local Model

Run a local text-classification pipeline, requesting the top label for
each text. Prediction never downloads a model or falls back to hosted
inference. Input order and duplicates are preserved. Missing and
empty-string texts retain missing labels and scores without being sent
to Python. Zero-length and entirely missing or empty inputs do not call
the backend.

## Usage

``` r
hf_classify_local(text, model, batch_size = 32L, truncation = TRUE)
```

## Arguments

- text:

  Character vector of texts to classify.

- model:

  An `hf_local_model` handle loaded with `task = "classify"`.

- batch_size:

  Positive scalar integer. Python inference batch size.

- truncation:

  Logical. Truncate long inputs to the tokenizer's maximum length. With
  `FALSE`, overlong inputs may produce a backend error.

## Value

A tibble with `text`, character `label`, and numeric `score`, matching
the schema of
[`hf_classify()`](https://farach.github.io/huggingfaceR/dev/reference/hf_classify.md).

## Details

Scores are model outputs, not calibrated measures of certainty. By
default, long texts are truncated to the tokenizer's supported maximum
length.

## See also

[`hf_load_local_model()`](https://farach.github.io/huggingfaceR/dev/reference/hf_load_local_model.md),
[`hf_embed_local()`](https://farach.github.io/huggingfaceR/dev/reference/hf_embed_local.md)

## Examples

``` r
if (FALSE) { # \dontrun{
model <- hf_load_local_model(task = "classify")
hf_classify_local(c("I like this.", NA, "I dislike this."), model)
} # }
```
