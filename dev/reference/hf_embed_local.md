# Generate Embeddings with a Loaded Local Model

Run SentenceTransformers on a reusable local handle. Prediction never
downloads a model or falls back to hosted inference. Input order and
duplicates are preserved. Missing texts retain `NULL` embeddings and
`NA_integer_` dimensions; zero-length and all-missing inputs do not call
Python. Empty strings are valid texts. The model's own maximum sequence
length controls truncation of long texts.

## Usage

``` r
hf_embed_local(
  text,
  model,
  batch_size = 32L,
  normalize = FALSE,
  progress = FALSE
)
```

## Arguments

- text:

  Character vector of texts to embed.

- model:

  An `hf_local_model` handle loaded with `task = "embed"`.

- batch_size:

  Positive scalar integer. Python inference batch size.

- normalize:

  Logical. Normalize embeddings to unit length.

- progress:

  Logical. Display the Python encoding progress bar.

## Value

A tibble with `text`, `embedding` (a list of numeric vectors), and
integer `n_dims`, matching the schema of
[`hf_embed()`](https://farach.github.io/huggingfaceR/dev/reference/hf_embed.md).

## See also

[`hf_load_local_model()`](https://farach.github.io/huggingfaceR/dev/reference/hf_load_local_model.md),
[`hf_classify_local()`](https://farach.github.io/huggingfaceR/dev/reference/hf_classify_local.md)

## Examples

``` r
if (FALSE) { # \dontrun{
model <- hf_load_local_model(task = "embed")
hf_embed_local(c("Hello", NA, "Hello"), model, normalize = TRUE)
} # }
```
