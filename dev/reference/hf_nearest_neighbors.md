# Find Nearest Neighbors by Semantic Similarity

Find the k most similar texts to a query text based on embedding
similarity. Similarities are computed with a blocked matrix-vector
product. Embeddings that are missing, non-numeric, zero-norm,
non-finite, or of a different dimension than the query embedding have an
\`NA\` similarity and sort last.

## Usage

``` r
hf_nearest_neighbors(
  data,
  query,
  k = 5,
  text_col = "text",
  model = hf_default_model("embed"),
  token = NULL
)
```

## Arguments

- data:

  A data frame with an 'embedding' column (from hf_embed_text).

- query:

  Character string. The query text to compare against.

- k:

  Integer. Number of nearest neighbors to return. Default: 5.

- text_col:

  Character string. Name of text column. Default: "text".

- model:

  Character string. Model to use for query embedding. Should match the
  model used for data embeddings.

- token:

  Character string or NULL. API token for authentication.

## Value

A tibble with the k nearest neighbors, sorted by similarity
(descending).

## Examples

``` r
if (FALSE) { # \dontrun{
docs_embedded |>
  hf_nearest_neighbors("machine learning", k = 5)
} # }
```
