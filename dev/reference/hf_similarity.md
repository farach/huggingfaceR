# Compute Pairwise Similarity

Compute cosine similarity between all pairs of embeddings. Embeddings
are grouped by dimension and each group is scored with one matrix
cross-product, so the work is done by compiled linear algebra rather
than by R code for each pair. Invalid, zero-length, zero-norm,
non-finite, or dimension-mismatched pairs return \`NA_real\_\`.

## Usage

``` r
hf_similarity(embeddings, text_col = "text")
```

## Arguments

- embeddings:

  A tibble with an 'embedding' column (from hf_embed).

- text_col:

  Character string. Name of the text column. Default: "text".

## Value

A tibble with columns: text_1, text_2, similarity.

## Examples

``` r
if (FALSE) { # \dontrun{
sentences <- c("I love cats", "I adore felines", "Dogs are great")
embeddings <- hf_embed(sentences)
similarities <- hf_similarity(embeddings)
} # }
```
