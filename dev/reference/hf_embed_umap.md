# Dimensionality Reduction with UMAP

Reduce embedding dimensions to 2D using UMAP for visualization. Requires
the 'uwot' package to be installed.

## Usage

``` r
hf_embed_umap(
  text,
  model = hf_default_model("embed"),
  token = NULL,
  endpoint_url = NULL,
  n_neighbors = 15,
  min_dist = 0.1,
  ...
)
```

## Arguments

- text:

  Character vector of text(s) to embed and reduce.

- model:

  Character string. Model ID for generating embeddings. Default:
  "BAAI/bge-small-en-v1.5".

- token:

  Character string or NULL. API token for authentication.

- endpoint_url:

  Character string or NULL. A custom Inference Endpoint URL.

- n_neighbors:

  Integer. UMAP n_neighbors parameter. Default: 15.

- min_dist:

  Numeric. UMAP min_dist parameter. Default: 0.1.

- ...:

  Additional arguments passed to uwot::umap().

## Value

A tibble with columns: text, umap_1, umap_2. It has one row per input,
and rows whose text is \`NA\` get \`NA\` coordinates.

## Examples

``` r
if (FALSE) { # \dontrun{
# Reduce and visualize
library(ggplot2)
texts <- c("cat", "dog", "kitten", "puppy", "car", "truck")
coords <- hf_embed_umap(texts)

ggplot(coords, aes(umap_1, umap_2, label = text)) +
  geom_text() +
  theme_minimal()
} # }
```
