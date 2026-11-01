# Cluster Texts by Semantic Similarity

Perform k-means clustering on text embeddings.

## Usage

``` r
hf_cluster_texts(data, k = 3, ...)
```

## Arguments

- data:

  A data frame with an 'embedding' column (from hf_embed_text).

- k:

  Integer. Number of clusters. Default: 3.

- ...:

  Additional arguments passed to stats::kmeans().

## Value

The input data frame with an added 'cluster' column.

## Examples

``` r
if (FALSE) { # \dontrun{
library(ggplot2)

# Cluster documents
docs_clustered <- docs_embedded |>
  hf_cluster_texts(k = 3)

# Reduce dimensions and visualize
library(uwot)
emb_matrix <- do.call(rbind, docs_clustered$embedding)
coords <- umap(emb_matrix)

docs_clustered |>
  mutate(umap_1 = coords[, 1], umap_2 = coords[, 2]) |>
  ggplot(aes(umap_1, umap_2, color = factor(cluster))) +
  geom_point(size = 3)
} # }
```
