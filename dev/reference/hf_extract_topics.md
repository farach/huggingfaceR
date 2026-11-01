# Extract Semantic Topics from Text

Identify semantic topics by clustering embeddings and extracting
representative keywords from each cluster.

## Usage

``` r
hf_extract_topics(data, text_col = "text", k = 5, top_n = 10)
```

## Arguments

- data:

  A data frame with text and embeddings.

- text_col:

  Character string. Name of text column.

- k:

  Integer. Number of topics/clusters. Default: 5.

- top_n:

  Integer. Number of top words per topic. Default: 10.

## Value

A tibble with topics and their top terms.

## Examples

``` r
if (FALSE) { # \dontrun{
library(tidytext)

# Extract topics
topics <- docs_embedded |>
  hf_extract_topics(text_col = "text", k = 3, top_n = 5)
} # }
```
