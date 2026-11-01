# Embed Text with Tidytext Compatibility

Generate embeddings for text in a tidy data frame. Designed to work
seamlessly with tidytext workflows.

## Usage

``` r
hf_embed_text(
  data,
  text_col,
  model = hf_default_model("embed"),
  token = NULL,
  keep_text = TRUE
)
```

## Arguments

- data:

  A data frame or tibble.

- text_col:

  Unquoted column name containing text to embed.

- model:

  Character string. Hugging Face model ID for embeddings. Default:
  "BAAI/bge-small-en-v1.5".

- token:

  Character string or NULL. API token for authentication.

- keep_text:

  Logical. Keep original text column? Default: TRUE.

## Value

The input data frame with added embedding and n_dims columns.

## Examples

``` r
if (FALSE) { # \dontrun{
library(dplyr)
library(tidytext)

# Embed documents
docs <- tibble(
  doc_id = 1:3,
  text = c("I love R", "Python is great", "Julia is fast")
)

docs_embedded <- docs |>
  hf_embed_text(text)

# Find similar documents
docs_embedded |>
  hf_nearest_neighbors("I love R", k = 2)
} # }
```
