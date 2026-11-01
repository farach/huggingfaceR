# Batch Embedding Generation (In-Memory)

Generate embeddings for multiple texts in parallel. This function
processes all inputs in memory and returns results in a single tibble.

## Usage

``` r
hf_embed_batch(
  text,
  model = hf_default_model("embed"),
  token = NULL,
  batch_size = 100L,
  max_active = 10L,
  progress = TRUE,
  endpoint_url = NULL
)
```

## Arguments

- text:

  Character vector of text(s) to embed.

- model:

  Character string. Model ID from Hugging Face Hub. Default:
  "BAAI/bge-small-en-v1.5" (384-dim embeddings).

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

A tibble with columns: - \`text\`: Original input text - \`embedding\`:
List-column of numeric vectors - \`n_dims\`: Dimension of embedding (or
NA on error) - \`.input_idx\`: Original position in input vector -
\`.error\`: TRUE if request failed - \`.error_msg\`: Error message or NA

## Details

Embedding responses are large arrays of numbers. When the optional
'RcppSimdJson' package is installed, well-formed responses are parsed
directly into numeric matrices, which is many times faster than the
default 'jsonlite' parser and returns the same embeddings. Set
\`options(huggingfaceR.simdjson = FALSE)\` to always use 'jsonlite'.

## Examples

``` r
if (FALSE) { # \dontrun{
# Embed many texts in parallel
texts <- c("Hello world", "Goodbye world", "R is great")
result <- hf_embed_batch(texts, max_active = 5)

# Check for errors
errors <- result[result$.error, ]
} # }
```
