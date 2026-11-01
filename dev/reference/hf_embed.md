# Generate Text Embeddings

Generate dense vector representations (embeddings) for text using
transformer models. Useful for semantic similarity, clustering, and as
features for ML models. Vector inputs are sent in a single batched
Inference API request when possible, which is substantially faster than
one API request per text.

## Usage

``` r
hf_embed(
  text,
  model = hf_default_model("embed"),
  token = NULL,
  endpoint_url = NULL,
  ...
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

- endpoint_url:

  Character string or NULL. A custom Inference Endpoint URL. When
  provided, requests are sent to this URL instead of the public
  Inference API. Use for models deployed on dedicated Inference
  Endpoints.

- ...:

  Additional arguments (currently unused).

## Value

A tibble with columns: text, embedding (list-column of numeric vectors),
n_dims

## Examples

``` r
if (FALSE) { # \dontrun{
# Generate embeddings
embeddings <- hf_embed(c("Hello world", "Goodbye world"))

# Access embedding vectors
embeddings$embedding[[1]]  # First embedding vector
} # }
```
