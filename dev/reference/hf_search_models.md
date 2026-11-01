# Search Models on Hugging Face Hub

Search for models using various filters. Returns a tibble of matching
models.

## Usage

``` r
hf_search_models(
  search = NULL,
  task = NULL,
  author = NULL,
  language = NULL,
  library = NULL,
  tags = NULL,
  sort = "downloads",
  direction = "desc",
  limit = 30,
  token = NULL,
  inference_provider = NULL
)
```

## Arguments

- search:

  Character string. Search query to filter models.

- task:

  Character string. Filter by task (e.g., "text-classification").

- author:

  Character string. Filter by model author/organization.

- language:

  Character string. Filter by language (e.g., "en").

- library:

  Character string. Filter by library (e.g., "pytorch", "transformers").

- tags:

  Character vector. Filter by tags.

- sort:

  Character string. Sort by field: "downloads", "likes", "created",
  "updated". Default: "downloads".

- direction:

  Character string. Sort direction: "asc" or "desc". Default: "desc".

- limit:

  Integer. Maximum number of models to return. Default: 30.

- token:

  Character string or NULL. API token for authentication.

- inference_provider:

  Character vector or NULL. Keep only models served by these Inference
  Providers (e.g., `"groq"` or `c("together", "fireworks-ai")`). Use
  `"all"` for models served by any provider. See \[hf_list_providers()\]
  for provider names.

## Value

A tibble with model information.

## Examples

``` r
if (FALSE) { # \dontrun{
# Search by task
hf_search_models(task = "text-classification", limit = 10)

# Search by author
hf_search_models(author = "facebook", sort = "downloads")

# Search with query
hf_search_models(search = "sentiment", task = "text-classification")

# Text classification models that some Inference Provider serves
hf_search_models(task = "text-classification", inference_provider = "all")
} # }
```
