# Search Papers on Hugging Face

Search papers indexed by Hugging Face.

## Usage

``` r
hf_search_papers(search = NULL, limit = 30, token = NULL)
```

## Arguments

- search:

  Character string or NULL. Search query.

- limit:

  Integer. Maximum number of papers to return.

- token:

  Character string or NULL. API token for authentication.

## Value

A tibble with paper metadata.

## Examples

``` r
if (FALSE) { # \dontrun{
hf_search_papers("transformers", limit = 10)
} # }
```
