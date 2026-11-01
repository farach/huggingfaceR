# Search Spaces on Hugging Face Hub

Search hosted Spaces and return one row per result.

## Usage

``` r
hf_search_spaces(
  search = NULL,
  author = NULL,
  sort = "likes",
  direction = "desc",
  limit = 30,
  token = NULL
)
```

## Arguments

- search:

  Character string or NULL. Search query.

- author:

  Character string or NULL. Filter by owner.

- sort:

  Character string. Sort field passed to the Hub API.

- direction:

  Character string. Sort direction: "asc" or "desc".

- limit:

  Integer. Maximum number of Spaces to return.

- token:

  Character string or NULL. API token for authentication.

## Value

A tibble with Space metadata.

## Examples

``` r
if (FALSE) { # \dontrun{
hf_search_spaces(search = "chat", limit = 10)
} # }
```
