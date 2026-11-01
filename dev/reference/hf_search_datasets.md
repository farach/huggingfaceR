# Search Datasets on Hugging Face Hub

Search for datasets using various filters.

## Usage

``` r
hf_search_datasets(
  search = NULL,
  task = NULL,
  language = NULL,
  size = NULL,
  sort = "downloads",
  limit = 30,
  token = NULL
)
```

## Arguments

- search:

  Character string. Search query to filter datasets.

- task:

  Character string. Filter by task.

- language:

  Character string. Filter by language.

- size:

  Character string. Filter by size: "small", "medium", "large".

- sort:

  Character string. Sort by: "downloads", "likes", "created", "updated".
  Default: "downloads".

- limit:

  Integer. Maximum number of datasets to return. Default: 30.

- token:

  Character string or NULL. API token for authentication.

## Value

A tibble with dataset information.

## Examples

``` r
if (FALSE) { # \dontrun{
# Search datasets
hf_search_datasets(search = "sentiment", limit = 10)
} # }
```
