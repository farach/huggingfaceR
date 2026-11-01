# List Authors

List Model Authors

## Usage

``` r
hf_list_authors(pattern = NULL)
```

## Arguments

- pattern:

  A search term or regular expression. Defaults to NULL (return all
  results).

## Value

A character vector of matching model authors.

## See also

<https://huggingface.co/docs/hub/searching-the-hub>

## Examples

``` r
if (FALSE) { # \dontrun{
hf_list_authors(pattern = "^sam")
} # }
```
