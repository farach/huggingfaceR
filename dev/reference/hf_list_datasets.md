# List Datasets

List Model Datasets

## Usage

``` r
hf_list_datasets(pattern = NULL)
```

## Arguments

- pattern:

  A search term or regular expression. Defaults to NULL (return all
  results).

## Value

A character vector of matching dataset names.

## See also

<https://huggingface.co/docs/hub/searching-the-hub>

## Examples

``` r
if (FALSE) { # \dontrun{
hf_list_datasets("imdb")
} # }
```
