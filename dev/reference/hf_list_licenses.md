# List Licenses

List Model Licenses

## Usage

``` r
hf_list_licenses(pattern = NULL)
```

## Arguments

- pattern:

  A search term or regular expression. Defaults to NULL (return all
  results).

## Value

A character vector of matching license identifiers.

## See also

<https://huggingface.co/docs/hub/searching-the-hub>

## Examples

``` r
if (FALSE) { # \dontrun{
hf_list_licenses("mit")
} # }
```
