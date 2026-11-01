# List Libraries

List Model Libraries

## Usage

``` r
hf_list_libraries(pattern = NULL)
```

## Arguments

- pattern:

  A search term or regular expression. Defaults to NULL (return all
  results).

## Value

A character vector of matching library names.

## See also

<https://huggingface.co/docs/hub/searching-the-hub>

## Examples

``` r
if (FALSE) { # \dontrun{
hf_list_libraries("pytorch|tensorflow")
} # }
```
