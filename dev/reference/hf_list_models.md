# List Models

List Model Names

## Usage

``` r
hf_list_models(pattern = NULL)
```

## Arguments

- pattern:

  A search term or regular expression. Defaults to NULL (return all
  results).

## Value

A tibble with a \`model\` column containing matching model IDs.

## See also

<https://huggingface.co/docs/hub/searching-the-hub>

## Examples

``` r
if (FALSE) { # \dontrun{
hf_list_models("bert-base-cased")
} # }
```
