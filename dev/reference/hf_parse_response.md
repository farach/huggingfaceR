# Parse API Response to Tibble

Internal helper to convert API JSON responses to tibbles.

## Usage

``` r
hf_parse_response(resp, input_text = NULL)
```

## Arguments

- resp:

  The response object from httr2.

- input_text:

  Optional. The original input text(s) to include in output.

## Value

A tibble with the parsed results.
