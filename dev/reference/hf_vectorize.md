# Vectorize API Calls

Internal helper to apply an API function to multiple inputs efficiently.

## Usage

``` r
hf_vectorize(inputs, fn, ..., .progress = FALSE)
```

## Arguments

- inputs:

  Character vector of inputs.

- fn:

  Function to apply to each input.

- ...:

  Additional arguments passed to fn.

- .progress:

  Logical. Show progress bar?

## Value

Combined results from all API calls.
