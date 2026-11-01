# Get Hugging Face API Token

Internal function to retrieve the API token from environment or
parameter.

## Usage

``` r
hf_get_token(token = NULL, required = FALSE)
```

## Arguments

- token:

  Character string or NULL

- required:

  Logical. If TRUE, throws error if no token found.

## Value

Character string with token, or NULL if not found and not required.
