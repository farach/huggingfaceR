# Get Model Information

Retrieve detailed information about a specific model.

## Usage

``` r
hf_model_info(model_id, token = NULL)
```

## Arguments

- model_id:

  Character string. The model ID (e.g., "bert-base-uncased").

- token:

  Character string or NULL. API token for authentication.

## Value

A list with detailed model information.

## Examples

``` r
if (FALSE) { # \dontrun{
# Get model details
hf_model_info("sentence-transformers/all-MiniLM-L6-v2")
} # }
```
