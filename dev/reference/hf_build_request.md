# Build a Hugging Face API Request

Internal factory function to construct httr2 request objects for batch
processing.

## Usage

``` r
hf_build_request(
  model_id,
  inputs,
  parameters = NULL,
  token = NULL,
  wait_for_model = TRUE,
  use_cache = TRUE,
  endpoint_url = NULL
)
```

## Arguments

- model_id:

  Character string. The model ID on Hugging Face Hub.

- inputs:

  The input data (usually character vector or list).

- parameters:

  Optional list of parameters for the inference.

- token:

  Character string or NULL. API token for authentication.

- wait_for_model:

  Deprecated no-op retained for internal compatibility.

- use_cache:

  Deprecated no-op retained for internal compatibility.

- endpoint_url:

  Character string or NULL. A custom Inference Endpoint URL.

## Value

An httr2 request object ready for execution.
