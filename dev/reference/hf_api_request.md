# Make a Request to Hugging Face Inference API

Internal helper function to construct and execute API requests.

## Usage

``` r
hf_api_request(
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

  Character string or NULL. A custom Inference Endpoint URL. When
  provided, requests are sent to this URL instead of the public
  Inference API. Use for dedicated Inference Endpoints.

## Value

The raw response from the API.
