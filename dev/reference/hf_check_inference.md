# Check Inference API Availability

Check whether a model supports the Hugging Face Serverless Inference
API. Not all models on the Hub are served by the Inference API. This
function queries model metadata to determine availability before you
make inference calls.

## Usage

``` r
hf_check_inference(model_id, token = NULL, quiet = FALSE)
```

## Arguments

- model_id:

  Character string. The model ID (e.g., "BAAI/bge-small-en-v1.5").

- token:

  Character string or NULL. API token for authentication.

- quiet:

  Logical. If TRUE, suppress console output and return result invisibly.
  Default: FALSE.

## Value

A list (invisibly if quiet = TRUE) with components:

- model_id:

  The model ID queried.

- available:

  Logical. TRUE if the model is available on the Inference API.

- pipeline_tag:

  The model's task type (e.g., "feature-extraction").

- inference_provider:

  The inference provider, if available.

## Examples

``` r
if (FALSE) { # \dontrun{
# Check if a model supports serverless inference
hf_check_inference("BAAI/bge-small-en-v1.5")

# Use programmatically
result <- hf_check_inference("some-org/some-model", quiet = TRUE)
if (result$available) {
  embeddings <- hf_embed("hello", model = "some-org/some-model")
}
} # }
```
