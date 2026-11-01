# Inference using a downloaded Hugging Face model or pipeline, or using the Inference API

If a model_id is provided, the Inference API will be used to make the
prediction. If you wish to download a model or pipeline rather than
running your predictions through the Inference API, download the model
with one of the hf_load\_\*\_model() or hf_load_pipeline() functions.

## Usage

``` r
hf_inference(
  model,
  payload,
  flatten = TRUE,
  use_gpu = FALSE,
  use_cache = FALSE,
  wait_for_model = FALSE,
  use_auth_token = NULL,
  stop_on_error = FALSE
)
```

## Arguments

- model:

  Either a downloaded model or pipeline from the Hugging Face Hub (using
  hf_load_pipeline()), or a model_id. Run hf_search_models(...) for
  model_ids.

- payload:

  The data to predict on. Use one of the hf\_\*\_payload() functions to
  create.

- flatten:

  Whether to flatten the results into a data frame. Default: TRUE
  (flatten the results)

- use_gpu:

  API Only - Whether to use GPU for inference.

- use_cache:

  API Only - Whether to use cached inference results for previously seen
  inputs.

- wait_for_model:

  API Only - Whether to wait for the model to be ready instead of
  receiving a 503 error after a certain amount of time.

- use_auth_token:

  API Only - The token to use as HTTP bearer authorization for the
  Inference API. Defaults to HUGGING_FACE_HUB_TOKEN environment
  variable.

- stop_on_error:

  API Only - Whether to throw an error if an API error is encountered.
  Defaults to FALSE (do not throw error).

## Value

The results of the inference

## See also

<https://huggingface.co/docs/inference-providers>

## Examples

``` r
if (FALSE) { # \dontrun{
payload <- hf_text_classification_payload("I love R.")
hf_inference("distilbert-base-uncased-finetuned-sst-2-english", payload)
} # }
```
