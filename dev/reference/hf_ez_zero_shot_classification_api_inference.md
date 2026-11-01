# Zero Shot Classification API Inference

Zero Shot Classification API Inference

## Usage

``` r
hf_ez_zero_shot_classification_api_inference(
  string,
  candidate_labels,
  multi_label = FALSE,
  tidy = TRUE,
  use_gpu = FALSE,
  use_cache = FALSE,
  wait_for_model = FALSE,
  use_auth_token = NULL,
  stop_on_error = FALSE,
  ...
)
```

## Arguments

- string:

  a string or list of strings

- candidate_labels:

  a list of strings that are potential classes for inputs. (max 10
  candidate_labels, for more, simply run multiple requests, results are
  going to be misleading if using too many candidate_labels anyway. If
  you want to keep the exact same, you can simply run multi_label=True
  and do the scaling on your end. )

- multi_label:

  (Default: false) Boolean that is set to True if classes can overlap

- tidy:

  Whether to tidy the results into a tibble. Default: TRUE (tidy the
  results)

- use_gpu:

  Whether to use GPU for inference.

- use_cache:

  Whether to use cached inference results for previously seen inputs.

- wait_for_model:

  Whether to wait for the model to be ready instead of receiving a 503
  error after a certain amount of time.

- use_auth_token:

  The token to use as HTTP bearer authorization for the Inference API.
  Defaults to HUGGING_FACE_HUB_TOKEN environment variable.

- stop_on_error:

  Whether to throw an error if an API error is encountered. Defaults to
  FALSE (do not throw error).

- ...:

  Additional arguments passed internally, including the model object or
  model ID.

## Value

The results of the inference

## See also

<https://huggingface.co/docs/inference-providers>
