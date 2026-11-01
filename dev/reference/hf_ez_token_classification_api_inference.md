# Token Classification API Inference

Token Classification API Inference

## Usage

``` r
hf_ez_token_classification_api_inference(
  string,
  aggregation_strategy = "simple",
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

  a string to be classified

- aggregation_strategy:

  (Default: simple). There are several aggregation strategies.  
  none: Every token gets classified without further aggregation.  
  simple: Entities are grouped according to the default schema (B-, I-
  tags get merged when the tag is similar).  
  first: Same as the simple strategy except words cannot end up with
  different tags. Words will use the tag of the first token when there
  is ambiguity.  
  average: Same as the simple strategy except words cannot end up with
  different tags. Scores are averaged across tokens and then the maximum
  label is applied.  
  max: Same as the simple strategy except words cannot end up with
  different tags. Word entity will be the token with the maximum
  score.  

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
