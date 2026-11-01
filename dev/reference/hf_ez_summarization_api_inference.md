# Summarization API Inference

Summarization API Inference

## Usage

``` r
hf_ez_summarization_api_inference(
  string,
  min_length = NULL,
  max_length = NULL,
  top_k = NULL,
  top_p = NULL,
  temperature = 1,
  repetition_penalty = NULL,
  max_time = NULL,
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

  a string to be summarized

- min_length:

  Integer to define the minimum length in tokens of the output summary.
  Default: NULL

- max_length:

  Integer to define the maximum length in tokens of the output summary.
  Default: NULL

- top_k:

  Integer to define the top tokens considered within the sample
  operation to create new text. Default: NULL

- top_p:

  Float to define the tokens that are within the sample operation of
  text generation. Add tokens in the sample for more probable to least
  probable until the sum of the probabilities is greater than top_p.
  Default: NULL

- temperature:

  Float (0.0-100.0). The temperature of the sampling operation. 1 means
  regular sampling, 0 means always take the highest score, 100.0 is
  getting closer to uniform probability. Default: 1.0

- repetition_penalty:

  Float (0.0-100.0). The more a token is used within generation the more
  it is penalized to not be picked in successive generation passes.
  Default: NULL

- max_time:

  Float (0-120.0). The amount of time in seconds that the query should
  take maximum. Network can cause some overhead so it will be a soft
  limit. Default: NULL

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
