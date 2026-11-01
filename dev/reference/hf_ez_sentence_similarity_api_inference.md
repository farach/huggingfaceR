# Sentence Similarity API Inference

Sentence Similarity API Inference

## Usage

``` r
hf_ez_sentence_similarity_api_inference(
  source_sentence,
  sentences,
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

- source_sentence:

  The string that you wish to compare the other strings with. This can
  be a phrase, sentence, or longer passage, depending on the model being
  used.

- sentences:

  A list of strings which will be compared against the source_sentence.

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
