# Conversational Local Inference

Conversational Local Inference

## Usage

``` r
hf_ez_conversational_local_inference(
  text,
  generated_responses = NULL,
  past_user_inputs = NULL,
  min_length = NULL,
  max_length = NULL,
  top_k = NULL,
  top_p = NULL,
  temperature = 1,
  max_time = NULL,
  tidy = TRUE,
  ...
)
```

## Arguments

- text:

  The last input from the user in the conversation.

- generated_responses:

  A list of strings corresponding to the earlier replies from the model.

- past_user_inputs:

  A list of strings corresponding to the earlier replies from the user.
  Should be of the same length of generated_responses.

- min_length:

  (Default: None). Integer to define the minimum length in tokens of the
  output summary.

- max_length:

  (Default: None). Integer to define the maximum length in tokens of the
  output summary.

- top_k:

  (Default: None). Integer to define the top tokens considered within
  the sample operation to create new text.

- top_p:

  (Default: None). Float to define the tokens that are within the sample
  operation of text generation. Add tokens in the sample for more
  probable to least probable until the sum of the probabilities is
  greater than top_p.

- temperature:

  (Default: 1.0). Float (0.0-100.0). The temperature of the sampling
  operation. 1 means regular sampling, 0 means always take the highest
  score, 100.0 is getting closer to uniform probability.

- max_time:

  (Default: None). Float (0-120.0). The amount of time in seconds that
  the query should take maximum. Network can cause some overhead so it
  will be a soft limit.

- tidy:

  Whether to tidy the results into a tibble. Default: TRUE (tidy the
  results)

- ...:

  Additional arguments passed internally, including the model object or
  model ID.

## Value

The results of the inference

## See also

<https://huggingface.co/docs/transformers/main/en/pipeline_tutorial>
