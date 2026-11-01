# Text Generation Payload

Use to continue text from a prompt. This is a very generic task.

## Usage

``` r
hf_text_generation_payload(
  string,
  top_k = NULL,
  top_p = NULL,
  temperature = 1,
  repetition_penalty = NULL,
  max_new_tokens = NULL,
  max_time = NULL,
  return_full_text = TRUE,
  num_return_sequences = 1L,
  do_sample = TRUE
)
```

## Arguments

- string:

  a string to be generated from

- top_k:

  (Default: None). Integer to define the top tokens considered within
  the sample operation to create new text.

- top_p:

  (Default: None). Float to define the tokens that are within the sample
  operation of text generation. Add tokens in the sample for more
  probable to least probable until the sum of the probabilities is
  greater than top_p

- temperature:

  Float (0.0-100.0). The temperature of the sampling operation. 1 means
  regular sampling, 0 means always take the highest score, 100.0 is
  getting closer to uniform probability. Default: 1.0

- repetition_penalty:

  (Default: None). Float (0.0-100.0). The more a token is used within
  generation the more it is penalized to not be picked in successive
  generation passes.

- max_new_tokens:

  (Default: None). Int (0-250). The amount of new tokens to be
  generated, this does not include the input length it is a estimate of
  the size of generated text you want. Each new tokens slows down the
  request, so look for balance between response times and length of text
  generated.

- max_time:

  (Default: None). Float (0-120.0). The amount of time in seconds that
  the query should take maximum. Network can cause some overhead so it
  will be a soft limit. Use that in combination with max_new_tokens for
  best results.

- return_full_text:

  (Default: True). Bool. If set to False, the return results will not
  contain the original query making it easier for prompting.

- num_return_sequences:

  (Default: 1). Integer. The number of proposition you want to be
  returned.

- do_sample:

  (Optional: True). Bool. Whether or not to use sampling, use greedy
  decoding otherwise.

## Value

An inference payload

## See also

<https://huggingface.co/docs/inference-providers/tasks/text-generation>

## Examples

``` r
hf_text_generation_payload("Once upon a time", max_new_tokens = 10)
```
