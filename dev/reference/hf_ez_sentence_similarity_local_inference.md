# Sentence Similarity Local Inference

Sentence Similarity Local Inference

## Usage

``` r
hf_ez_sentence_similarity_local_inference(
  source_sentence,
  sentences,
  tidy = TRUE,
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

- ...:

  Additional arguments passed internally, including the model object or
  model ID.

## Value

The results of the inference

## See also

<https://huggingface.co/docs/transformers/main/en/pipeline_tutorial>
