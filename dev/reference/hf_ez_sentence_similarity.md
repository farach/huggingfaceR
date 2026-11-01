# Compare Sentence Similarity Semantically

Calculate the semantic similarity between one text and a list of other
sentences by comparing their embeddings.

## Usage

``` r
hf_ez_sentence_similarity(
  model_id = "sentence-transformers/all-MiniLM-L6-v2",
  use_api = FALSE
)
```

## Arguments

- model_id:

  A model_id. Run hf_search_models(...) for model_ids. Defaults to
  'sentence-transformers/all-MiniLM-L6-v2'.

- use_api:

  Whether to use the Inference API to run the model (TRUE) or download
  and run the model locally (FALSE). Defaults to FALSE

## Value

A sentence similarity object

## See also

<https://huggingface.co/docs/inference-providers/tasks/feature-extraction>

## Examples

``` r
if (FALSE) { # \dontrun{
# Load the default model and use local inference
ez <- hf_ez_sentence_similarity()
ez$infer(
  source_sentence = "That is a happy person",
  sentences = list(
    "That is a happy dog",
    "That is a very happy person",
    "Today is a sunny day"
  )
)

# Use the API for inference.
ez <- hf_ez_sentence_similarity(use_api = TRUE)
ez$infer(
  source_sentence = "That is a happy person",
  sentences = list(
    "That is a happy dog",
    "That is a very happy person",
    "Today is a sunny day"
  )
)
} # }
```
