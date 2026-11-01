# Answer Questions about a Text based on Context

Want to have a nice know-it-all bot that can answer any question?

## Usage

``` r
hf_ez_question_answering(
  model_id = "deepset/roberta-base-squad2",
  use_api = FALSE
)
```

## Arguments

- model_id:

  A model_id. Run hf_search_models(...) for model_ids. Defaults to
  'deepset/roberta-base-squad2'.

- use_api:

  Whether to use the Inference API to run the model (TRUE) or download
  and run the model locally (FALSE). Defaults to FALSE

## Value

A question answering object

## See also

<https://huggingface.co/docs/inference-providers/tasks/question-answering>

## Examples

``` r
if (FALSE) { # \dontrun{
# Load the default model and use local inference
ez <- hf_ez_question_answering()
ez$infer(question = "What's my name?", context = "My name is Clara and I live in Berkeley.")

# Use the api for inference.
ez <- hf_ez_fill_mask(use_api = TRUE)
ez$infer(question = "What's my name?", context = "My name is Clara and I live in Berkeley.")
} # }
```
