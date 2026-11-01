# Perform Text Classification with No Context Required

This task is super useful to try out classification with zero code, you
simply pass a sentence/paragraph and the possible labels for that
sentence, and you get a result.

## Usage

``` r
hf_ez_zero_shot_classification(
  model_id = "facebook/bart-large-mnli",
  use_api = FALSE
)
```

## Arguments

- model_id:

  A model_id. Run hf_search_models(...) for model_ids. Defaults to
  'facebook/bart-large-mnli'.

- use_api:

  Whether to use the Inference API to run the model (TRUE) or download
  and run the model locally (FALSE). Defaults to FALSE

## Value

A zero shot classification object

## See also

<https://huggingface.co/docs/inference-providers/tasks/zero-shot-classification>

## Examples

``` r
if (FALSE) { # \dontrun{
# Load the default model
ez <- hf_ez_zero_shot_classification()

# Classify the string
ez$infer(
  string = paste(
    "Hi, I recently bought a device from your company but it is not working",
    "as advertised and I would like to get reimbursed!"
  ),
  candidate_labels = c("refund", "legal", "faq")
)
} # }
```
