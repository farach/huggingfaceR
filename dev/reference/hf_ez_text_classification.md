# Classify Texts into Pre-trained Categories

Usually used for sentiment-analysis this will output the likelihood of
classes of an input.

## Usage

``` r
hf_ez_text_classification(
  model_id = "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
  use_api = FALSE
)
```

## Arguments

- model_id:

  A model_id. Run hf_search_models(...) for model_ids. Defaults to
  'distilbert-base-uncased-finetuned-sst-2-english'.

- use_api:

  Whether to use the Inference API to run the model (TRUE) or download
  and run the model locally (FALSE). Defaults to FALSE

## Value

A text classification object

## See also

<https://huggingface.co/docs/inference-providers/tasks/text-classification>

## Examples

``` r
if (FALSE) { # \dontrun{
# Load the default model and use local inference
ez <- hf_ez_text_classification()
ez$infer(string = c('I like you. I love you'), flatten = FALSE)

# Use the API for inference.
ez <- hf_ez_text_classification(use_api = TRUE)
ez$infer(string = c('I like you. I love you'), flatten = FALSE)
} # }
```
