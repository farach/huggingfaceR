# Classify parts of a Text

Usually used for sentence parsing, either grammatical, or Named Entity
Recognition (NER) to understand keywords contained within text.

## Usage

``` r
hf_ez_token_classification(
  model_id = "dbmdz/bert-large-cased-finetuned-conll03-english",
  use_api = FALSE
)
```

## Arguments

- model_id:

  A model_id. Run hf_search_models(...) for model_ids. Defaults to
  'dbmdz/bert-large-cased-finetuned-conll03-english'.

- use_api:

  Whether to use the Inference API to run the model (TRUE) or download
  and run the model locally (FALSE). Defaults to FALSE

## Value

A text2text generation object

## See also

<https://huggingface.co/docs/inference-providers/tasks/token-classification>

## Examples

``` r
if (FALSE) { # \dontrun{
# Load the default named entity recognition model
ez <- hf_ez_token_classification()

# Run NER. Note how the full name is aggregated into one named entity.
ez$infer(
  string = "My name is Sarah Jessica Parker but you can call me Jessica",
  aggregation_strategy = "simple"
)

# Run NER without aggregation.
# Note how the full name is separated into distinct named entities.
ez$infer(
  string = "My name is Sarah Jessica Parker but you can call me Jessica",
  aggregation_strategy = "none"
)
} # }
```
