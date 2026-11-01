# Translate between Languages

This task is well known to translate text from one language to another

## Usage

``` r
hf_ez_translation(model_id = "Helsinki-NLP/opus-mt-en-es", use_api = FALSE)
```

## Arguments

- model_id:

  A model_id. Run hf_search_models(...) for model_ids. Defaults to
  'Helsinki-NLP/opus-mt-en-es'.

- use_api:

  Whether to use the Inference API to run the model (TRUE) or download
  and run the model locally (FALSE). Defaults to FALSE

## Value

A translation object

## See also

<https://huggingface.co/docs/inference-providers/tasks/translation>

## Examples

``` r
if (FALSE) { # \dontrun{
# Load the default translation model
ez <- hf_ez_translation()

# Translate from English to Spanish.
ez$infer(string = "My name is Sarah and I live in London")
} # }
```
