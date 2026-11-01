# Perform Fill-in-the-Blank Tasks

Tries to fill in a hole with a missing word (token to be precise).
That’s the base task for BERT models.

## Usage

``` r
hf_ez_fill_mask(model_id = "google-bert/bert-base-uncased", use_api = FALSE)
```

## Arguments

- model_id:

  A model_id. Run hf_search_models(...) for model_ids. Defaults to
  'bert-base-uncased'.

- use_api:

  Whether to use the Inference API to run the model (TRUE) or download
  and run the model locally (FALSE). Defaults to FALSE

## Value

A fill mask object

## See also

<https://huggingface.co/docs/inference-providers/tasks/fill-mask>

## Examples

``` r
if (FALSE) { # \dontrun{
# Load the default model and use local inference
ez <- hf_ez_fill_mask()
ez$infer(string = "The answer to the universe is [MASK].")

# Load a specific model and use the API for inference.
# Note the mask is different for different models.
ez <- hf_ez_fill_mask(model_id = 'xlm-roberta-base', use_api = TRUE)
ez$infer(string = "The answer to the universe is <MASK>.")
} # }
```
