# Answer General Questions

Essentially Text-generation task. But uses Encoder-Decoder architecture,
so might change in the future for more options.

## Usage

``` r
hf_ez_text2text_generation(model_id = "google/flan-t5-large", use_api = FALSE)
```

## Arguments

- model_id:

  A model_id. Run hf_search_models(...) for model_ids. Defaults to
  'google/flan-t5-small'.

- use_api:

  Whether to use the Inference API to run the model (TRUE) or download
  and run the model locally (FALSE). Defaults to FALSE

## Value

A text2text generation object

## See also

<https://huggingface.co/docs/inference-providers/tasks/text-generation>

## Examples

``` r
if (FALSE) { # \dontrun{
# Load the default model and use local inference
ez <- hf_ez_text2text_generation()
ez$infer("Please answer the following question. What is the boiling point of Nitrogen?")

# Use the api for inference.
ez <- hf_ez_text2text_generation(use_api = TRUE)
ez$infer("Please answer the following question. What is the boiling point of Nitrogen?")
} # }
```
