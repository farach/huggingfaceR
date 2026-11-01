# Summarize a Text

This task is well known to summarize longer text into shorter text. Be
careful, some models have a maximum length of input. That means that the
summary cannot handle full books for instance. Be careful when choosing
your model.

## Usage

``` r
hf_ez_summarization(model_id = "facebook/bart-large-cnn", use_api = FALSE)
```

## Arguments

- model_id:

  A model_id. Run hf_search_models(...) for model_ids. Defaults to
  'facebook/bart-large-cnn'.

- use_api:

  Whether to use the Inference API to run the model (TRUE) or  
  download and run the model locally (FALSE). Defaults to FALSE

## Value

A summarization object

## See also

<https://huggingface.co/docs/inference-providers/tasks/summarization>

## Examples

``` r
if (FALSE) { # \dontrun{
# Load the default model and use local inference
ez <- hf_ez_summarization()
text <- paste(
  "The Eiffel Tower is 324 metres tall and located in Paris.",
  "It was the world's tallest man-made structure for 41 years."
)
ez$infer(string = text, min_length = 10, max_length = 40)

# Load a specific model and use the API for inference.
ez <- hf_ez_summarization(model_id = 'xlm-roberta-base', use_api = TRUE)
ez$infer(string = "huggingface is the <mask>!")
} # }
```
