# Generate Text from a Prompt

Use to continue text from a prompt. This is a very generic task.

## Usage

``` r
hf_ez_text_generation(model_id = "openai-community/gpt2", use_api = FALSE)
```

## Arguments

- model_id:

  A model_id. Run hf_search_models(...) for model_ids. Defaults to
  'gpt2'.

- use_api:

  Whether to use the Inference API to run the model (TRUE) or download
  and run the model locally (FALSE). Defaults to FALSE

## Value

A text generation object

## See also

<https://huggingface.co/docs/inference-providers/tasks/text-generation>

## Examples

``` r
if (FALSE) { # \dontrun{
# Load the default model and use local inference
ez <- hf_ez_text_generation()
ez$infer(string = 'The answer to the universe is')

# Use the api for inference.
ez <- hf_ez_text_generation(use_api = TRUE)
ez$infer(string = 'The answer to the universe is')
} # }
```
