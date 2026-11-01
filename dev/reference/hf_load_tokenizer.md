# Load an AutoTokenizer from a pre-tained model

Load Tokenizer for Hugging Face Model

## Usage

``` r
hf_load_tokenizer(model_id, ...)
```

## Arguments

- model_id:

  The id of the model given in the url by
  https://huggingface.co/model_name.

- ...:

  sent to the \`AutoTokenizer.from_pretained()\`, accepts named
  arguments e.g. use_fast
  <https://huggingface.co/docs/transformers/main_classes/tokenizer>

## Value

A Hugging Face model's tokenizer.

## See also

<https://huggingface.co/docs/transformers/main/en/pipeline_tutorial>

## Examples

``` r
if (FALSE) { # \dontrun{
tokenizer <- hf_load_tokenizer("distilbert-base-uncased")
tokenizer$tokenize("Hello from R")
} # }
```
