# Load a pre-trained AutoModel object from Hugging Face

Load a pre-trained AutoModel object from Hugging Face

## Usage

``` r
hf_load_model(model_id, ...)
```

## Arguments

- model_id:

  model_id The id of the model given in the url by
  https://huggingface.co/model_name.

- ...:

  sent to AutoModel.from_pretrained()

## Value

a pre-trained model object

## Examples

``` r
if (FALSE) { # \dontrun{
model <- hf_load_model("distilbert-base-uncased")
} # }
```
