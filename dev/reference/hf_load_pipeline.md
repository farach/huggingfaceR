# Load a pipeline object from Hugging Face - pipelines usually include a model, tokenizer and task.

Load Model from Hugging Face

## Usage

``` r
hf_load_pipeline(model_id, tokenizer = NULL, task = NULL, ...)
```

## Arguments

- model_id:

  The id of the model given in the url by
  https://huggingface.co/model_name.

- tokenizer:

  The tokenizer function used to tokenize inputs. Defaults to NULL (one
  will be automatically loaded).

- task:

  The task the model will accomplish. Run hf_list_tasks() for options.

- ...:

  Fed to the hf_pipeline function

## Value

A Hugging Face model ready for prediction.

## See also

<https://huggingface.co/docs/transformers/main/en/pipeline_tutorial>

## Examples

``` r
if (FALSE) { # \dontrun{
pipe <- hf_load_pipeline("distilbert-base-uncased-finetuned-sst-2-english",
                         task = "sentiment-analysis")
pipe("I love R.")
} # }
```
