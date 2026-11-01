# Import a pre-trained AutoModel object for a specific task.

Use this function when you need to load an AutoModel for a specific task
separate to a pipeline. The model's config should come ready-equipped
for a specific task according to what you input as model_type.

## Usage

``` r
hf_load_AutoModel_for_task(
  model_type = "AutoModelForSequenceClassification",
  model_id,
  use_auth_token = NULL
)
```

## Arguments

- model_type:

  The AutoModel's type passed as a string e.g.
  c("AutoModelForQuestionAnswering", "AutoModelForTokenClassification",
  "AutoModelForSequenceClassification")

- model_id:

  The model's name or id on the Hugging Face hub

- use_auth_token:

  For private models, copy and paste your auth token in as a string.

## Value

an AutoModel object for a specific task

## See also

<https://huggingface.co/transformers/v3.0.2/model_doc/auto.html>

## Examples

``` r
if (FALSE) { # \dontrun{
model <- hf_load_AutoModel_for_task(
  "AutoModelForSequenceClassification",
  "distilbert-base-uncased-finetuned-sst-2-english"
)
} # }
```
