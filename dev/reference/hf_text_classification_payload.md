# Text Classification Payload

Usually used for sentiment-analysis this will output the likelihood of
classes of an input.

## Usage

``` r
hf_text_classification_payload(string)
```

## Arguments

- string:

  a string to be classified

## Value

An inference payload

## See also

<https://huggingface.co/docs/inference-providers/tasks/text-classification>

## Examples

``` r
hf_text_classification_payload("I love using R.")
```
