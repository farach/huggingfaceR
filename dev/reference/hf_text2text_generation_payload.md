# Text2Text Generation Payload

takes an input containing the sentence including the task and returns
the output of the accomplished task.

## Usage

``` r
hf_text2text_generation_payload(string)
```

## Arguments

- string:

  a string containing a question or task and a sentence from which the
  answer is derived

## Value

An inference payload

## See also

<https://huggingface.co/docs/inference-providers/tasks/text-generation>

## Examples

``` r
hf_text2text_generation_payload("translate English to French: Hello")
```
