# Translation Payload

This task is well known to translate text from one language to another

## Usage

``` r
hf_translation_payload(string)
```

## Arguments

- string:

  a string to be translated in the original languages

## Value

An inference payload

## See also

<https://huggingface.co/docs/inference-providers/tasks/translation>

## Examples

``` r
hf_translation_payload("Hello, world.")
```
