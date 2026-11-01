# Fill Mask Task Payload

Tries to fill in a hole with a missing word (token to be precise).
That’s the base task for BERT models.

## Usage

``` r
hf_fill_mask_payload(string)
```

## Arguments

- string:

  a string to be filled from, must contain the \[MASK\] token (check
  model card for exact name of the mask)

## Value

An inference payload

## See also

<https://huggingface.co/docs/inference-providers/tasks/fill-mask>

## Examples

``` r
hf_fill_mask_payload("The capital of France is [MASK].")
```
