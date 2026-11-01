# Fill Mask Local Inference

Fill Mask Local Inference

## Usage

``` r
hf_ez_fill_mask_local_inference(string, tidy = TRUE, ...)
```

## Arguments

- string:

  a string to be filled from, must contain the \[MASK\] token (check
  model card for exact name of the mask)

- tidy:

  Whether to tidy the results into a tibble. Default: TRUE (tidy the
  results)

- ...:

  Additional arguments passed internally, including the model object or
  model ID.

## Value

The results of the inference

## See also

<https://huggingface.co/docs/transformers/main/en/pipeline_tutorial>
