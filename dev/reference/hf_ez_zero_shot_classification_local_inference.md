# Zero Shot Classification Local Inference

Zero Shot Classification Local Inference

## Usage

``` r
hf_ez_zero_shot_classification_local_inference(
  string,
  candidate_labels,
  multi_label = FALSE,
  tidy = TRUE,
  ...
)
```

## Arguments

- string:

  a string or list of strings

- candidate_labels:

  a list of strings that are potential classes for inputs. (max 10
  candidate_labels, for more, simply run multiple requests, results are
  going to be misleading if using too many candidate_labels anyway. If
  you want to keep the exact same, you can simply run multi_label=True
  and do the scaling on your end. )

- multi_label:

  (Default: false) Boolean that is set to True if classes can overlap

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
