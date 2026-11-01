# Zero Shot Classification Payload

This task is super useful to try out classification with zero code, you
simply pass a sentence/paragraph and the possible labels for that
sentence, and you get a result.

## Usage

``` r
hf_zero_shot_classification_payload(
  string,
  candidate_labels,
  multi_label = FALSE
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

## Value

An inference payload

## See also

<https://huggingface.co/docs/inference-providers/tasks/zero-shot-classification>

## Examples

``` r
hf_zero_shot_classification_payload(
  "I need help with my laptop.",
  candidate_labels = c("technology", "sports", "food")
)
```
