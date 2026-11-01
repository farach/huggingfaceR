# Token Classification Local Inference

Token Classification Local Inference

## Usage

``` r
hf_ez_token_classification_local_inference(
  string,
  aggregation_strategy = "simple",
  tidy = TRUE,
  ...
)
```

## Arguments

- string:

  a string to be classified

- aggregation_strategy:

  (Default: simple). There are several aggregation strategies.  
  none: Every token gets classified without further aggregation.  
  simple: Entities are grouped according to the default schema (B-, I-
  tags get merged when the tag is similar).  
  first: Same as the simple strategy except words cannot end up with
  different tags. Words will use the tag of the first token when there
  is ambiguity.  
  average: Same as the simple strategy except words cannot end up with
  different tags. Scores are averaged across tokens and then the maximum
  label is applied.  
  max: Same as the simple strategy except words cannot end up with
  different tags. Word entity will be the token with the maximum
  score.  

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
