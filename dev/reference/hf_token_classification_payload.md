# Token Classification Payload

Usually used for sentence parsing, either grammatical, or Named Entity
Recognition (NER) to understand keywords contained within text.

## Usage

``` r
hf_token_classification_payload(string, aggregation_strategy = "simple")
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

## Value

An inference payload

## See also

<https://huggingface.co/docs/inference-providers/tasks/token-classification>

## Examples

``` r
hf_token_classification_payload("My name is Sarah Jessica Parker.")
```
