# Sentence Similarity Payload

Calculate the semantic similarity between one text and a list of other
sentences by comparing their embeddings.

## Usage

``` r
hf_sentence_similarity_payload(source_sentence, sentences)
```

## Arguments

- source_sentence:

  The string that you wish to compare the other strings with. This can
  be a phrase, sentence, or longer passage, depending on the model being
  used.

- sentences:

  A list of strings which will be compared against the source_sentence.

## Value

An inference payload

## See also

<https://huggingface.co/docs/inference-providers/tasks/feature-extraction>

## Examples

``` r
hf_sentence_similarity_payload(
  source_sentence = "A happy person",
  sentences = list("A joyful person", "A rainy day")
)
```
