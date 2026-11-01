# Question Answering Payload

Want to have a nice know-it-all bot that can answer any question?

## Usage

``` r
hf_question_answering_payload(question, context)
```

## Arguments

- question:

  a question to be answered based on the provided context

- context:

  the context to consult for answering the question

## Value

An inference payload

## See also

<https://huggingface.co/docs/inference-providers/tasks/question-answering>

## Examples

``` r
hf_question_answering_payload(
  question = "What is R?",
  context = "R is a language for statistical computing."
)
```
