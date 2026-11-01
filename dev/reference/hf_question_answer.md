# Extractive Question Answering

Answer a question from a supplied context passage using an extractive
question-answering model via the Hugging Face Inference Providers API.
The answer is a span extracted verbatim from the context. \`question\`
and \`context\` are recycled to a common length (each may be length 1).

## Usage

``` r
hf_question_answer(
  question,
  context,
  model = hf_default_model("question_answer"),
  token = NULL,
  endpoint_url = NULL,
  max_active = getOption("huggingfaceR.max_active", 1L),
  ...
)
```

## Arguments

- question:

  Character vector of question(s).

- context:

  Character vector of context passage(s) to answer from.

- model:

  Character string. Model ID from the Hugging Face Hub. Append
  \`":provider"\` to select an inference provider. Default:
  "deepset/roberta-base-squad2".

- token:

  Character string or NULL. API token for authentication.

- endpoint_url:

  Character string or NULL. A custom Inference Endpoint URL.

- max_active:

  Integer. Maximum number of inputs to send at once. Default:
  \`getOption("huggingfaceR.max_active", 1)\`, which sends one input at
  a time. Larger values send inputs concurrently. This is faster for
  long vectors but spends provider credits and rate limits faster.
  Results keep input order, transient failures (HTTP 429 and 5xx) are
  retried a bounded number of times, and the first failing input stops
  the call with its error, as in sequential mode. Set
  \`options(huggingfaceR.max_active = 4)\` to change the default for
  every per-input function.

- ...:

  Additional arguments (currently unused).

## Value

A tibble with columns: question, answer, score, start, end.

## See also

<https://huggingface.co/docs/inference-providers/tasks/question-answering>

## Examples

``` r
if (FALSE) { # \dontrun{
hf_question_answer(
  question = "Where was Obama born?",
  context = "Barack Obama was born in Honolulu, Hawaii."
)

# One context, several questions
hf_question_answer(
  question = c("Who?", "Where?"),
  context = "Ada Lovelace worked in London."
)
} # }
```
