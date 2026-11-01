# Table Question Answering

Ask a question in plain language about a data frame, using a table
question-answering model (such as TAPAS) via the Hugging Face Inference
Providers API. The data frame is converted to the string-cell format the
API expects; all values are coerced to character.

## Usage

``` r
hf_table_question_answer(
  query,
  table,
  model = hf_default_model("table_question_answer"),
  token = NULL,
  endpoint_url = NULL,
  max_active = getOption("huggingfaceR.max_active", 1L),
  ...
)
```

## Arguments

- query:

  Character vector of question(s) to ask about the table.

- table:

  A data frame to query.

- model:

  Character string. Model ID from the Hugging Face Hub. Append
  \`":provider"\` to select an inference provider. Default:
  "google/tapas-base-finetuned-wtq".

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

A tibble with columns: query, answer, aggregator, cells (a list-column
of the source cells the answer was drawn from).

## See also

<https://huggingface.co/docs/inference-providers/tasks/table-question-answering>

## Examples

``` r
if (FALSE) { # \dontrun{
sales <- data.frame(
  product = c("Widgets", "Gadgets", "Gizmos"),
  revenue = c(120, 80, 50)
)
hf_table_question_answer("Which product had the highest revenue?", sales)
hf_table_question_answer("What is the total revenue?", sales)
} # }
```
