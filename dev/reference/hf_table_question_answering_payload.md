# Table Question Answering Payload

Don’t know SQL? Don’t want to dive into a large spreadsheet? Ask
questions in plain english!

## Usage

``` r
hf_table_question_answering_payload(query, table)
```

## Arguments

- query:

  The query in plain text that you want to ask the table

- table:

  A table of data represented as a dict of list where entries are
  headers and the lists are all the values, all lists must have the same
  size.

## Value

An inference payload

## See also

<https://huggingface.co/docs/inference-providers/tasks/table-question-answering>

## Examples

``` r
hf_table_question_answering_payload(
  query = "How many rows are shown?",
  table = list(name = c("Alice", "Bob"), rows = c("1", "2"))
)
```
