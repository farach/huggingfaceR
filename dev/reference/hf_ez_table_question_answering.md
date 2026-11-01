# Answer Questions about a Data Table

Don’t know SQL? Don’t want to dive into a large spreadsheet? Ask
questions in plain english!

## Usage

``` r
hf_ez_table_question_answering(
  model_id = "google/tapas-base-finetuned-wtq",
  use_api = FALSE
)
```

## Arguments

- model_id:

  A model_id. Run hf_search_models(...) for model_ids. Defaults to
  'google/tapas-base-finetuned-wtq'.

- use_api:

  Whether to use the Inference API to run the model (TRUE) or download
  and run the model locally (FALSE). Defaults to FALSE

## Value

A table question answering object

## See also

<https://huggingface.co/docs/inference-providers/tasks/table-question-answering>

## Examples

``` r
if (FALSE) { # \dontrun{
# Create a table to query
qa_table <-
  tibble::tibble(Repository = c('Transformers', 'Datasets', 'Tokenizers'),
                 Stars = c('36542', '4512', '3934'),
                 Contributors = c('651', '77', '34'),
                 Programming.language = c('Python', 'Python', 'Rust, Python and NodeJS'))

# Load the default model and use local inference
ez <- hf_ez_table_question_answering()
ez$infer(query = "How many stars does the transformers repository have?", table = qa_table)

# Use the api for inference.
ez <- hf_ez_fill_mask(use_api = TRUE)
ez$infer(query = "How many stars does the transformers repository have?", table = qa_table)
} # }
```
