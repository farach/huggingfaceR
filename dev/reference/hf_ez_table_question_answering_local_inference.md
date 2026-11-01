# Table Question Answering Local Inference

Table Question Answering Local Inference

## Usage

``` r
hf_ez_table_question_answering_local_inference(query, table, tidy = TRUE, ...)
```

## Arguments

- query:

  The query in plain text that you want to ask the table

- table:

  A dataframe with all text columns.

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
