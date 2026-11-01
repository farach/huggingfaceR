# Get Column Statistics for a Dataset Split

Retrieve the summary statistics that the Hugging Face dataset viewer
computes for each column of a split, without downloading the data.

## Usage

``` r
hf_dataset_statistics(dataset, split = "train", config = NULL, token = NULL)
```

## Arguments

- dataset:

  Character string. Dataset ID (e.g., "stanfordnlp/imdb"). Short names
  are resolved as in \[hf_load_dataset()\].

- split:

  Character string. Split name. Default: "train". Slice syntax is not
  supported here.

- config:

  Character string or NULL. Dataset configuration. If NULL (default), it
  is detected from the dataset's available configs.

- token:

  Character string or NULL. API token. The search and filter endpoints
  can refuse anonymous requests, so a token is recommended.

## Value

A tibble with one row per column: `column`, `type`, `nan_count`,
`nan_proportion`, `n_unique`, `min`, `max`, `mean`, `median`, `std`,
`histogram` (a list of tibbles with `bin_start`, `bin_end`, and
`count`), `frequencies` (a list of tibbles with `value` and `count`),
and `statistics` (the raw API fields). The number of rows in the split
is stored in the `"num_examples"` attribute.

## Details

Which statistics are filled in depends on the column type. Numeric
columns have `min`, `max`, `mean`, `median`, `std`, and a histogram.
Text and list columns describe their lengths in the same way. Label and
boolean columns have `n_unique` and `frequencies`. The `statistics`
column keeps every field the API returned, including summaries with no
numeric form such as datetime ranges.

## Examples

``` r
if (FALSE) { # \dontrun{
stats <- hf_dataset_statistics("cornell-movie-review-data/rotten_tomatoes")
stats
stats$frequencies[[1]]
attr(stats, "num_examples")
} # }
```
