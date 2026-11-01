# Filter Dataset Rows with a Condition

Select rows of one dataset split with a SQL-style condition, run by the
Hugging Face dataset viewer. Only the matching rows are downloaded.

## Usage

``` r
hf_dataset_filter(
  dataset,
  where,
  orderby = NULL,
  split = "train",
  config = NULL,
  limit = 100,
  offset = 0,
  max_active = 4L,
  token = NULL
)
```

## Arguments

- dataset:

  Character string. Dataset ID (e.g., "stanfordnlp/imdb"). Short names
  are resolved as in \[hf_load_dataset()\].

- where:

  Character string. The condition rows must meet.

- orderby:

  Character string or NULL. Sort order, for example `"\"score\" DESC"`.
  Default: NULL (split order).

- split:

  Character string. Split name. Default: "train". Slice syntax is not
  supported here.

- config:

  Character string or NULL. Dataset configuration. If NULL (default), it
  is detected from the dataset's available configs.

- limit:

  Integer. Maximum number of matching rows to return. Default: 100. Use
  Inf for all matches.

- offset:

  Integer. Number of matching rows to skip. Default: 0.

- max_active:

  Integer. The most page requests sent at once. Default: 4.

- token:

  Character string or NULL. API token. The search and filter endpoints
  can refuse anonymous requests, so a token is recommended.

## Value

A tibble with one row per match, plus `.row_idx`, `.dataset`, and
`.split`.

## Details

The `where` condition compares columns with `=`, `<>`, `>`, `>=`, `<`,
and `<=`, and joins comparisons with `AND`, `OR`, and `NOT`. Put column
names in double quotes and text values in single quotes, for example
`"\"label\" = 1 AND \"lang\" = 'en'"`. Filtering works for datasets that
have a Parquet export.

## See also

\[hf_dataset_search()\] for full-text search.

## Examples

``` r
if (FALSE) { # \dontrun{
hf_dataset_filter("cornell-movie-review-data/rotten_tomatoes",
                  where = "\"label\" = 1", limit = 50)
} # }
```
