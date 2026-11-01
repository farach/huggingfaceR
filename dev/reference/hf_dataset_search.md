# Search Dataset Rows by Text

Full-text search over the string columns of one dataset split, run by
the Hugging Face dataset viewer. Only the matching rows are downloaded.

## Usage

``` r
hf_dataset_search(
  dataset,
  query,
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

- query:

  Character string. The text to search for.

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

A tibble with one row per match, plus `.row_idx` (the row's 0-based
position in the split), `.dataset`, and `.split`.

## Details

Search works for datasets that have a Parquet export. Use
\[hf_dataset_validity()\] to check before searching. For large datasets
the viewer indexes only the first part of the split, and a message says
so.

## See also

\[hf_dataset_filter()\] to select rows with a condition.

## Examples

``` r
if (FALSE) { # \dontrun{
hf_dataset_search("cornell-movie-review-data/rotten_tomatoes", "dull",
                  limit = 20)
} # }
```
