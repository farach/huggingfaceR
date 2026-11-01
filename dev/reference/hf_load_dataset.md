# Load Dataset via Hugging Face Datasets Server API

Load a dataset from Hugging Face Hub using the Datasets Server API. This
is an API-first approach that doesn't require Python. For local dataset
loading with Python, see the legacy function or advanced vignette.

## Usage

``` r
hf_load_dataset(
  dataset,
  split = "train",
  config = NULL,
  limit = 1000,
  offset = 0,
  token = NULL,
  format = c("rows", "parquet"),
  max_active = 4L
)
```

## Arguments

- dataset:

  Character string. Dataset name (e.g., "imdb", "squad").

- split:

  Character string. Dataset split: "train", "test", "validation", etc.
  Supports Hugging Face slice syntax such as \`"train\[100:200\]"\`.
  Percentage slices like \`"train\[:10\\ "train".

- config:

  Character string or NULL. Dataset configuration/subset name. If NULL
  (default), auto-detected from the dataset's available configs.

- limit:

  Integer. Maximum number of rows to fetch. Default: 1000. Set to Inf to
  fetch all rows (may be slow for large datasets).

- offset:

  Integer. Row offset for pagination. Default: 0.

- token:

  Character string or NULL. API token for private datasets.

- format:

  Either `"rows"` (default), which reads JSON pages from the dataset
  viewer, or `"parquet"`, which reads the split's Parquet export with
  the arrow package.

- max_active:

  Integer. The most page or shard requests sent at once. Dataset viewer
  reads are not billed, so the default is 4. Use 1 to send requests one
  at a time.

## Value

A tibble with dataset rows, plus .dataset and .split columns.

## Details

Rows are requested in pages of 100. The first page reports the size of
the split, so the remaining pages are requested together, up to
`max_active` at a time, and combined in order.

With `format = "parquet"`, the split's Parquet export is downloaded and
read with the arrow package instead. Shards of the export hold about 500
MB each and are read whole, so this format suits large or complete
splits (`limit = Inf`). For a few rows the default `"rows"` format is
faster. Column types follow the Parquet schema, so they can differ from
`"rows"`: images and audio arrive as data-frame columns of raw bytes and
file names rather than URLs.

Each `"rows"` request returns at most 100 rows, and Hugging Face caps
how many requests a user or IP address can send in each 5-minute window.
Anonymous requests get the lowest cap, so set a token with
\[hf_set_token()\]. To load more than a few thousand rows, use
`format = "parquet"`, which needs far fewer requests. Requests that hit
the cap are retried after the wait the server asks for.

## See also

\[hf_dataset_search()\], \[hf_dataset_filter()\], and
\[hf_dataset_statistics()\] to query a split without loading it.

## Examples

``` r
if (FALSE) { # \dontrun{
# Load first 1000 rows of IMDB train set
imdb <- hf_load_dataset("imdb", split = "train", limit = 1000)

# Load test set
imdb_test <- hf_load_dataset("imdb", split = "test", limit = 500)

# Load a slice of a split
imdb_sample <- hf_load_dataset("imdb", split = "train[100:200]", limit = Inf)

# Load a whole split from its Parquet export (requires arrow)
imdb_all <- hf_load_dataset("imdb", split = "train", limit = Inf,
                            format = "parquet")
} # }
```
