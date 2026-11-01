# Push a Data Frame as a Dataset File

Write a data frame to CSV or Parquet and upload it to a Hub dataset
repository.

## Usage

``` r
hf_push_dataset(
  data,
  repo_id,
  path_in_repo = NULL,
  format = c("csv", "parquet"),
  create_repo = FALSE,
  private = FALSE,
  commit_message = NULL,
  token = NULL,
  overwrite = FALSE,
  confirm = FALSE
)
```

## Arguments

- data:

  A data frame.

- repo_id:

  Character string. Dataset repository ID.

- path_in_repo:

  Character string or NULL. Destination path. Defaults to "data.csv" or
  "data.parquet".

- format:

  Character string. One of "csv" or "parquet".

- create_repo:

  Logical. If TRUE, create the dataset repo first with \`exist_ok =
  TRUE\`.

- private:

  Logical. Used when \`create_repo = TRUE\`.

- commit_message:

  Character string or NULL.

- token:

  Character string or NULL. API token with write scope.

- overwrite:

  Logical. If FALSE, error when the destination already exists.

- confirm:

  Logical. Must be TRUE to perform write operations.

## Value

The result from \`hf_upload_file()\`.

## Examples

``` r
if (FALSE) { # \dontrun{
hf_push_dataset(mtcars, "me/mtcars-small", create_repo = TRUE, confirm = TRUE)
} # }
```
