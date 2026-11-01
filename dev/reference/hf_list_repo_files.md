# List Files in a Hub Repository

List files and directories in a model, dataset, or Space repository.

## Usage

``` r
hf_list_repo_files(
  repo_id,
  repo_type = "model",
  revision = "main",
  recursive = TRUE,
  token = NULL
)
```

## Arguments

- repo_id:

  Character string. Repository ID, e.g. "BAAI/bge-small-en-v1.5".

- repo_type:

  Character string. One of "model", "dataset", or "space".

- revision:

  Character string. Git revision, branch, or tag. Default: "main".

- recursive:

  Logical. If TRUE, list files recursively.

- token:

  Character string or NULL. API token for private repositories.

## Value

A tibble with columns: path, type, size, oid.

## Examples

``` r
if (FALSE) { # \dontrun{
hf_list_repo_files("BAAI/bge-small-en-v1.5")
} # }
```
