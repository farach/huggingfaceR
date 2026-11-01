# Create a Hub Repository

Create a model, dataset, or Space repository. This is a write operation
and requires an API token with write scope plus \`confirm = TRUE\`.

## Usage

``` r
hf_create_repo(
  repo_id,
  repo_type = "model",
  private = FALSE,
  exist_ok = FALSE,
  token = NULL,
  confirm = FALSE
)
```

## Arguments

- repo_id:

  Character string. Repository name or "namespace/name".

- repo_type:

  Character string. One of "model", "dataset", or "space".

- private:

  Logical. Whether to create a private repository.

- exist_ok:

  Logical. If TRUE, do not fail when the repo already exists.

- token:

  Character string or NULL. API token with write scope.

- confirm:

  Logical. Must be TRUE to perform the write operation.

## Value

The parsed Hub API response.

## Examples

``` r
if (FALSE) { # \dontrun{
hf_create_repo("my-dataset", repo_type = "dataset", private = TRUE, confirm = TRUE)
} # }
```
