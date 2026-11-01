# Delete a Hub Repository

Delete a model, dataset, or Space repository. This destructive operation
is guarded: it requires \`confirm = TRUE\` and is refused when
\`CI=true\`.

## Usage

``` r
hf_delete_repo(repo_id, repo_type = "model", token = NULL, confirm = FALSE)
```

## Arguments

- repo_id:

  Character string. Repository ID.

- repo_type:

  Character string. One of "model", "dataset", or "space".

- token:

  Character string or NULL. API token with write scope.

- confirm:

  Logical. Must be TRUE to delete the repository.

## Value

The parsed Hub API response.

## Examples

``` r
if (FALSE) { # \dontrun{
hf_delete_repo("me/old-test-dataset", repo_type = "dataset", confirm = TRUE)
} # }
```
