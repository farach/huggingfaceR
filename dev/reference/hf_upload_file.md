# Upload a File to a Hub Repository

Upload a local file into a model, dataset, or Space repository. This is
a write operation and requires an API token with write scope plus
\`confirm = TRUE\`.

## Usage

``` r
hf_upload_file(
  path,
  repo_id,
  path_in_repo = NULL,
  repo_type = "model",
  commit_message = NULL,
  token = NULL,
  overwrite = FALSE,
  confirm = FALSE
)
```

## Arguments

- path:

  Local file path to upload.

- repo_id:

  Character string. Repository ID.

- path_in_repo:

  Character string or NULL. Destination path in the repo. Defaults to
  \`basename(path)\`.

- repo_type:

  Character string. One of "model", "dataset", or "space".

- commit_message:

  Character string or NULL. Commit message when supported by the Hub
  upload endpoint.

- token:

  Character string or NULL. API token with write scope.

- overwrite:

  Logical. If FALSE, error when \`path_in_repo\` already exists.

- confirm:

  Logical. Must be TRUE to perform the write operation.

## Value

The parsed Hub API response, or the response path when the endpoint
returns no JSON body.

## Examples

``` r
if (FALSE) { # \dontrun{
hf_upload_file("results.csv", "me/my-dataset", repo_type = "dataset", confirm = TRUE)
} # }
```
