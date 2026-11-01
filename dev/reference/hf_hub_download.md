# Download a File from the Hub

Download a single file from a model, dataset, or Space repository.

## Usage

``` r
hf_hub_download(
  repo_id,
  filename,
  repo_type = "model",
  revision = "main",
  dest = NULL,
  token = NULL,
  overwrite = FALSE
)
```

## Arguments

- repo_id:

  Character string. Repository ID.

- filename:

  Character string. Path to the file inside the repository.

- repo_type:

  Character string. One of "model", "dataset", or "space".

- revision:

  Character string. Git revision, branch, or tag.

- dest:

  Character path or NULL. If NULL, writes to a temporary file. If an
  existing directory, the repository filename basename is used inside
  it.

- token:

  Character string or NULL. API token for private repositories.

- overwrite:

  Logical. If TRUE, overwrite an existing destination file.

## Value

The downloaded file path.

## Examples

``` r
if (FALSE) { # \dontrun{
hf_hub_download("BAAI/bge-small-en-v1.5", "README.md")
} # }
```
