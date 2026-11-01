# Get Current Hugging Face User Information

Retrieve information about the currently authenticated user. Requires a
valid Hugging Face token to be set.

## Usage

``` r
hf_whoami(token = NULL)
```

## Arguments

- token:

  Character string containing your HF token. If NULL, uses the
  \`HF_TOKEN\` environment variable, falling back to the legacy
  \`HUGGING_FACE_HUB_TOKEN\`.

## Value

A tibble with user, billing, organization, and token-scope metadata.

## Examples

``` r
if (FALSE) { # \dontrun{
# Check current user
hf_whoami()
} # }
```
