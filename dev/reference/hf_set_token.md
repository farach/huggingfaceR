# Set Hugging Face API Token

Set or update your Hugging Face API token for authentication. The token
is written to the \`HF_TOKEN\` environment variable used by current
Hugging Face tooling, and mirrored to the legacy
\`HUGGING_FACE_HUB_TOKEN\` name.

## Usage

``` r
hf_set_token(token = NULL, store = FALSE)
```

## Arguments

- token:

  Character string containing your HF token, or NULL to set
  interactively. If NULL, will prompt for token input (not echoed to
  console).

- store:

  Logical. If TRUE, stores the token in .Renviron for future sessions.
  Default: FALSE (token only available for current session).

## Value

Invisibly returns TRUE if token was set successfully.

## Details

Inference calls require a token with the "Make calls to Inference
Providers" permission. See
<https://huggingface.co/docs/hub/security-tokens> for token setup.

## Examples

``` r
if (FALSE) { # \dontrun{
# Set token for current session only
hf_set_token("hf_xxxxxxxxxxxxx")

# Set token interactively and store permanently
hf_set_token(store = TRUE)
} # }
```
