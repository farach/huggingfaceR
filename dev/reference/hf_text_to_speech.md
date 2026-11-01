# Convert Text to Speech

Generate speech audio from text and write it to disk using a
text-to-speech model.

## Usage

``` r
hf_text_to_speech(
  text,
  output = NULL,
  model = hf_default_model("text_to_speech"),
  token = NULL,
  endpoint_url = NULL,
  overwrite = FALSE,
  max_active = getOption("huggingfaceR.max_active", 1L),
  ...
)
```

## Arguments

- text:

  Character vector of text to synthesize.

- output:

  Character path(s) or NULL. When NULL, files are written to temporary
  paths with an extension inferred from the response content type.

- model:

  Character string. Model ID from Hugging Face Hub. Default:
  "hexgrad/Kokoro-82M".

- token:

  Character string or NULL. API token for authentication.

- endpoint_url:

  Character string or NULL. A custom Inference Endpoint URL.

- overwrite:

  Logical. If TRUE, overwrite existing output files.

- max_active:

  Integer. Maximum number of inputs to send at once. Default:
  \`getOption("huggingfaceR.max_active", 1)\`, which sends one input at
  a time. Larger values send inputs concurrently. This is faster for
  long vectors but spends provider credits and rate limits faster.
  Results keep input order, transient failures (HTTP 429 and 5xx) are
  retried a bounded number of times, and the first failing input stops
  the call with its error, as in sequential mode. Set
  \`options(huggingfaceR.max_active = 4)\` to change the default for
  every per-input function.

- ...:

  Additional generation parameters passed to the model.

## Value

A tibble with columns: text, path, content_type, audio.

## Provider availability

huggingfaceR's task functions speak the Hugging Face task API contract,
which only the first-party \`hf-inference\` provider implements. At the
time of writing \`hf-inference\` serves no text-to-speech model, so
serverless text-to-speech is not reachable through this function: calls
raise an error naming the third-party providers that do serve the model.
To synthesise speech today, deploy the model as a dedicated Inference
Endpoint and pass \`endpoint_url\`.

## See also

<https://huggingface.co/tasks/text-to-speech>

## Examples

``` r
if (FALSE) { # \dontrun{
hf_text_to_speech("Hello from R.")
} # }
```
