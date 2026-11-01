# Generate an Image from Text

Generate an image from a prompt and write it to disk using a
text-to-image model via the Hugging Face Inference Providers API.

## Usage

``` r
hf_text_to_image(
  prompt,
  output = NULL,
  seed = NULL,
  model = hf_default_model("text_to_image"),
  token = NULL,
  endpoint_url = NULL,
  overwrite = FALSE,
  max_active = getOption("huggingfaceR.max_active", 1L),
  ...
)
```

## Arguments

- prompt:

  Character vector of prompts.

- output:

  Character path(s) or NULL. When NULL, files are written to temporary
  paths with an extension inferred from the response content type.

- seed:

  Integer or NULL. Optional random seed for reproducibility when the
  provider/model supports it.

- model:

  Character string. Model ID from Hugging Face Hub. Default:
  "stabilityai/stable-diffusion-3-medium-diffusers". This is the model
  served for text-to-image by the \`hf-inference\` provider; it is
  gated, so accept the licence on the model page once before first use.

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

A tibble with columns: prompt, path, content_type, image.

## Provider availability

huggingfaceR's task functions speak the Hugging Face task API contract,
which only the first-party \`hf-inference\` provider implements. Popular
third-party image models (for example
\`black-forest-labs/FLUX.1-schnell\`, served by nscale, Fal AI, and
WaveSpeed) expose their own request and response formats and are
therefore not reachable through this function; passing one raises an
error naming the providers that serve it. Use a model served by
\`hf-inference\`, or pass \`endpoint_url\` for a dedicated Inference
Endpoint.

## See also

<https://huggingface.co/docs/inference-providers/tasks/text-to-image>

## Examples

``` r
if (FALSE) { # \dontrun{
img <- hf_text_to_image("a small red cube on a white background", seed = 42)
img$path
} # }
```
