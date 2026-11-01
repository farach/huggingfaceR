# Describe an Image

Ask a vision-capable chat model to describe an image.

## Usage

``` r
hf_describe_image(
  image,
  prompt = "Describe this image.",
  model = hf_default_model("vision_chat"),
  max_tokens = 200,
  token = NULL,
  endpoint_url = NULL,
  max_active = getOption("huggingfaceR.max_active", 1L),
  ...
)
```

## Arguments

- image:

  Image URL, local file path, raw vector, or a character vector/list of
  image URLs or paths.

- prompt:

  Prompt to send with each image. Default: "Describe this image."

- model:

  Character string. Vision-capable chat model ID.

- max_tokens:

  Integer. Maximum tokens to generate. Default: 200.

- token:

  Character string or NULL. API token for authentication.

- endpoint_url:

  Character string or NULL. A custom Inference Endpoint URL.

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

  Additional parameters passed to
  [`hf_chat()`](https://farach.github.io/huggingfaceR/dev/reference/hf_chat.md).
  When `stream = TRUE` is passed, images are described one at a time
  whatever the value of `max_active`.

## Value

A tibble with columns: image, description.

## Examples

``` r
if (FALSE) { # \dontrun{
image <- paste0(
  "https://huggingface.co/datasets/huggingface/",
  "documentation-images/resolve/main/cat.png"
)
hf_describe_image(image)
} # }
```
