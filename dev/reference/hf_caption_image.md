# Caption Images

Generate short image captions. This uses a vision-capable chat model by
default because the public \`hf-inference\` provider did not expose a
broadly available image-to-text captioning model during verification.

## Usage

``` r
hf_caption_image(
  image,
  prompt = "Write a short, factual caption for this image.",
  model = hf_default_model("caption_image"),
  max_tokens = 80,
  token = NULL,
  endpoint_url = NULL,
  max_active = getOption("huggingfaceR.max_active", 1L),
  ...
)
```

## Arguments

- image:

  Image input: a local file path, URL, raw vector, or vector/list of
  paths/URLs.

- prompt:

  Prompt used to request the caption.

- model:

  Character string. Vision-capable chat model ID. Default:
  "google/gemma-3-4b-it".

- max_tokens:

  Integer. Maximum tokens to generate.

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

  Additional arguments passed to
  [`hf_describe_image()`](https://farach.github.io/huggingfaceR/dev/reference/hf_describe_image.md).

## Value

A tibble with columns: image, caption.

## See also

[`hf_describe_image`](https://farach.github.io/huggingfaceR/dev/reference/hf_describe_image.md)

## Examples

``` r
if (FALSE) { # \dontrun{
hf_caption_image("cat.png")
} # }
```
