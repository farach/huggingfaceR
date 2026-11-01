# Classify Images

Classify images using an image-classification model via the Hugging Face
Inference Providers API.

## Usage

``` r
hf_classify_image(
  image,
  top_k = 5L,
  model = hf_default_model("classify_image"),
  token = NULL,
  endpoint_url = NULL,
  content_type = NULL,
  max_active = getOption("huggingfaceR.max_active", 1L),
  ...
)
```

## Arguments

- image:

  Image input: a local file path, URL, raw vector, or vector/list of
  paths/URLs.

- top_k:

  Integer. Maximum labels to return per image.

- model:

  Character string. Model ID from Hugging Face Hub. Default:
  "google/vit-base-patch16-224".

- token:

  Character string or NULL. API token for authentication.

- endpoint_url:

  Character string or NULL. A custom Inference Endpoint URL.

- content_type:

  Character string or NULL. MIME type to use for raw image inputs. Paths
  and URLs are inferred when possible.

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

  Additional arguments (currently unused).

## Value

A tibble with columns: image, label, score.

## See also

<https://huggingface.co/docs/inference-providers/tasks/image-classification>

## Examples

``` r
if (FALSE) { # \dontrun{
hf_classify_image("cat.png", top_k = 3)
} # }
```
