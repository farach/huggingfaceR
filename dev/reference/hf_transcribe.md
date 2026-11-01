# Transcribe Audio

Transcribe speech from an audio file, URL, or raw vector using automatic
speech recognition via the Hugging Face Inference Providers API.

## Usage

``` r
hf_transcribe(
  audio,
  return_timestamps = FALSE,
  model = hf_default_model("transcribe"),
  token = NULL,
  endpoint_url = NULL,
  content_type = NULL,
  max_active = getOption("huggingfaceR.max_active", 1L),
  ...
)
```

## Arguments

- audio:

  Audio input: a local file path, URL, raw vector, or vector/list of
  paths/URLs.

- return_timestamps:

  Logical or character. Use \`FALSE\` for text only, \`TRUE\` for chunk
  timestamps, or a model-supported value such as \`"word"\`.

- model:

  Character string. Model ID from Hugging Face Hub. Default:
  "openai/whisper-large-v3".

- token:

  Character string or NULL. API token for authentication.

- endpoint_url:

  Character string or NULL. A custom Inference Endpoint URL.

- content_type:

  Character string or NULL. MIME type to use for raw audio inputs. Paths
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

A tibble with columns: audio, text, chunks.

## See also

<https://huggingface.co/docs/inference-providers/tasks/automatic-speech-recognition>

## Examples

``` r
if (FALSE) { # \dontrun{
hf_transcribe("interview.flac")
hf_transcribe("interview.flac", return_timestamps = "word")
} # }
```
