# Chunked Text Classification (Disk Checkpoints)

Classify large datasets with automatic checkpointing to disk. Supports
resuming interrupted processing.

## Usage

``` r
hf_classify_chunks(
  text,
  output_dir,
  model = hf_default_model("classify"),
  token = NULL,
  chunk_size = 1000L,
  batch_size = 100L,
  max_active = 10L,
  resume = TRUE,
  progress = TRUE,
  endpoint_url = NULL
)
```

## Arguments

- text:

  Character vector of text(s) to classify.

- output_dir:

  Character string. Directory to write chunk files.

- model:

  Character string. Model ID from Hugging Face Hub. Default:
  "distilbert/distilbert-base-uncased-finetuned-sst-2-english".

- token:

  Character string or NULL. API token for authentication.

- chunk_size:

  Integer. Number of texts per disk chunk. Default: 1000.

- batch_size:

  Integer. Number of texts per API request. Default: 100.

- max_active:

  Integer. Maximum concurrent requests. Default: 10.

- resume:

  Logical. Skip already-completed chunks. Default: TRUE.

- progress:

  Logical. Show progress bar. Default: TRUE.

- endpoint_url:

  Character string or NULL. A custom Inference Endpoint URL.

## Value

Invisibly returns the output directory path. Use \`hf_read_chunks()\` to
read results.

## Examples

``` r
if (FALSE) { # \dontrun{
# Process large dataset with checkpoints
texts <- rep("sample text", 5000)
hf_classify_chunks(texts, output_dir = "classify_output", chunk_size = 1000)

# Read results
results <- hf_read_chunks("classify_output")
} # }
```
