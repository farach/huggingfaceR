# Download a Model Snapshot for Local Inference

Use the official `huggingface_hub.snapshot_download()` cache to download
a model's configuration, tokenizer, and safetensors weights. Repository
subdirectories and immutable snapshot paths are preserved. Online calls
can download files; `local_files_only = TRUE` never fetches missing Hub
files.

## Usage

``` r
hf_download_model(
  model,
  revision = "main",
  cache_dir = NULL,
  token = NULL,
  local_files_only = FALSE
)
```

## Arguments

- model:

  Character string. Model ID on the Hugging Face Hub.

- revision:

  Character string. Branch, tag, or commit to download. Use a commit
  hash for reproducibility.

- cache_dir:

  Character string or `NULL`. Official Hugging Face cache directory.
  `NULL` uses the Hub library's configured cache.

- token:

  Character string or `NULL`. Optional authentication token. `NULL` uses
  `HF_TOKEN`, then legacy `HUGGING_FACE_HUB_TOKEN`, via the package's
  token helper.

- local_files_only:

  Logical. Use only already cached model files. An incomplete offline
  cache fails rather than going online.

## Value

A normalized, existing snapshot-directory character path.

## Details

Only standard models with safetensors weights are supported, not pickle
weights, adapters, or custom Python code. The download patterns are
`*.json`, `*.safetensors`, `*.txt`, `*.model`, and `*.tiktoken`,
including matching files in subdirectories. Embedding module stacks,
when present, must use a root Transformer followed by built-in Pooling
or Normalize modules. Missing required files or referenced weight shards
cause an error.

Python dependencies must already be available for fully offline
operation; `local_files_only` controls model files, not reticulate's
environment setup.

## See also

[`hf_local_setup()`](https://farach.github.io/huggingfaceR/dev/reference/hf_local_setup.md),
[`hf_load_local_model()`](https://farach.github.io/huggingfaceR/dev/reference/hf_load_local_model.md)

## Examples

``` r
if (FALSE) { # \dontrun{
path <- hf_download_model(hf_default_model("embed"))
model <- hf_load_local_model(path, task = "embed")

# Reuse the official cache without downloading model files
path <- hf_download_model(hf_default_model("embed"), local_files_only = TRUE)
} # }
```
