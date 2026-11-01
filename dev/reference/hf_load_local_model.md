# Load a Reusable Local Model

Load an opt-in local embedding or text-classification model. The default
device is CPU. A Hub ID is first resolved by
[`hf_download_model()`](https://farach.github.io/huggingfaceR/dev/reference/hf_download_model.md);
an existing local directory skips downloading. All backend components
load from that exact directory with `local_files_only = TRUE` and
`trust_remote_code = FALSE`. There is no hosted inference fallback.

## Usage

``` r
hf_load_local_model(
  model = NULL,
  task = c("embed", "classify"),
  revision = "main",
  cache_dir = NULL,
  token = NULL,
  local_files_only = FALSE,
  device = "cpu"
)

# S3 method for class 'hf_local_model'
print(x, ...)
```

## Arguments

- model:

  Character string or `NULL`. A Hub model ID or an existing local model
  directory. `NULL` resolves
  [`hf_default_model()`](https://farach.github.io/huggingfaceR/dev/reference/hf_default_model.md)
  for `task`.

- task:

  Character string. Either `"embed"` or `"classify"`.

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

- device:

  Character string. `"cpu"` by default. Other devices, such as
  `"cuda:0"` or `"mps"`, are passed to the user's compatible Python
  stack. This function does not install CUDA or select a GPU
  automatically.

- x:

  An `hf_local_model` handle.

- ...:

  Additional arguments (currently unused).

## Value

An `hf_local_model` handle containing `task`, `model`, `source` (`"hub"`
or `"local"`), `path`, resolved `revision` when known (otherwise
`NULL`), `requested_revision`, `device`, and `backend`. Authentication
tokens are not stored in the handle.

## Details

Standard safetensors models are required. Embedding models must include
`modules.json` describing a root Transformer followed by built-in
Pooling or Normalize modules. Missing module metadata is rejected rather
than falling back to a different pooling configuration. Arbitrary module
loaders, adapters, and custom code are unsupported.

Python handles are session-specific. Do not use
[`saveRDS()`](https://rdrr.io/r/base/readRDS.html) to transfer a loaded
handle between R sessions. Save the snapshot path instead and call
`hf_load_local_model()` again in the new session.

## See also

[`hf_local_setup()`](https://farach.github.io/huggingfaceR/dev/reference/hf_local_setup.md),
[`hf_embed_local()`](https://farach.github.io/huggingfaceR/dev/reference/hf_embed_local.md),
[`hf_classify_local()`](https://farach.github.io/huggingfaceR/dev/reference/hf_classify_local.md)

## Examples

``` r
if (FALSE) { # \dontrun{
model <- hf_load_local_model(task = "embed")
hf_embed_local(c("Hello world", "Goodbye world"), model)

classifier <- hf_load_local_model(task = "classify")
hf_classify_local("I enjoy programming in R.", classifier)

# Reload an already downloaded snapshot in a new session
snapshot_path <- model$path
model <- hf_load_local_model(snapshot_path, task = "embed")
} # }
```
