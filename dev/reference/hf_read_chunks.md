# Read All Chunks from Directory

Read and combine all parquet chunk files from a directory.

## Usage

``` r
hf_read_chunks(output_dir, pattern = "*.parquet")
```

## Arguments

- output_dir:

  Character string. Directory containing chunk files.

- pattern:

  Character string. Glob pattern to match files. Default: "\*.parquet".

## Value

A tibble combining all chunks, sorted by \`.input_idx\` if present.

## Examples

``` r
if (FALSE) { # \dontrun{
# After running hf_embed_chunks()
results <- hf_read_chunks("my_output_dir")
} # }
```
