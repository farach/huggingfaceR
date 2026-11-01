# Get Existing Chunk IDs

Internal function to find which chunks have already been written.

## Usage

``` r
hf_get_existing_chunks(output_dir, prefix = "chunk")
```

## Arguments

- output_dir:

  Character string. Directory to check.

- prefix:

  Character string. File name prefix to match.

## Value

Integer vector of chunk IDs that already exist.
