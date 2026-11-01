# Write Chunk to Parquet

Internal function to write a data chunk to a parquet file.

## Usage

``` r
hf_write_chunk(data, output_dir, chunk_id, prefix = "chunk")
```

## Arguments

- data:

  A tibble to write.

- output_dir:

  Character string. Directory to write to.

- chunk_id:

  Integer. Chunk identifier.

- prefix:

  Character string. File name prefix. Default: "chunk".

## Value

Invisibly returns the file path written.
