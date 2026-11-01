# Split Vector into Batches

Internal helper to split a vector into batches with index tracking.

## Usage

``` r
batch_vector(x, batch_size)
```

## Arguments

- x:

  Vector to split into batches.

- batch_size:

  Integer. Maximum size of each batch.

## Value

A list of lists, each containing: - \`value\`: The batch values -
\`indices\`: The original indices of the values
