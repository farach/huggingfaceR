# Perform Parallel Batch Requests

Internal function to execute multiple API requests in parallel with
error handling.

## Usage

``` r
hf_perform_batch(
  reqs,
  input_indices,
  max_active = 10L,
  progress = TRUE,
  max_tries = 3L,
  paths = NULL
)
```

## Arguments

- reqs:

  List of httr2 request objects.

- input_indices:

  Integer vector. Original indices corresponding to each request.

- max_active:

  Integer. Maximum concurrent requests. Default: 10.

- progress:

  Logical. Show progress bar. Default: TRUE.

- max_tries:

  Integer. Maximum attempts per request, including the first. Default:
  3.

- paths:

  Character vector or NULL. When given, each response body is written to
  the matching file instead of being kept in memory.

## Value

A tibble with columns: - \`.input_idx\`: Original input index -
\`response\`: The response object (or error) - \`.error\`: Logical, TRUE
if request failed - \`.error_msg\`: Error message or NA

## Details

\`httr2::req_perform_parallel()\` retries transient failures without
honouring \`req_retry(max_tries)\`, so a model that keeps returning 429
or 5xx responses would never return. Each request's own retry policy is
therefore disabled and transient failures, including dropped connections
and timeouts, are retried here in bounded rounds instead. Between rounds
it waits as long as the server asks, through \`Retry-After\` or the
Hub's \`RateLimit\` header, or else backs off exponentially.
