#' Batch Embedding Generation (In-Memory)
#'
#' Generate embeddings for multiple texts in parallel. This function processes
#' all inputs in memory and returns results in a single tibble.
#'
#' @param text Character vector of text(s) to embed.
#' @param model Character string. Model ID from Hugging Face Hub.
#'   Default: "BAAI/bge-small-en-v1.5" (384-dim embeddings).
#' @param token Character string or NULL. API token for authentication.
#' @param batch_size Integer. Number of texts per API request. Default: 100.
#' @param max_active Integer. Maximum concurrent requests. Default: 10.
#' @param progress Logical. Show progress bar. Default: TRUE.
#' @param endpoint_url Character string or NULL. A custom Inference Endpoint URL.
#'
#' @returns A tibble with columns:
#'   - `text`: Original input text
#'   - `embedding`: List-column of numeric vectors
#'   - `n_dims`: Dimension of embedding (or NA on error)
#'   - `.input_idx`: Original position in input vector
#'   - `.error`: TRUE if request failed
#'   - `.error_msg`: Error message or NA
#' @details
#' Embedding responses are large arrays of numbers. When the optional
#' 'RcppSimdJson' package is installed, well-formed responses are parsed
#' directly into numeric matrices, which is many times faster than the default
#' 'jsonlite' parser and returns the same embeddings. Set
#' `options(huggingfaceR.simdjson = FALSE)` to always use 'jsonlite'.
#' @export
#'
#' @examples
#' \dontrun{
#' # Embed many texts in parallel
#' texts <- c("Hello world", "Goodbye world", "R is great")
#' result <- hf_embed_batch(texts, max_active = 5)
#'
#' # Check for errors
#' errors <- result[result$.error, ]
#' }
hf_embed_batch <- function(text,
                            model = hf_default_model("embed"),
                            token = NULL,
                            batch_size = 100L,
                            max_active = 10L,
                            progress = TRUE,
                            endpoint_url = NULL) {

  if (length(text) == 0) {
    return(tibble::tibble(
      text = character(),
      embedding = list(),
      n_dims = integer(),
      .input_idx = integer(),
      .error = logical(),
      .error_msg = character()
    ))
  }

  # Split into batches
  batches <- batch_vector(text, batch_size)

  # Build requests for each batch from one resolved template
  reqs <- hf_build_batch_requests(
    model_id = model,
    inputs = lapply(batches, `[[`, "value"),
    token = token,
    endpoint_url = endpoint_url
  )

  # Perform parallel requests
  batch_results <- hf_perform_batch(
    reqs, seq_along(batches), max_active = max_active, progress = progress
  )
  failed <- batch_results$.error
  messages <- batch_results$.error_msg
  responses <- batch_results$response

  # Write each batch into preallocated per-text columns. Batches are contiguous
  # and in input order, so no per-batch tibbles, row binding, or sorting are
  # needed.
  embedding <- vector("list", length(text))
  error <- logical(length(text))
  error_msg <- rep(NA_character_, length(text))

  for (i in seq_along(batches)) {
    rows <- batches[[i]]$indices
    if (failed[[i]]) {
      # All texts in this batch failed
      error[rows] <- TRUE
      error_msg[rows] <- messages[[i]]
    } else {
      embedding[rows] <- hf_parse_embeddings(responses[[i]], length(rows))
    }
  }

  n_dims <- lengths(embedding)
  n_dims[vapply(embedding, is.null, logical(1))] <- NA_integer_

  tibble::tibble(
    text = text,
    embedding = embedding,
    n_dims = n_dims,
    .input_idx = seq_along(text),
    .error = error,
    .error_msg = error_msg
  )
}


# Parses one feature-extraction response into one embedding per input.
#
# jsonlite allocates an R object for every number, which dominates the R-side
# cost of large embedding payloads. When the optional 'RcppSimdJson' package is
# installed, a 2-D array of non-missing numbers is parsed straight into a double
# matrix instead. Any other shape, or a body it cannot parse, falls back to
# jsonlite, so the choice of parser does not change which embeddings are
# returned.
hf_parse_embeddings <- function(resp, n) {
  if (hf_use_simdjson()) {
    httr2::resp_check_content_type(resp, "application/json", "json")
    parsed <- tryCatch(
      RcppSimdJson::fparse(httr2::resp_body_raw(resp), max_simplify_lvl = "matrix"),
      error = function(e) NULL
    )
    if (is.matrix(parsed) && is.double(parsed) && nrow(parsed) == n &&
        ncol(parsed) > 0L && !anyNA(parsed)) {
      return(lapply(seq_len(n), function(i) parsed[i, ]))
    }
  }

  hf_embedding_rows(httr2::resp_body_json(resp), n)
}


# Whether to parse embedding payloads with the optional 'RcppSimdJson' package.
# Set `options(huggingfaceR.simdjson = FALSE)` to always use jsonlite.
hf_use_simdjson <- function() {
  isTRUE(getOption("huggingfaceR.simdjson", TRUE)) &&
    requireNamespace("RcppSimdJson", quietly = TRUE)
}


# Splits one parsed feature-extraction response into one embedding per input.
#
# A response can be a list of embeddings, a single embedding for a one-text
# batch, or nested token-level vectors, which are flattened. A response with
# the wrong number of embeddings is padded with NULL or truncated, with a
# warning.
hf_embedding_rows <- function(result, n) {
  embeddings <- if (is.list(result) && length(result) > 0) {
    if (is.list(result[[1]]) && all(sapply(result[[1]], is.numeric))) {
      # Nested list of numerics - single text returned as [[1]]
      if (n == 1) list(unlist(result[[1]])) else lapply(result, unlist)
    } else if (is.numeric(result[[1]])) {
      # List of lists of single numerics
      if (n == 1) list(unlist(result)) else lapply(result, unlist)
    } else {
      # Each element is a list representing one embedding
      lapply(result, function(emb) if (is.list(emb)) unlist(emb) else emb)
    }
  } else {
    vector("list", n)
  }

  if (length(embeddings) != n) {
    cli::cli_warn("Embedding count mismatch for batch. Expected {n}, got {length(embeddings)}")
    length(embeddings) <- n
  }

  embeddings
}


#' Chunked Embedding Generation (Disk Checkpoints)
#'
#' Generate embeddings for large datasets with automatic checkpointing to disk.
#' Supports resuming interrupted processing.
#'
#' @param text Character vector of text(s) to embed.
#' @param output_dir Character string. Directory to write chunk files.
#' @param model Character string. Model ID from Hugging Face Hub.
#'   Default: "BAAI/bge-small-en-v1.5".
#' @param token Character string or NULL. API token for authentication.
#' @param chunk_size Integer. Number of texts per disk chunk. Default: 1000.
#' @param batch_size Integer. Number of texts per API request. Default: 100.
#' @param max_active Integer. Maximum concurrent requests. Default: 10.
#' @param resume Logical. Skip already-completed chunks. Default: TRUE.
#' @param progress Logical. Show progress bar. Default: TRUE.
#' @param endpoint_url Character string or NULL. A custom Inference Endpoint URL.
#'
#' @returns Invisibly returns the output directory path. Use `hf_read_chunks()`
#'   to read results.
#' @export
#'
#' @examples
#' \dontrun{
#' # Process large dataset with checkpoints
#' texts <- rep("sample text", 5000)
#' hf_embed_chunks(texts, output_dir = "embeddings_output", chunk_size = 1000)
#'
#' # Read results
#' results <- hf_read_chunks("embeddings_output")
#'
#' # Resume interrupted processing
#' hf_embed_chunks(more_texts, output_dir = "embeddings_output", resume = TRUE)
#' }
hf_embed_chunks <- function(text,
                             output_dir,
                             model = hf_default_model("embed"),
                             token = NULL,
                             chunk_size = 1000L,
                             batch_size = 100L,
                             max_active = 10L,
                             resume = TRUE,
                             progress = TRUE,
                             endpoint_url = NULL) {

  if (!requireNamespace("arrow", quietly = TRUE)) {
    stop("Package 'arrow' is required for chunk operations. Install with: install.packages('arrow')",
         call. = FALSE)
  }

  if (length(text) == 0) {
    cli::cli_alert_info("No texts to process")
    return(invisible(output_dir))
  }

  # Create output directory
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }

  # Split into chunks
  chunks <- batch_vector(text, chunk_size)
  n_chunks <- length(chunks)

  # Get existing chunk IDs if resuming
  existing_ids <- if (resume) {
    hf_get_existing_chunks(output_dir, prefix = "embed_chunk")
  } else {
    integer()
  }

  if (length(existing_ids) > 0 && resume) {
    cli::cli_alert_info("Found {length(existing_ids)} existing chunk(s), resuming...")
  }

  # Process each chunk
  for (chunk_id in seq_along(chunks)) {
    if (chunk_id %in% existing_ids && resume) {
      if (progress) {
        cli::cli_alert_success("Chunk {chunk_id}/{n_chunks} already exists, skipping")
      }
      next
    }

    if (progress) {
      cli::cli_alert_info("Processing chunk {chunk_id}/{n_chunks} ({length(chunks[[chunk_id]]$value)} texts)")
    }

    # Process this chunk
    chunk_result <- hf_embed_batch(
      text = chunks[[chunk_id]]$value,
      model = model,
      token = token,
      batch_size = batch_size,
      max_active = max_active,
      progress = progress,
      endpoint_url = endpoint_url
    )

    # Adjust input indices to global positions
    chunk_result$.input_idx <- chunks[[chunk_id]]$indices

    # Write to disk
    hf_write_chunk(chunk_result, output_dir, chunk_id, prefix = "embed_chunk")

    if (progress) {
      n_errors <- sum(chunk_result$.error)
      if (n_errors > 0) {
        cli::cli_alert_warning("Chunk {chunk_id} completed with {n_errors} error(s)")
      } else {
        cli::cli_alert_success("Chunk {chunk_id} completed successfully")
      }
    }
  }

  cli::cli_alert_success("All chunks processed. Use hf_read_chunks('{output_dir}') to read results.")
  invisible(output_dir)
}
