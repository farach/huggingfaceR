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
#'
#' @returns A tibble with columns:
#'   - `text`: Original input text
#'   - `embedding`: List-column of numeric vectors
#'   - `n_dims`: Dimension of embedding (or NA on error)
#'   - `.input_idx`: Original position in input vector
#'   - `.error`: TRUE if request failed
#'   - `.error_msg`: Error message or NA
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
                            model = "BAAI/bge-small-en-v1.5",
                            token = NULL,
                            batch_size = 100L,
                            max_active = 10L,
                            progress = TRUE) {

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

  # Build requests for each batch
  reqs <- purrr::map(batches, function(batch) {
    hf_build_request(
      model_id = model,
      inputs = batch$value,
      token = token
    )
  })

  # Collect batch indices (we need to track which batch each request belongs to)
  batch_indices <- seq_along(batches)

  # Perform parallel requests
  batch_results <- hf_perform_batch(reqs, batch_indices, max_active = max_active, progress = progress)

  # Parse results and expand to individual texts
  results <- purrr::pmap_dfr(
    list(
      batch_result = split(batch_results, seq_len(nrow(batch_results))),
      batch = batches
    ),
    function(batch_result, batch) {
      batch_result <- batch_result[1, ]  # Ensure single row

      if (batch_result$.error) {
        # All texts in this batch failed
        tibble::tibble(
          text = batch$value,
          embedding = rep(list(NULL), length(batch$value)),
          n_dims = rep(NA_integer_, length(batch$value)),
          .input_idx = batch$indices,
          .error = TRUE,
          .error_msg = batch_result$.error_msg
        )
      } else {
        # Parse successful response
        resp <- batch_result$response[[1]]
        result <- httr2::resp_body_json(resp)

        # Handle response format - could be list of embeddings or single embedding
        embeddings <- if (is.list(result) && length(result) > 0) {
          if (is.list(result[[1]]) && all(sapply(result[[1]], is.numeric))) {
            # Nested list of numerics - single text returned as [[1]]
            if (length(batch$value) == 1) {
              list(unlist(result[[1]]))
            } else {
              purrr::map(result, unlist)
            }
          } else if (is.numeric(result[[1]])) {
            # List of lists of single numerics
            if (length(batch$value) == 1) {
              list(unlist(result))
            } else {
              purrr::map(result, unlist)
            }
          } else {
            # Each element is a list representing one embedding
            purrr::map(result, function(emb) {
              if (is.list(emb)) unlist(emb) else emb
            })
          }
        } else {
          rep(list(NULL), length(batch$value))
        }

        # Ensure we have the right number of embeddings
        if (length(embeddings) != length(batch$value)) {
          # Mismatch - something unexpected
          cli::cli_warn("Embedding count mismatch for batch. Expected {length(batch$value)}, got {length(embeddings)}")
          embeddings <- c(embeddings, rep(list(NULL), length(batch$value) - length(embeddings)))
          embeddings <- embeddings[seq_along(batch$value)]
        }

        tibble::tibble(
          text = batch$value,
          embedding = embeddings,
          n_dims = purrr::map_int(embeddings, ~ if (is.null(.x)) NA_integer_ else length(.x)),
          .input_idx = batch$indices,
          .error = FALSE,
          .error_msg = NA_character_
        )
      }
    }
  )

  # Sort by original input index
  dplyr::arrange(results, .data$.input_idx)
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
                             model = "BAAI/bge-small-en-v1.5",
                             token = NULL,
                             chunk_size = 1000L,
                             batch_size = 100L,
                             max_active = 10L,
                             resume = TRUE,
                             progress = TRUE) {

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
      progress = progress
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
