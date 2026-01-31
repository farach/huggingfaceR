#' Batch Text Classification (In-Memory)
#'
#' Classify multiple texts in parallel. This function processes all inputs
#' in memory and returns results in a single tibble.
#'
#' @param text Character vector of text(s) to classify.
#' @param model Character string. Model ID from Hugging Face Hub.
#'   Default: "distilbert/distilbert-base-uncased-finetuned-sst-2-english".
#' @param token Character string or NULL. API token for authentication.
#' @param batch_size Integer. Number of texts per API request. Default: 100.
#' @param max_active Integer. Maximum concurrent requests. Default: 10.
#' @param progress Logical. Show progress bar. Default: TRUE.
#'
#' @returns A tibble with columns:
#'   - `text`: Original input text
#'   - `label`: Predicted label
#'   - `score`: Confidence score
#'   - `.input_idx`: Original position in input vector
#'   - `.error`: TRUE if request failed
#'   - `.error_msg`: Error message or NA
#' @export
#'
#' @examples
#' \dontrun{
#' # Classify many texts in parallel
#' texts <- c("I love this!", "This is terrible.", "Meh, it's okay.")
#' result <- hf_classify_batch(texts, max_active = 5)
#'
#' # Check for errors
#' errors <- result[result$.error, ]
#' }
hf_classify_batch <- function(text,
                               model = "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                               token = NULL,
                               batch_size = 100L,
                               max_active = 10L,
                               progress = TRUE) {

  if (length(text) == 0) {
    return(tibble::tibble(
      text = character(),
      label = character(),
      score = numeric(),
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

  # Collect batch indices
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
          label = rep(NA_character_, length(batch$value)),
          score = rep(NA_real_, length(batch$value)),
          .input_idx = batch$indices,
          .error = TRUE,
          .error_msg = batch_result$.error_msg
        )
      } else {
        # Parse successful response
        resp <- batch_result$response[[1]]
        result <- httr2::resp_body_json(resp)

        # Classification returns [[1]][[1]]$label/score for each text
        # When batched: list of list of classifications
        classifications <- if (is.list(result) && length(result) > 0) {
          purrr::map(result, function(item) {
            if (is.list(item) && length(item) > 0 && !is.null(item[[1]]$label)) {
              # Get top classification
              list(label = item[[1]]$label, score = item[[1]]$score)
            } else if (!is.null(item$label)) {
              list(label = item$label, score = item$score)
            } else {
              list(label = NA_character_, score = NA_real_)
            }
          })
        } else {
          rep(list(list(label = NA_character_, score = NA_real_)), length(batch$value))
        }

        # Ensure we have the right number of results
        if (length(classifications) != length(batch$value)) {
          cli::cli_warn("Classification count mismatch for batch")
          classifications <- c(classifications,
            rep(list(list(label = NA_character_, score = NA_real_)),
                length(batch$value) - length(classifications)))
          classifications <- classifications[seq_along(batch$value)]
        }

        tibble::tibble(
          text = batch$value,
          label = purrr::map_chr(classifications, ~ .x$label %||% NA_character_),
          score = purrr::map_dbl(classifications, ~ .x$score %||% NA_real_),
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


#' Chunked Text Classification (Disk Checkpoints)
#'
#' Classify large datasets with automatic checkpointing to disk.
#' Supports resuming interrupted processing.
#'
#' @param text Character vector of text(s) to classify.
#' @param output_dir Character string. Directory to write chunk files.
#' @param model Character string. Model ID from Hugging Face Hub.
#'   Default: "distilbert/distilbert-base-uncased-finetuned-sst-2-english".
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
#' hf_classify_chunks(texts, output_dir = "classify_output", chunk_size = 1000)
#'
#' # Read results
#' results <- hf_read_chunks("classify_output")
#' }
hf_classify_chunks <- function(text,
                                output_dir,
                                model = "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
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
    hf_get_existing_chunks(output_dir, prefix = "classify_chunk")
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
    chunk_result <- hf_classify_batch(
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
    hf_write_chunk(chunk_result, output_dir, chunk_id, prefix = "classify_chunk")

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


#' Batch Zero-Shot Classification (In-Memory)
#'
#' Classify multiple texts into custom categories in parallel without training.
#'
#' @param text Character vector of text(s) to classify.
#' @param labels Character vector of candidate labels/categories.
#' @param model Character string. Model ID from Hugging Face Hub.
#'   Default: "facebook/bart-large-mnli".
#' @param multi_label Logical. If TRUE, allows multiple labels per text.
#'   Default: FALSE.
#' @param token Character string or NULL. API token for authentication.
#' @param batch_size Integer. Number of texts per API request. Default: 50.
#' @param max_active Integer. Maximum concurrent requests. Default: 10.
#' @param progress Logical. Show progress bar. Default: TRUE.
#'
#' @returns A tibble with columns:
#'   - `text`: Original input text
#'   - `label`: Predicted label (or labels if multi_label)
#'   - `score`: Confidence score(s)
#'   - `.input_idx`: Original position in input vector
#'   - `.error`: TRUE if request failed
#'   - `.error_msg`: Error message or NA
#' @export
#'
#' @examples
#' \dontrun{
#' texts <- c("I love my new laptop", "The game was exciting", "This recipe is delicious")
#' labels <- c("technology", "sports", "food")
#' result <- hf_classify_zero_shot_batch(texts, labels, max_active = 5)
#' }
hf_classify_zero_shot_batch <- function(text,
                                         labels,
                                         model = "facebook/bart-large-mnli",
                                         multi_label = FALSE,
                                         token = NULL,
                                         batch_size = 50L,
                                         max_active = 10L,
                                         progress = TRUE) {

  if (length(labels) == 0) {
    stop("At least one label must be provided", call. = FALSE)
  }

  if (length(text) == 0) {
    return(tibble::tibble(
      text = character(),
      label = character(),
      score = numeric(),
      .input_idx = integer(),
      .error = logical(),
      .error_msg = character()
    ))
  }

  # For zero-shot, we typically process one text at a time due to label requirements
  # But we can parallelize across texts
  batches <- batch_vector(text, batch_size = 1L)  # One text per request for zero-shot

  # Build requests
  reqs <- purrr::map(batches, function(batch) {
    hf_build_request(
      model_id = model,
      inputs = batch$value,
      parameters = list(
        candidate_labels = labels,
        multi_label = multi_label
      ),
      token = token
    )
  })

  # Flatten indices for individual texts
  all_indices <- purrr::map_int(batches, ~ .x$indices[1])

  # Perform parallel requests
  batch_results <- hf_perform_batch(reqs, all_indices, max_active = max_active, progress = progress)

  # Parse results
  results <- purrr::pmap_dfr(
    list(
      batch_result = split(batch_results, seq_len(nrow(batch_results))),
      batch = batches
    ),
    function(batch_result, batch) {
      batch_result <- batch_result[1, ]
      input_text <- batch$value[1]
      input_idx <- batch$indices[1]

      if (batch_result$.error) {
        tibble::tibble(
          text = rep(input_text, length(labels)),
          label = rep(NA_character_, length(labels)),
          score = rep(NA_real_, length(labels)),
          .input_idx = rep(input_idx, length(labels)),
          .error = TRUE,
          .error_msg = batch_result$.error_msg
        )
      } else {
        resp <- batch_result$response[[1]]
        result <- httr2::resp_body_json(resp)

        # Zero-shot returns: [{label, score}, ...] or {labels: [], scores: []}
        if (is.list(result) && length(result) > 0 && !is.null(result[[1]]$label)) {
          purrr::map_dfr(result, function(item) {
            tibble::tibble(
              text = input_text,
              label = item$label %||% NA_character_,
              score = item$score %||% NA_real_,
              .input_idx = input_idx,
              .error = FALSE,
              .error_msg = NA_character_
            )
          })
        } else if (!is.null(result$labels) && !is.null(result$scores)) {
          tibble::tibble(
            text = input_text,
            label = unlist(result$labels),
            score = unlist(result$scores),
            .input_idx = input_idx,
            .error = FALSE,
            .error_msg = NA_character_
          )
        } else {
          tibble::tibble(
            text = rep(input_text, length(labels)),
            label = rep(NA_character_, length(labels)),
            score = rep(NA_real_, length(labels)),
            .input_idx = rep(input_idx, length(labels)),
            .error = FALSE,
            .error_msg = NA_character_
          )
        }
      }
    }
  )

  # Sort by input index and label score
  results <- dplyr::arrange(results, .data$.input_idx, dplyr::desc(.data$score))
  results
}
