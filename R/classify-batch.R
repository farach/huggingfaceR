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
#' @param endpoint_url Character string or NULL. A custom Inference Endpoint URL.
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
                               model = hf_default_model("classify"),
                               token = NULL,
                               batch_size = 100L,
                               max_active = 10L,
                               progress = TRUE,
                               endpoint_url = NULL) {

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
  label <- rep(NA_character_, length(text))
  score <- rep(NA_real_, length(text))
  error <- logical(length(text))
  error_msg <- rep(NA_character_, length(text))

  for (i in seq_along(batches)) {
    rows <- batches[[i]]$indices
    if (failed[[i]]) {
      # All texts in this batch failed
      error[rows] <- TRUE
      error_msg[rows] <- messages[[i]]
      next
    }

    result <- httr2::resp_body_json(responses[[i]])
    classifications <- hf_classification_rows(result, length(rows))
    label[rows] <- purrr::map_chr(classifications, ~ .x$label %||% NA_character_)
    score[rows] <- purrr::map_dbl(classifications, ~ .x$score %||% NA_real_)
  }

  tibble::tibble(
    text = text,
    label = label,
    score = score,
    .input_idx = seq_along(text),
    .error = error,
    .error_msg = error_msg
  )
}


# Extracts one {label, score} pair per input from a parsed classification
# response.
#
# The classification API returns a nested structure whose first element holds
# the classifications. For a batch of N texts it holds N {label, score} items,
# the top classification for each text. For a single text it holds every label
# for that text, so the first (top) one is used. Anything else yields missing
# values for every text.
hf_classification_rows <- function(result, n) {
  missing_row <- list(label = NA_character_, score = NA_real_)

  if (!is.list(result) || length(result) == 0 || !is.list(result[[1]])) {
    return(rep(list(missing_row), n))
  }

  inner <- result[[1]]
  if (length(inner) == n && n > 1) {
    purrr::map(inner, function(item) {
      list(label = item$label %||% NA_character_,
           score = item$score %||% NA_real_)
    })
  } else if (n == 1 && !is.null(inner[[1]]$label)) {
    list(list(label = inner[[1]]$label, score = inner[[1]]$score))
  } else {
    rep(list(missing_row), n)
  }
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
#' hf_classify_chunks(texts, output_dir = "classify_output", chunk_size = 1000)
#'
#' # Read results
#' results <- hf_read_chunks("classify_output")
#' }
hf_classify_chunks <- function(text,
                                output_dir,
                                model = hf_default_model("classify"),
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
      progress = progress,
      endpoint_url = endpoint_url
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
#' @param endpoint_url Character string or NULL. A custom Inference Endpoint URL.
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
                                         model = hf_default_model("zero_shot"),
                                         multi_label = FALSE,
                                         token = NULL,
                                         batch_size = 50L,
                                         max_active = 10L,
                                         progress = TRUE,
                                         endpoint_url = NULL) {

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

  # Build requests from one resolved template
  reqs <- hf_build_batch_requests(
    model_id = model,
    inputs = lapply(batches, `[[`, "value"),
    parameters = list(
      candidate_labels = labels,
      multi_label = multi_label
    ),
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

  # Parse each response into plain column vectors and combine them once, rather
  # than building a tibble for every label of every text.
  pieces <- lapply(seq_along(batches), function(i) {
    input_text <- batches[[i]]$value[1]
    input_idx <- batches[[i]]$indices[1]

    if (failed[[i]]) {
      return(hf_zero_shot_missing(
        input_text, input_idx, length(labels),
        error = TRUE, message = messages[[i]]
      ))
    }

    hf_zero_shot_rows(
      httr2::resp_body_json(responses[[i]]), input_text, input_idx, length(labels)
    )
  })

  columns <- c("text", "label", "score", ".input_idx", ".error", ".error_msg")
  results <- lapply(columns, function(column) {
    unlist(lapply(pieces, .subset2, column), use.names = FALSE)
  })
  names(results) <- columns
  results <- tibble::as_tibble(results)

  # Sort by input index and label score
  results <- dplyr::arrange(results, .data$.input_idx, dplyr::desc(.data$score))
  results
}


# Converts one parsed zero-shot response into result columns for one input
# text. Responses are either a list of {label, score} objects or the legacy
# {labels, scores} object; anything else yields one missing row per candidate
# label.
hf_zero_shot_rows <- function(result, input_text, input_idx, n_labels) {
  if (is.list(result) && length(result) > 0 && !is.null(result[[1]]$label)) {
    n <- length(result)
    return(list(
      text = rep(input_text, n),
      label = unlist(
        lapply(result, function(item) item$label %||% NA_character_),
        use.names = FALSE
      ),
      score = unlist(
        lapply(result, function(item) item$score %||% NA_real_),
        use.names = FALSE
      ),
      .input_idx = rep(input_idx, n),
      .error = rep(FALSE, n),
      .error_msg = rep(NA_character_, n)
    ))
  }

  if (!is.null(result$labels) && !is.null(result$scores)) {
    # Legacy shape. tibble() keeps its recycling and length checks.
    return(as.list(tibble::tibble(
      text = input_text,
      label = unlist(result$labels),
      score = unlist(result$scores),
      .input_idx = input_idx,
      .error = FALSE,
      .error_msg = NA_character_
    )))
  }

  hf_zero_shot_missing(
    input_text, input_idx, n_labels,
    error = FALSE, message = NA_character_
  )
}


# One missing row per candidate label, for a text whose request failed or whose
# response could not be parsed.
hf_zero_shot_missing <- function(input_text, input_idx, n_labels, error, message) {
  list(
    text = rep(input_text, n_labels),
    label = rep(NA_character_, n_labels),
    score = rep(NA_real_, n_labels),
    .input_idx = rep(input_idx, n_labels),
    .error = rep(error, n_labels),
    .error_msg = rep(message, n_labels)
  )
}
