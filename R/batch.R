#' @importFrom rlang .data
NULL

#' Split Vector into Batches
#'
#' Internal helper to split a vector into batches with index tracking.
#'
#' @param x Vector to split into batches.
#' @param batch_size Integer. Maximum size of each batch.
#'
#' @returns A list of lists, each containing:
#'   - `value`: The batch values
#'   - `indices`: The original indices of the values
#' @keywords internal
batch_vector <- function(x, batch_size) {
  if (length(x) == 0) {
    return(list())
  }

  n <- length(x)
  n_batches <- ceiling(n / batch_size)

  lapply(seq_len(n_batches), function(i) {
    start_idx <- (i - 1) * batch_size + 1
    end_idx <- min(i * batch_size, n)
    list(
      value = x[start_idx:end_idx],
      indices = start_idx:end_idx
    )
  })
}


#' Build a Hugging Face API Request
#'
#' Internal factory function to construct httr2 request objects for batch processing.
#'
#' @param model_id Character string. The model ID on Hugging Face Hub.
#' @param inputs The input data (usually character vector or list).
#' @param parameters Optional list of parameters for the inference.
#' @param token Character string or NULL. API token for authentication.
#' @param wait_for_model Logical. Wait for model to load if not ready. Default: TRUE.
#' @param use_cache Logical. Use cached results for identical inputs. Default: TRUE.
#'
#' @returns An httr2 request object ready for execution.
#' @keywords internal
hf_build_request <- function(model_id,
                              inputs,
                              parameters = NULL,
                              token = NULL,
                              wait_for_model = TRUE,
                              use_cache = TRUE) {

  token <- hf_get_token(token, required = FALSE)

  # Build request body
  body <- list(inputs = inputs)
  if (!is.null(parameters)) {
    body$parameters <- parameters
  }
  if (wait_for_model) {
    body$options <- list(wait_for_model = TRUE)
  }
  if (!is.null(use_cache)) {
    if (is.null(body$options)) body$options <- list()
    body$options$use_cache <- use_cache
  }

  # Build request
 req <- httr2::request(paste0("https://router.huggingface.co/hf-inference/models/", model_id))

  if (!is.null(token)) {
    req <- httr2::req_auth_bearer_token(req, token)
  }

  req |>
    httr2::req_body_json(body) |>
    httr2::req_retry(max_tries = 3) |>
    httr2::req_throttle(rate = 10 / 1)
}


#' Perform Parallel Batch Requests
#'
#' Internal function to execute multiple API requests in parallel with error handling.
#'
#' @param reqs List of httr2 request objects.
#' @param input_indices Integer vector. Original indices corresponding to each request.
#' @param max_active Integer. Maximum concurrent requests. Default: 10.
#' @param progress Logical. Show progress bar. Default: TRUE.
#'
#' @returns A tibble with columns:
#'   - `.input_idx`: Original input index
#'   - `response`: The response object (or error)
#'   - `.error`: Logical, TRUE if request failed
#'   - `.error_msg`: Error message or NA
#' @keywords internal
hf_perform_batch <- function(reqs, input_indices, max_active = 10L, progress = TRUE) {

  if (length(reqs) == 0) {
    return(tibble::tibble(
      .input_idx = integer(),
      response = list(),
      .error = logical(),
      .error_msg = character()
    ))
  }

  results <- httr2::req_perform_parallel(
    reqs,
    on_error = "continue",
    progress = progress,
    max_active = max_active
  )

  tibble::tibble(
    .input_idx = input_indices,
    response = results,
    .error = purrr::map_lgl(results, inherits, "error"),
    .error_msg = purrr::map_chr(results, ~ {
      if (inherits(.x, "error")) conditionMessage(.x) else NA_character_
    })
  )
}


#' Write Chunk to Parquet
#'
#' Internal function to write a data chunk to a parquet file.
#'
#' @param data A tibble to write.
#' @param output_dir Character string. Directory to write to.
#' @param chunk_id Integer. Chunk identifier.
#' @param prefix Character string. File name prefix. Default: "chunk".
#'
#' @returns Invisibly returns the file path written.
#' @keywords internal
hf_write_chunk <- function(data, output_dir, chunk_id, prefix = "chunk") {

  if (!requireNamespace("arrow", quietly = TRUE)) {
    stop("Package 'arrow' is required for chunk operations. Install with: install.packages('arrow')",
         call. = FALSE)
  }

  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }

  file_path <- file.path(output_dir, sprintf("%s_%04d.parquet", prefix, chunk_id))
  arrow::write_parquet(data, file_path)

  invisible(file_path)
}


#' Read All Chunks from Directory
#'
#' Read and combine all parquet chunk files from a directory.
#'
#' @param output_dir Character string. Directory containing chunk files.
#' @param pattern Character string. Glob pattern to match files. Default: "*.parquet".
#'
#' @returns A tibble combining all chunks, sorted by `.input_idx` if present.
#' @export
#'
#' @examples
#' \dontrun{
#' # After running hf_embed_chunks()
#' results <- hf_read_chunks("my_output_dir")
#' }
hf_read_chunks <- function(output_dir, pattern = "*.parquet") {

  if (!requireNamespace("arrow", quietly = TRUE)) {
    stop("Package 'arrow' is required for chunk operations. Install with: install.packages('arrow')",
         call. = FALSE)
  }

  if (!dir.exists(output_dir)) {
    stop(paste0("Directory not found: ", output_dir), call. = FALSE)
  }

  files <- list.files(output_dir, pattern = utils::glob2rx(pattern), full.names = TRUE)

  if (length(files) == 0) {
    cli::cli_warn("No parquet files found in {.path {output_dir}}")
    return(tibble::tibble())
  }

  # Read all files
  chunks <- purrr::map(files, arrow::read_parquet)
  combined <- dplyr::bind_rows(chunks)

  # Sort by input index if present
 if (".input_idx" %in% names(combined)) {
    combined <- dplyr::arrange(combined, .data$.input_idx)
  }

  combined
}


#' Get Existing Chunk IDs
#'
#' Internal function to find which chunks have already been written.
#'
#' @param output_dir Character string. Directory to check.
#' @param prefix Character string. File name prefix to match.
#'
#' @returns Integer vector of chunk IDs that already exist.
#' @keywords internal
hf_get_existing_chunks <- function(output_dir, prefix = "chunk") {

  if (!dir.exists(output_dir)) {
    return(integer())
  }

  files <- list.files(output_dir, pattern = paste0("^", prefix, "_\\d{4}\\.parquet$"))

  if (length(files) == 0) {
    return(integer())
  }

  # Extract chunk IDs from filenames
  matches <- regmatches(files, regexpr("\\d{4}", files))
  as.integer(matches)
}
