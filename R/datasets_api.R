#' Detect Default Config for a Dataset
#'
#' Query the splits endpoint to find the default config for a given dataset and split.
#'
#' @param dataset Character string. Dataset name.
#' @param split Character string. The split to find a config for.
#' @param token Character string or NULL. API token.
#'
#' @returns Character string with the config name.
#' @keywords internal
hf_detect_config <- function(dataset, split, token = NULL) {
  req <- httr2::request("https://datasets-server.huggingface.co/splits") |>
    httr2::req_url_query(dataset = dataset)

  if (!is.null(token)) {
    req <- httr2::req_auth_bearer_token(req, token)
  }

  resp <- tryCatch(
    req |>
      httr2::req_error(body = function(resp) {
        paste0("Failed to detect config for dataset '", dataset, "'")
      }) |>
      httr2::req_perform(),
    error = function(e) NULL
  )

  if (is.null(resp)) {
    # Dataset not found - try resolving the full ID via Hub API
    resolved <- hf_resolve_dataset(dataset, token)
    if (!is.null(resolved)) {
      return(hf_detect_config(resolved, split, token))
    }
    stop(
      paste0("Could not auto-detect config for dataset '", dataset,
             "'. Please specify the full dataset ID (e.g., 'org/dataset') ",
             "and/or the 'config' parameter explicitly."),
      call. = FALSE
    )
  }

  result <- httr2::resp_body_json(resp)

  if (is.null(result$splits) || length(result$splits) == 0) {
    stop(
      paste0("No configs found for dataset '", dataset,
             "'. Please specify the 'config' parameter explicitly."),
      call. = FALSE
    )
  }

  # Find configs that match the requested split
  matching <- purrr::keep(result$splits, function(s) s$split == split)

  if (length(matching) > 0) {
    return(matching[[1]]$config)
  }

  # If no match for the split, use the first available config
  result$splits[[1]]$config
}


#' Resolve a Short Dataset Name to Full ID
#'
#' Uses the Hub API to find the full org/name ID for a dataset.
#'
#' @param dataset Character string. Short dataset name.
#' @param token Character string or NULL. API token.
#'
#' @returns Character string with full dataset ID, or NULL if not found.
#' @keywords internal
hf_resolve_dataset <- function(dataset, token = NULL) {
  req <- httr2::request("https://huggingface.co/api/datasets") |>
    httr2::req_url_query(search = dataset, limit = 5)

  if (!is.null(token)) {
    req <- httr2::req_auth_bearer_token(req, token)
  }

  resp <- tryCatch(
    req |> httr2::req_perform(),
    error = function(e) NULL
  )

  if (is.null(resp)) return(NULL)

  results <- httr2::resp_body_json(resp)

  if (length(results) == 0) return(NULL)

  # Look for an exact match on the dataset name portion
  for (r in results) {
    id <- r$id %||% ""
    # Match if the name after "/" equals the search term
    name_part <- sub("^.*/", "", id)
    if (tolower(name_part) == tolower(dataset)) {
      return(id)
    }
  }

  NULL
}


#' Load Dataset via Hugging Face Datasets Server API
#'
#' Load a dataset from Hugging Face Hub using the Datasets Server API.
#' This is an API-first approach that doesn't require Python.
#' For local dataset loading with Python, see the legacy function or advanced vignette.
#'
#' @param dataset Character string. Dataset name (e.g., "imdb", "squad").
#' @param split Character string. Dataset split: "train", "test", "validation", etc.
#'   Default: "train".
#' @param config Character string or NULL. Dataset configuration/subset name.
#'   If NULL (default), auto-detected from the dataset's available configs.
#' @param limit Integer. Maximum number of rows to fetch. Default: 1000.
#'   Set to Inf to fetch all rows (may be slow for large datasets).
#' @param offset Integer. Row offset for pagination. Default: 0.
#' @param token Character string or NULL. API token for private datasets.
#'
#' @returns A tibble with dataset rows, plus .dataset and .split columns.
#' @export
#'
#' @examples
#' \dontrun{
#' # Load first 1000 rows of IMDB train set
#' imdb <- hf_load_dataset("imdb", split = "train", limit = 1000)
#'
#' # Load test set
#' imdb_test <- hf_load_dataset("imdb", split = "test", limit = 500)
#' }
hf_load_dataset <- function(dataset,
                            split = "train",
                            config = NULL,
                            limit = 1000,
                            offset = 0,
                            token = NULL) {

  token <- hf_get_token(token, required = FALSE)

  # Resolve short dataset names (e.g., "imdb" -> "stanfordnlp/imdb")
  if (!grepl("/", dataset)) {
    resolved <- hf_resolve_dataset(dataset, token)
    if (!is.null(resolved)) {
      dataset <- resolved
    }
  }

  # Auto-detect config if not provided
  if (is.null(config)) {
    config <- hf_detect_config(dataset, split, token)
  }

  # Build API URL with all required parameters
  url <- paste0(
    "https://datasets-server.huggingface.co/rows?dataset=",
    utils::URLencode(dataset, reserved = TRUE),
    "&config=", utils::URLencode(config, reserved = TRUE),
    "&split=", utils::URLencode(split, reserved = TRUE)
  )
  
  # Fetch data in batches if limit > 100
  all_rows <- list()
  current_offset <- offset
  rows_fetched <- 0
  batch_size <- min(100, limit)  # API max is usually 100 per request
  
  while (rows_fetched < limit) {
    # Adjust batch size for last batch
    current_batch_size <- min(batch_size, limit - rows_fetched)
    
    req <- httr2::request(url)
    
    if (!is.null(token)) {
      req <- httr2::req_auth_bearer_token(req, token)
    }
    
    resp <- req |>
      httr2::req_url_query(
        offset = current_offset,
        length = current_batch_size
      ) |>
      httr2::req_error(body = function(resp) {
        body <- tryCatch(
          httr2::resp_body_json(resp),
          error = function(e) list(error = "Unknown error")
        )
        error_msg <- body$error %||% "Unknown error"
        
        if (grepl("not found", error_msg, ignore.case = TRUE)) {
          paste0("Dataset '", dataset, "' or split '", split, "' not found")
        } else {
          paste0("API error: ", error_msg)
        }
      }) |>
      httr2::req_perform()
    
    result <- httr2::resp_body_json(resp)
    
    # Extract rows
    if (!is.null(result$rows) && length(result$rows) > 0) {
      batch_rows <- purrr::map(result$rows, function(row) {
        row$row  # The actual data is in the 'row' field
      })
      
      all_rows <- c(all_rows, batch_rows)
      rows_fetched <- rows_fetched + length(batch_rows)
      current_offset <- current_offset + length(batch_rows)
      
      # If we got fewer rows than requested, we've reached the end
      if (length(batch_rows) < current_batch_size) {
        break
      }
    } else {
      break
    }
  }
  
  if (length(all_rows) == 0) {
    cli::cli_warn("No rows returned. Dataset may be empty or unavailable via API.")
    return(tibble::tibble())
  }
  
  # Convert to tibble
  df <- purrr::map_dfr(all_rows, function(row) {
    # Convert each row to a tibble
    tibble::as_tibble(lapply(row, function(x) {
      if (is.null(x)) NA else x
    }))
  })
  
  # Add metadata columns
  df$.dataset <- dataset
  df$.split <- split
  
  df
}


#' Get Dataset Information
#'
#' Retrieve metadata about a dataset from Hugging Face Hub.
#'
#' @param dataset Character string. Dataset name.
#' @param token Character string or NULL. API token for private datasets.
#'
#' @returns A list with dataset information.
#' @export
#'
#' @examples
#' \dontrun{
#' hf_dataset_info("imdb")
#' }
hf_dataset_info <- function(dataset, token = NULL) {

  token <- hf_get_token(token, required = FALSE)

  # Resolve short dataset names
  if (!grepl("/", dataset)) {
    resolved <- hf_resolve_dataset(dataset, token)
    if (!is.null(resolved)) {
      dataset <- resolved
    }
  }

  req <- httr2::request(
    paste0("https://datasets-server.huggingface.co/info?dataset=",
           utils::URLencode(dataset, reserved = TRUE))
  )
  
  if (!is.null(token)) {
    req <- httr2::req_auth_bearer_token(req, token)
  }
  
  resp <- req |>
    httr2::req_error(body = function(resp) {
      paste0("Dataset '", dataset, "' not found or unavailable")
    }) |>
    httr2::req_perform()
  
  httr2::resp_body_json(resp)
}
