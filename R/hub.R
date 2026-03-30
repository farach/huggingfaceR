#' Search Models on Hugging Face Hub
#'
#' Search for models using various filters. Returns a tibble of matching models.
#'
#' @param search Character string. Search query to filter models.
#' @param task Character string. Filter by task (e.g., "text-classification").
#' @param author Character string. Filter by model author/organization.
#' @param language Character string. Filter by language (e.g., "en").
#' @param library Character string. Filter by library (e.g., "pytorch", "transformers").
#' @param tags Character vector. Filter by tags.
#' @param sort Character string. Sort by field: "downloads", "likes", "created", "updated".
#'   Default: "downloads".
#' @param direction Character string. Sort direction: "asc" or "desc". Default: "desc".
#' @param limit Integer. Maximum number of models to return. Default: 30.
#' @param token Character string or NULL. API token for authentication.
#'
#' @returns A tibble with model information.
#' @export
#'
#' @examples
#' \dontrun{
#' # Search by task
#' hf_search_models(task = "text-classification", limit = 10)
#'
#' # Search by author
#' hf_search_models(author = "facebook", sort = "downloads")
#'
#' # Search with query
#' hf_search_models(search = "sentiment", task = "text-classification")
#' }
hf_search_models <- function(search = NULL,
                             task = NULL,
                             author = NULL,
                             language = NULL,
                             library = NULL,
                             tags = NULL,
                             sort = "downloads",
                             direction = "desc",
                             limit = 30,
                             token = NULL) {
  
  token <- hf_get_token(token, required = FALSE)
  
  # Build query parameters
  params <- list(
    limit = limit,
    sort = sort,
    direction = if (direction == "desc") -1 else 1
  )
  
  if (!is.null(search)) params$search <- search
  if (!is.null(task)) params$pipeline_tag <- task
  if (!is.null(author)) params$author <- author
  if (!is.null(language)) params$language <- language
  if (!is.null(library)) params$library <- library
  if (!is.null(tags) && length(tags) > 0) {
    params$tags <- paste(tags, collapse = ",")
  }
  
  # Make API request
  req <- httr2::request("https://huggingface.co/api/models")
  
  if (!is.null(token)) {
    req <- httr2::req_auth_bearer_token(req, token)
  }
  
  resp <- req |>
    httr2::req_url_query(!!!params) |>
    httr2::req_error(body = function(resp) {
      "Failed to search models"
    }) |>
    httr2::req_perform()
  
  models <- httr2::resp_body_json(resp)
  
  # Convert to tibble
  if (length(models) == 0) {
    return(tibble::tibble(
      model_id = character(),
      author = character(),
      task = character(),
      downloads = integer(),
      likes = integer(),
      tags = list()
    ))
  }
  
  purrr::map_dfr(models, function(model) {
    tibble::tibble(
      model_id = model$id %||% NA_character_,
      author = model$author %||% NA_character_,
      task = model$pipeline_tag %||% NA_character_,
      downloads = model$downloads %||% 0L,
      likes = model$likes %||% 0L,
      tags = list(unlist(model$tags) %||% character(0)),
      library = model$library_name %||% NA_character_
    )
  })
}


#' Get Model Information
#'
#' Retrieve detailed information about a specific model.
#'
#' @param model_id Character string. The model ID (e.g., "bert-base-uncased").
#' @param token Character string or NULL. API token for authentication.
#'
#' @returns A list with detailed model information.
#' @export
#'
#' @examples
#' \dontrun{
#' # Get model details
#' hf_model_info("sentence-transformers/all-MiniLM-L6-v2")
#' }
hf_model_info <- function(model_id, token = NULL) {
  
  token <- hf_get_token(token, required = FALSE)
  
  req <- httr2::request(paste0("https://huggingface.co/api/models/", model_id))
  
  if (!is.null(token)) {
    req <- httr2::req_auth_bearer_token(req, token)
  }
  
  resp <- req |>
    httr2::req_error(body = function(resp) {
      paste0("Model '", model_id, "' not found")
    }) |>
    httr2::req_perform()
  
  httr2::resp_body_json(resp)
}


#' Check Inference API Availability
#'
#' Check whether a model supports the Hugging Face Serverless Inference API.
#' Not all models on the Hub are served by the Inference API. This function
#' queries model metadata to determine availability before you make inference
#' calls.
#'
#' @param model_id Character string. The model ID (e.g., "BAAI/bge-small-en-v1.5").
#' @param token Character string or NULL. API token for authentication.
#' @param quiet Logical. If TRUE, suppress console output and return result
#'   invisibly. Default: FALSE.
#'
#' @returns A list (invisibly if quiet = TRUE) with components:
#'   \item{model_id}{The model ID queried.}
#'   \item{available}{Logical. TRUE if the model is available on the Inference API.}
#'   \item{pipeline_tag}{The model's task type (e.g., "feature-extraction").}
#'   \item{inference_provider}{The inference provider, if available.}
#' @export
#'
#' @examples
#' \dontrun{
#' # Check if a model supports serverless inference
#' hf_check_inference("BAAI/bge-small-en-v1.5")
#'
#' # Use programmatically
#' result <- hf_check_inference("some-org/some-model", quiet = TRUE)
#' if (result$available) {
#'   embeddings <- hf_embed("hello", model = "some-org/some-model")
#' }
#' }
hf_check_inference <- function(model_id, token = NULL, quiet = FALSE) {

  token <- hf_get_token(token, required = FALSE)

  # Query the model metadata from the Hub API
  info <- tryCatch(
    hf_model_info(model_id, token = token),
    error = function(e) NULL
  )

  if (is.null(info)) {
    result <- list(
      model_id = model_id,
      available = FALSE,
      pipeline_tag = NA_character_,
      inference_provider = NA_character_
    )
    if (!quiet) {
      cli::cli_alert_danger("Model {.val {model_id}} was not found on the Hub.")
    }
    return(if (quiet) invisible(result) else result)
  }

  pipeline_tag <- info$pipeline_tag %||% NA_character_

  # Check for inference provider availability via tags
  tags <- unlist(info$tags) %||% character(0)

  # Detect inference provider from tags (these indicate the model is served)
  inference_provider <- NA_character_
  provider_tags <- grep(
    "^text-embeddings-inference$|^text-generation-inference$",
    tags, value = TRUE
  )
  if (length(provider_tags) > 0) {
    inference_provider <- provider_tags[1]
  }

  # A model is likely available on the Inference API if:

  # 1. It has a recognized inference provider tag, OR
  # 2. It is marked endpoints_compatible AND has a pipeline_tag
  has_provider <- !is.na(inference_provider)
  has_endpoints <- "endpoints_compatible" %in% tags && !is.na(pipeline_tag)
  available <- has_provider || has_endpoints

  result <- list(
    model_id = model_id,
    available = available,
    pipeline_tag = pipeline_tag,
    inference_provider = inference_provider
  )

  if (!quiet) {
    if (available) {
      cli::cli_alert_success(
        "Model {.val {model_id}} is available on the Inference API (task: {.val {pipeline_tag}})."
      )
    } else {
      cli::cli_alert_warning(
        paste0(
          "Model {.val {model_id}} exists on the Hub (task: {.val {pipeline_tag}}) ",
          "but may not be available for serverless inference. ",
          "Check the model card at {.url https://huggingface.co/{model_id}} ",
          "for an Inference API widget."
        )
      )
    }
  }

  if (quiet) invisible(result) else result
}


#' List Available Tasks
#'
#' List all available task types on Hugging Face.
#'
#' @param pattern Character string or NULL. Optional regex pattern to filter tasks.
#'
#' @returns A character vector of task names.
#' @export
#'
#' @examples
#' \dontrun{
#' # List all tasks
#' hf_list_tasks()
#'
#' # Filter tasks
#' hf_list_tasks(pattern = "classification")
#' }
hf_list_tasks <- function(pattern = NULL) {
  
  # Common HF tasks (as of 2024)
  tasks <- c(
    "text-classification",
    "token-classification",
    "question-answering",
    "summarization",
    "translation",
    "text-generation",
    "text2text-generation",
    "fill-mask",
    "sentence-similarity",
    "feature-extraction",
    "zero-shot-classification",
    "conversational",
    "image-classification",
    "image-segmentation",
    "object-detection",
    "image-to-text",
    "text-to-image",
    "audio-classification",
    "automatic-speech-recognition",
    "text-to-speech",
    "table-question-answering",
    "visual-question-answering"
  )
  
  if (!is.null(pattern)) {
    tasks <- grep(pattern, tasks, value = TRUE, ignore.case = TRUE)
  }
  
  tasks
}


#' Search Datasets on Hugging Face Hub
#'
#' Search for datasets using various filters.
#'
#' @param search Character string. Search query to filter datasets.
#' @param task Character string. Filter by task.
#' @param language Character string. Filter by language.
#' @param size Character string. Filter by size: "small", "medium", "large".
#' @param sort Character string. Sort by: "downloads", "likes", "created", "updated".
#'   Default: "downloads".
#' @param limit Integer. Maximum number of datasets to return. Default: 30.
#' @param token Character string or NULL. API token for authentication.
#'
#' @returns A tibble with dataset information.
#' @export
#'
#' @examples
#' \dontrun{
#' # Search datasets
#' hf_search_datasets(search = "sentiment", limit = 10)
#' }
hf_search_datasets <- function(search = NULL,
                               task = NULL,
                               language = NULL,
                               size = NULL,
                               sort = "downloads",
                               limit = 30,
                               token = NULL) {
  
  token <- hf_get_token(token, required = FALSE)
  
  # Build query parameters
  params <- list(
    limit = limit,
    sort = sort
  )
  
  if (!is.null(search)) params$search <- search
  if (!is.null(task)) params$task <- task
  if (!is.null(language)) params$language <- language
  if (!is.null(size)) params$size <- size
  
  # Make API request
  req <- httr2::request("https://huggingface.co/api/datasets")
  
  if (!is.null(token)) {
    req <- httr2::req_auth_bearer_token(req, token)
  }
  
  resp <- req |>
    httr2::req_url_query(!!!params) |>
    httr2::req_error(body = function(resp) {
      "Failed to search datasets"
    }) |>
    httr2::req_perform()
  
  datasets <- httr2::resp_body_json(resp)
  
  # Convert to tibble
  if (length(datasets) == 0) {
    return(tibble::tibble(
      dataset_id = character(),
      author = character(),
      downloads = integer(),
      likes = integer(),
      tags = list()
    ))
  }
  
  purrr::map_dfr(datasets, function(dataset) {
    tibble::tibble(
      dataset_id = dataset$id %||% NA_character_,
      author = dataset$author %||% NA_character_,
      downloads = dataset$downloads %||% 0L,
      likes = dataset$likes %||% 0L,
      tags = list(unlist(dataset$tags) %||% character(0))
    )
  })
}
