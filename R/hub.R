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
  
  params <- list(
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
  
  models <- hf_hub_paginated_get(
    "https://huggingface.co/api/models",
    params = params,
    limit = limit,
    token = token,
    error = "Failed to search models"
  )
  
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

  providers <- hf_list_providers(model_id, token = token)
  live_providers <- providers$provider[providers$status == "live"]

  # Check for inference provider availability via tags
  tags <- unlist(info$tags) %||% character(0)

  # Detect inference provider from tags (these indicate the model is served)
  inference_provider <- NA_character_
  provider_tags <- grep(
    "^text-embeddings-inference$|^text-generation-inference$",
    tags, value = TRUE
  )
  if (length(live_providers) > 0) {
    inference_provider <- live_providers[[1]]
  } else if (length(provider_tags) > 0) {
    inference_provider <- provider_tags[1]
  }

  # A model is likely available on the Inference API if:

  # 1. It has a recognized inference provider tag, OR
  # 2. It is marked endpoints_compatible AND has a pipeline_tag
  has_provider <- !is.na(inference_provider)
  has_endpoints <- "endpoints_compatible" %in% tags && !is.na(pipeline_tag)
  available <- has_provider || has_endpoints || length(live_providers) > 0

  result <- list(
    model_id = model_id,
    available = available,
    pipeline_tag = pipeline_tag,
    inference_provider = inference_provider,
    providers = providers
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
  
  params <- list(sort = sort)
  
  if (!is.null(search)) params$search <- search
  if (!is.null(task)) params$task <- task
  if (!is.null(language)) params$language <- language
  if (!is.null(size)) params$size <- size
  
  datasets <- hf_hub_paginated_get(
    "https://huggingface.co/api/datasets",
    params = params,
    limit = limit,
    token = token,
    error = "Failed to search datasets"
  )
  
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


#' Search Spaces on Hugging Face Hub
#'
#' Search hosted Spaces and return one row per result.
#'
#' @param search Character string or NULL. Search query.
#' @param author Character string or NULL. Filter by owner.
#' @param sort Character string. Sort field passed to the Hub API.
#' @param direction Character string. Sort direction: "asc" or "desc".
#' @param limit Integer. Maximum number of Spaces to return.
#' @param token Character string or NULL. API token for authentication.
#'
#' @returns A tibble with Space metadata.
#' @export
#'
#' @examples
#' \dontrun{
#' hf_search_spaces(search = "chat", limit = 10)
#' }
hf_search_spaces <- function(search = NULL,
                             author = NULL,
                             sort = "likes",
                             direction = "desc",
                             limit = 30,
                             token = NULL) {
  token <- hf_get_token(token, required = FALSE)
  params <- list(sort = sort, direction = if (direction == "desc") -1 else 1)
  if (!is.null(search)) params$search <- search
  if (!is.null(author)) params$author <- author

  spaces <- hf_hub_paginated_get(
    "https://huggingface.co/api/spaces",
    params = params,
    limit = limit,
    token = token,
    error = "Failed to search Spaces"
  )

  if (length(spaces) == 0) {
    return(tibble::tibble(
      space_id = character(),
      author = character(),
      sdk = character(),
      likes = integer(),
      tags = list()
    ))
  }

  purrr::map_dfr(spaces, function(space) {
    tibble::tibble(
      space_id = space$id %||% NA_character_,
      author = sub("/.*$", "", space$id %||% NA_character_),
      sdk = space$sdk %||% NA_character_,
      likes = space$likes %||% 0L,
      last_modified = space$lastModified %||% NA_character_,
      tags = list(unlist(space$tags) %||% character(0))
    )
  })
}


#' Search Papers on Hugging Face
#'
#' Search papers indexed by Hugging Face.
#'
#' @param search Character string or NULL. Search query.
#' @param limit Integer. Maximum number of papers to return.
#' @param token Character string or NULL. API token for authentication.
#'
#' @returns A tibble with paper metadata.
#' @export
#'
#' @examples
#' \dontrun{
#' hf_search_papers("transformers", limit = 10)
#' }
hf_search_papers <- function(search = NULL, limit = 30, token = NULL) {
  token <- hf_get_token(token, required = FALSE)
  params <- list()
  if (!is.null(search)) params$search <- search

  papers <- hf_hub_paginated_get(
    "https://huggingface.co/api/papers",
    params = params,
    limit = limit,
    token = token,
    error = "Failed to search papers"
  )

  if (length(papers) == 0) {
    return(tibble::tibble(
      paper_id = character(),
      title = character(),
      upvotes = integer(),
      authors = list(),
      url = character()
    ))
  }

  purrr::map_dfr(papers, function(paper) {
    tibble::tibble(
      paper_id = paper$id %||% NA_character_,
      title = paper$title %||% NA_character_,
      upvotes = paper$upvotes %||% 0L,
      published_at = paper$publishedAt %||% NA_character_,
      authors = list(vapply(paper$authors %||% list(), function(author) {
        author$name %||% NA_character_
      }, character(1))),
      url = paste0("https://huggingface.co/papers/", paper$id %||% "")
    )
  })
}


#' List Files in a Hub Repository
#'
#' List files and directories in a model, dataset, or Space repository.
#'
#' @param repo_id Character string. Repository ID, e.g. "BAAI/bge-small-en-v1.5".
#' @param repo_type Character string. One of "model", "dataset", or "space".
#' @param revision Character string. Git revision, branch, or tag. Default:
#'   "main".
#' @param recursive Logical. If TRUE, list files recursively.
#' @param token Character string or NULL. API token for private repositories.
#'
#' @returns A tibble with columns: path, type, size, oid.
#' @export
#'
#' @examples
#' \dontrun{
#' hf_list_repo_files("BAAI/bge-small-en-v1.5")
#' }
hf_list_repo_files <- function(repo_id,
                               repo_type = "model",
                               revision = "main",
                               recursive = TRUE,
                               token = NULL) {
  repo_type <- hf_match_repo_type(repo_type)
  token <- hf_get_token(token, required = FALSE)

  url <- hf_repo_api_url(repo_type, repo_id, "tree", revision)
  params <- list(recursive = if (isTRUE(recursive)) "true" else "false")
  files <- hf_hub_get_json(url, params = params, token = token,
                           error = paste0("Failed to list files for repo '", repo_id, "'"))

  if (length(files) == 0) {
    return(tibble::tibble(
      path = character(),
      type = character(),
      size = integer(),
      oid = character()
    ))
  }

  purrr::map_dfr(files, function(file) {
    tibble::tibble(
      path = file$path %||% NA_character_,
      type = file$type %||% NA_character_,
      size = file$size %||% NA_integer_,
      oid = file$oid %||% NA_character_
    )
  })
}


#' Download a File from the Hub
#'
#' Download a single file from a model, dataset, or Space repository.
#'
#' @param repo_id Character string. Repository ID.
#' @param filename Character string. Path to the file inside the repository.
#' @param repo_type Character string. One of "model", "dataset", or "space".
#' @param revision Character string. Git revision, branch, or tag.
#' @param dest Character path or NULL. If NULL, writes to a temporary file. If an
#'   existing directory, the repository filename basename is used inside it.
#' @param token Character string or NULL. API token for private repositories.
#' @param overwrite Logical. If TRUE, overwrite an existing destination file.
#'
#' @returns The downloaded file path.
#' @export
#'
#' @examples
#' \dontrun{
#' hf_hub_download("BAAI/bge-small-en-v1.5", "README.md")
#' }
hf_hub_download <- function(repo_id,
                            filename,
                            repo_type = "model",
                            revision = "main",
                            dest = NULL,
                            token = NULL,
                            overwrite = FALSE) {
  repo_type <- hf_match_repo_type(repo_type)
  token <- hf_get_token(token, required = FALSE)
  url <- hf_repo_resolve_url(repo_type, repo_id, revision, filename)
  path <- hf_download_dest(filename, dest)

  if (file.exists(path) && !isTRUE(overwrite)) {
    stop("Destination file already exists: ", path, call. = FALSE)
  }
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)

  req <- httr2::request(url) |>
    httr2::req_error(body = function(resp) {
      paste0("Failed to download '", filename, "' from repo '", repo_id, "'.")
    })
  if (!is.null(token)) {
    req <- httr2::req_auth_bearer_token(req, token)
  }

  httr2::req_perform(req, path = path)
  normalizePath(path, winslash = "/", mustWork = FALSE)
}


#' List Inference Providers for a Model
#'
#' Query Hugging Face router metadata for provider availability, pricing, latency,
#' and capabilities. Router provider metadata is available for OpenAI-compatible
#' models; non-router task models return an empty tibble.
#'
#' @param model_id Character string. Model ID.
#' @param token Character string or NULL. API token for authentication.
#'
#' @returns A tibble with one row per provider.
#' @export
#'
#' @examples
#' \dontrun{
#' hf_list_providers("Qwen/Qwen2.5-72B-Instruct")
#' }
hf_list_providers <- function(model_id, token = NULL) {
  token <- hf_get_token(token, required = FALSE)
  req <- httr2::request(paste0(
    "https://router.huggingface.co/v1/models/",
    utils::URLencode(model_id, reserved = FALSE)
  ))
  if (!is.null(token)) {
    req <- httr2::req_auth_bearer_token(req, token)
  }

  resp <- tryCatch(
    req |> httr2::req_perform(),
    error = function(e) NULL
  )
  if (is.null(resp)) {
    return(hf_empty_providers(model_id))
  }

  body <- httr2::resp_body_json(resp, simplifyVector = FALSE)
  providers <- body$data$providers %||% list()
  if (length(providers) == 0) {
    return(hf_empty_providers(model_id))
  }

  purrr::map_dfr(providers, function(provider_info) {
    tibble::tibble(
      model_id = model_id,
      provider = provider_info$provider %||% NA_character_,
      status = provider_info$status %||% NA_character_,
      context_length = provider_info$context_length %||% NA_integer_,
      input_price = provider_info$pricing$input %||% NA_real_,
      output_price = provider_info$pricing$output %||% NA_real_,
      is_free = provider_info$is_free %||% NA,
      supports_tools = provider_info$supports_tools %||% NA,
      supports_structured_output = provider_info$supports_structured_output %||% NA,
      first_token_latency_ms = provider_info$first_token_latency_ms %||% NA_real_,
      throughput = provider_info$throughput %||% NA_real_,
      is_model_author = provider_info$is_model_author %||% NA
    )
  })
}


#' Create a Hub Repository
#'
#' Create a model, dataset, or Space repository. This is a write operation and
#' requires an API token with write scope plus `confirm = TRUE`.
#'
#' @param repo_id Character string. Repository name or "namespace/name".
#' @param repo_type Character string. One of "model", "dataset", or "space".
#' @param private Logical. Whether to create a private repository.
#' @param exist_ok Logical. If TRUE, do not fail when the repo already exists.
#' @param token Character string or NULL. API token with write scope.
#' @param confirm Logical. Must be TRUE to perform the write operation.
#'
#' @returns The parsed Hub API response.
#' @export
#'
#' @examples
#' \dontrun{
#' hf_create_repo("my-dataset", repo_type = "dataset", private = TRUE, confirm = TRUE)
#' }
hf_create_repo <- function(repo_id,
                           repo_type = "model",
                           private = FALSE,
                           exist_ok = FALSE,
                           token = NULL,
                           confirm = FALSE) {
  hf_confirm_write(confirm, "create a Hub repository")
  repo_type <- hf_match_repo_type(repo_type)
  token <- hf_get_token(token, required = TRUE)
  parts <- hf_split_repo_id(repo_id)

  body <- purrr::compact(list(
    name = parts$name,
    organization = parts$namespace,
    type = repo_type,
    private = private,
    exist_ok = exist_ok
  ))

  httr2::request("https://huggingface.co/api/repos/create") |>
    httr2::req_auth_bearer_token(token) |>
    httr2::req_body_json(body) |>
    httr2::req_error(body = function(resp) {
      body <- tryCatch(httr2::resp_body_json(resp), error = function(e) list())
      paste0("Failed to create repo '", repo_id, "': ", body$error %||% "Unknown error")
    }) |>
    httr2::req_perform() |>
    httr2::resp_body_json()
}


#' Upload a File to a Hub Repository
#'
#' Upload a local file into a model, dataset, or Space repository. This is a
#' write operation and requires an API token with write scope plus
#' `confirm = TRUE`.
#'
#' @param path Local file path to upload.
#' @param repo_id Character string. Repository ID.
#' @param path_in_repo Character string or NULL. Destination path in the repo.
#'   Defaults to `basename(path)`.
#' @param repo_type Character string. One of "model", "dataset", or "space".
#' @param commit_message Character string or NULL. Commit message when supported
#'   by the Hub upload endpoint.
#' @param token Character string or NULL. API token with write scope.
#' @param overwrite Logical. If FALSE, error when `path_in_repo` already exists.
#' @param confirm Logical. Must be TRUE to perform the write operation.
#'
#' @returns The parsed Hub API response, or the response path when the endpoint
#'   returns no JSON body.
#' @export
#'
#' @examples
#' \dontrun{
#' hf_upload_file("results.csv", "me/my-dataset", repo_type = "dataset", confirm = TRUE)
#' }
hf_upload_file <- function(path,
                           repo_id,
                           path_in_repo = NULL,
                           repo_type = "model",
                           commit_message = NULL,
                           token = NULL,
                           overwrite = FALSE,
                           confirm = FALSE) {
  hf_confirm_write(confirm, "upload a file to the Hub")
  repo_type <- hf_match_repo_type(repo_type)
  token <- hf_get_token(token, required = TRUE)
  if (!file.exists(path)) {
    stop("File not found: ", path, call. = FALSE)
  }
  path_in_repo <- path_in_repo %||% basename(path)

  if (!isTRUE(overwrite)) {
    existing <- hf_list_repo_files(repo_id, repo_type = repo_type, token = token)
    if (path_in_repo %in% existing$path) {
      stop("File already exists in repo: ", path_in_repo, call. = FALSE)
    }
  }

  params <- purrr::compact(list(commit_message = commit_message))
  url <- hf_repo_api_url(repo_type, repo_id, "upload", path_in_repo)
  req <- httr2::request(url) |>
    httr2::req_auth_bearer_token(token) |>
    httr2::req_body_file(path, type = "application/octet-stream") |>
    httr2::req_error(body = function(resp) {
      body <- tryCatch(httr2::resp_body_json(resp), error = function(e) list())
      paste0("Failed to upload '", path, "': ", body$error %||% "Unknown error")
    })
  if (length(params) > 0) {
    req <- do.call(httr2::req_url_query, c(list(req), params))
  }

  resp <- httr2::req_perform(req)
  tryCatch(
    httr2::resp_body_json(resp),
    error = function(e) list(path = path_in_repo, repo_id = repo_id)
  )
}


#' Push a Data Frame as a Dataset File
#'
#' Write a data frame to CSV or Parquet and upload it to a Hub dataset
#' repository.
#'
#' @param data A data frame.
#' @param repo_id Character string. Dataset repository ID.
#' @param path_in_repo Character string or NULL. Destination path. Defaults to
#'   "data.csv" or "data.parquet".
#' @param format Character string. One of "csv" or "parquet".
#' @param create_repo Logical. If TRUE, create the dataset repo first with
#'   `exist_ok = TRUE`.
#' @param private Logical. Used when `create_repo = TRUE`.
#' @param commit_message Character string or NULL.
#' @param token Character string or NULL. API token with write scope.
#' @param overwrite Logical. If FALSE, error when the destination already exists.
#' @param confirm Logical. Must be TRUE to perform write operations.
#'
#' @returns The result from `hf_upload_file()`.
#' @export
#'
#' @examples
#' \dontrun{
#' hf_push_dataset(mtcars, "me/mtcars-small", create_repo = TRUE, confirm = TRUE)
#' }
hf_push_dataset <- function(data,
                            repo_id,
                            path_in_repo = NULL,
                            format = c("csv", "parquet"),
                            create_repo = FALSE,
                            private = FALSE,
                            commit_message = NULL,
                            token = NULL,
                            overwrite = FALSE,
                            confirm = FALSE) {
  hf_confirm_write(confirm, "push a dataset to the Hub")
  if (!is.data.frame(data)) {
    stop("`data` must be a data frame.", call. = FALSE)
  }
  format <- match.arg(format)
  token <- hf_get_token(token, required = TRUE)

  if (isTRUE(create_repo)) {
    hf_create_repo(
      repo_id,
      repo_type = "dataset",
      private = private,
      exist_ok = TRUE,
      token = token,
      confirm = confirm
    )
  }

  path_in_repo <- path_in_repo %||% paste0("data.", if (format == "csv") "csv" else "parquet")
  local_path <- tempfile(fileext = paste0(".", format))
  on.exit(unlink(local_path), add = TRUE)

  if (format == "csv") {
    utils::write.csv(data, local_path, row.names = FALSE)
  } else {
    if (!requireNamespace("arrow", quietly = TRUE)) {
      stop("The arrow package is required to write Parquet datasets.", call. = FALSE)
    }
    arrow::write_parquet(data, local_path)
  }

  hf_upload_file(
    local_path,
    repo_id = repo_id,
    path_in_repo = path_in_repo,
    repo_type = "dataset",
    commit_message = commit_message %||% paste0("Upload ", path_in_repo),
    token = token,
    overwrite = overwrite,
    confirm = confirm
  )
}


#' Delete a Hub Repository
#'
#' Delete a model, dataset, or Space repository. This destructive operation is
#' guarded: it requires `confirm = TRUE` and is refused when `CI=true`.
#'
#' @param repo_id Character string. Repository ID.
#' @param repo_type Character string. One of "model", "dataset", or "space".
#' @param token Character string or NULL. API token with write scope.
#' @param confirm Logical. Must be TRUE to delete the repository.
#'
#' @returns The parsed Hub API response.
#' @export
#'
#' @examples
#' \dontrun{
#' hf_delete_repo("me/old-test-dataset", repo_type = "dataset", confirm = TRUE)
#' }
hf_delete_repo <- function(repo_id,
                           repo_type = "model",
                           token = NULL,
                           confirm = FALSE) {
  hf_confirm_write(confirm, "delete a Hub repository")
  if (tolower(Sys.getenv("CI")) %in% c("true", "1", "yes")) {
    stop("Refusing to delete a Hub repository while CI is set.", call. = FALSE)
  }
  repo_type <- hf_match_repo_type(repo_type)
  token <- hf_get_token(token, required = TRUE)
  parts <- hf_split_repo_id(repo_id)

  body <- purrr::compact(list(
    name = parts$name,
    organization = parts$namespace,
    type = repo_type
  ))

  httr2::request("https://huggingface.co/api/repos/delete") |>
    httr2::req_auth_bearer_token(token) |>
    httr2::req_body_json(body) |>
    httr2::req_method("DELETE") |>
    httr2::req_error(body = function(resp) {
      body <- tryCatch(httr2::resp_body_json(resp), error = function(e) list())
      paste0("Failed to delete repo '", repo_id, "': ", body$error %||% "Unknown error")
    }) |>
    httr2::req_perform() |>
    httr2::resp_body_json()
}


hf_hub_get_json <- function(url, params = NULL, token = NULL,
                            error = "Hub API request failed") {
  req <- httr2::request(url)
  if (!is.null(token)) {
    req <- httr2::req_auth_bearer_token(req, token)
  }
  params <- purrr::compact(params %||% list())
  if (length(params) > 0) {
    req <- do.call(httr2::req_url_query, c(list(req), params))
  }
  req |>
    httr2::req_error(body = function(resp) error) |>
    httr2::req_perform() |>
    httr2::resp_body_json()
}


hf_hub_paginated_get <- function(url, params = NULL, limit = 30, token = NULL,
                                 error = "Hub API request failed") {
  if (!is.numeric(limit) || length(limit) != 1L || is.na(limit) || limit < 0) {
    stop("`limit` must be a non-negative number.", call. = FALSE)
  }
  if (limit == 0) {
    return(list())
  }

  out <- list()
  next_url <- NULL
  remaining <- limit

  repeat {
    page_limit <- if (is.infinite(remaining)) 100L else min(100L, remaining)
    req <- httr2::request(next_url %||% url)
    if (!is.null(token)) {
      req <- httr2::req_auth_bearer_token(req, token)
    }
    if (is.null(next_url)) {
      page_params <- c(purrr::compact(params %||% list()), list(limit = page_limit))
      req <- do.call(httr2::req_url_query, c(list(req), page_params))
    }
    resp <- req |>
      httr2::req_error(body = function(resp) error) |>
      httr2::req_perform()
    page <- httr2::resp_body_json(resp)
    if (length(page) == 0) {
      break
    }
    out <- c(out, page)

    if (!is.infinite(remaining)) {
      remaining <- limit - length(out)
      if (remaining <= 0) break
    }

    next_url <- hf_link_next(httr2::resp_header(resp, "link"))
    if (is.null(next_url)) {
      break
    }
  }

  if (!is.infinite(limit)) {
    out <- utils::head(out, limit)
  }
  out
}


hf_link_next <- function(link) {
  if (is.null(link) || !nzchar(link)) {
    return(NULL)
  }
  links <- strsplit(link, ",", fixed = TRUE)[[1]]
  next_link <- links[grepl('rel="next"', links, fixed = TRUE)]
  if (length(next_link) == 0) {
    return(NULL)
  }
  sub("^\\s*<([^>]+)>.*$", "\\1", next_link[[1]])
}


hf_match_repo_type <- function(repo_type) {
  match.arg(repo_type, c("model", "dataset", "space"))
}


hf_repo_api_plural <- function(repo_type) {
  switch(repo_type,
    model = "models",
    dataset = "datasets",
    space = "spaces"
  )
}


hf_repo_url_prefix <- function(repo_type) {
  switch(repo_type,
    model = "",
    dataset = "datasets/",
    space = "spaces/"
  )
}


hf_repo_api_url <- function(repo_type, repo_id, ...) {
  parts <- vapply(list(...), utils::URLencode, character(1), reserved = FALSE)
  paste(
    c("https://huggingface.co/api", hf_repo_api_plural(repo_type), repo_id, parts),
    collapse = "/"
  )
}


hf_repo_resolve_url <- function(repo_type, repo_id, revision, filename) {
  paste0(
    "https://huggingface.co/",
    hf_repo_url_prefix(repo_type),
    repo_id,
    "/resolve/",
    utils::URLencode(revision, reserved = TRUE),
    "/",
    utils::URLencode(filename, reserved = FALSE)
  )
}


hf_download_dest <- function(filename, dest = NULL) {
  if (is.null(dest)) {
    ext <- tools::file_ext(filename)
    return(tempfile(fileext = if (nzchar(ext)) paste0(".", ext) else ""))
  }
  if (dir.exists(dest)) {
    return(file.path(dest, basename(filename)))
  }
  dest
}


hf_split_repo_id <- function(repo_id) {
  if (!is.character(repo_id) || length(repo_id) != 1L || is.na(repo_id) ||
      !nzchar(repo_id)) {
    stop("`repo_id` must be a non-empty character scalar.", call. = FALSE)
  }
  parts <- strsplit(repo_id, "/", fixed = TRUE)[[1]]
  if (length(parts) == 1L) {
    return(list(namespace = NULL, name = parts[[1]]))
  }
  if (length(parts) == 2L && all(nzchar(parts))) {
    return(list(namespace = parts[[1]], name = parts[[2]]))
  }
  stop("`repo_id` must be a repository name or 'namespace/name'.", call. = FALSE)
}


hf_confirm_write <- function(confirm, action) {
  if (!identical(confirm, TRUE)) {
    stop(
      "Refusing to ", action, " without `confirm = TRUE`.",
      call. = FALSE
    )
  }
  invisible(TRUE)
}


hf_empty_providers <- function(model_id) {
  tibble::tibble(
    model_id = character(),
    provider = character(),
    status = character(),
    context_length = integer(),
    input_price = numeric(),
    output_price = numeric(),
    is_free = logical(),
    supports_tools = logical(),
    supports_structured_output = logical(),
    first_token_latency_ms = numeric(),
    throughput = numeric(),
    is_model_author = logical()
  )
}
