#' Set Hugging Face API Token
#'
#' Set or update your Hugging Face API token for authentication.
#' The token can be obtained from https://huggingface.co/settings/tokens
#'
#' @param token Character string containing your HF token, or NULL to set interactively.
#'   If NULL, will prompt for token input (not echoed to console).
#' @param store Logical. If TRUE, stores the token in .Renviron for future sessions.
#'   Default: FALSE (token only available for current session).
#'
#' @returns Invisibly returns TRUE if token was set successfully.
#' @export
#'
#' @examples
#' \dontrun{
#' # Set token for current session only
#' hf_set_token("hf_xxxxxxxxxxxxx")
#'
#' # Set token interactively and store permanently
#' hf_set_token(store = TRUE)
#' }
hf_set_token <- function(token = NULL, store = FALSE) {
  
  if (is.null(token)) {
    # Interactive token entry
    if (!interactive()) {
      stop("Token must be provided in non-interactive sessions", call. = FALSE)
    }
    
    token <- readline(prompt = "Enter your Hugging Face token: ")
    token <- trimws(token)
  }
  
  if (!is.character(token) || nchar(token) == 0) {
    stop("Token must be a non-empty character string", call. = FALSE)
  }
  
  # Validate token format (basic check)
  if (!grepl("^hf_[A-Za-z0-9]{20,}$", token)) {
    cli::cli_warn("Token format looks unusual. HF tokens usually start with 'hf_'")
  }
  
  # Set for current session
  Sys.setenv(HUGGING_FACE_HUB_TOKEN = token)
  
  if (store) {
    # Store permanently in .Renviron
    renviron_path <- file.path(Sys.getenv("HOME"), ".Renviron")
    
    # Read existing .Renviron if it exists
    if (file.exists(renviron_path)) {
      renviron_lines <- readLines(renviron_path)
      # Remove any existing HUGGING_FACE_HUB_TOKEN line
      renviron_lines <- renviron_lines[!grepl("^HUGGING_FACE_HUB_TOKEN=", renviron_lines)]
    } else {
      renviron_lines <- character(0)
    }
    
    # Add new token
    renviron_lines <- c(renviron_lines, paste0("HUGGING_FACE_HUB_TOKEN=", token))
    
    writeLines(renviron_lines, renviron_path)
    
    cli::cli_alert_success("Token stored in {.file ~/.Renviron}")
    cli::cli_alert_info("Restart R for the token to be available in new sessions")
  } else {
    cli::cli_alert_success("Token set for current session")
  }
  
  invisible(TRUE)
}


#' Get Current Hugging Face User Information
#'
#' Retrieve information about the currently authenticated user.
#' Requires a valid Hugging Face token to be set.
#'
#' @param token Character string containing your HF token. If NULL, uses the
#'   HUGGING_FACE_HUB_TOKEN environment variable.
#'
#' @returns A tibble with user information including name, email, and organizations.
#' @export
#'
#' @examples
#' \dontrun{
#' # Check current user
#' hf_whoami()
#' }
hf_whoami <- function(token = NULL) {
  
  if (is.null(token)) {
    token <- Sys.getenv("HUGGING_FACE_HUB_TOKEN")
    if (token == "") {
      stop("No token found. Set one with hf_set_token() or pass it as an argument.", 
           call. = FALSE)
    }
  }
  
  # Make request to whoami endpoint
  resp <- httr2::request("https://huggingface.co/api/whoami-v2") |>
    httr2::req_auth_bearer_token(token) |>
    httr2::req_error(body = function(resp) {
      body <- httr2::resp_body_json(resp)
      paste0("API error: ", body$error %||% "Unknown error")
    }) |>
    httr2::req_perform()
  
  user_data <- httr2::resp_body_json(resp)
  
  # Convert to tibble
  tibble::tibble(
    name = user_data$name %||% NA_character_,
    email = user_data$email %||% NA_character_,
    orgs = list(unlist(user_data$orgs) %||% character(0))
  )
}


#' Get Hugging Face API Token
#'
#' Internal function to retrieve the API token from environment or parameter.
#'
#' @param token Character string or NULL
#' @param required Logical. If TRUE, throws error if no token found.
#'
#' @returns Character string with token, or NULL if not found and not required.
#' @keywords internal
hf_get_token <- function(token = NULL, required = FALSE) {
  
  if (is.null(token)) {
    token <- Sys.getenv("HUGGING_FACE_HUB_TOKEN")
    if (token == "") {
      token <- NULL
    }
  }
  
  if (required && is.null(token)) {
    stop("API token required. Set one with hf_set_token() or pass it as an argument.",
         call. = FALSE)
  }
  
  token
}
