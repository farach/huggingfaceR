#' Set Hugging Face API Token
#'
#' Set or update your Hugging Face API token for authentication. The token is
#' written to the `HF_TOKEN` environment variable used by current Hugging Face
#' tooling, and mirrored to the legacy `HUGGING_FACE_HUB_TOKEN` name.
#'
#' Inference calls require a token with the "Make calls to Inference Providers"
#' permission. See \url{https://huggingface.co/docs/hub/security-tokens} for
#' token setup.
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
  
  # Set for current session. Both names are populated so that code (and Python
  # tooling via reticulate) reading either variable sees the token.
  Sys.setenv(HF_TOKEN = token)
  Sys.setenv(HUGGING_FACE_HUB_TOKEN = token)

  if (store) {
    # Store permanently in .Renviron
    home <- Sys.getenv("HOME", unset = "")
    if (!nzchar(home)) {
      home <- path.expand("~")
    }
    renviron_path <- file.path(home, ".Renviron")

    # Read existing .Renviron if it exists
    if (file.exists(renviron_path)) {
      renviron_lines <- readLines(renviron_path, warn = FALSE)
      # Remove any existing token lines so the file does not accumulate stale
      # values under either variable name.
      renviron_lines <- renviron_lines[
        !grepl("^(HF_TOKEN|HUGGING_FACE_HUB_TOKEN)=", renviron_lines)
      ]
    } else {
      renviron_lines <- character(0)
    }

    # Add new token under the current canonical name
    renviron_lines <- c(renviron_lines, paste0("HF_TOKEN=", token))

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
#'   `HF_TOKEN` environment variable, falling back to the legacy
#'   `HUGGING_FACE_HUB_TOKEN`.
#'
#' @returns A tibble with user, billing, organization, and token-scope metadata.
#' @export
#'
#' @examples
#' \dontrun{
#' # Check current user
#' hf_whoami()
#' }
hf_whoami <- function(token = NULL) {

  if (is.null(token)) {
    token <- hf_token_from_env()
    if (is.null(token)) {
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
  
  tibble::tibble(
    type = user_data$type %||% NA_character_,
    id = user_data$id %||% NA_character_,
    name = user_data$name %||% NA_character_,
    fullname = user_data$fullname %||% NA_character_,
    email = user_data$email %||% NA_character_,
    email_verified = user_data$emailVerified %||% NA,
    can_pay = user_data$canPay %||% NA,
    billing_mode = user_data$billingMode %||% NA_character_,
    is_pro = user_data$isPro %||% NA,
    token_name = user_data$auth$accessToken$displayName %||% NA_character_,
    token_role = user_data$auth$accessToken$role %||% NA_character_,
    token_created_at = user_data$auth$accessToken$createdAt %||% NA_character_,
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
    token <- hf_token_from_env()
  }

  if (required && is.null(token)) {
    stop("API token required. Set one with hf_set_token() or pass it as an argument.",
         call. = FALSE)
  }

  token
}


# Environment variables searched for a token, in priority order.
#
# `HF_TOKEN` is the variable used across current Hugging Face tooling (the `hf`
# CLI, huggingface_hub, Spaces). `HUGGING_FACE_HUB_TOKEN` is the legacy name and
# is still honoured so existing setups keep working.
hf_token_env_vars <- function() {
  c("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
}


# Returns the first token found in the environment, or NULL when none is set.
hf_token_from_env <- function() {
  for (name in hf_token_env_vars()) {
    value <- Sys.getenv(name, unset = "")
    if (nzchar(value)) {
      return(value)
    }
  }
  NULL
}
