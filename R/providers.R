# Provider resolution for task-style Inference Providers requests.
#
# Hugging Face routes task requests through provider-specific paths
# (https://router.huggingface.co/{provider}/models/{model}). Historically every
# model was served by the first-party "hf-inference" provider, so hardcoding it
# was safe. Since mid-2025 hf-inference focuses on CPU-friendly classic models,
# and many popular models (text-to-image, text-to-speech, large LLMs) are served
# only by third-party providers. Resolving the provider from the Hub's
# `inferenceProviderMapping` keeps requests routable as the ecosystem shifts.


# Routing policies accepted by the router in place of a provider slug. These are
# resolved server-side for chat completions, so they must never be spliced into
# a task-style URL path.
hf_routing_policies <- function() {
  c("auto", "fastest", "cheapest", "preferred")
}


# Session-scoped cache of Hub provider mappings, keyed by model ID.
hf_provider_cache <- new.env(parent = emptyenv())


# Seconds to remember a failed lookup before trying the Hub again.
hf_provider_cache_ttl <- function() 60


# Seconds to remember a successful lookup. Provider availability changes over
# time, so a long-running process (a Shiny app, a scheduled job) must not be
# pinned to a mapping it read hours earlier.
hf_provider_cache_ok_ttl <- function() 900


# Maximum cached mappings. Prevents unbounded growth in processes that touch
# many models; the cache is a latency optimisation, not a store of record.
hf_provider_cache_max <- function() 500


#' Clear the Cached Inference Provider Mappings
#'
#' `huggingfaceR` caches each model's Hugging Face provider mapping for the
#' duration of the R session so that repeated inference calls do not re-query the
#' Hub. Call this function to discard the cache after provider availability
#' changes, or in tests.
#'
#' @returns `NULL`, invisibly.
#' @export
#'
#' @examples
#' hf_clear_provider_cache()
hf_clear_provider_cache <- function() {
  rm(
    list = ls(envir = hf_provider_cache, all.names = TRUE),
    envir = hf_provider_cache
  )
  invisible(NULL)
}


# Normalizes the two shapes the Hub uses for `inferenceProviderMapping`.
#
# The single-model endpoint (/api/models/{id}) returns an object keyed by
# provider slug, while the list endpoint (/api/models?...) returns an array of
# objects carrying a `provider` field. Both are normalized to a list of records
# with `provider`, `status`, `task`, and `provider_id`.
hf_normalize_provider_mapping <- function(mapping) {
  if (is.null(mapping) || length(mapping) == 0) {
    return(list())
  }

  entry <- function(record, fallback_name = NULL) {
    if (!is.list(record)) {
      return(NULL)
    }
    # `[[` is used rather than `$` because `$` partially matches on lists, and
    # the Hub payload contains `providerId` alongside `provider`.
    provider <- record[["provider"]] %||% fallback_name
    if (is.null(provider) || !nzchar(provider)) {
      return(NULL)
    }
    list(
      provider = as.character(provider),
      status = as.character(record[["status"]] %||% NA_character_),
      task = as.character(record[["task"]] %||% NA_character_),
      provider_id = as.character(record[["providerId"]] %||% NA_character_)
    )
  }

  names_present <- names(mapping)
  records <- if (is.null(names_present)) {
    lapply(mapping, entry)
  } else {
    purrr::imap(mapping, function(record, name) entry(record, name))
  }

  purrr::compact(unname(records))
}


# Retrieves a model's provider mapping from the Hub, caching per session.
#
# Network and parsing failures return NULL rather than raising, so that a
# transient Hub outage degrades to the legacy hf-inference route instead of
# breaking inference calls outright.
hf_fetch_provider_mapping <- function(model, token = NULL, refresh = FALSE) {
  key <- paste0(model, "|", if (is.null(token)) "anon" else "auth")
  cached <- hf_provider_cache[[key]]

  if (!refresh && !is.null(cached)) {
    age <- as.numeric(difftime(Sys.time(), cached$time, units = "secs"))
    ttl <- if (isTRUE(cached$ok)) {
      hf_provider_cache_ok_ttl()
    } else {
      hf_provider_cache_ttl()
    }
    if (age < ttl) {
      return(cached$value)
    }
  }

  result <- tryCatch(
    {
      req <- httr2::request("https://huggingface.co/api/models") |>
        httr2::req_url_path_append(model) |>
        httr2::req_url_query(`expand[]` = "inferenceProviderMapping") |>
        httr2::req_retry(max_tries = 2, is_transient = hf_is_transient) |>
        httr2::req_timeout(15)

      if (!is.null(token)) {
        req <- httr2::req_auth_bearer_token(req, token)
      }

      body <- httr2::resp_body_json(
        httr2::req_perform(req),
        simplifyVector = FALSE
      )
      hf_normalize_provider_mapping(body[["inferenceProviderMapping"]])
    },
    error = function(e) NULL
  )

  if (length(ls(envir = hf_provider_cache)) >= hf_provider_cache_max()) {
    hf_clear_provider_cache()
  }

  hf_provider_cache[[key]] <- list(
    ok = !is.null(result),
    value = result,
    time = Sys.time()
  )

  result
}


# Chooses the provider to route a task request through.
#
# Task-style requests in this package use the Hugging Face task contract:
# POST {base}/models/{hf_model_id} with an `{"inputs": ...}` body and a
# task-shaped JSON (or binary) response. Only the first-party `hf-inference`
# provider implements that contract. Third-party providers on the router expose
# their own native routes, payloads, and response shapes (for example Fal AI
# text-to-speech takes `{"text": ...}` at `/{provider_id}` and returns a JSON
# document containing an audio URL that must then be downloaded). Silently
# routing to them would send a request they cannot answer.
#
# So resolution deliberately does NOT pick an arbitrary live provider. It
# returns:
#   * an explicit provider when the caller asked for one,
#   * "hf-inference" when that provider serves the model,
#   * "hf-inference" when the Hub cannot be reached, preserving the historical
#     route rather than failing on metadata, or
#   * NULL when the Hub is reachable and says hf-inference does not serve the
#     model, so the caller can raise an actionable error.
hf_resolve_provider <- function(model, provider = NULL, token = NULL,
                                task = NULL) {
  if (!is.null(provider) && !provider %in% hf_routing_policies()) {
    return(provider)
  }

  mapping <- hf_fetch_provider_mapping(model, token = token)

  # Unknown mapping (Hub unreachable) or an empty one keeps the historical
  # route. Hub metadata can lag reality, so an absent or empty mapping must not
  # block a request that would previously have worked: let the API be the
  # arbiter rather than refusing client-side.
  if (is.null(mapping) || length(mapping) == 0) {
    return("hf-inference")
  }

  hf_inference <- Filter(
    function(x) identical(x$provider, "hf-inference"),
    mapping
  )
  if (!is.null(task)) {
    hf_inference <- Filter(
      function(x) is.na(x$task) || identical(x$task, task),
      hf_inference
    )
  }

  live <- Filter(function(x) identical(x$status, "live"), hf_inference)
  if (length(live) > 0) {
    return("hf-inference")
  }

  # The Hub positively lists providers for this model and hf-inference is not
  # among the live ones. This is the only case we are certain about, so it is
  # the only case where the request is refused.
  NULL
}


# Lists the providers currently serving a model, optionally for one task.
hf_live_providers <- function(model, token = NULL, task = NULL) {
  mapping <- hf_fetch_provider_mapping(model, token = token) %||% list()
  live <- Filter(function(x) identical(x$status, "live"), mapping)
  if (!is.null(task)) {
    live <- Filter(function(x) is.na(x$task) || identical(x$task, task), live)
  }
  unique(vapply(live, function(x) x$provider, character(1)))
}


# Builds the guidance shown when no supported route exists for a model.
hf_no_provider_message <- function(model, task = NULL, providers = character()) {
  task_label <- if (!is.null(task)) paste0(" for task '", task, "'") else ""

  if (length(providers) > 0) {
    return(paste0(
      "Model '", model, "' is not served by the 'hf-inference' provider",
      task_label, ". It is currently served by: ",
      paste(providers, collapse = ", "), ". huggingfaceR's task functions ",
      "speak the Hugging Face task API contract, which only 'hf-inference' ",
      "implements; the other providers expose their own request and response ",
      "formats. Use a model served by 'hf-inference', supply `endpoint_url` ",
      "for a dedicated Inference Endpoint, or call that provider directly. ",
      "See https://huggingface.co/docs/inference-providers."
    ))
  }

  paste0(
    "Model '", model, "' is not currently served by any Hugging Face ",
    "Inference Provider", task_label, ". Run hf_check_inference('", model,
    "') to inspect availability, choose a different model, or supply ",
    "`endpoint_url` for a dedicated Inference Endpoint. See ",
    "https://huggingface.co/docs/inference-providers."
  )
}
