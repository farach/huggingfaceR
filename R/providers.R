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


# Seconds to remember a failed lookup before trying the Hub again. Successful
# lookups are cached for the whole session; failures expire so that a transient
# outage does not pin the session to the fallback route.
hf_provider_cache_ttl <- function() 60


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
    if (isTRUE(cached$ok)) {
      return(cached$value)
    }
    # A failed lookup is only trusted for a short window.
    if (difftime(Sys.time(), cached$time, units = "secs") < hf_provider_cache_ttl()) {
      return(NULL)
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

  hf_provider_cache[[key]] <- list(
    ok = !is.null(result),
    value = result,
    time = Sys.time()
  )

  result
}


# Chooses the provider to route a task request through.
#
# Preference order: an explicit provider, then hf-inference when it is live
# (first-party and covered by the free allowance), then any other live provider,
# then any mapped provider. Returns "hf-inference" when nothing can be resolved
# so behaviour matches earlier releases when the Hub is unreachable.
hf_resolve_provider <- function(model, provider = NULL, token = NULL,
                                task = NULL) {
  if (!is.null(provider) && !provider %in% hf_routing_policies()) {
    return(provider)
  }

  mapping <- hf_fetch_provider_mapping(model, token = token)
  if (length(mapping) == 0) {
    return("hf-inference")
  }

  if (!is.null(task)) {
    matched <- Filter(
      function(x) is.na(x$task) || identical(x$task, task),
      mapping
    )
    if (length(matched) > 0) {
      mapping <- matched
    }
  }

  live <- Filter(function(x) identical(x$status, "live"), mapping)
  pool <- if (length(live) > 0) live else mapping

  slugs <- vapply(pool, function(x) x$provider, character(1))
  if ("hf-inference" %in% slugs) {
    return("hf-inference")
  }

  slugs[[1]]
}


# Builds the guidance shown when a model has no route for the requested task.
hf_no_provider_message <- function(model, task = NULL) {
  paste0(
    "Model '", model, "' is not currently served by any Hugging Face ",
    "Inference Provider",
    if (!is.null(task)) paste0(" for task '", task, "'") else "",
    ". Run hf_check_inference('", model, "') to inspect availability, pick a ",
    "different model, or supply `endpoint_url` for a dedicated Inference ",
    "Endpoint. See https://huggingface.co/docs/inference-providers."
  )
}
