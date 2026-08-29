# Offline tests for provider resolution (R/providers.R). Hugging Face routes
# task requests through provider-specific paths, so picking the wrong provider
# produces a 404 even when the model is perfectly healthy. These tests are
# deterministic: the Hub lookup is always mocked.

test_that("hf_normalize_provider_mapping reads the keyed-object shape", {
  # /api/models/{id} returns an object keyed by provider slug.
  mapping <- hf_normalize_provider_mapping(list(
    `hf-inference` = list(
      status = "live",
      providerId = "BAAI/bge-small-en-v1.5",
      task = "feature-extraction"
    )
  ))

  expect_length(mapping, 1)
  expect_equal(mapping[[1]]$provider, "hf-inference")
  expect_equal(mapping[[1]]$status, "live")
  expect_equal(mapping[[1]]$task, "feature-extraction")
})

test_that("hf_normalize_provider_mapping reads the array shape", {
  # /api/models?... returns an array whose records carry a `provider` field.
  mapping <- hf_normalize_provider_mapping(list(
    list(provider = "fal-ai", status = "live", task = "text-to-speech"),
    list(provider = "deepinfra", status = "live", task = "text-to-speech")
  ))

  expect_length(mapping, 2)
  expect_equal(
    vapply(mapping, function(x) x$provider, character(1)),
    c("fal-ai", "deepinfra")
  )
})

test_that("hf_normalize_provider_mapping tolerates empty and malformed input", {
  expect_equal(hf_normalize_provider_mapping(NULL), list())
  expect_equal(hf_normalize_provider_mapping(list()), list())
  # An unnamed record with no provider field carries no routing information.
  expect_equal(hf_normalize_provider_mapping(list(list(status = "live"))), list())
})

test_that("an explicit provider suffix wins without a Hub lookup", {
  called <- FALSE
  testthat::local_mocked_bindings(
    hf_fetch_provider_mapping = function(...) {
      called <<- TRUE
      list()
    }
  )

  expect_equal(hf_resolve_provider("a/b", provider = "together"), "together")
  expect_false(called)
})

test_that("hf-inference is used when it serves the model", {
  testthat::local_mocked_bindings(
    hf_fetch_provider_mapping = function(...) {
      list(
        list(provider = "deepinfra", status = "live", task = "feature-extraction"),
        list(provider = "hf-inference", status = "live", task = "feature-extraction")
      )
    }
  )

  expect_equal(hf_resolve_provider("a/b"), "hf-inference")
})

test_that("resolution refuses to route to a provider it cannot speak to", {
  # This is the FLUX.1-schnell / Kokoro case: live third-party providers exist,
  # but they expose their own request and response formats. Silently routing
  # there would send a request the provider cannot answer, so resolution must
  # return NULL and let the caller raise an actionable error.
  testthat::local_mocked_bindings(
    hf_fetch_provider_mapping = function(...) {
      list(
        list(provider = "together", status = "error", task = "text-to-image"),
        list(provider = "nscale", status = "live", task = "text-to-image"),
        list(provider = "fal-ai", status = "live", task = "text-to-image")
      )
    }
  )

  expect_null(hf_resolve_provider("a/b", task = "text-to-image"))
})

test_that("an unroutable model raises an error naming the real providers", {
  testthat::local_mocked_bindings(
    hf_fetch_provider_mapping = function(...) {
      list(
        list(provider = "nscale", status = "live", task = "text-to-image"),
        list(provider = "fal-ai", status = "live", task = "text-to-image")
      )
    }
  )

  parsed <- hf_parse_model("a/b")
  expect_error(
    hf_task_provider(parsed, task = "text-to-image"),
    "nscale"
  )
  expect_error(
    hf_task_provider(parsed, task = "text-to-image"),
    "hf-inference"
  )
})

test_that("an empty or unknown mapping still attempts the historical route", {
  # Hub metadata can lag reality. Models such as prajjwal1/bert-tiny report an
  # empty inferenceProviderMapping, and refusing client-side would block calls
  # that previously worked. Only a positively-populated mapping without a live
  # hf-inference entry is treated as proof of no route.
  testthat::local_mocked_bindings(
    hf_fetch_provider_mapping = function(...) list()
  )
  expect_equal(hf_resolve_provider("a/b", task = "summarization"), "hf-inference")

  testthat::local_mocked_bindings(
    hf_fetch_provider_mapping = function(...) NULL
  )
  expect_equal(hf_resolve_provider("a/b", task = "summarization"), "hf-inference")
})

test_that("a dead hf-inference entry is not treated as routable", {
  testthat::local_mocked_bindings(
    hf_fetch_provider_mapping = function(...) {
      list(
        list(provider = "hf-inference", status = "error", task = "summarization"),
        list(provider = "together", status = "live", task = "summarization")
      )
    }
  )

  expect_null(hf_resolve_provider("a/b", task = "summarization"))
})

test_that("provider resolution respects the requested task", {
  # hf-inference serves this model, but for a different task, and the mapping is
  # populated - so the request is refused rather than sent to a route that
  # cannot answer it.
  testthat::local_mocked_bindings(
    hf_fetch_provider_mapping = function(...) {
      list(
        list(provider = "hf-inference", status = "live", task = "feature-extraction"),
        list(provider = "fal-ai", status = "live", task = "text-to-speech")
      )
    }
  )

  expect_equal(hf_resolve_provider("a/b", task = "feature-extraction"), "hf-inference")
  expect_null(hf_resolve_provider("a/b", task = "text-to-speech"))
})

test_that("resolution falls back to hf-inference when the Hub is unreachable", {
  # A Hub outage must degrade to the historical route rather than erroring.
  testthat::local_mocked_bindings(
    hf_fetch_provider_mapping = function(...) NULL
  )

  expect_equal(hf_resolve_provider("a/b"), "hf-inference")
})

test_that("a routing policy suffix does not become a URL path segment", {
  testthat::local_mocked_bindings(
    hf_fetch_provider_mapping = function(...) {
      list(list(provider = "hf-inference", status = "live", task = "summarization"))
    }
  )

  parsed <- hf_parse_model("a/b:cheapest")
  provider <- hf_resolve_provider(parsed$model, provider = parsed$provider)

  expect_equal(provider, "hf-inference")
  expect_equal(
    hf_inference_url(parsed$model, provider),
    "https://router.huggingface.co/hf-inference/models/a/b"
  )
})

test_that("hf_req_bill_to only sets the org billing header when configured", {
  req <- httr2::request("https://example.com")

  old <- Sys.getenv("HF_BILL_TO", unset = NA)
  on.exit(
    if (is.na(old)) Sys.unsetenv("HF_BILL_TO") else Sys.setenv(HF_BILL_TO = old),
    add = TRUE
  )

  Sys.unsetenv("HF_BILL_TO")
  expect_null(hf_req_bill_to(req)$headers$`X-HF-Bill-To`)

  # An explicit argument works even when the environment variable is unset.
  expect_equal(hf_req_bill_to(req, bill_to = "explicit-org")$headers$`X-HF-Bill-To`,
               "explicit-org")

  Sys.setenv(HF_BILL_TO = "my-org")
  expect_equal(hf_req_bill_to(req)$headers$`X-HF-Bill-To`, "my-org")
})

test_that("hf_clear_provider_cache empties the session cache", {
  assign("sentinel", list(), envir = hf_provider_cache)
  expect_true(length(ls(envir = hf_provider_cache)) > 0)

  hf_clear_provider_cache()
  expect_equal(length(ls(envir = hf_provider_cache)), 0)
})
