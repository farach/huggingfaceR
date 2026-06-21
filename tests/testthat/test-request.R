# Offline tests for the shared request engine (R/request.R). These are
# deterministic and run everywhere (no network, no token required).

test_that("hf_parse_model splits a provider suffix", {
  expect_equal(
    hf_parse_model("meta-llama/Model"),
    list(model = "meta-llama/Model", provider = NULL)
  )
  expect_equal(
    hf_parse_model("meta-llama/Model:together"),
    list(model = "meta-llama/Model", provider = "together")
  )
})

test_that("hf_inference_url routes by default, provider, and dedicated endpoint", {
  expect_equal(
    hf_inference_url("a/b"),
    "https://router.huggingface.co/hf-inference/models/a/b"
  )
  expect_equal(
    hf_inference_url("a/b", provider = "together"),
    "https://router.huggingface.co/together/models/a/b"
  )
  # A dedicated endpoint URL wins and has its trailing slash trimmed.
  expect_equal(
    hf_inference_url("a/b", provider = "together", endpoint_url = "https://x.example/"),
    "https://x.example"
  )
})

test_that("hf_is_transient flags only retryable status codes", {
  expect_true(hf_is_transient(httr2::response(429)))
  expect_true(hf_is_transient(httr2::response(503)))
  expect_false(hf_is_transient(httr2::response(200)))
  expect_false(hf_is_transient(httr2::response(404)))
})

test_that("hf_error_body gives model-specific guidance on a 404", {
  resp <- httr2::response(
    status_code = 404,
    headers = list("Content-Type" = "application/json"),
    body = charToRaw('{"error":"Model not found"}')
  )
  msg <- hf_error_body("foo/bar")(resp)
  expect_match(msg, "foo/bar")
  expect_match(msg, "hf_check_inference")
})

test_that("hf_error_body translates token and rate-limit errors", {
  token_resp <- httr2::response(
    status_code = 401,
    headers = list("Content-Type" = "application/json"),
    body = charToRaw('{"error":"Authorization header is invalid"}')
  )
  expect_match(hf_error_body()(token_resp), "token")

  rate_resp <- httr2::response(
    status_code = 429,
    headers = list("Content-Type" = "application/json"),
    body = charToRaw('{"error":"Rate limit reached"}')
  )
  expect_match(hf_error_body()(rate_resp), "Rate limit")
})
