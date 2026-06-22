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

test_that("hf_inference_body omits deprecated options and drops NULL parameters", {
  body <- hf_inference_body(
    inputs = "hello",
    parameters = list(top_k = 3, unused = NULL)
  )

  expect_named(body, c("inputs", "parameters"))
  expect_equal(body$inputs, "hello")
  expect_named(body$parameters, "top_k")
  expect_equal(body$parameters$top_k, 3)
  expect_null(body$options)
})

test_that("hf_build_request uses provider routing and the shared body", {
  req <- hf_build_request(
    model_id = "org/model:together",
    inputs = "hello",
    parameters = list(top_k = 2),
    token = "test_token",
    wait_for_model = TRUE,
    use_cache = TRUE
  )

  expect_equal(req$url, "https://router.huggingface.co/together/models/org/model")
  expect_equal(req$body$data$inputs, "hello")
  expect_equal(req$body$data$parameters$top_k, 2)
  expect_null(req$body$data$options)
})

test_that("hf_chat_body and request builder preserve chat arguments", {
  messages <- list(list(role = "user", content = "hello"))
  body <- hf_chat_body(
    model = "meta-llama/Test",
    messages = messages,
    max_tokens = 10,
    temperature = 0,
    top_p = NULL,
    seed = 123
  )

  expect_named(body, c("model", "messages", "max_tokens", "temperature", "seed"))
  expect_equal(body$messages, messages)
  expect_equal(body$seed, 123)

  req <- hf_build_chat_request(
    body,
    token = "test_token",
    endpoint_url = "https://dedicated.example/"
  )

  expect_equal(req$url, "https://dedicated.example/v1/chat/completions")
  expect_equal(req$body$data$model, "meta-llama/Test")
  expect_equal(req$body$data$messages, messages)
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

test_that("hf_error_body reads newer message-shaped errors", {
  resp <- httr2::response(
    status_code = 400,
    headers = list("Content-Type" = "application/json"),
    body = charToRaw('{"code":400,"reason":"INVALID_REQUEST_BODY","message":"bad body"}')
  )
  expect_match(hf_error_body()(resp), "bad body")
})
