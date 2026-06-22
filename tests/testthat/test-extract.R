test_that("hf_extract expands a lightweight schema and parses JSON", {
  captured <- NULL
  testthat::local_mocked_bindings(
    hf_perform_chat_request = function(body, token = NULL, endpoint_url = NULL) {
      captured <<- body
      list(choices = list(list(message = list(
        content = '{"name":"Amelie","city":"Paris","score":0.95}'
      ))))
    }
  )

  res <- hf_extract(
    "Amelie is a chef in Paris.",
    c(name = "string", city = "string", score = "number")
  )

  expect_s3_class(res, "tbl_df")
  expect_named(res, c("name", "city", "score"))
  expect_equal(res$name, "Amelie")
  expect_equal(res$city, "Paris")
  expect_equal(res$score, 0.95)
  expect_equal(captured$response_format$type, "json_schema")
  expect_equal(
    names(captured$response_format$json_schema$schema$properties),
    c("name", "city", "score")
  )
  expect_equal(captured$temperature, 0)
})

test_that("hf_extract preserves input order and typed NA rows", {
  testthat::local_mocked_bindings(
    hf_perform_chat_request = function(body, token = NULL, endpoint_url = NULL) {
      value <- body$messages[[2]]$content
      list(choices = list(list(message = list(
        content = paste0('{"label":"', value, '","flag":true}')
      ))))
    }
  )

  res <- hf_extract(c("a", NA, "b"), c(label = "string", flag = "boolean"))

  expect_equal(res$label, c("a", NA, "b"))
  expect_equal(res$flag, c(TRUE, NA, TRUE))
})

test_that("hf_extract returns a typed empty tibble for empty input", {
  res <- hf_extract(
    character(),
    c(name = "string", score = "number", flag = "boolean")
  )

  expect_s3_class(res, "tbl_df")
  expect_named(res, c("name", "score", "flag"))
  expect_equal(nrow(res), 0)
  expect_type(res$name, "character")
  expect_type(res$score, "double")
  expect_type(res$flag, "logical")
})

test_that("hf_extract accepts a full JSON Schema list", {
  captured <- NULL
  schema <- list(
    type = "object",
    properties = list(
      tags = list(type = "array"),
      metadata = list(type = "object")
    ),
    required = c("tags", "metadata")
  )
  testthat::local_mocked_bindings(
    hf_perform_chat_request = function(body, token = NULL, endpoint_url = NULL) {
      captured <<- body$response_format$json_schema
      list(choices = list(list(message = list(
        content = '{"tags":["r","ai"],"metadata":{"source":"note"}}'
      ))))
    }
  )

  res <- hf_extract("R and AI note", schema, strict = FALSE)

  expect_equal(captured$strict, FALSE)
  expect_equal(res$tags[[1]], c("r", "ai"))
  expect_equal(res$metadata[[1]]$source, "note")
})

test_that("hf_extract falls back to json_object when json_schema is unsupported", {
  formats <- character()
  testthat::local_mocked_bindings(
    hf_perform_chat_request = function(body, token = NULL, endpoint_url = NULL) {
      formats <<- c(formats, body$response_format$type)
      if (identical(body$response_format$type, "json_schema")) {
        stop("Model does not support json_schema response format", call. = FALSE)
      }
      list(choices = list(list(message = list(
        content = '{"name":"Amelie","city":"Paris"}'
      ))))
    }
  )

  res <- hf_extract(
    "Amelie is a chef in Paris.",
    c(name = "string", city = "string")
  )

  expect_equal(formats, c("json_schema", "json_object"))
  expect_equal(res$name, "Amelie")
  expect_equal(res$city, "Paris")
})

test_that("hf_extract validates schema inputs", {
  expect_error(hf_extract("x", c(field = "factor")), "field types")
  expect_error(hf_extract("x", list(type = "array")), "object JSON Schema")
})
