# Offline tests for opt-in concurrency (`max_active`) in per-input wrappers.
# Requests run through httr2's mocked responses, so no network is used.

local_offline_routing <- function(env = parent.frame()) {
  testthat::local_mocked_bindings(
    hf_task_provider = function(parsed, token = NULL, task = NULL) "hf-inference",
    hf_sleep = function(seconds) invisible(NULL),
    .env = env
  )
}

# Fails if a wrapper falls back to its one-request-at-a-time helper.
forbid_sequential_path <- function(env = parent.frame()) {
  sequential <- function(...) stop("sequential path used", call. = FALSE)
  testthat::local_mocked_bindings(
    hf_task_request = sequential,
    hf_binary_task_request = sequential,
    hf_binary_generation_request = sequential,
    hf_perform_chat_request = sequential,
    hf_chat = sequential,
    .env = env
  )
}

json_response <- function(body, status = 200L) {
  httr2::response(
    status_code = status,
    headers = list("Content-Type" = "application/json"),
    body = charToRaw(as.character(jsonlite::toJSON(body, auto_unbox = TRUE)))
  )
}

chat_response <- function(content) {
  json_response(list(choices = list(list(
    message = list(role = "assistant", content = content)
  ))))
}


test_that("hf_max_active accepts whole numbers of at least 1", {
  expect_identical(hf_max_active(NULL), 1L)
  expect_identical(hf_max_active(4), 4L)
  for (bad in list(0, -1, 1.5, NA_real_, "2", c(2, 3))) {
    expect_error(hf_max_active(bad), "`max_active` must be")
  }
  expect_error(
    hf_summarize("text", max_active = 0, token = "hf_test"),
    "`max_active` must be"
  )
})

test_that("hf_record_rows keeps or drops inputs without records", {
  records <- list(
    list(list(a = "x"), list(a = "y")),
    list(),
    "not a record list",
    list(NULL, list(a = "z"))
  )
  fields <- list(a = function(r) hf_chr(r$a))

  kept <- hf_record_rows(list(id = 1:4), records, fields)
  expect_equal(kept$id, c(1L, 1L, 2L, 3L, 4L, 4L))
  expect_equal(kept$a, c("x", "y", NA, NA, NA, "z"))

  dropped <- hf_record_rows(list(id = 1:4), records, fields, keep_empty = FALSE)
  expect_equal(dropped$id, c(1L, 1L, 4L, 4L))
  expect_equal(dropped$a, c("x", "y", NA, "z"))

  empty <- hf_record_rows(list(id = integer()), list(), fields)
  expect_equal(nrow(empty), 0)
  expect_type(empty$a, "character")
})

test_that("concurrent task requests keep input order and skip NA inputs", {
  local_offline_routing()
  forbid_sequential_path()
  sent <- character()
  httr2::local_mocked_responses(function(req) {
    input <- req$body$data$inputs
    sent <<- c(sent, input)
    json_response(list(list(summary_text = toupper(input))))
  })

  res <- hf_summarize(c("a", NA, "b", "c"), max_active = 3, token = "hf_test")

  expect_equal(res$text, c("a", NA, "b", "c"))
  expect_equal(res$summary, c("A", NA, "B", "C"))
  expect_setequal(sent, c("a", "b", "c"))
})

test_that("the huggingfaceR.max_active option enables concurrency", {
  local_offline_routing()
  forbid_sequential_path()
  withr::local_options(huggingfaceR.max_active = 2)
  httr2::local_mocked_responses(function(req) {
    json_response(list(list(translation_text = paste0(req$body$data$inputs, "!"))))
  })

  res <- hf_translate(c("one", "two"), token = "hf_test")

  expect_equal(res$translation, c("one!", "two!"))
})

test_that("the first failing input stops the call in either mode", {
  local_offline_routing()
  httr2::local_mocked_responses(function(req) {
    input <- req$body$data$inputs
    if (input %in% c("b", "c")) {
      return(json_response(list(error = paste("bad", input)), status = 422L))
    }
    json_response(list(list(summary_text = input)))
  })

  expect_error(
    hf_summarize(c("a", "b", "c"), max_active = 3, token = "hf_test"),
    "bad b",
    class = "httr2_http_422"
  )
  expect_error(
    hf_summarize(c("a", "b", "c"), token = "hf_test"),
    "bad b",
    class = "httr2_http_422"
  )
})

test_that("hf_fill_mask expands predictions and keeps unsent inputs", {
  testthat::local_mocked_bindings(
    hf_task_request = function(model, inputs, parameters = NULL, token = NULL,
                               endpoint_url = NULL, task = NULL) {
      expect_equal(task, "fill-mask")
      expect_equal(parameters$top_k, 2)
      list(
        list(token_str = "capital", score = 0.9),
        list(token_str = "heart", score = 0.1)
      )
    }
  )

  expect_warning(
    res <- hf_fill_mask(c("Paris is the [MASK].", "no mask", NA), top_k = 2),
    "does not contain mask token"
  )

  expect_named(res, c("text", "token", "score", "filled"))
  expect_equal(res$text, c(rep("Paris is the [MASK].", 2), "no mask", NA))
  expect_equal(res$token, c("capital", "heart", NA, NA))
  expect_equal(res$score, c(0.9, 0.1, NA, NA))
  expect_equal(
    res$filled,
    c("Paris is the capital.", "Paris is the heart.", NA, NA)
  )
})

test_that("hf_classify_zero_shot reads current, legacy, and empty responses", {
  responses <- list(
    list(list(label = "tech", score = 0.8), list(label = "food", score = 0.2)),
    list(sequence = "pizza", labels = list("food", "tech"), scores = list(0.7, 0.3)),
    list()
  )
  calls <- 0L
  testthat::local_mocked_bindings(
    hf_task_request = function(model, inputs, parameters = NULL, token = NULL,
                               endpoint_url = NULL, task = NULL) {
      expect_equal(task, "zero-shot-classification")
      expect_equal(parameters$candidate_labels, c("tech", "food"))
      calls <<- calls + 1L
      responses[[calls]]
    }
  )

  res <- hf_classify_zero_shot(
    c("laptop", "pizza", NA, "unknown"),
    labels = c("tech", "food")
  )

  expect_equal(calls, 3L)
  expect_equal(res$text, rep(c("laptop", "pizza", NA, "unknown"), each = 2))
  expect_equal(res$label, c("tech", "food", "food", "tech", NA, NA, NA, NA))
  expect_equal(res$score, c(0.8, 0.2, 0.7, 0.3, NA, NA, NA, NA))
  expect_error(
    hf_classify_zero_shot("x", labels = character()),
    "At least one label"
  )
})

test_that("concurrent chat requests keep prompt order", {
  local_offline_routing()
  forbid_sequential_path()
  httr2::local_mocked_responses(function(req) {
    chat_response(paste0(req$body$data$messages[[1]]$content, "!"))
  })

  res <- hf_generate(c("a", NA, "b"), max_active = 2, token = "hf_test")

  expect_equal(res$prompt, c("a", NA, "b"))
  expect_equal(res$generated_text, c("a!", NA, "b!"))
})

test_that("hf_extract falls back to json_object once, then reuses it", {
  local_offline_routing()
  formats <- character()
  httr2::local_mocked_responses(function(req) {
    body <- req$body$data
    formats <<- c(formats, body$response_format$type)
    if (identical(body$response_format$type, "json_schema")) {
      return(json_response(
        list(error = list(message = "response format json_schema is not supported")),
        status = 400L
      ))
    }
    text <- body$messages[[2]]$content
    chat_response(as.character(jsonlite::toJSON(list(name = text), auto_unbox = TRUE)))
  })

  res <- hf_extract(
    c("a", NA, "b", "c"),
    c(name = "string"),
    max_active = 3,
    token = "hf_test"
  )

  expect_equal(res$name, c("a", NA, "b", "c"))
  expect_equal(formats, c("json_schema", rep("json_object", 3)))
})

test_that("concurrent image classification keeps image order", {
  local_offline_routing()
  forbid_sequential_path()
  httr2::local_mocked_responses(function(req) {
    byte <- as.integer(req$body$data[[1]])
    json_response(list(
      list(label = paste0("img", byte), score = 0.9),
      list(label = "other", score = 0.1)
    ))
  })

  res <- hf_classify_image(
    list(as.raw(1), NA_character_, as.raw(2)),
    top_k = 1,
    max_active = 2,
    token = "hf_test"
  )

  expect_equal(res$image, c("<raw>", NA, "<raw>"))
  expect_equal(res$label, c("img1", NA, "img2"))
})

test_that("concurrent text-to-image writes each output in order", {
  local_offline_routing()
  forbid_sequential_path()
  httr2::local_mocked_responses(function(req) {
    httr2::response(
      status_code = 200,
      headers = list("Content-Type" = "image/png"),
      body = charToRaw(req$body$data$inputs)
    )
  })
  outs <- vapply(1:3, function(i) tempfile(fileext = ".png"), character(1))

  res <- hf_text_to_image(
    c("one", NA, "three"),
    output = outs,
    max_active = 2,
    token = "hf_test"
  )

  expect_named(res, c("prompt", "path", "content_type", "image"))
  expect_equal(res$path[c(1, 3)], normalizePath(outs[c(1, 3)], winslash = "/", mustWork = FALSE))
  expect_true(is.na(res$path[[2]]))
  expect_false(file.exists(outs[[2]]))
  expect_equal(rawToChar(readBin(outs[[3]], "raw", n = 10)), "three")
  expect_equal(res$content_type, c("image/png", NA, "image/png"))
})

test_that("existing output files are refused before any request is sent", {
  local_offline_routing()
  calls <- 0L
  httr2::local_mocked_responses(function(req) {
    calls <<- calls + 1L
    httr2::response(status_code = 200, body = as.raw(1))
  })
  existing <- tempfile(fileext = ".wav")
  writeBin(as.raw(0), existing)

  expect_error(
    hf_text_to_speech("hello", output = existing, token = "hf_test"),
    "already exists"
  )
  expect_equal(calls, 0L)
})

test_that("concurrent image descriptions keep image order", {
  local_offline_routing()
  forbid_sequential_path()
  httr2::local_mocked_responses(function(req) {
    url <- req$body$data$messages[[1]]$content[[2]]$image_url$url
    chat_response(paste("about", url))
  })
  images <- c("https://example.com/a.png", "https://example.com/b.png")

  res <- hf_describe_image(images, max_active = 2, token = "hf_test")

  expect_equal(res$image, images)
  expect_equal(res$description, paste("about", images))
})

test_that("streaming image descriptions run one image at a time", {
  calls <- list()
  testthat::local_mocked_bindings(hf_chat = function(message, ..., image = NULL) {
    calls[[length(calls) + 1L]] <<- list(...)
    tibble::tibble(content = paste("streamed", image))
  })
  withr::local_options(huggingfaceR.max_active = 4)

  one <- hf_describe_image("a.png", stream = TRUE)
  two <- hf_describe_image(c("a.png", "b.png"), stream = TRUE, max_active = 3)

  expect_equal(one$description, "streamed a.png")
  expect_equal(two$description, c("streamed a.png", "streamed b.png"))
  expect_length(calls, 3)
  expect_true(all(vapply(calls, function(x) isTRUE(x$stream), logical(1))))
})

test_that("generated files are saved before a later failure stops the call", {
  local_offline_routing()
  httr2::local_mocked_responses(function(req) {
    input <- req$body$data$inputs
    if (identical(input, "bad")) {
      return(json_response(list(error = "rejected"), status = 422L))
    }
    httr2::response(
      status_code = 200,
      headers = list("Content-Type" = "image/png"),
      body = charToRaw(input)
    )
  })

  for (max_active in c(1, 3)) {
    outs <- vapply(1:3, function(i) tempfile(fileext = ".png"), character(1))
    expect_error(
      hf_text_to_image(
        c("first", "bad", "third"),
        output = outs,
        max_active = max_active,
        token = "hf_test"
      ),
      "rejected",
      class = "httr2_http_422"
    )
    expect_true(file.exists(outs[[1]]))
    expect_false(file.exists(outs[[2]]))
    # Sequential calls stop at the failure; concurrent calls have already
    # generated the later outputs, so those are kept too.
    expect_identical(file.exists(outs[[3]]), max_active > 1)
  }
})
