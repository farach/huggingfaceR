test_that("hf_transcribe parses text and timestamp chunks", {
  testthat::local_mocked_bindings(
    hf_binary_task_request = function(model, input, token = NULL,
                                      endpoint_url = NULL, content_type = NULL,
                                      query = NULL) {
      expect_equal(query$return_timestamps, "word")
      list(
        text = "hello world",
        chunks = list(list(text = "hello", timestamp = list(0, 1)))
      )
    }
  )

  res <- hf_transcribe("speech.flac", return_timestamps = "word")

  expect_named(res, c("audio", "text", "chunks"))
  expect_equal(res$text, "hello world")
  expect_length(res$chunks[[1]], 1)
})

test_that("hf_text_to_image writes binary image outputs", {
  testthat::local_mocked_bindings(
    hf_binary_generation_request = function(model, inputs, parameters = NULL,
                                            token = NULL, endpoint_url = NULL) {
      expect_equal(parameters$seed, 42)
      list(raw = as.raw(c(1, 2, 3)), content_type = "image/png")
    }
  )

  out <- tempfile(fileext = ".png")
  res <- hf_text_to_image("a red cube", output = out, seed = 42)

  expect_true(file.exists(out))
  expect_equal(readBin(out, "raw", n = 3), as.raw(c(1, 2, 3)))
  expect_named(res, c("prompt", "path", "content_type", "image"))
  expect_equal(res$content_type, "image/png")
})

test_that("hf_text_to_speech refuses accidental overwrite", {
  testthat::local_mocked_bindings(
    hf_binary_generation_request = function(...) {
      list(raw = as.raw(c(1, 2, 3)), content_type = "audio/wav")
    }
  )

  out <- tempfile(fileext = ".wav")
  writeBin(as.raw(0), out)

  expect_error(
    hf_text_to_speech("hello", output = out),
    "already exists"
  )
})

test_that("hf_classify_image parses top labels", {
  testthat::local_mocked_bindings(
    hf_binary_task_request = function(...) {
      list(
        list(label = "cat", score = 0.9),
        list(label = "dog", score = 0.1)
      )
    }
  )

  res <- hf_classify_image("cat.png", top_k = 1)

  expect_named(res, c("image", "label", "score"))
  expect_equal(nrow(res), 1)
  expect_equal(res$label, "cat")
})

test_that("hf_caption_image delegates to vision chat captions", {
  testthat::local_mocked_bindings(
    hf_describe_image = function(image, prompt, model, max_tokens, token = NULL,
                                 endpoint_url = NULL, ...) {
      tibble::tibble(image = image, description = "a cat on a couch")
    }
  )

  res <- hf_caption_image("cat.png")

  expect_named(res, c("image", "caption"))
  expect_equal(res$caption, "a cat on a couch")
})

test_that("hf_detect_objects parses boxes and threshold", {
  testthat::local_mocked_bindings(
    hf_binary_task_request = function(...) {
      list(
        list(
          label = "cat",
          score = 0.95,
          box = list(xmin = 1, ymin = 2, xmax = 3, ymax = 4)
        ),
        list(
          label = "chair",
          score = 0.2,
          box = list(xmin = 5, ymin = 6, xmax = 7, ymax = 8)
        )
      )
    }
  )

  res <- hf_detect_objects("cat.png", threshold = 0.5)

  expect_named(res, c("image", "label", "score", "xmin", "ymin", "xmax", "ymax"))
  expect_equal(nrow(res), 1)
  expect_equal(res$label, "cat")
  expect_equal(res$xmax, 3)
})

test_that("media helpers infer content types and labels", {
  png <- tempfile(fileext = ".png")
  writeBin(as.raw(c(0x89, 0x50, 0x4e, 0x47)), png)

  media <- hf_media_input(png)

  expect_equal(media$content_type, "image/png")
  expect_equal(hf_media_label(as.raw(1:3)), "<raw>")
  expect_error(hf_media_input("missing.png"), "not found")
})
